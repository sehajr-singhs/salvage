"""Brief 03: the multi-model part readers.

:func:`read_part` asks every enabled provider to identify one Part crop
and returns one Reading per reliably-parsed answer.

Rules kept from 00-README and the brief, and how they are met here:

- Providers come from ``config.PROVIDERS``; they are OpenAI-compatible
  HTTP endpoints, one prompt (``readers/prompt.READ_PROMPT``) for all.
- Providers run in parallel under asyncio, each capped with a 30 s
  timeout (both the per-call HTTP timeout and an asyncio :mod:`wait_for`
  outer bound).
- Every call goes through ``cache.cached_call``, keyed by the
  provider-qualified model id and the sha256 of the exact request
  payload, so rereading the same crop with the same models costs
  nothing. Provider keys live only in the auth header: they are never
  part of the payload, the digest, or the cache file.
- The output is validated into a ``Reading`` with pydantic. A parse
  failure is retried once — with a corrective note appended, so the
  retry payload hashes differently and cannot be served from the failed
  attempt's cache entry — and then the reading is dropped.
- A failure of any kind (network error, timeout, HTTP error, garbage
  answer, missing crop) contributes no vote and never crashes: the
  caller simply gets one fewer reading, and the gate abstains rather
  than guessing.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from pathlib import Path
from typing import Any, Sequence

import httpx
from pydantic import ValidationError

from salvage.cache import cached_call
from salvage.config import PROVIDERS, Provider
from salvage.readers.prompt import READ_PROMPT, RETRY_NOTE
from salvage.schemas import Condition, Part, Reading

# Brief: "Run the providers in parallel with asyncio and a 30 s timeout each."
TIMEOUT_SECONDS = 30.0
MAX_PARSE_RETRIES = 1

_GARBAGE_BODY = {"error": "unparseable"}  # a body with no choices at all


def _mime_type(image_bytes: bytes) -> str:
    """The data-URL mime for the crop bytes (sniffed, not guessed)."""
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8"):
        return "image/jpeg"
    return "image/png"


def _cache_model_id(provider: Provider) -> str:
    """Provider-qualified cache/model id.

    Two providers can serve the same underlying model string; the
    provider name prefixes it so their responses can never share a cache
    entry. The key still includes both the model id and the payload, as
    the README requires.
    """
    return f"{provider.name}:{provider.model_id}"


def request_payload(
    provider: Provider, image_bytes: bytes, *, retry: bool = False
) -> bytes:
    """The exact chat-completions request body, serialized for the cache.

    Deterministic (sorted keys, fixed separators) so the same crop asked
    of the same provider hashes to the same cache entry every run. The
    API key is deliberately not part of the request body.
    """
    data_url = (
        "data:"
        + _mime_type(image_bytes)
        + ";base64,"
        + base64.b64encode(image_bytes).decode("ascii")
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": READ_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    if retry:
        messages.append({"role": "user", "content": RETRY_NOTE})
    body = {
        "model": provider.model_id,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _post(provider: Provider, request: dict[str, Any]) -> dict[str, Any] | None:
    """One OpenAI-compatible chat-completions call, or None on any failure."""
    url = provider.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {provider.api_key()}",
        "Content-Type": "application/json",
    }
    try:
        response = httpx.post(url, json=request, headers=headers, timeout=30.0)
        response.raise_for_status()
        body = response.json()
    except Exception:  # noqa: BLE001 - README: a failed model call never crashes
        return None
    return body if isinstance(body, dict) else None


def _content(body: dict[str, Any]) -> str | None:
    """The text content of the first reply, whatever shape it arrives in."""
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if content is None:
        return None
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts = [
            str(item.get("text"))
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        text = "".join(parts)
        return text or None
    return None


def _extract_json(content: str) -> Any | None:
    """The JSON object in a reply, tolerating a markdown fence."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _optional_text(value: Any) -> str | None:
    """A marking as the model sent it, cleaned; None when nothing is legible."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def parse_reading(
    provider: Provider, part_id: str, body: dict[str, Any]
) -> Reading | None:
    """Validate one call body into a Reading, or None (a parse failure).

    ``part_id`` and ``model_id`` come from the caller, never from the
    model: a reader cannot rename the part it was asked about or pass
    itself off as another reader.
    """
    content = _content(body)
    if content is None:
        return None
    data = _extract_json(content)
    if not isinstance(data, dict):
        return None

    part_type = data.get("part_type")
    if isinstance(part_type, str):
        part_type = "_".join(part_type.split()).lower()
    if not isinstance(part_type, str) or not part_type:
        return None
    condition = data.get("condition")
    if not isinstance(condition, str) or condition.strip().lower() not in {
        c.value for c in Condition
    }:
        return None
    evidence = data.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        return None
    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        return None

    fields = {
        "part_type": part_type,
        "part_number": _optional_text(data.get("part_number")),
        "key_rating": _optional_text(data.get("key_rating")),
        "condition": condition.strip().lower(),
        "confidence": float(confidence),
        "evidence": evidence.strip(),
    }
    try:
        return Reading(part_id=part_id, model_id=provider.model_id, **fields)
    except ValidationError:
        # most likely a confidence outside [0, 1]: an answer we cannot
        # calibrate is an answer we refuse, with a retry, not a shrug
        return None


async def _read_via(
    provider: Provider,
    part: Part,
    image_bytes: bytes,
    cache_dir: str | Path | None,
    timeout: float,
) -> Reading | None:
    """One provider's full chain: cached HTTP call, parse, one retry."""
    attempts = (
        request_payload(provider, image_bytes, retry=False),
        request_payload(provider, image_bytes, retry=True),
    )

    def call(payload: bytes) -> dict[str, Any] | None:
        return cached_call(
            _cache_model_id(provider),
            payload,
            lambda data: _post(provider, json.loads(data)),
            cache_dir=cache_dir,
        )

    try:
        body = await asyncio.wait_for(asyncio.to_thread(call, attempts[0]), timeout)
    except Exception:  # noqa: BLE001 - timeout/transport error = no vote
        return None
    if body is None:
        return None  # a failed call is not parse-failure; no retry
    reading = parse_reading(provider, part.id, body)
    if reading is not None:
        return reading

    # Exactly one retry, and only on a parse failure. The corrective
    # note changes the payload, so the retry cannot be served from the
    # failed attempt's cache entry.
    try:
        body = await asyncio.wait_for(asyncio.to_thread(call, attempts[1]), timeout)
    except Exception:  # noqa: BLE001
        return None
    return parse_reading(provider, part.id, body) if body is not None else None


async def _read_all(
    providers: Sequence[Provider],
    part: Part,
    image_bytes: bytes,
    cache_dir: str | Path | None,
    timeout: float,
) -> list[Reading]:
    votes = await asyncio.gather(
        *(
            _read_via(provider, part, image_bytes, cache_dir, timeout)
            for provider in providers
        )
    )
    return [vote for vote in votes if isinstance(vote, Reading)]


def read_part(
    part: Part,
    providers: Sequence[Provider] | None = None,
    *,
    cache_dir: str | Path | None = None,
    timeout: float = TIMEOUT_SECONDS,
) -> list[Reading]:
    """One Reading per provider that returns a parseable, validated answer.

    ``providers`` defaults to the configured providers that are enabled
    and keyed right now; explicitly passed providers are used as given
    still filtered to the ones that are currently available. With no
    key anywhere the honest answer is zero votes — the offline mock
    reader (``salvage.web.pipeline.offline_mock_readings``) owns the
    keyless path, per briefs 06 and 08.
    """
    chosen = (
        [p for p in providers if p.is_available()]
        if providers is not None
        else [p for p in PROVIDERS if p.is_available()]
    )
    if not chosen:
        return []
    try:
        image_bytes = Path(part.crop_path).read_bytes()
    except OSError:
        # no crop on disk: nothing has actually looked at this part,
        # so the reader votes nothing rather than inventing something
        return []
    try:
        return asyncio.run(_read_all(chosen, part, image_bytes, cache_dir, timeout))
    except Exception:  # noqa: BLE001 - README: a failed reader never crashes
        return []
