"""Request parsing (brief 05): free text -> structured filters.

One cached LLM call per distinct request: the payload is the strict-JSON
prompt, hashed with the reader model id by ``salvage.cache.cached_call``, so
asking the same question twice costs nothing. The real call only happens
when the meta-api provider has a key in ``.env``; every failure — no key,
timeout, bad HTTP status, unparseable reply, a reply with no usable filters
— falls back to a deterministic local heuristic parser. That keeps every
test and every offline run working with no keys and no network, per the
00-README rule that model failures return ``None`` and never crash.
"""

from __future__ import annotations

import json
import re
import os
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from salvage import cache
from salvage.config import get_provider

READER_MODEL = get_provider("meta-api").models["reader"]
"""The model that parses a request into filters (Muse Spark 1.3)."""

_LLM_TIMEOUT_SECONDS = 20.0

# --- vocabulary used by the heuristic parser -------------------------------

FILLER_WORDS = frozenset(
    """i need want looking please give my me a an the that has in on at to of
    for with and or any some one something anything got have be is there it""".split()
)
UNIT_WORDS = frozenset(
    """v volt volts a amp amps ampere amperes ma milliamps ohm ohms uf f w watt
    watts hz khz kv""".split()
)
_NUMBER = r"(\d+(?:\.\d+)?)"

VOLTS_RE = re.compile(rf"{_NUMBER}\s*(?:-|–)?\s*(?:volts?\b|v\b)")
"""``12 V motor`` -> 12. Guarded against ``2300KV``: a k/oxide between the
number and the v stops the match."""

CURRENT_RE = re.compile(
    rf"{_NUMBER}(\s*(?:milliamp(?:ere)?s?\b|amperes?\b|amps?\b|ma\b|a\b))"
)
"""``500 mA``, ``2 A``, ``30a`` -> current. Group 2 keeps the unit so
``heuristic_filters`` can normalise milliamp values down to amps."""

INTERFACE_PATTERNS: dict[str, re.Pattern[str]] = {
    "i2c": re.compile(r"\bi2c\b|\btwi\b"),
    "spi": re.compile(r"\bspi\b"),
    "uart": re.compile(r"\buart\b|\bserial\b"),
    "usb": re.compile(r"\busb\b"),
    "pwm": re.compile(r"\bpwm\b"),
    "servo": re.compile(r"\bservo\b"),
    "can": re.compile(r"\bcan[\s-]?bus\b|\bcanbus\b"),
    "one_wire": re.compile(r"\bone[\s-]?wire\b|\b1[\s-]?wire\b"),
    "analog": re.compile(r"\banalogs?\b|\banalogue\b"),
    "esc": re.compile(r"\besc\b"),
    "driver": re.compile(r"\bdrivers?\b|\bh[\s-]?bridge\b"),
}
"""Ordered, so ``parse_filters`` output is always deterministic."""

TOKEN_ALIASES: dict[str, tuple[str, ...]] = {
    "battery": ("battery", "cell", "lipo", "li ion", "liion", "nimh", "pack", "18650", "21700"),
    "cell": ("cell", "lipo", "li ion", "liion", "battery", "nimh", "pack", "18650", "21700"),
    "cap": ("capacitor", "cap"),
    "sensor": ("sensor", "imu", "accelerometer", "gyro", "gyroscope", "detector", "probe"),
    "regulator": ("regulator", "buck", "boost", "ldo"),
    "motor": ("motor",),
}
"""Synonyms a request may use for one type token, OR-grouped in SQL."""

INTERFACE_ALIASES: dict[str, tuple[str, ...]] = {
    "driver": ("driver", "esc", "h bridge"),
}
"""A request for ``a driver`` is satisfied by an ESC or an h-bridge too."""


class Filters(BaseModel):
    """Structured filters parsed out of one free-text request.

    ``type`` holds the leftover part-family words after voltage, current and
    interface phrases are removed (``"motor"``, ``"dc motor"``). ``interface``
    holds space-joined tokens from ``INTERFACE_PATTERNS``.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = ""
    voltage: float | None = Field(default=None, gt=0.0)
    current: float | None = Field(default=None, gt=0.0)
    interface: str = ""

    @property
    def type_tokens(self) -> list[str]:
        return _clean_tokens(self.type)

    @property
    def interface_tokens(self) -> list[str]:
        return _clean_tokens(self.interface)

    @classmethod
    def try_from(cls, raw: Mapping[str, object]) -> "Filters | None":
        """Validate an LLM-style dict; None if it is not a usable filter set.

        A reply that yields no type and no interface tokens is treated as
        unusable: the caller falls back to the heuristic parser instead of
        answering an empty filter set.
        """
        if not isinstance(raw, dict):
            return None
        data: dict[str, object] = {}
        for key in ("type", "interface"):
            value = raw.get(key)
            data[key] = " ".join(_clean_tokens(str(value))) if value is not None else ""
        for key in ("voltage", "current"):
            value = raw.get(key)
            if isinstance(value, (int, float)) and value > 0:
                data[key] = float(value)
        try:
            parsed = cls.model_validate(data)
        except ValidationError:
            return None
        if not parsed.type_tokens and not parsed.interface_tokens:
            return None
        return parsed


def _clean_tokens(text: str) -> list[str]:
    """Lowercase words with numbers and unit/filler words removed."""
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    return [
        word
        for word in words
        if word not in FILLER_WORDS and word not in UNIT_WORDS and not word.isdigit()
    ]


def heuristic_filters(text: str) -> Filters:
    """Parse a request with regexes only. No model, no cache, no network."""
    t = text.lower()
    voltage_match = VOLTS_RE.search(t)
    current_match = CURRENT_RE.search(t)
    t = VOLTS_RE.sub(" ", t)
    t = CURRENT_RE.sub(" ", t)
    current: float | None = None
    if current_match:
        current = float(current_match.group(1))
        unit = current_match.group(2).strip().lower()
        if unit.startswith("milli") or unit == "ma":
            current /= 1000.0
    interface = " ".join(
        token for token, pattern in INTERFACE_PATTERNS.items() if pattern.search(t)
    )
    for pattern in INTERFACE_PATTERNS.values():
        t = pattern.sub(" ", t)
    type = " ".join(_clean_tokens(t))
    return Filters(
        type=type,
        voltage=float(voltage_match.group(1)) if voltage_match else None,
        current=current,
        interface=interface,
    )


def token_variants(token: str) -> tuple[str, ...]:
    """Every search-text substring that satisfies a type token."""
    return TOKEN_ALIASES.get(token, (token,))


def interface_variants(token: str) -> tuple[str, ...]:
    """Every search-text substring that satisfies an interface token."""
    return INTERFACE_ALIASES.get(token, (token,))


def _llm_parse(payload: bytes) -> dict | None:
    """The one real model call behind ``parse_filters``.

    Returns the strict-JSON filters dict, or None on every failure: no key,
    no network, timeout, non-200 status, or an unparseable reply.
    """
    try:
        request = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    provider = get_provider("meta-api")
    if not provider.is_available():
        return None  # offline: no key, so no call and no cache write
    try:
        import httpx

        response = httpx.post(
            provider.base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {provider.api_key()}"},
            json={
                "model": READER_MODEL,
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You parse a hardware parts request into strict JSON. "
                            'Reply with only {\"type\": str, \"voltage\": number|null, '
                            '"current": number|null, "interface": str}. '
                            "type: space-separated part-family words without units. "
                            "interface: space-separated tokens from i2c spi uart usb "
                            "pwm can one_wire analog servo esc driver, or \"\"."
                        ),
                    },
                    {"role": "user", "content": str(request.get("text", ""))},
                ],
            },
            timeout=_LLM_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            return None
        content = response.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", str(content), re.DOTALL)
        return json.loads(match.group(0)) if match else None
    except Exception:  # noqa: BLE001 - a failed model call returns None, never crashes
        return None


def parse_filters(text: str, *, cache_dir: str | os.PathLike[str] | None = None) -> Filters:
    """Structured filters for one request via the one cached LLM call.

    The payload embeds the task name, so hashes never collide with other
    modules' calls to the same reader model. On any LLM failure the
    deterministic ``heuristic_filters`` result is used instead.
    """
    payload = json.dumps(
        {"task": "parse-parts-request", "v": 1, "text": text}, sort_keys=True
    ).encode("utf-8")
    raw = cache.cached_call(READER_MODEL, payload, _llm_parse, cache_dir=cache_dir)
    filters = Filters.try_from(raw) if isinstance(raw, dict) else None
    return filters if filters is not None else heuristic_filters(text)
