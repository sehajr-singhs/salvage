"""Tests for brief 03: multi-model part readers.

All offline. Providers are fake ``salvage.config.Provider`` objects whose
keys are monkeypatched into the environment and whose HTTP is mocked at
``httpx.post`` (the same trick the 02 backend tests use), so pytest never
touches a real endpoint and no real key ever exists. A short outer
timeout is passed to :func:`salvage.readers.read_part` purely to keep the
tests fast; the production path is the 30 s cap the brief specifies.
"""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlparse

import httpx
import pytest
from PIL import Image

from salvage.config import Provider
from salvage.readers import reader as reader_module
from salvage.readers.agreement import agreement, part_number_family
from salvage.readers.prompt import READ_PROMPT, RETRY_NOTE
from salvage.readers.reader import parse_reading, read_part, request_payload
from salvage.schemas import Part, Reading

# All provider key env vars, cleared defensively so the lazy-export gate
# and the no-provider default paths see a genuinely keyless environment
# even if a stray var leaks in from somewhere.
KEY_ENV_VARS = (
    "META_API_KEY",
    "BASETEN_API_KEY",
    "FIREWORKS_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "NVIDIA_NIM_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
)

SENTINEL_TIMEOUT = object()  # the provider hangs on the network
SENTINEL_HTTP_500 = object()  # the provider answers 500
SENTINEL_GARBAGE = (
    "This crop is too small for me to be sure, so I will not claim an id."
)


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def crop_png(tmp_path: Any) -> Any:
    path = tmp_path / "crop.png"
    image = Image.new("RGB", (64, 48), (120, 120, 120))
    image.save(path)
    return str(path)


@pytest.fixture()
def part(crop_png: str) -> Part:
    return Part(
        id="p_001",
        device_id="dev001",
        crop_path=crop_png,
        bbox=(0.0, 0.0, 10.0, 10.0),
        mask_path="mask.png",
    )


def provider(name: str, model: str | None = None) -> Provider:
    return Provider(
        name=name,
        base_url=f"https://{name}.readers.test/v1",
        api_key_env=f"{name.upper()}_KEY",
        model_id=model or f"reader-{name}",
    )


def agree_reply(**overrides: Any) -> str:
    """A parseable Reading JSON: a legible marking, honestly quoted."""
    data = {
        "part_type": "bldc_motor",
        "part_number": "2205",
        "key_rating": "2300KV",
        "condition": "works",
        "confidence": 0.87,
        "evidence": 'the rotor can is stencilled "2205" over "2300KV"',
    }
    data.update(overrides)
    return json.dumps(data)


def _host_of(url: str) -> str:
    return urlparse(url).hostname.split(".", 1)[0]


def install_fake_http(
    monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]], behavior: dict[str, Any]
) -> None:
    """Route ``httpx.post`` to per-provider scripted behavior."""

    def fake_post(
        url: str,
        json: Any = None,  # noqa: A002 - matches httpx.post's own signature
        headers: dict[str, str] | None = None,
        timeout: Any = None,
        **kwargs: Any,
    ) -> httpx.Response:
        host = _host_of(url)
        calls.append({"host": host, "request": json, "headers": headers})
        assert timeout is not None  # every network call carries a timeout
        assert headers is not None and headers["Authorization"].startswith("Bearer ")
        scripted = behavior[host]
        if scripted is SENTINEL_TIMEOUT:
            raise httpx.ReadTimeout("simulated hung endpoint", request=httpx.Request("POST", url))
        if scripted is SENTINEL_HTTP_500:
            return httpx.Response(500, request=httpx.Request("POST", url))
        reply = scripted if isinstance(scripted, str) else scripted()
        body = {"choices": [{"message": {"content": reply}}]}
        return httpx.Response(
            200, json=body, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx, "post", fake_post)


# --- the brief's done-when scenario ------------------------------------------


def test_done_when_three_agree_one_times_out_one_returns_garbage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, part: Part
) -> None:
    """3 agreeing + 1 timing out + 1 garbage -> 3 Readings, agreement 1.0."""
    specs = [
        provider("alpha"),
        provider("bravo"),
        provider("charlie"),
        provider("delta"),
        provider("echo"),
    ]
    for spec in specs:
        monkeypatch.setenv(spec.api_key_env, "fake-test-key")
    calls: list[dict[str, Any]] = []
    install_fake_http(
        monkeypatch,
        calls,
        {
            "alpha": agree_reply(),
            "bravo": "```json\n" + agree_reply(part_number="2205") + "\n```",
            "charlie": agree_reply(condition="works", confidence=1.0),
            "delta": SENTINEL_TIMEOUT,
            "echo": SENTINEL_GARBAGE,
        },
    )

    readings = read_part(part, specs, cache_dir=tmp_path / "cache", timeout=2.0)

    assert len(readings) == 3
    assert {r.model_id for r in readings} == {"reader-alpha", "reader-bravo", "reader-charlie"}
    assert all(r.part_id == "p_001" for r in readings)
    assert all(r.part_type == "bldc_motor" for r in readings)
    assert all(r.part_number == "2205" for r in readings)
    assert agreement(readings) == 1.0

    # attempt profile: agree x1 each, timeout x1 (no retry on timeouts),
    # garbage x2 (one parse retry, then dropped)
    counts: dict[str, int] = {}
    for call in calls:
        counts[call["host"]] = counts.get(call["host"], 0) + 1
    assert counts == {"alpha": 1, "bravo": 1, "charlie": 1, "delta": 1, "echo": 2}
    # the garbage retry carried the corrective note and still failed twice
    second_junk = [c for c in calls if c["host"] == "echo"][1]["request"]
    note_texts = [
        m["content"] for m in second_junk["messages"] if isinstance(m["content"], str)
    ]
    assert RETRY_NOTE in note_texts


def test_readers_are_asked_for_an_image_not_a_guess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, part: Part
) -> None:
    """Every request is the shared prompt plus the crop as a data URL, no key."""
    spec = provider("solo")
    monkeypatch.setenv(spec.api_key_env, "fake-test-key")
    calls: list[dict[str, Any]] = []
    install_fake_http(monkeypatch, calls, {"solo": agree_reply()})

    readings = read_part(part, [spec], cache_dir=tmp_path / "cache", timeout=2.0)
    assert len(readings) == 1
    request = calls[0]["request"]
    assert request["model"] == "reader-solo"
    assert json.dumps(request).count("fake-test-key") == 0  # keys stay out of the payload
    first_message = request["messages"][0]["content"]
    assert first_message[0]["text"] == READ_PROMPT
    image_part = first_message[1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


# --- caching -----------------------------------------------------------------


def test_rereading_the_same_crop_costs_no_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, part: Part
) -> None:
    spec = provider("solo")
    monkeypatch.setenv(spec.api_key_env, "fake-test-key")
    calls: list[dict[str, Any]] = []
    install_fake_http(monkeypatch, calls, {"solo": agree_reply()})
    cache_dir = tmp_path / "cache"

    first = read_part(part, [spec], cache_dir=cache_dir, timeout=2.0)
    assert len(first) == 1
    assert len(calls) == 1

    install_fake_http(monkeypatch, calls, {"solo": SENTINEL_HTTP_500})
    second = read_part(part, [spec], cache_dir=cache_dir, timeout=2.0)
    # cache hit: the network was never touched again (no new call), and the
    # rehydrated readings parse identically
    assert len(calls) == 1
    assert first == second


def test_cache_keys_never_collide_across_providers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, part: Part
) -> None:
    """Two providers serving the same model string still cache separately."""
    one = provider("one", model="same-model")
    two = provider("two", model="same-model")
    for spec in (one, two):
        monkeypatch.setenv(spec.api_key_env, "fake-test-key")
    calls: list[dict[str, Any]] = []
    install_fake_http(
        monkeypatch,
        calls,
        {"one": agree_reply(evidence="provider one saw it"), "two": agree_reply(evidence="provider two saw it")},
    )

    readings = read_part(part, [one, two], cache_dir=tmp_path / "cache", timeout=2.0)
    assert len(readings) == 2
    assert {r.evidence for r in readings} == {"provider one saw it", "provider two saw it"}
    assert len(calls) == 2


# --- failures and retries ----------------------------------------------------


def test_parse_failure_is_retried_once_then_dropped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, part: Part
) -> None:
    spec = provider("junk")
    monkeypatch.setenv(spec.api_key_env, "fake-test-key")
    calls: list[dict[str, Any]] = []
    install_fake_http(monkeypatch, calls, {"junk": SENTINEL_GARBAGE})

    assert read_part(part, [spec], cache_dir=tmp_path / "cache", timeout=2.0) == []
    assert len(calls) == 2  # exactly one retry, then the vote is dropped


def test_parse_retry_recovers_a_valid_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, part: Part
) -> None:
    spec = provider("moody")
    monkeypatch.setenv(spec.api_key_env, "fake-test-key")
    calls: list[dict[str, Any]] = []
    # script a two-shot provider: prose first, valid JSON on the retry
    attempt = {"n": 0}

    def fake_post(url, json=None, headers=None, timeout=None, **kwargs):  # noqa: A002
        calls.append({"host": _host_of(url), "request": json})
        assert timeout is not None
        attempt["n"] += 1
        reply = SENTINEL_GARBAGE if attempt["n"] == 1 else agree_reply()
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": reply}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    readings = read_part(part, [spec], cache_dir=tmp_path / "cache", timeout=2.0)
    assert len(readings) == 1
    assert readings[0].model_id == "reader-moody"
    assert len(calls) == 2


def test_http_error_is_dropped_without_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, part: Part
) -> None:
    spec = provider("broken")
    monkeypatch.setenv(spec.api_key_env, "fake-test-key")
    calls: list[dict[str, Any]] = []
    install_fake_http(monkeypatch, calls, {"broken": SENTINEL_HTTP_500})

    readings = read_part(part, [spec], cache_dir=tmp_path / "cache", timeout=2.0)
    assert readings == []  # a failed reader votes nothing, never crashes
    assert len(calls) == 1  # network errors are not parse failures: no retry


def test_asyncio_timeout_drops_a_slow_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, part: Part
) -> None:
    """The provider answers *eventually*; the outer asyncio cap still drops it."""
    spec = provider("sloth")
    monkeypatch.setenv(spec.api_key_env, "fake-test-key")

    def fake_post(url, json=None, headers=None, timeout=None, **kwargs):  # noqa: A002
        time.sleep(1.0)  # too slow for the 0.05 s outer timeout
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": agree_reply()}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    readings = read_part(part, [spec], cache_dir=tmp_path / "cache", timeout=0.05)
    assert readings == []


def test_slow_provider_never_blocks_the_parallel_providers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, part: Part
) -> None:
    """One provider hanging 1 s does not stop the fast ones voting at once."""
    slow = provider("sloth", model="m")
    fast = provider("hare", model="m")
    for spec in (slow, fast):
        monkeypatch.setenv(spec.api_key_env, "fake-test-key")

    def fake_post(url, json=None, headers=None, timeout=None, **kwargs):  # noqa: A002
        if _host_of(url) == "sloth":
            time.sleep(30.0)  # far beyond the outer timeout; never returned to
            raise AssertionError("the slow provider should have been abandoned")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": agree_reply()}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    readings = read_part(part, [slow, fast], cache_dir=tmp_path / "cache", timeout=0.05)
    assert [r.model_id for r in readings] == ["m"]


def test_missing_crop_votes_nothing(part: Part, tmp_path: Any) -> None:
    part = part.model_copy(update={"crop_path": str(tmp_path / "nope.png")})
    assert read_part(part, [provider("any")], cache_dir=tmp_path / "cache") == []


def test_unavailable_providers_vote_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, part: Part
) -> None:
    spec = provider("ghost")
    monkeypatch.delenv(spec.api_key_env, raising=False)
    assert read_part(part, [spec], cache_dir=tmp_path / "cache") == []


def test_no_providers_at_all_votes_nothing(tmp_path: Any, part: Part) -> None:
    assert read_part(part, [], cache_dir=tmp_path / "cache") == []


def test_default_providers_are_keyless_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, part: Part
) -> None:
    """No keys anywhere: the default provider set is empty and nothing is sent."""
    calls: list[dict[str, Any]] = []
    install_fake_http(monkeypatch, calls, {})
    assert read_part(part, cache_dir=tmp_path / "cache") == []
    assert calls == []


def test_part_id_and_model_id_come_from_the_caller(
    tmp_path: Any, part: Part
) -> None:
    """A reader cannot rename the part it was asked about or pose as another."""
    spec = provider("solo")
    body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "part_id": "p_999",
                            "model_id": "someone-else",
                            "part_type": "bldc_motor",
                            "part_number": "2205",
                            "key_rating": None,
                            "condition": "works",
                            "confidence": 0.9,
                            "evidence": 'marking "2205" legible',
                            "extra_key": "ignored",
                        }
                    )
                }
            }
        ]
    }
    reading = parse_reading(spec, part.id, body)
    assert reading is not None
    assert reading.part_id == "p_001"
    assert reading.model_id == "reader-solo"
    assert reading.key_rating is None


# --- parse_reading, field level ----------------------------------------------


def body_with(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


def test_parse_tolerates_fenced_json_and_extra_keys() -> None:
    spec = provider("solo")
    fenced = (
        "Here is my answer:\n```json\n"
        + json.dumps(
            {
                "part_type": "Electrolytic Capacitor",
                "part_number": "470uF 25V",
                "condition": "unknown",
                "confidence": 0.7,
                "evidence": "the can is marked 470uF 25V",
                "unrequested": True,
            }
        )
        + "\n```"
    )
    reading = parse_reading(spec, "p_001", body_with(fenced))
    assert reading is not None
    assert reading.part_type == "electrolytic_capacitor"
    assert reading.part_number == "470uF 25V"
    assert reading.condition.value == "unknown"
    assert reading.key_rating is None


def test_parse_normalizes_case_and_spacing_of_the_type() -> None:
    reading = parse_reading(provider("s"), "p", body_with(agree_reply(part_type="BLDC  Motor")))
    assert reading is not None and reading.part_type == "bldc_motor"
    reading = parse_reading(provider("s"), "p", body_with(agree_reply(part_number="IRF 540-N")))
    assert reading is not None and reading.part_number == "IRF 540-N"


def test_confidence_must_be_a_real_0_1_number() -> None:
    assert parse_reading(provider("s"), "p", body_with(agree_reply(confidence=87.0))) is None
    assert parse_reading(provider("s"), "p", body_with(agree_reply(confidence="high"))) is None


def test_missing_or_legible_evidence_is_refused() -> None:
    assert parse_reading(provider("s"), "p", body_with(agree_reply(evidence="  "))) is None
    assert parse_reading(provider("s"), "p", body_with(agree_reply(evidence=42))) is None


def test_garbage_body_shapes_are_parse_failures() -> None:
    assert parse_reading(provider("s"), "p", body_with("not json at all")) is None
    assert parse_reading(provider("s"), "p", body_with("prefix {\"confidence\": 0.2, \"evidence\": \"x\", \"condition\": \"unknown\", \"part_type\": \"motor\"} suffix")) is not None
    assert parse_reading(provider("s"), "p", {"error": "totally unparseable"}) is None
    assert parse_reading(provider("s"), "p", {"choices": [{"message": {"content": None}}]}) is None
    assert parse_reading(provider("s"), "p", {"error": "rate limited"}) is None


def test_the_call_body_is_stable_for_the_cache() -> None:
    spec = provider("solo")
    image = b"\x89PNG\r\n\x1a\nfake"
    first = request_payload(spec, image)
    second = request_payload(spec, image)
    assert first == second
    assert request_payload(spec, image, retry=True) != first  # retry hashes apart
    assert b"fake-test-key" not in first  # keys never enter payload or cache


# --- agreement ---------------------------------------------------------------


def _reading(model: str, ptype: str, number: str | None = None) -> Reading:
    return Reading(
        part_id="p",
        model_id=model,
        part_type=ptype,
        part_number=number,
        condition="unknown",
        confidence=0.5,
        evidence="e",
    )


def test_agreement_empty_is_zero() -> None:
    assert agreement([]) == 0.0


def test_agreement_counts_unanimous_family_agreement() -> None:
    readings = [
        _reading("m1", "bldc_motor", "2205"),
        _reading("m2", "bldc_motor", "2205"),
        _reading("m3", "bldc_motor", "2205"),
    ]
    assert agreement(readings) == 1.0


def test_agreement_splits_on_part_type() -> None:
    readings = [
        _reading("m1", "bldc_motor", "2205"),
        _reading("m2", "bldc_motor", "2205"),
        _reading("m3", "dc_motor", "2205"),
    ]
    assert agreement(readings) == pytest.approx(2 / 3)


def test_agreement_counts_a_number_family_disagreement() -> None:
    readings = [
        _reading("m1", "bldc_motor", "IRF540"),
        _reading("m2", "bldc_motor", "IRF540"),
        _reading("m3", "bldc_motor", "IRF540N"),
    ]
    # conservative families: IRF540 and IRF540N are different families
    assert agreement(readings) == pytest.approx(2 / 3)


def test_agreement_when_nobody_can_read_a_marking() -> None:
    readings = [
        _reading("m1", "dc_motor", None),
        _reading("m2", "dc_motor", None),
    ]
    assert agreement(readings) == 1.0


def test_a_missing_marking_does_not_agree_with_a_read_one() -> None:
    readings = [
        _reading("m1", "bldc_motor", "2205"),
        _reading("m2", "bldc_motor", "2205"),
        _reading("m3", "bldc_motor", None),
    ]
    assert agreement(readings) == pytest.approx(2 / 3)


def test_agreement_ignores_case_and_spacing_in_types() -> None:
    readings = [
        _reading("m1", "BLDC motor", None),
        _reading("m2", "bldc_motor", None),
    ]
    assert agreement(readings) == 1.0


def test_part_number_family_flattens_variants() -> None:
    assert part_number_family("470uF 25V") == "470uf25v"
    assert part_number_family("IRF 540-N") == part_number_family("irf540n")
    assert part_number_family(None) is None
    assert part_number_family("  ") is None


# --- the one shared prompt ----------------------------------------------------


def test_prompt_demands_the_reading_schema_and_honesty() -> None:
    for field in (
        "part_type",
        "part_number",
        "key_rating",
        "condition",
        "confidence",
        "evidence",
    ):
        assert f'"{field}"' in READ_PROMPT
    text = READ_PROMPT.lower()
    assert "only one json object" in text
    assert "never guess" in text
    assert "cannot read" in text
    assert "null" in text  # unreadable markings become null, not a guess
    assert "<works | dead | unknown>" in READ_PROMPT


def test_retry_note_keeps_the_answer_to_json() -> None:
    assert "json" in RETRY_NOTE.lower()
    assert "no prose" in RETRY_NOTE.lower()


# --- package surface (what briefs 06 and 08 import) ---------------------------


def test_read_part_imports_only_when_a_key_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ImportError):
        from salvage.readers import read_part as _offline  # noqa: F401

    with_keys = reader_module.read_part
    import salvage.readers.reader as fresh  # the module was imported above

    assert fresh.read_part is with_keys
    # with one key configured, the lazy export resolves to the real reader
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    from salvage.readers import read_part as keyed

    assert keyed is fresh.read_part


def test_read_part_module_docs_claim_the_brief() -> None:
    module_text = reader_module.__doc__ or ""
    assert "read_part" in module_text
    assert "30 s" in module_text
