"""Tests for the shared Salvage contract (schemas), config, and cache.

The brief's acceptance test: writing a Reading to JSON and reading it back
gives the same object. Also covered: Part, Verdict, validation bounds, the
PROVIDERS contract, and cached_call's sha256 file caching.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from salvage import cache
from salvage.config import PROVIDERS, Provider, available_providers, get_provider
from salvage.schemas import BBox, Condition, Part, Reading, State, Verdict


def make_part() -> Part:
    return Part(
        id="p_001",
        device_id="dev_01",
        crop_path="data/crops/dev_01/p_001.jpg",
        bbox=(0.1, 0.2, 0.3, 0.4),
        mask_path="data/masks/dev_01/p_001.png",
    )


def make_reading() -> Reading:
    return Reading(
        part_id="p_001",
        model_id="muse-spark-1.3",
        part_type="electrolytic_capacitor",
        part_number="UPW1V101MDD1TD",
        key_rating="100uF 35V",
        condition=Condition.WORKS,
        confidence=0.87,
        evidence="Marking matches part family; voltage rating exceeds rail.",
    )


def make_verdict() -> Verdict:
    return Verdict(
        part_id="p_001",
        state=State.NEEDS_MEASUREMENT,
        label="capacitor 100uF 35V",
        reasons=["2 of 3 readers agree", "ESR not measurable from photo"],
        requested_measurement="Put the meter across the two leads; expect < 5 ohms.",
    )


# --- schemas ------------------------------------------------------------


class TestSchemas:
    def test_reading_round_trip(self) -> None:
        """The brief's done-when: JSON write then read gives the same object."""
        reading = make_reading()
        raw = reading.model_dump_json()
        assert json.loads(raw)  # it is valid JSON on disk
        again = Reading.model_validate_json(raw)
        assert again == reading

    def test_reading_dump_shape(self) -> None:
        fields = set(make_reading().model_dump())
        assert fields == {
            "part_id",
            "model_id",
            "part_type",
            "part_number",
            "key_rating",
            "condition",
            "confidence",
            "evidence",
        }

    def test_part_round_trip(self) -> None:
        part = make_part()
        again = Part.model_validate_json(part.model_dump_json())
        assert again == part
        assert part.bbox == (0.1, 0.2, 0.3, 0.4)

    def test_verdict_round_trip_all_states(self) -> None:
        for state in State:
            verdict = make_verdict().model_copy(update={"state": state})
            again = Verdict.model_validate_json(verdict.model_dump_json())
            assert again == verdict
            assert again.state == state

    def test_state_values_match_readme(self) -> None:
        assert [s.value for s in State] == [
            "VERIFIED",
            "NEEDS_MEASUREMENT",
            "ABSTAIN",
            "HAZARD",
        ]

    def test_condition_values_match_readme(self) -> None:
        assert {c.value for c in Condition} == {"works", "dead", "unknown"}

    def test_confidence_bounds(self) -> None:
        assert Reading.model_validate(
            {**make_reading().model_dump(), "confidence": 0.0}
        ).confidence == 0.0
        assert Reading.model_validate(
            {**make_reading().model_dump(), "confidence": 1.0}
        ).confidence == 1.0
        for bad in (-0.01, 1.01, 2):
            with pytest.raises(ValidationError):
                Reading.model_validate({**make_reading().model_dump(), "confidence": bad})

    def test_bbox_must_have_four_values(self) -> None:
        data = make_part().model_dump()
        data["bbox"] = [0.1, 0.2, 0.3]
        with pytest.raises(ValidationError):
            Part.model_validate(data)
        data["bbox"] = [0.1, 0.2, 0.3, 0.4]
        Part.model_validate(data)

    def test_verdict_defaults(self) -> None:
        verdict = Verdict(part_id="p_002", state=State.HAZARD, label="swollen cell")
        assert verdict.reasons == []
        assert verdict.requested_measurement is None

    def test_optional_fields_default_none(self) -> None:
        reading = Reading(
            part_id="p_003",
            model_id="muse-glimmer-30b",
            part_type="unknown_marking",
            condition=Condition.UNKNOWN,
            confidence=0.1,
            evidence="No legible part number.",
        )
        assert reading.part_number is None
        assert reading.key_rating is None

    def test_bbox_type_alias(self) -> None:
        assert BBox((0.0, 0.0, 1.0, 1.0)) == (0.0, 0.0, 1.0, 1.0)


# --- config -------------------------------------------------------------


class TestConfig:
    def test_providers_have_unique_names(self) -> None:
        names = [p.name for p in PROVIDERS]
        assert len(names) == len(set(names))
        assert "meta-api" in names

    def test_provider_contract(self) -> None:
        """Each provider exposes exactly the fields the 01 brief demands."""
        for provider in PROVIDERS:
            assert provider.name
            assert provider.base_url.startswith(("http://", "https://"))
            assert provider.api_key_env
            assert isinstance(provider.model_id, str) and provider.model_id
            assert isinstance(provider.enabled, bool)

    def test_env_example_lists_every_key(self) -> None:
        text = Path("tests/../.env.example").read_text(encoding="utf-8")
        for provider in PROVIDERS:
            assert provider.api_key_env in text, provider.api_key_env

    def test_no_real_key_in_env_example(self) -> None:
        text = Path(".env.example").read_text(encoding="utf-8")
        for line in text.splitlines():
            if "=" in line and not line.startswith("#"):
                assert line.split("=", 1)[1] == ""

    def test_key_read_from_env_never_hardcoded(self) -> None:
        provider = get_provider("meta-api")
        assert provider.api_key() is None  # no .env on disk in this checkout
        import os

        os.environ["META_API_KEY"] = "test-key-xyz"
        try:
            assert provider.api_key() == "test-key-xyz"
        finally:
            del os.environ["META_API_KEY"]

    def test_available_providers_respects_key_and_flag(self) -> None:
        import os

        off = Provider(
            name="disabled", base_url="https://x/v1", api_key_env="X_KEY",
            model_id="m", enabled=False,
        )
        os.environ["X_KEY"] = "k"
        try:
            assert off.is_available() is False  # disabled wins
            os.environ["X2_KEY"] = "k"
            on = Provider(
                name="enabled", base_url="https://x/v1", api_key_env="X2_KEY",
                model_id="m",
            )
            assert on.is_available() is True
        finally:
            os.environ.pop("X_KEY", None)
            os.environ.pop("X2_KEY", None)

    def test_meta_provider_roles(self) -> None:
        meta = get_provider("meta-api")
        assert meta.models == {
            "vision": "sam-3.1",
            "foreman": "muse-wonder-1.0",
            "reader": "muse-spark-1.3",
            "voice": "muse-voice-transcribe",
        }

    def test_open_weights_providers_present(self) -> None:
        assert get_provider("baseten").model_id == "meta-models/muse-spark-open-1026"
        assert "muse-glimmer" in get_provider("fireworks").model_id

    def test_get_provider_unknown(self) -> None:
        with pytest.raises(KeyError):
            get_provider("nope")

    def test_available_providers_empty_without_keys(self) -> None:
        import os

        saved = {
            p.api_key_env: os.environ.pop(p.api_key_env, None) for p in PROVIDERS
        }
        try:
            assert available_providers() == []
        finally:
            for env, value in saved.items():
                if value is not None:
                    os.environ[env] = value


# --- cache --------------------------------------------------------------


class TestCache:
    def test_same_payload_hits_cache(self, tmp_path: Path) -> None:
        calls = []

        def fn(payload: bytes) -> dict:
            calls.append(payload)
            return {"answer": "ok"}

        payload = b'{"image": "abc"}'
        first = cache.cached_call("muse-spark-1.3", payload, fn, cache_dir=tmp_path)
        second = cache.cached_call("muse-spark-1.3", payload, fn, cache_dir=tmp_path)
        assert first == second == {"answer": "ok"}
        assert len(calls) == 1  # second call cost nothing

    def test_cache_file_is_sha256_of_model_and_payload(self, tmp_path: Path) -> None:
        cache.cached_call(
            "muse-spark-1.3", b"x", lambda p: {"a": 1}, cache_dir=tmp_path
        )
        digest = cache.payload_digest("muse-spark-1.3", b"x")
        assert len(digest) == 64
        assert (tmp_path / f"{digest}.json").exists()
        # different model, same payload -> different file
        other = cache.payload_digest("muse-spark-open", b"x")
        assert other != digest

    def test_fn_raises_returns_none_no_crash(self, tmp_path: Path) -> None:
        def boom(payload: bytes) -> dict:
            raise TimeoutError("network down")

        assert cache.cached_call("m1", b"p", boom, cache_dir=tmp_path) is None
        assert list(tmp_path.iterdir()) == []  # nothing cached

    def test_none_result_not_cached(self, tmp_path: Path) -> None:
        assert cache.cached_call("m1", b"p", lambda p: None, cache_dir=tmp_path) is None
        assert list(tmp_path.iterdir()) == []

    def test_pydantic_result_serialized(self, tmp_path: Path) -> None:
        result = cache.cached_call("m1", b"p", lambda p: make_reading(), cache_dir=tmp_path)
        first = result.model_dump() if isinstance(result, Reading) else result
        again = cache.cached_call("m1", b"p", lambda p: make_reading(), cache_dir=tmp_path)
        assert again == first  # the JSON copy on a cache hit

    def test_corrupt_cache_treated_as_miss(self, tmp_path: Path) -> None:
        path = tmp_path / f"{cache.payload_digest('m1', b'p')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        calls = []

        def fn(payload: bytes) -> dict:
            calls.append(1)
            return {"fresh": True}

        assert cache.cached_call("m1", b"p", fn, cache_dir=tmp_path) == {"fresh": True}
        assert len(calls) == 1

    def test_cache_dir_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SALVAGE_CACHE_DIR", str(tmp_path / "envdir"))
        assert cache.cache_root() == tmp_path / "envdir"
        cache.cached_call("m1", b"p", lambda p: {"m": "env"})
        assert list((tmp_path / "envdir").iterdir())
