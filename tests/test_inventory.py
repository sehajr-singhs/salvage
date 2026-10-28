"""Tests for the inventory and reverse search (brief 05).

The brief's done-when: tests on a seeded DB cover an exact match, the
fallback path, and hazard exclusion. Around those three: filter parsing,
state ranking (VERIFIED before NEEDS_MEASUREMENT), ABSTAIN exclusion, the
one-cached-LLM-call contract, and the module-level conveniences — all
completely offline (no keys on this machine, and the LLM path is stubbed
rather than simulated over HTTP).
"""

from __future__ import annotations

import sqlite3

import pytest
from pydantic import ValidationError

from salvage import inventory
from salvage.inventory import filters as filters_mod
from salvage.inventory import store as store_mod
from salvage.inventory.common_sources import COMMON_SOURCES, fallback_matches
from salvage.inventory.filters import Filters, heuristic_filters
from salvage.inventory.models import Device, Match
from salvage.inventory.store import DEFAULT_DB_PATH, Inventory, rating_amps, rating_volts
from salvage.schemas import Condition, Part, Reading, State, Verdict


# --- builders ---------------------------------------------------------------


def make_device(device_id: str, name: str, source: str | None = None) -> Device:
    return Device(id=device_id, name=name, source=source)


def make_part(part_id: str, device_id: str) -> Part:
    return Part(
        id=part_id,
        device_id=device_id,
        crop_path=f"data/crops/{device_id}/{part_id}.png",
        bbox=(0.1, 0.2, 0.3, 0.4),
        mask_path=f"data/masks/{device_id}/{part_id}.png",
    )


def make_reading(
    part_id: str,
    model: str,
    part_type: str,
    rating: str | None = None,
    condition: str = "works",
    evidence: str = "marking legible on the crop",
) -> Reading:
    return Reading(
        part_id=part_id,
        model_id=model,
        part_type=part_type,
        key_rating=rating,
        condition=Condition(condition),
        confidence=0.8,
        evidence=evidence,
    )


def make_verdict(part_id: str, state: State, label: str, why: str) -> Verdict:
    return Verdict(part_id=part_id, state=state, label=label, reasons=[why])


def seeded_inventory() -> Inventory:
    """A bench inventory with two scanned devices: a drone and a router."""
    store = Inventory(":memory:")

    drone = make_device("drone_01", "Camera drone", source="drawer")
    drone_readings = [
        make_reading(
            "p_001",
            "muse-spark-1.3",
            "bldc_motor",
            rating="2205 12V 20A 2300KV",
            evidence="kv marking legible; pairs with a 30A esc, pwm throttle",
        ),
        make_reading(
            "p_001",
            "muse-spark-open",
            "bldc_motor",
            rating="12V 20A",
            evidence="phase wires to the esc under the top plate",
        ),
        make_reading(
            "p_002",
            "muse-spark-1.3",
            "lipo_cell",
            rating="11.1V",
            evidence="3S pack pouch flat, no swelling",
        ),
        make_reading(
            "p_003",
            "muse-spark-1.3",
            "power_supply",
            rating="12V",
            evidence="found right after the 120 V AC mains input bridge",
        ),
    ]
    store.add(
        drone,
        [make_part("p_001", "drone_01"), make_part("p_002", "drone_01"), make_part("p_003", "drone_01")],
        [
            make_verdict("p_001", State.VERIFIED, "BLDC 2205 2300KV", "two independent readers agree"),
            make_verdict("p_002", State.VERIFIED, "LiPo 3S 11.1V", "pouch intact"),
            make_verdict(
                "p_003",
                State.HAZARD,
                "mains PSU primary side",
                "evidence mentions mains: do not open mains-side boards",
            ),
        ],
        readings=drone_readings,
    )

    router = make_device("router_01", "Wifi router", source="school IT closet")
    router_readings = [
        make_reading(
            "p_010",
            "muse-spark-1.3",
            "bldc_motor",
            rating="12V",
            condition="unknown",
            evidence="sticker unreadable; phase wires go to its esc, winding test needed",
        ),
        make_reading(
            "p_011",
            "muse-spark-1.3",
            "cooling_fan",
            rating="12V 0.2A",
            condition="unknown",
            evidence="hub spins by hand but no meter reading yet",
        ),
    ]
    store.add(
        router,
        [make_part("p_010", "router_01"), make_part("p_011", "router_01")],
        [
            make_verdict(
                "p_010",
                State.NEEDS_MEASUREMENT,
                "unknown 12V motor",
                "single reader only; meter the winding resistance",
            ),
            make_verdict(
                "p_011",
                State.ABSTAIN,
                "unknown fan",
                "readers contradict",
            ),
        ],
        readings=router_readings,
    )
    return store


@pytest.fixture(autouse=True)
def offline_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """No keys exist: no provider can be reached from these tests."""
    for key in (
        "META_API_KEY",
        "BASETEN_API_KEY",
        "FIREWORKS_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def default_db_path(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Keep the default store off the real repo's data/ during tests."""
    monkeypatch.setattr(store_mod, "DEFAULT_DB_PATH", tmp_path / "inventory.db")
    yield


# --- filter parsing ---------------------------------------------------------


class TestParseFilters:
    def test_the_briefs_example_request(self) -> None:
        parsed = heuristic_filters("I need a 12 V motor with a driver")
        assert parsed.type_tokens == ["motor"]
        assert parsed.voltage == pytest.approx(12.0)
        assert "driver" in parsed.interface_tokens

    def test_kv_never_reads_as_volts(self) -> None:
        assert heuristic_filters("a 2300KV outrunner").voltage is None

    def test_current_in_milliamps_units_removed_from_type(self) -> None:
        parsed = heuristic_filters("500 mA dc fan")
        assert parsed.current == pytest.approx(0.5)
        assert parsed.type_tokens == ["dc", "fan"]

    def test_units_removed_from_type(self) -> None:
        parsed = heuristic_filters("12 volt motor")
        assert "volt" not in parsed.type_tokens

    def test_offline_fallback_writes_no_cache(self, tmp_path) -> None:
        filters_mod.parse_filters("I need a 12 V motor with a driver", cache_dir=tmp_path)
        assert list(tmp_path.iterdir()) == []  # the fallback path is free

    def test_llm_parse_is_cached_and_reused(self, tmp_path, monkeypatch) -> None:
        calls = {"n": 0}

        def fake_llm(payload: bytes) -> dict:
            calls["n"] += 1
            return {"type": "motor", "voltage": 12.0, "current": None, "interface": "driver"}

        monkeypatch.setattr(filters_mod, "_llm_parse", fake_llm)
        text = "I need a 12 V motor with a driver"
        first = filters_mod.parse_filters(text, cache_dir=tmp_path)
        assert first.voltage == pytest.approx(12.0)
        assert first.type_tokens == ["motor"]

        def exploding_llm(payload: bytes) -> dict:
            raise AssertionError("second call must be served from cache")

        monkeypatch.setattr(filters_mod, "_llm_parse", exploding_llm)
        assert filters_mod.parse_filters(text, cache_dir=tmp_path) == first
        assert calls["n"] == 1
        assert len(list(tmp_path.iterdir())) == 1

    def test_unusable_llm_reply_falls_back(self, monkeypatch) -> None:
        monkeypatch.setattr(
            filters_mod,
            "_llm_parse",
            lambda payload: {"type": "", "voltage": None, "current": None, "interface": ""},
        )
        parsed = filters_mod.parse_filters("I need a 12 V motor with a driver")
        assert parsed.type_tokens == ["motor"]  # heuristic result, not the empty reply


# --- ratings parsing --------------------------------------------------------


class TestRatingParsing:
    def test_2300kv_is_not_2300_volts(self) -> None:
        assert rating_volts("2205 12V 2300KV") == pytest.approx(12.0)

    def test_mv_scales_down(self) -> None:
        assert rating_volts("3300mV") == pytest.approx(3.3)

    def test_va_is_not_amps(self) -> None:
        assert rating_amps("25VA") is None

    def test_amps_and_milliamps(self) -> None:
        assert rating_amps("12V 30A") == pytest.approx(30.0)
        assert rating_amps("12V 500mA") == pytest.approx(0.5)


# --- the query itself -------------------------------------------------------


class TestExactMatch:
    def test_exact_match_from_seeded_db(self) -> None:
        store = seeded_inventory()
        matches = store.query("I need a 12 V motor with a driver")
        assert matches, "the seeded 12 V BLDC must answer the brief's request"
        assert matches[0].source == "inventory"
        assert matches[0].part_id == "p_001"
        assert matches[0].device == "Camera drone"
        assert matches[0].state == State.VERIFIED
        assert matches[0].voltage == pytest.approx(12.0)
        assert matches[0].part_type == "bldc_motor"

    def test_match_names_device_and_part(self) -> None:
        store = seeded_inventory()
        top = store.query("I need a 12 V motor with a driver")[0]
        assert top.device_id == "drone_01"
        assert top.part_id == "p_001"
        assert top.crop_path  # the bench can show the part's crop

    def test_driver_is_satisfied_by_esc_evidence(self) -> None:
        store = seeded_inventory()
        matches = store.query("I need a driver")
        assert matches[0].part_id == "p_001"  # its esc evidence satisfies "driver"

    def test_ranked_verified_before_needs_measurement(self) -> None:
        store = seeded_inventory()
        matches = store.query("I need a 12 V motor with a driver")
        states = [m.state for m in matches]
        assert states == sorted(states, key=lambda s: s != State.VERIFIED)
        assert states[0] == State.VERIFIED
        assert State.NEEDS_MEASUREMENT in states  # the router motor matched too

    def test_voltage_tolerance_lets_11v_pack_answer_12v(self) -> None:
        store = seeded_inventory()
        assert any(m.part_id == "p_001" for m in store.query("I need a 12 V motor"))

    def test_far_off_voltage_does_not_match_motor_tokens(self) -> None:
        store = seeded_inventory()
        matches = store.query("I need a 48 V motor")
        assert all(m.state is None for m in matches)  # fallback advice, if any

    def test_empty_request_answers_nothing(self) -> None:
        store = seeded_inventory()
        assert store.query("i need something i guess") == []

    def test_nothing_scanned_yet_answers_nothing_from_stock(self) -> None:
        store = Inventory(":memory:")
        matches = store.query("I need a 12 V motor with a driver")
        assert all(m.source == "common_sources" for m in matches)


# --- hazard and abstain exclusion -------------------------------------------


class TestExclusion:
    def test_hazard_parts_are_never_returned(self) -> None:
        store = seeded_inventory()
        for text in (
            "I need a 12 V power supply",
            "open up the power supply on that drone",
            "I need a mains board",
        ):
            matches = store.query(text)
            assert all(m.part_id != "p_003" for m in matches)

    def test_mains_requests_get_no_stock_and_no_advice(self) -> None:
        store = seeded_inventory()
        assert store.query("I need a 12 V switching power supply") == []

    def test_abstain_parts_are_never_returned(self) -> None:
        store = seeded_inventory()
        matches = store.query("I need a 12 V fan")
        assert all(m.part_id != "p_011" for m in matches)

    def test_unverdicted_part_is_invisible_to_queries(self) -> None:
        store = seeded_inventory()
        store.add(
            make_device("cam_01", "Dash cam"),
            [make_part("p_020", "cam_01")],
            [],  # no verdict yet
        )
        assert all(m.source == "common_sources" for m in store.query("I need a cam"))

    def test_both_unsafe_states_really_exist_in_the_seed(self) -> None:
        store = seeded_inventory()
        states = {row[0] for row in store._conn.execute("SELECT state FROM verdicts")}
        assert {"HAZARD", "ABSTAIN"} <= states


# --- the fallback path ------------------------------------------------------


class TestFallback:
    def test_5v_motor_falls_back_to_toy_cars(self) -> None:
        store = Inventory(":memory:")
        matches = store.query("I need a 5 V DC motor")
        assert matches
        assert all(m.source == "common_sources" for m in matches)
        best = matches[0]
        assert best.part_type == "brushed_dc_motor"
        assert best.device == "toy cars"
        assert "old tray-loading CD and DVD drives" in best.alternatives
        assert best.part_id is None
        assert best.state is None

    def test_fallback_prefers_requested_voltage(self) -> None:
        matches = fallback_matches(Filters(type="motor", voltage=5.0))
        assert matches[0].part_type == "brushed_dc_motor"

    def test_fallback_notes_interface_gaps(self) -> None:
        matches = fallback_matches(Filters(type="motor", voltage=12.0, interface="driver"))
        motor = next(m for m in matches if m.part_type == "brushed_dc_motor")
        assert any("driver" in reason for reason in motor.reasons)

    def test_fallback_flags_voltage_mismatch(self) -> None:
        matches = fallback_matches(Filters(type="motor", voltage=24.0))
        motor = next(m for m in matches if m.part_type == "brushed_dc_motor")
        assert any("24" in reason for reason in motor.reasons)

    def test_fallback_empty_for_no_type(self) -> None:
        assert fallback_matches(Filters(type="", voltage=5.0)) == []

    def test_common_sources_are_mains_safe(self) -> None:
        banned = ("crt", "microwave", "mains", "wall adapter", "wall wart")
        text = " ".join(
            f"{source.part_type} {source.note} {' '.join(source.devices)}"
            for source in COMMON_SOURCES
        )
        for word in banned:
            assert word not in text.lower()


# --- provenance and persistence ---------------------------------------------


class TestPersistence:
    def test_seeded_rows_are_retrievable(self) -> None:
        store = seeded_inventory()
        conn = store._conn
        assert conn.execute("SELECT COUNT(*) FROM devices").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM parts").fetchone()[0] == 5
        assert conn.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0] == 5
        assert conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0] == 6

    def test_file_backed_db_round_trips(self, tmp_path) -> None:
        db_path = tmp_path / "bench" / "inventory.db"
        first = Inventory(db_path)
        first.add(
            make_device("d1", "Toy car"),
            [make_part("p_001", "d1")],
            [make_verdict("p_001", State.VERIFIED, "DC motor", "readers agree")],
        )
        first.close()
        assert db_path.exists()

        second = Inventory(db_path)
        matches = second.query("I need a motor")
        assert matches and matches[0].part_id == "p_001"

    def test_default_path_is_data_inventory_db(self) -> None:
        assert DEFAULT_DB_PATH.name == "inventory.db"
        assert DEFAULT_DB_PATH.parts[-2:] == ("data", "inventory.db")

    def test_corrupt_db_file_raises_not_hangs(self, tmp_path) -> None:
        bad = tmp_path / "inventory.db"
        bad.write_bytes(b"this is not a database")
        with pytest.raises(sqlite3.DatabaseError):
            Inventory(bad)


# --- the module-level conveniences ------------------------------------------


class TestModuleSurface:
    def test_add_and_query_over_default_store(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(inventory, "_DEFAULT", Inventory(tmp_path / "inventory.db"))
        inventory.add(
            make_device("d1", "Toy car"),
            [make_part("p_001", "d1")],
            [make_verdict("p_001", State.VERIFIED, "DC motor", "readers agree")],
        )
        matches = inventory.query("I need a motor")
        assert [m.part_id for m in matches] == ["p_001"]

    def test_match_model_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            Match(source="inventory", part_id="p", not_a_field=1)
