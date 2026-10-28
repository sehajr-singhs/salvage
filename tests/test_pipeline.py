"""Tests for brief 08: the one chain, photo → segment → read → gate →
inventory, plus the measurement dialogue that turns a deferred or abstained
part into a VERIFIED one.

Everything runs offline per the 00-README rules: the segmentation backend is
the deterministic SAM mock (02), the readers are a scripted in-memory
provider standing in for the real 03 multi-model swarm, the gate is 04's
plain code, and the inventory is 05's SQLite store on a tmp path. No key is
ever read; no network call is ever made.

The brief's done-when, verbatim: "an end-to-end test on a fixture with
mocked models goes from a photo to inventory, including one ABSTAIN turning
into VERIFIED after a simulated measurement."
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from salvage.__main__ import main
from salvage.inventory import Inventory
from salvage.measure.dialogue import ScriptPrompter, state_transitions
from salvage.pipeline import DeviceScan, apply_measurements, run_device, run_dialogue
from salvage.schemas import Part, Reading, State
from salvage.segment.backends import Detection, MockSAMBackend

WIDTH, HEIGHT = 320, 240

# One detection per part, far enough apart that no overlap pass drops them.
_DETECTIONS = [
    # (y1, x1) sort order -> part ids p_001..p_004
    Detection(prompt="brushless motor", bbox=(20.0, 20.0, 100.0, 80.0)),
    Detection(prompt="cooling fan", bbox=(140.0, 20.0, 220.0, 80.0)),
    Detection(prompt="electrolytic capacitor", bbox=(40.0, 120.0, 120.0, 180.0)),
    Detection(prompt="lithium polymer cell", bbox=(200.0, 120.0, 280.0, 180.0)),
]


@pytest.fixture()
def fixture_image(tmp_path: Path) -> Path:
    """A deterministic RGB device photo: gray plate, darker components."""
    rng = np.random.default_rng(8)
    base = np.full((HEIGHT, WIDTH, 3), 150, dtype=np.uint8)
    base[30:90, 30:110] = 90
    base[30:90, 150:230] = 95
    base[130:190, 50:130] = 100
    base[130:190, 210:290] = 105
    base += rng.integers(0, 8, size=base.shape).astype(np.uint8)
    path = tmp_path / "drone.png"
    Image.fromarray(base, "RGB").save(path)
    return path


def _reading(
    part_id: str,
    model_id: str,
    part_type: str,
    *,
    condition: str = "works",
    part_number: str | None = None,
    key_rating: str | None = None,
    evidence: str = "the marking says so",
    confidence: float = 0.9,
) -> Reading:
    return Reading(
        part_id=part_id,
        model_id=model_id,
        part_type=part_type,
        part_number=part_number,
        key_rating=key_rating,
        condition=condition,
        confidence=confidence,
        evidence=evidence,
    )


def _scripted_readings() -> dict[str, list[Reading]]:
    """What the mock swarm says about the four parts of the fixture drone.

    - p_001 `bldc_motor`, two agreeing readers, works: the gate wants the
      winding measurement (NEEDS_MEASUREMENT) — or, when the device-wide
      note is an open circuit, ABSTAIN on contradiction.
    - p_002 `cooling_fan`, two agreeing readers, works: spin check.
    - p_003 `electrolytic_capacitor`, two readers that agree on the type but
      contradict each other on condition: a hard ABSTAIN.
    - p_004 `lipo_cell` with a swollen cell in the evidence: HAZARD.
    """
    return {
        "p_001": [
            _reading(
                "p_001",
                "muse-spark-1.3",
                "bldc_motor",
                part_number="2205",
                key_rating="2300KV",
                evidence=(
                    "magnet can: BLDC 2205 2300KV, two motor leads, "
                    "runs on the esc driver board"
                ),
            ),
            _reading(
                "p_001",
                "muse-spark-open",
                "bldc_motor",
                part_number="2205",
                key_rating="2300KV",
                evidence=(
                    "magnet can: BLDC 2205 2300KV, two motor leads, "
                    "runs on the esc driver board"
                ),
            ),
        ],
        "p_002": [
            _reading("p_002", "muse-spark-1.3", "cooling_fan", part_number="4010"),
            _reading("p_002", "muse-spark-open", "cooling_fan", part_number="4010"),
        ],
        "p_003": [
            _reading(
                "p_003",
                "muse-spark-1.3",
                "electrolytic_capacitor",
                part_number="470uF",
                condition="works",
                evidence="can marked 470uF, leads intact",
            ),
            _reading(
                "p_003",
                "muse-spark-open",
                "electrolytic_capacitor",
                part_number="470uF",
                condition="dead",
                evidence="can marked 470uF, the vent is popped",
            ),
        ],
        "p_004": [
            _reading(
                "p_004",
                "muse-spark-1.3",
                "lipo_cell",
                evidence="a swollen pouch cell, see the puffing on the foil",
                confidence=0.95,
            ),
        ],
    }


def _provider(readings_by_part: dict[str, list[Reading]]):
    def read(part: Part) -> list[Reading]:
        return list(readings_by_part.get(part.id, []))

    return read


def _scan(
    fixture_image: Path,
    tmp_path: Path,
    *,
    readings_by_part: dict[str, list[Reading]] | None = None,
    inventory: Inventory | None = None,
    measurement: str | None = None,
) -> DeviceScan:
    return run_device(
        fixture_image,
        "drone1",
        backend=MockSAMBackend(_DETECTIONS),
        data_root=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        readings_provider=_provider(readings_by_part or _scripted_readings()),
        measurement=measurement,
        inventory=inventory,
    )


# --- run_device: the one chain ----------------------------------------------


def test_run_device_end_to_end_with_mocks(fixture_image: Path, tmp_path: Path) -> None:
    """Photo to inventory in one call, mocked models, no keys."""
    inventory = Inventory(tmp_path / "inventory.db")
    scan = _scan(fixture_image, tmp_path, inventory=inventory)

    assert scan.ok
    assert scan.device_name == "drone"  # the display name is the photo's stem
    assert [p.id for p in scan.parts] == ["p_001", "p_002", "p_003", "p_004"]
    assert all(p.device_id == "drone1" for p in scan.parts)

    # 02 wrote every artifact under the request's data root
    for part in scan.parts:
        assert (tmp_path / "data" / "crops" / "drone1" / f"{part.id}.png").is_file()
        assert (tmp_path / "data" / "masks" / "drone1" / f"{part.id}.png").is_file()
    assert (tmp_path / "data" / "scan" / "drone1" / "overlay.png").is_file()
    assert (tmp_path / "data" / "scan" / "drone1" / "photo.png").is_file()
    assert scan.overlay_path and Path(scan.overlay_path).is_file()

    # 03 produced one Reading per reader; 04 produced one Verdict per Part
    assert len(scan.readings) == 7  # 2 + 2 + 2 + 1
    assert [v.part_id for v in scan.verdicts] == [p.id for p in scan.parts]
    states = {v.part_id: v.state for v in scan.verdicts}
    assert states["p_001"] is State.NEEDS_MEASUREMENT
    assert states["p_002"] is State.NEEDS_MEASUREMENT
    assert states["p_003"] is State.ABSTAIN
    assert states["p_004"] is State.HAZARD
    motor_verdict = next(v for v in scan.verdicts if v.part_id == "p_001")
    assert "winding" in (motor_verdict.requested_measurement or "")

    # 05 remembered the device and its parts
    matches = inventory.query("motor", cache_dir=tmp_path / "cache")
    inventory_matches = [m for m in matches if m.source == "inventory"]
    assert any(m.part_id == "p_001" and m.device_id == "drone1" for m in inventory_matches)
    # HAZARD and ABSTAIN parts never surface in a search
    assert not any(m.part_id in ("p_003", "p_004") for m in inventory_matches)


def test_run_device_without_backend_is_soft(fixture_image: Path, tmp_path: Path) -> None:
    """No backend (no keys, no Modal app) is an ok:false scan, not a crash."""
    scan = _scan(fixture_image, tmp_path, readings_by_part={})
    # first_available_backend() finds nothing and nothing is injected either,
    # because run_device only auto-selects when backend is None; force that
    # path by passing backend=None through the thin public wrapper.
    auto = run_device(
        fixture_image,
        "drone1",
        backend=None,
        data_root=tmp_path / "data2",
        cache_dir=tmp_path / "cache2",
        readings_provider=_provider({}),
    )
    assert not auto.ok
    assert auto.parts == []
    assert auto.verdicts == []
    assert auto.message and "sample" in auto.message

    scan_ok_state = scan.ok  # the mock-backend scan above ran fine
    assert scan_ok_state


def test_failed_reader_adds_no_votes(fixture_image: Path, tmp_path: Path) -> None:
    """A reader failure yields no vote; the gate gets more cautious."""
    readings = _scripted_readings()
    readings["p_002"] = []  # both fan readers failed
    scan = _scan(fixture_image, tmp_path, readings_by_part=readings)
    assert scan.ok
    verdict = next(v for v in scan.verdicts if v.part_id == "p_002")
    assert verdict.state is State.ABSTAIN
    assert any("independent reader" in reason for reason in verdict.reasons)


# --- the measurement dialogue ------------------------------------------------


def test_run_dialogue_turns_abstain_into_verified(
    fixture_image: Path, tmp_path: Path
) -> None:
    """One ABSTAIN becomes VERIFIED after a simulated meter reading.

    The scan runs with the device-wide note 'open circuit', so the motor's
    'works' claim contradicts an open winding: ABSTAIN. The user then reads
    the meter as 8.2 ohms; the gate re-runs with that measurement and the
    part verifies. That is the D3 beat the brief's done-when asks for.
    """
    inventory = Inventory(tmp_path / "inventory.db")
    scan = _scan(
        fixture_image,
        tmp_path,
        inventory=inventory,
        measurement="open circuit",
    )
    before = {v.part_id: v.state for v in scan.verdicts}
    assert before == {
        "p_001": State.ABSTAIN,
        "p_002": State.NEEDS_MEASUREMENT,
        "p_003": State.ABSTAIN,
        "p_004": State.HAZARD,
    }

    script = ScriptPrompter(
        {
            "p_001": "8.2 ohms",  # the meter, across the two motor leads
            "p_002": "it spins",
            "p_003": "3 ohms",  # resolvable in principle, still contradiction
        }
    )
    updated, transitions = run_dialogue(scan, script, inventory=inventory)

    # exactly the two resolvable parts moved; the contradiction stays ABSTAIN
    moved = {(t.part_id, t.before, t.after) for t in transitions}
    assert ("p_001", State.ABSTAIN, State.VERIFIED) in moved
    assert ("p_002", State.NEEDS_MEASUREMENT, State.VERIFIED) in moved
    assert all(t.part_id != "p_003" for t in transitions)
    after = {v.part_id: v.state for v in updated.verdicts}
    assert after == {
        "p_001": State.VERIFIED,
        "p_002": State.VERIFIED,
        "p_003": State.ABSTAIN,
        "p_004": State.HAZARD,
    }
    # the transitions helper agrees with what the dialogue just did
    assert {(t.part_id, t.before, t.after) for t in state_transitions(
        scan.verdicts, updated.verdicts
    )} == moved

    # HAZARD parts are never asked: there is no measurement that helps
    assert "p_004" not in [part_id for part_id, _ in script.asked]
    assert [part_id for part_id, _ in script.asked].count("p_004") == 0

    # the re-run's reasons say why the motor now verifies
    motor = next(v for v in updated.verdicts if v.part_id == "p_001")
    assert any("8.2" in reason and "in range" in reason for reason in motor.reasons)

    # the inventory shows the new states (05 upserts by device)
    matches = [
        m
        for m in inventory.query("motor", cache_dir=tmp_path / "cache")
        if m.source == "inventory" and m.part_id == "p_001"
    ]
    assert matches and matches[0].state is State.VERIFIED


def test_run_dialogue_retries_unparseable_replies(
    fixture_image: Path, tmp_path: Path
) -> None:
    """Gibberish gets asked once more; a part skipped twice keeps its state."""
    scan = _scan(fixture_image, tmp_path)
    script = ScriptPrompter(
        {
            "p_001": "huh?",  # unparseable
            "p_002": ["also nothing", "spins"],  # fail once, then a real reading
            # p_003 / p_004: scripted, so the user is skipping them
        }
    )
    updated, transitions = run_dialogue(scan, script)

    # p_001 was asked exactly twice (one retry) and never resolved
    asked = [part_id for part_id, _ in script.asked]
    assert asked.count("p_001") == 2
    assert next(v for v in updated.verdicts if v.part_id == "p_001").state is (
        State.NEEDS_MEASUREMENT
    )
    # p_002 failed once, then parsed: two asks, one retry, then verified
    assert asked.count("p_002") == 2
    moved = {(t.part_id, t.after) for t in transitions}
    assert ("p_002", State.VERIFIED) in moved

    # a no-answer dialogue changes nothing at all
    silent_scan = _scan(fixture_image, tmp_path)
    untouched, no_transitions = run_dialogue(silent_scan, ScriptPrompter({}))
    assert no_transitions == []
    assert untouched.verdicts == silent_scan.verdicts


def test_apply_measurements_overrides_device_note(
    fixture_image: Path, tmp_path: Path
) -> None:
    """A per-part answer wins over the device-wide note; others keep it."""
    scan = _scan(fixture_image, tmp_path, measurement="open circuit")
    updated = apply_measurements(scan, {"p_001": "15 ohms"})
    after = {v.part_id: v.state for v in updated.verdicts}
    # 15 ohms is in the winding range
    assert after["p_001"] is State.VERIFIED
    # the fan never got an answer, so it keeps the device-wide note's effect
    assert after["p_002"] is State.NEEDS_MEASUREMENT
    # everything else of the scan is carried through unchanged
    assert updated.parts == scan.parts
    assert updated.readings == scan.readings
    assert updated.device_id == scan.device_id


# --- the CLI: python -m salvage ------------------------------------------------


def test_cli_scan_then_need(fixture_image: Path, tmp_path: Path, capsys) -> None:
    """`scan --measure` then `need`, the two advertised commands, end to end."""
    db = tmp_path / "inventory.db"
    data_root = tmp_path / "cli-data"
    script = ScriptPrompter(
        {
            "p_001": "8.2 ohms",
            "p_002": "spins",
            "p_003": "3 ohms",
        }
    )
    code = main(
        [
            "scan",
            str(fixture_image),
            "--device",
            "drone2",
            "--measure",
            "--db",
            str(db),
            "--data-root",
            str(data_root),
        ],
        prompter=script,
        scan_backend=MockSAMBackend(_DETECTIONS),
        readings_provider=_provider(_scripted_readings()),
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "drone2" in out
    assert "readings: injected" in out
    assert "[VERIFIED]" in out
    assert "[ABSTAIN]" in out  # the capacitor contradiction stays unproven
    assert "[HAZARD]" in out
    assert out.count("-> [VERIFIED]") == 2  # two parts resolved by measurement

    # the one command remembered everything; `need` finds it again
    # ("a motor": the interface tokens the swarm saw — esc, driver — are
    # stored, but a "for an ESC" request also filters on them, so the
    # plain query is what the CLI demos use)
    code = main(["need", "a motor", "--db", str(db)])
    assert code == 0
    out = capsys.readouterr().out
    assert "p_001" in out
    assert "drone" in out

    # `need` never points at hazard or abstain parts
    assert "p_003" not in out
    assert "p_004" not in out


def test_cli_scan_without_backend_exits_soft_error(
    fixture_image: Path, tmp_path: Path, capsys
) -> None:
    """No backend means exit code 1 with the advice message, not a crash."""
    code = main(
        [
            "scan",
            str(fixture_image),
            "--device",
            "drone3",
            "--db",
            str(tmp_path / "inventory.db"),
            "--data-root",
            str(tmp_path / "cli-data"),
        ],
        scan_backend=None,
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "sample" in out


def test_device_scan_round_trips(fixture_image: Path, tmp_path: Path) -> None:
    """The scan object is pydantic/JSON-ready, so CLI and UI share it."""
    scan = _scan(fixture_image, tmp_path)
    payload = scan.model_dump_json()
    rehydrated = DeviceScan.model_validate_json(payload)
    assert rehydrated == scan
    assert rehydrated.verdicts[0].state is scan.verdicts[0].state
