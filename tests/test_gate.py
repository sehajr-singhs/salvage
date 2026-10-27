"""Table-driven tests for the verify-or-abstain gate (brief 04).

The brief's done-when: at least 25 table-driven tests covering every state
(VERIFIED / NEEDS_MEASUREMENT / ABSTAIN / HAZARD) and every downgrade rule,
all passing. One parametrized table drives the scenarios; a few unit tests
lock the invariants (determinism, caution under missing voters) in directly.
"""

from __future__ import annotations

import pytest

from salvage.gate import decide
from salvage.gate.checks import (
    CHECKS,
    consistency_problems,
    extract_rail,
    find_check,
    parse_ohms,
    parse_volts,
    rating_volts,
)
from salvage.gate.hazards import HAZARD_RULES, find_hazard
from salvage.schemas import Condition, Part, Reading, State

PART = Part(
    id="p_001",
    device_id="dev_01",
    crop_path="data/crops/dev_01/p_001.jpg",
    bbox=(0.1, 0.2, 0.3, 0.4),
    mask_path="data/masks/dev_01/p_001.png",
)


def rd(
    model: str,
    part_type: str,
    condition: str = "works",
    evidence: str = "marking legible in the crop",
    pn: str | None = None,
    rating: str | None = None,
    confidence: float = 0.8,
    part_id: str = "p_001",
) -> Reading:
    return Reading(
        part_id=part_id,
        model_id=model,
        part_type=part_type,
        part_number=pn,
        key_rating=rating,
        condition=Condition(condition),
        confidence=confidence,
        evidence=evidence,
    )


def two(
    ptype: str,
    condition: str = "works",
    e1: str = "marking legible",
    e2: str = "marking legible",
    pn: str | None = None,
    rating: str | None = None,
) -> list[Reading]:
    """Two independent agreeing readers: the smallest passing quorum."""
    return [
        rd("muse-spark-1.3", ptype, condition=condition, evidence=e1, pn=pn, rating=rating),
        rd("muse-spark-open", ptype, condition=condition, evidence=e2, pn=pn, rating=rating),
    ]


def run(readings: list[Reading], measurement: str | None = None):
    return decide(PART, readings, measurement)


# --- the scenario table -----------------------------------------------------
# Each row: readings, optional measurement, expected state, and substrings that
# must appear in reasons / requested_measurement / label.

SCENARIOS = [
    # 1. HAZARD -- every hazard rule via type, evidence, or the measurement
    # note; hazard wins with a single reader; context guards hold.
    dict(
        id="hazard-swollen-lipo-via-evidence",
        readings=[
            rd("m1", "lipo_cell", "unknown", evidence="pouch is swollen and puffed"),
            rd("m2", "lipo_cell", "unknown", evidence="marks normal"),
        ],
        expect=State.HAZARD,
        reason_contains=["swollen or damaged lithium cell", "instead"],
        label_contains=["swollen or damaged lithium cell"],
    ),
    dict(
        id="hazard-lipo-punctured-one-reader",
        readings=[rd("m1", "18650_cell", "unknown", evidence="wrapper punctured during removal")],
        expect=State.HAZARD,
        reason_contains=["instead"],
    ),
    dict(
        id="hazard-crt-via-type",
        readings=two("crt_tube", e1="monitor tube still charged", e2="do not open"),
        expect=State.HAZARD,
        reason_contains=["CRT"],
    ),
    dict(
        id="hazard-crt-via-evidence",
        readings=two(
            "unknown_module",
            e1="screen glass and cathode ray tube from an old TV",
            e2="heavy glass envelope",
        ),
        expect=State.HAZARD,
        reason_contains=["e-waste"],
    ),
    dict(
        id="hazard-microwave-hv-cap-via-type",
        readings=two("microwave_hv_capacitor", e1="from the oven cavity", e2="large oil-filled can"),
        expect=State.HAZARD,
        reason_contains=["lethal charge"],
    ),
    dict(
        id="hazard-microwave-via-evidence",
        readings=two(
            "hv_capacitor",
            e1="pulled from a microwave oven, never discharged",
            e2="can-style HV capacitor",
        ),
        expect=State.HAZARD,
        reason_contains=["instead"],
    ),
    dict(
        id="hazard-mains-psu-via-type",
        readings=two("power_supply", e1="switch-mode brick, 120 V input", e2="transformer inline"),
        expect=State.HAZARD,
        reason_contains=["mains primary side"],
    ),
    dict(
        id="hazard-mains-via-evidence",
        readings=two(
            "ac_board",
            e1="sits right after the 120 V AC mains input bridge",
            e2="before the switching transformer",
        ),
        expect=State.HAZARD,
        reason_contains=["instead"],
    ),
    dict(
        id="hazard-via-measurement-note",
        readings=two("lipo_cell", e1="clean pouch", e2="tags intact"),
        measurement="3.8 V but the cell feels swollen",
        expect=State.HAZARD,
        reason_contains=["swollen or damaged lithium cell"],
    ),
    dict(
        id="no-hazard-healthy-cell-asks-for-voltage",
        readings=two(
            "lipo_cell",
            e1="pouch flat and clean, no swelling",
            e2="stable, no damage marks",
        ),
        expect=State.NEEDS_MEASUREMENT,
        request_contains=["2.5 to 4.25 V"],
    ),
    dict(
        id="no-hazard-swollen-plastic-needs-cell-context",
        readings=two(
            "plastic_bezel",
            e1="the plastic bezel is swollen where a screw over-tightened",
            e2="cosmetic warping only",
        ),
        expect=State.VERIFIED,
    ),
    # 2. ABSTAIN -- quorum, agreement, contradictions, voter hygiene.
    dict(
        id="abstain-no-readings",
        readings=[],
        expect=State.ABSTAIN,
        reason_contains=["at least 2"],
    ),
    dict(
        id="abstain-single-reader-even-unanimous",
        readings=[rd("m1", "ribbon_connector")],
        expect=State.ABSTAIN,
        reason_contains=["1 independent reader"],
    ),
    dict(
        id="abstain-duplicate-reader-counts-once",
        readings=[
            rd("m1", "fan", confidence=0.9),
            rd("m1", "fan", confidence=0.7),  # the same model voting twice
        ],
        expect=State.ABSTAIN,
        reason_contains=["1 independent reader"],
    ),
    dict(
        id="abstain-agreement-2-of-3-below-threshold",
        readings=[
            rd("m1", "motor"),
            rd("m2", "motor"),
            rd("m3", "speaker"),
        ],
        expect=State.ABSTAIN,
        reason_contains=["below 0.67", "agreement 0.67"],
    ),
    dict(
        id="abstain-works-vs-dead-contradiction",
        readings=[
            rd("m1", "connector", "works"),
            rd("m2", "connector", "dead"),
        ],
        expect=State.ABSTAIN,
        reason_contains=["contradict"],
    ),
    dict(
        id="abstain-readings-for-other-parts-never-vote",
        readings=[
            rd("m1", "fan", part_id="p_999"),
            rd("m2", "fan", part_id="p_999"),
        ],
        expect=State.ABSTAIN,
        reason_contains=["0 independent reader"],
    ),
    # The caution invariant, end to end: three agreeing readers verify, and
    # losing one voter may only make the gate more cautious.
    dict(
        id="caution-three-agreeing-readers-verify",
        readings=[rd("m1", "speaker"), rd("m2", "speaker"), rd("m3", "speaker")],
        expect=State.VERIFIED,
    ),
    dict(
        id="caution-only-one-voter-left-abstains",
        readings=[rd("m1", "speaker")],  # m2 and m3 both failed to vote
        expect=State.ABSTAIN,
        reason_contains=["more cautious"],
    ),
    # 3. NEEDS_MEASUREMENT -- unknown condition, or a cheap check exists.
    dict(
        id="nm-unknown-condition-generic-ask",
        readings=two("ribbon_connector", condition="unknown"),
        expect=State.NEEDS_MEASUREMENT,
        reason_contains=["not on whether it works"],
        label_contains=["ribbon_connector"],
    ),
    dict(
        id="nm-motor-asks-for-winding-resistance",
        readings=two("bldc_motor", pn="2205"),
        expect=State.NEEDS_MEASUREMENT,
        request_contains=["0.5 to 20 ohms", "leads"],
    ),
    dict(
        id="nm-motor-unparseable-measurement-asks-again",
        readings=two("brushed_motor"),
        measurement="the banana slipped and I read gopher",
        expect=State.NEEDS_MEASUREMENT,
        reason_contains=["could not be parsed"],
        request_contains=["0.5 to 20 ohms"],
    ),
    dict(
        id="nm-capacitor-asks-for-esr",
        readings=two("electrolytic_capacitor", rating="470uF 25V"),
        expect=State.NEEDS_MEASUREMENT,
        request_contains=["under 5 ohms", "Discharge"],
    ),
    dict(
        id="nm-fuse-asks-for-continuity-even-if-readers-say-works",
        readings=two("blade_fuse"),
        expect=State.NEEDS_MEASUREMENT,
        request_contains=["continuity", "beep"],
    ),
    dict(
        id="nm-dead-claim-still-asks-when-type-has-a-check",
        readings=two("blade_fuse", condition="dead"),
        expect=State.NEEDS_MEASUREMENT,
        request_contains=["continuity"],
    ),
    # 4. VERIFIED -- measurement in range, or the type needs no measurement.
    dict(
        id="verify-motor-in-range",
        readings=two("stepper_motor", pn="42BYGH"),
        measurement="8.2 ohms",
        expect=State.VERIFIED,
        reason_contains=["in range", "8.2"],
    ),
    dict(
        id="motor-kilo-ohm-reading-contradicts-works-claim",
        readings=two("dc_motor"),
        measurement="1.2 k ohms",  # above 20 ohms even after the k multiplier
        expect=State.ABSTAIN,
        reason_contains=["out of range"],
    ),
    dict(
        id="verify-cell-in-range-fills-unknown-condition",
        readings=two("lipo_cell", condition="unknown", e1="flat pack", e2="hardcase intact"),
        measurement="3.7 V",
        expect=State.VERIFIED,
        reason_contains=["2.5 to 4.25 V"],
    ),
    dict(
        id="verify-fan-spins",
        readings=two("cooling_fan", pn="9733"),
        measurement="spins freely at 5 V rated",
        expect=State.VERIFIED,
        reason_contains=["in range"],
    ),
    dict(
        id="verify-fuse-continuity",
        readings=two("glass_fuse"),
        measurement="meter beeps, continuity holds",
        expect=State.VERIFIED,
    ),
    dict(
        id="verify-capacitor-esr-in-range",
        readings=two("ceramic_capacitor", rating="100uF 35V"),
        measurement="0.8 ohm",
        expect=State.VERIFIED,
    ),
    dict(
        id="verified-dead-no-check-type",
        readings=two("ribbon_connector", condition="dead"),
        expect=State.VERIFIED,
        label_contains=["dead"],
    ),
    dict(
        id="verified-dead-motor-open-leads-agrees-with-meter",
        readings=two("dc_motor", condition="dead"),
        measurement="OL, open circuit",
        expect=State.VERIFIED,
        label_contains=["dead"],
    ),
    dict(
        id="verified-dead-cell-zero-volts",
        readings=two("lipo_cell", condition="dead", e1="no cracks", e2="case intact"),
        measurement="0 V, reads flat",
        expect=State.VERIFIED,
    ),
    # 5. Consistency downgrades -- rail rule and part-family rule.
    dict(
        id="downgrade-cap-rating-below-its-rail",
        readings=two(
            "electrolytic_capacitor",
            rating="100uF 10V",
            e1="across the 12 V rail near the VRM",
            e2="sits on the 12v rail",
        ),
        measurement="1.1 ohm",
        expect=State.ABSTAIN,
        reason_contains=["below the rail"],
    ),
    dict(
        id="cap-on-rail-within-rating-verifies",
        readings=two(
            "electrolytic_capacitor",
            rating="470uF 35V",
            e1="across the 12 V rail near the VRM",
            e2="12v rail input filter",
        ),
        measurement="0.9 ohm",
        expect=State.VERIFIED,
    ),
    dict(
        id="downgrade-part-number-does-not-match-family",
        readings=two("bldc_motor", pn="MTR-99"),
        measurement="8 ohms",
        expect=State.ABSTAIN,
        reason_contains=["part number", "family"],
    ),
    dict(
        id="part-number-family-ok",
        readings=two("bldc_motor", pn="2205 2300KV BLDC"),
        measurement="8 ohms",
        expect=State.VERIFIED,
    ),
    dict(
        id="missing-part-number-never-downgrades",
        readings=two("bldc_motor", pn=None),
        measurement="8 ohms",
        expect=State.VERIFIED,
    ),
]

ROW_IDS = [row["id"] for row in SCENARIOS]


@pytest.mark.parametrize("row", SCENARIOS, ids=ROW_IDS)
def test_scenario(row: dict) -> None:
    verdict = run(row["readings"], row.get("measurement"))
    assert verdict.state == row["expect"], verdict.reasons
    for text in row.get("reason_contains", []):
        assert text in " ".join(verdict.reasons), verdict.reasons
    for text in row.get("request_contains", []):
        assert verdict.requested_measurement is not None, verdict.reasons
        assert text in verdict.requested_measurement, verdict.requested_measurement
    for text in row.get("label_contains", []):
        assert text in verdict.label, verdict.label


# --- invariants beyond the table -------------------------------------------


class TestGateInvariants:
    def test_deterministic(self) -> None:
        readings = [
            rd("m1", "bldc_motor", pn="2205", evidence="kv marking"),
            rd("m2", "bldc_motor", pn="2205", evidence="same"),
        ]
        first = decide(PART, readings, "8 ohms")
        second = decide(PART, list(reversed(readings)), "8 ohms")
        assert first == second  # input order must not matter

    def test_hazard_beats_lack_of_quorum(self) -> None:
        assert decide(PART, two("power_supply")[:1]).state == State.HAZARD

    def test_three_agreeing_readers_meet_the_threshold(self) -> None:
        # 3 of 3 is agreement 1.0; the table also encodes that 2 of 3
        # (0.667 < 0.67) must abstain.
        readings = [rd("m1", "speaker"), rd("m2", "speaker"), rd("m3", "speaker")]
        assert decide(PART, readings).state == State.VERIFIED

    def test_unparseable_measurement_beats_nothing(self) -> None:
        readings = two("bldc_motor", pn="2205")
        plain = decide(PART, readings)  # asks for ohms
        garbage = decide(PART, readings, "read it wrong")
        assert plain.state == garbage.state == State.NEEDS_MEASUREMENT
        assert plain.requested_measurement == garbage.requested_measurement

    def test_verdict_contract(self) -> None:
        verdict = run(two("speaker"))
        assert verdict.part_id == PART.id
        assert verdict.reasons  # never an unexplained verdict

    def test_empty_measurement_string_is_no_measurement(self) -> None:
        readings = two("speaker")
        assert run(readings, "   ") == run(readings, None)

    def test_measurement_ignored_for_types_with_no_check(self) -> None:
        # A made-up number for a check-less type neither verifies nor hurts:
        # the verdict comes from the readers alone.
        readings = two("panel_connector")
        assert run(readings, "999 ohms").state == State.VERIFIED


# --- unit tests for the table modules themselves ----------------------------


class TestChecksUnits:
    @pytest.mark.parametrize(
        "text, value",
        [
            ("8.2 ohms", 8.2),
            ("8", 8.0),
            ("1.2k ohm", 1200.0),
            ("OL, open circuit", 0.0),
        ],
    )
    def test_parse_ohms(self, text: str, value: float) -> None:
        assert parse_ohms(text) == pytest.approx(value)

    def test_parse_ohms_garbage_is_none(self) -> None:
        assert parse_ohms("gopher") is None

    @pytest.mark.parametrize(
        "text, value",
        [("3.7 V", 3.7), ("3.7", 3.7), ("open", 0.0)],
    )
    def test_parse_volts(self, text: str, value: float) -> None:
        assert parse_volts(text) == pytest.approx(value)

    def test_every_check_row_is_self_consistent(self) -> None:
        for check in CHECKS:
            assert check.instruction
            assert check.expected in check.instruction or check.name
            assert check.parse("unparseable nonsense") in (0.0, None)

    def test_find_check_by_type(self) -> None:
        assert find_check("bldc_motor") is not None
        assert find_check("lipo_cell") is not None
        assert find_check("cooling_fan") is not None
        assert find_check("blade_fuse") is not None
        assert find_check("electrolytic_capacitor") is not None
        assert find_check("ribbon_connector") is None

    def test_rating_and_rail_extraction(self) -> None:
        assert rating_volts("470uF 25V") == 25.0
        assert rating_volts("100uF, no volts") is None
        assert extract_rail("across the 12 V rail near the VRM") == 12.0
        assert extract_rail("rail of 5V feed") == 5.0
        assert extract_rail("no rail mentioned") is None

    def test_consistency_family_via_helper(self) -> None:
        readings = two("bldc_motor", pn="MTR-99")
        problems = consistency_problems("bldc_motor", readings)
        assert any("family" in p for p in problems)
        assert consistency_problems("ribbon_connector", readings) == []


class TestHazardsUnits:
    @pytest.mark.parametrize(
        "part_type, evidence, hazardous",
        [
            ("lipo_cell", "swollen pouch", True),
            ("lipo_cell", "clean pouch", False),
            ("connector", "plastic is swollen from heat shrink", False),
            ("connector", "a leaking capacitor and a bent cell", True),
            ("crt_tube", "charged", True),
            ("microwave_capacitor", "oily can", True),
            ("power_supply", "120 V AC in", True),
        ],
    )
    def test_find_hazard(self, part_type: str, evidence: str, hazardous: bool) -> None:
        assert (find_hazard(part_type, evidence) is not None) is hazardous

    def test_hazard_instead_is_actionable(self) -> None:
        for rule in HAZARD_RULES:
            assert "do not" in rule.instead.lower()
