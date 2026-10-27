"""decide(): the deterministic verify-or-abstain gate. Plain code, no models.

Turns a Part, the readers' Readings, and one optional free-text measurement
into a single Verdict, applying the brief's rules strictly in order:

1. HAZARD if any reading or rule flags a dangerous part (gate/hazards.py).
   The reason says what to do instead.
2. ABSTAIN if fewer than 2 valid readings vote, or agreement < 0.67.
3. NEEDS_MEASUREMENT if the readers agree on the type but the condition is
   unknown, or the type has a cheap check in gate/checks.py and no valid
   measurement has been taken yet.
4. VERIFIED only when the readings agree and either the measurement is in
   range or the type needs no measurement.
5. Consistency failures (capacitor rated below its rail, a part_number that
   matches no family marking) downgrade to ABSTAIN.

Two invariants drive every tie:
- A missing or unparseable measurement is treated the same as no
  measurement: missing evidence may only make the gate more cautious,
  never more confident.
- The gate counts exactly one vote per reader model, so duplicate readers
  cannot inflate agreement.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from salvage.gate.checks import Check, consistency_problems, find_check
from salvage.gate.hazards import find_hazard, find_hazard_in_readings
from salvage.schemas import Condition, Part, Reading, State, Verdict

MIN_READERS = 2
MIN_AGREEMENT = 0.67


def _votes_for(part: Part, readings: Sequence[Reading]) -> list[Reading]:
    """Readings that actually vote: right part, one per reader model."""
    votes: dict[str, Reading] = {}
    for reading in readings:
        if reading.part_id != part.id:
            continue
        prior = votes.get(reading.model_id)
        if prior is None or (reading.confidence, reading.evidence) > (
            prior.confidence,
            prior.evidence,
        ):
            votes[reading.model_id] = reading
    return sorted(votes.values(), key=lambda r: (r.model_id, r.confidence))


def _agree_condition(readings: Sequence[Reading]) -> Condition | None:
    """The condition the agreeing readers support, or None on contradiction."""
    conds = {reading.condition for reading in readings}
    if Condition.WORKS in conds and Condition.DEAD in conds:
        return None  # readers contradict each other outright
    if len(conds) == 1:
        return conds.pop()
    return Condition.UNKNOWN  # anything mixed with unknown stays unknown


def _consensus_label(part_type: str, readings: Sequence[Reading]) -> str:
    numbers = sorted({r.part_number for r in readings if r.part_number})
    condition = _agree_condition(readings)
    label = part_type
    if len(numbers) == 1:
        label += f" {numbers[0]}"
    if condition is Condition.DEAD:
        label += " (dead)"
    return label


def _abstain(
    part_id: str,
    reasons: list[str],
    requested_measurement: str | None = None,
) -> Verdict:
    return Verdict(
        part_id=part_id,
        state=State.ABSTAIN,
        label="unproven",
        reasons=reasons,
        requested_measurement=requested_measurement,
    )


def _needs_measurement(
    part_id: str,
    label: str,
    reasons: list[str],
    instruction: str,
) -> Verdict:
    return Verdict(
        part_id=part_id,
        state=State.NEEDS_MEASUREMENT,
        label=label,
        reasons=reasons,
        requested_measurement=instruction,
    )


def decide(
    part: Part,
    readings: Sequence[Reading],
    measurement: str | None = None,
) -> Verdict:
    """Turn readings plus an optional measurement into one Verdict."""
    own = [r for r in readings if r.part_id == part.id]
    note = (measurement or "").strip() or None

    # 1. HAZARD beats everything, even one reading, even on no readings.
    hazard = find_hazard_in_readings(own, extra=note or "")
    if hazard is not None:
        return Verdict(
            part_id=part.id,
            state=State.HAZARD,
            label=hazard.name,
            reasons=[
                f"refusing: {hazard.name} ({hazard.trigger})",
                f"instead: {hazard.instead}",
            ],
        )

    # 2. Enough voters, and they agree on what the part is.
    votes = _votes_for(part, readings)
    if len(votes) < MIN_READERS:
        supplied = len(own)
        return _abstain(
            part.id,
            [
                f"only {len(votes)} independent reader(s) voted "
                f"({supplied} reading(s) supplied); the gate needs at least "
                f"{MIN_READERS}",
                "missing readers may only make the gate more cautious, so it abstains",
            ],
        )

    counts = Counter(reading.part_type for reading in votes)
    top_type, top_count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    agreement = top_count / len(votes)
    agree = [reading for reading in votes if reading.part_type == top_type]

    reasons = [
        f"{top_count} of {len(votes)} readers agree the part is {top_type!r} "
        f"(agreement {agreement:.2f})",
    ]
    if agreement < MIN_AGREEMENT:
        return _abstain(
            part.id,
            reasons
            + [
                f"agreement is below {MIN_AGREEMENT}: the readers do not "
                "agree on what this part is, so nothing can be verified",
            ],
        )

    condition = _agree_condition(agree)
    if condition is None:
        return _abstain(
            part.id,
            reasons
            + [
                "readers contradict each other on the condition (one says it "
                "works, another says it is dead)",
            ],
        )

    label = _consensus_label(top_type, agree)
    check: Check | None = find_check(top_type)

    # 5. Consistency downgrades. A part whose markings do not fit the story
    # the readers tell must abstain, whether it was headed for VERIFIED or
    # NEEDS_MEASUREMENT; the ask is kept so the bench agent can still propose
    # the measurement that would settle it.
    problems = consistency_problems(top_type, agree)
    if problems:
        instruction = (
            check.instruction
            if check is not None
            else "Report one direct observation that proves this part works "
            "(its full marking and any rating, and what you probed)."
        )
        return _abstain(part.id, reasons + problems, requested_measurement=instruction)

    # 3 + 4. States depend on the claimed condition and the measurement.
    parsed: float | None = None
    if check is not None and note:
        parsed = check.parse(note)

    if parsed is not None:
        verdict_word = "in range" if check.verdict(parsed) else "out of range"
        reasons.append(
            f"measurement {note!r} parses to {parsed:g} ({check.name}); "
            f"expected {check.expected}; it is {verdict_word}"
        )
        if not check.verdict(parsed):
            if condition is Condition.DEAD:
                # Out of range is exactly what a dead part looks like: the
                # claim and the meter agree.
                reasons.append("readers also report the part is dead")
                return Verdict(
                    part_id=part.id,
                    state=State.VERIFIED,
                    label=label,
                    reasons=reasons,
                )
            return _abstain(
                part.id,
                reasons
                + [
                    "the measurement contradicts the readers' condition, so "
                    "the gate proves nothing either way",
                ],
            )
        # In range and (works or unknown): the meter fills the evidence gap.
        return Verdict(
            part_id=part.id,
            state=State.VERIFIED,
            label=label,
            reasons=reasons,
        )

    if condition is Condition.UNKNOWN or check is not None:
        if condition is Condition.UNKNOWN:
            reasons.append(
                "readers agree on the type but not on whether it works"
            )
        if check is None:
            reasons.append("no cheap check exists for this type")
            instruction = (
                "Report one direct observation that proves this part works "
                "(its full marking and any rating, and what you probed); the "
                "readers cannot decide the condition from photos alone."
            )
        else:
            if parsed is None and note:
                reasons.append(
                    f"the measurement {note!r} could not be parsed as "
                    f"{check.expected}"
                )
            else:
                reasons.append(f"this type has a cheap check: {check.name}")
            instruction = check.instruction
        return _needs_measurement(part.id, label, reasons, instruction)

    # No measurement asked for and the readers agree on works/dead.
    if condition is Condition.WORKS:
        reasons.append("this type needs no measurement to verify")
    else:
        reasons.append("readers unanimously report the part is dead")
    return Verdict(
        part_id=part.id,
        state=State.VERIFIED,
        label=label,
        reasons=reasons,
    )
