"""Cheap physical checks and consistency rules for the gate.

Two tables live here:

``CHECKS``
    One row per part type that has a cheap bench check the agent can ask for
    out loud (rule 3 of the brief). Each row carries the plain-English
    instruction, the expected range as text, a ``parse`` function that turns a
    free-text measurement into a float (``None`` = could not parse), and a
    ``verdict`` predicate saying whether the number proves the part works. A
    parsed number outside the range contradicts a "works" claim but is fully
    consistent with a "dead" claim.

``FAMILY_PATTERNS``
    Marking-style part numbers that genuinely belong to each part family
    (consistency rule: a part_number that matches none of its family's
    patterns downgrades the part to ABSTAIN). Missing part numbers never
    downgrade: missing evidence may only make the gate more cautious, and
    here it simply has nothing to check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

from salvage.schemas import Reading


def _normalize(text: str) -> str:
    """snake_case and kebab-case part types become space-separated words."""
    return text.lower().replace("_", " ").replace("-", " ")


def _rx(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


_NUMBER = r"(\d+(?:\.\d+)?)"

# A meter reading that reports nothing readable is treated as 0.0: never in
# range, so a "works" claim is contradicted (cautious), and it never raises
# the gate's confidence.
_OPEN_CIRCUIT = re.compile(
    r"\b(?:ol|o\.l|open(?: circuit)?|infinit\w*|no ?continuity|seiz\w*|jamm\w*)\b",
    re.IGNORECASE,
)


def parse_ohms(text: str) -> float | None:
    """Resistance: ``8``, ``8.2 ohms``, ``1.8k``. Units default to ohms."""
    if not text or _OPEN_CIRCUIT.search(text):
        return 0.0
    match = re.search(rf"{_NUMBER}\s*(k|m)?\s*(?:ohms?|Ω)?", text, re.IGNORECASE)
    if match is None:
        return None
    value = float(match.group(1))
    if match.group(2):
        value *= 1000.0 if match.group(2).lower() == "k" else 1_000_000.0
    return value


def parse_volts(text: str) -> float | None:
    """DC voltage: ``3.7``, ``3.7 V``. Units default to volts."""
    if not text or _OPEN_CIRCUIT.search(text):
        return 0.0
    match = re.search(rf"{_NUMBER}\s*v?", text, re.IGNORECASE)
    return float(match.group(1)) if match else None


def parse_spin(text: str) -> float | None:
    """Fan spin test: 1.0 = spun freely at rated voltage, 0.0 = did not."""
    t = text.lower()
    if "spin" not in t:
        return None
    if _OPEN_CIRCUIT.search(t) or re.search(r"(no|not|doesn'?t|didn'?t) spin", t):
        return 0.0
    return 1.0 if "spin" in t else None


def parse_continuity(text: str) -> float | None:
    """Fuse continuity: 1.0 = beeps/held, 0.0 = open or blown."""
    t = text.lower()
    if _OPEN_CIRCUIT.search(t) or "blown" in t:
        return 0.0
    if re.search(r"\bcontinu\w*|\bbeeps?\b", t):
        return 1.0
    return None


@dataclass(frozen=True)
class Check:
    """One cheap measurement the gate can request for a family of types."""

    name: str
    type_patterns: tuple[re.Pattern[str], ...]
    instruction: str
    """Plain-English, spoken at the bench. Must include the expected range."""
    expected: str
    """The expected range as short text, e.g. ``0.5 to 20 ohms``."""
    parse: Callable[[str], float | None]
    verdict: Callable[[float], bool]

    def matches_type(self, part_type: str) -> bool:
        part_type = _normalize(part_type)
        return any(p.search(part_type) for p in self.type_patterns)


def between(lo: float, hi: float) -> Callable[[float], bool]:
    return lambda v: lo <= v <= hi


CHECKS: tuple[Check, ...] = (
    Check(
        name="winding resistance",
        type_patterns=_rx(r"\bmotors?\b", r"\bbldc\b", r"\bsteppers?\b", r"\bbrush(ed|less)?\b"),
        instruction=(
            "Put the meter across the two motor leads on ohms and read it out; "
            "I expect 0.5 to 20 ohms of winding resistance."
        ),
        expected="0.5 to 20 ohms",
        parse=parse_ohms,
        verdict=between(0.5, 20.0),
    ),
    Check(
        name="open-circuit voltage",
        type_patterns=_rx(
            r"lipo", r"li[-_ ]?ion", r"lithium", r"18\d50", r"21\d70",
            r"\bcell\b", r"\bbatt(?:ery)?\b",
        ),
        instruction=(
            "Measure the cell's open-circuit voltage across its two terminals "
            "with the meter on DC volts; I expect 2.5 to 4.25 V."
        ),
        expected="2.5 to 4.25 V",
        parse=parse_volts,
        verdict=between(2.5, 4.25),
    ),
    Check(
        name="spin test at rated voltage",
        type_patterns=_rx(r"\bfans?\b", r"\bblowers?\b"),
        instruction=(
            "Power the fan at its rated voltage and watch the blades; I expect "
            "it to spin freely, no grind, no stall."
        ),
        expected="spins freely at rated voltage",
        parse=parse_spin,
        verdict=lambda v: v == 1.0,
    ),
    Check(
        name="continuity",
        type_patterns=_rx(r"\bfuses?\b", r"\bfusibles?\b", r"thermal[_ -]fuse"),
        instruction=(
            "Put the meter on continuity across the fuse's two pads; I expect "
            "a beep, open means it blew."
        ),
        expected="continuity",
        parse=parse_continuity,
        verdict=lambda v: v == 1.0,
    ),
    Check(
        name="ESR",
        type_patterns=_rx(r"\bcapacitors?\b", r"\belectrolytics?\b"),
        instruction=(
            "Discharge the capacitor through a resistor first, then measure "
            "ESR across the leads; I expect under 5 ohms."
        ),
        expected="under 5 ohms",
        parse=parse_ohms,
        verdict=lambda v: 0.0 < v < 5.0,
    ),
)


def find_check(part_type: str) -> Check | None:
    """The first check table row whose type patterns match, or None."""
    for check in CHECKS:
        if check.matches_type(part_type):
            return check
    return None


# --- consistency rules (rule 5: downgrade to ABSTAIN) ----------------------


FAMILY_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "capacitor": _rx(r"\d+\s*[pnuµu]f\b", r"[a-z]{2,4}\d{2}"),
    "motor": _rx(r"\d{4}", r"\bkv\b", r"bldc", r"brush", r"\d{2}\s?by\w*"),
    "fuse": _rx(r"fusible", r"\bfuse\b", r"\bf\d{2,}"),
    "fan": _rx(r"\bfans?\b", r"blower", r"\bcfm\b", r"\d+\s?mm\b", r"\d{4}"),
}

_RATING_VOLTS = re.compile(rf"{_NUMBER}\s*v", re.IGNORECASE)
_RAIL_VOLTS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        rf"{_NUMBER}\s*v(?:olts?)?[- _]?rail",
        rf"rail\s*(?:of\s*|at\s*|is\s*)?{_NUMBER}\s*v",
        rf"sits?\s*on\s*(?:the\s*)?{_NUMBER}\s*v",
    )
)


def extract_rail(evidence: str) -> float | None:
    """The rail voltage a reader says the part sits on, if it says one."""
    for pattern in _RAIL_VOLTS:
        match = pattern.search(evidence)
        if match:
            return float(match.group(1))
    return None


def rating_volts(key_rating: str) -> float | None:
    """The voltage out of a key_rating string like ``100uF 35V``."""
    match = _RATING_VOLTS.search(key_rating)
    return float(match.group(1)) if match else None


def family_name(part_type: str) -> str | None:
    """The family table key contained in a part_type, if any."""
    normalized = _normalize(part_type)
    for name in FAMILY_PATTERNS:
        if name in normalized:
            return name
    return None


def consistency_problems(
    part_type: str,
    readings: Sequence[Reading],
) -> list[str]:
    """Human-readable consistency failures for the agreeing readings.

    Two rules, both downgrade to ABSTAIN:
    - a capacitor whose rating voltage is below the rail a reader says it
      sits on (checked only when both voltages are actually known),
    - a part_number that matches none of its type family's marking patterns.
    """
    problems: list[str] = []
    if "capacitor" in part_type.lower():
        rails = [
            v
            for reading in readings
            for v in (extract_rail(reading.evidence),)
            if v is not None
        ]
        ratings = [
            v
            for reading in readings
            if reading.key_rating
            for v in (rating_volts(reading.key_rating),)
            if v is not None
        ]
        if rails and ratings:
            rail, rating = min(rails), min(ratings)
            if rating < rail:
                problems.append(
                    f"capacitor rated {rating:g} V sits on a {rail:g} V rail; "
                    "its voltage rating is below the rail it is on"
                )
    family = family_name(part_type)
    if family is not None:
        patterns = FAMILY_PATTERNS[family]
        for reading in readings:
            if not reading.part_number:
                continue
            if not any(p.search(reading.part_number) for p in patterns):
                problems.append(
                    f"part number {reading.part_number!r} does not match the "
                    f"{family} part family"
                )
    return problems
