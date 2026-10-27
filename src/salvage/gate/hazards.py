"""Hazard rules for the gate: when to refuse a teardown outright.

A hazard match beats every other gate state. Each rule is a small table entry:

- ``type_patterns``: if the claimed ``part_type`` matches, refuse (these are
  categories that are unsafe to open as a class).
- ``evidence_patterns``: if a reader's evidence (or a user-supplied
  measurement note) mentions a dangerous condition, refuse. Rules with a
  ``context_patterns`` field only fire when the dangerous wording co-occurs
  with the risky context, so a reader saying "swollen" about, say, a plastic
  bezel does not trip the lithium rule.

Every rule carries an ``instead`` action: the HAZARD reason must say what to
do instead, never just that the gate refuses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from salvage.schemas import Reading


def _rx(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


# Words that say the thing in front of you is a lithium cell. Used both as a
# part_type detector and as the context guard on the damage-evidence rule.
LITHIUM_CONTEXT = _rx(
    r"li[-\s_]*(?:ion|po|poly)",
    r"lipo",
    r"li-?ion",
    r"lithium",
    r"18\d50|21\d700",
    r"\bcell\b",
    r"\bbatt(?:ery)?\b",
    r"\bpack\b",
)


@dataclass(frozen=True)
class Hazard:
    """One matched hazard, ready to become a Verdict's reasons."""

    name: str
    """Short label for the Verdict."""

    trigger: str
    """What matched (for the gate's reasons list)."""

    instead: str
    """What to do instead of the teardown the agent would have suggested."""


@dataclass(frozen=True)
class HazardRule:
    name: str
    instead: str
    type_patterns: tuple[re.Pattern[str], ...] = ()
    """Match these against the claimed part_type alone."""
    evidence_patterns: tuple[re.Pattern[str], ...] = ()
    """Match these against evidence / measurement note text."""
    context_patterns: tuple[re.Pattern[str], ...] | None = None
    """If given, evidence patterns only fire when context words appear too."""

    def as_hazard(self, trigger: re.Pattern[str]) -> Hazard:
        return Hazard(
            name=self.name,
            trigger=trigger.pattern,
            instead=self.instead,
        )


HAZARD_RULES: tuple[HazardRule, ...] = (
    HazardRule(
        name="swollen or damaged lithium cell",
        instead=(
            "do not puncture, bend or short the cell and do not build a pack "
            "with it; put it straight into a sand bucket or fireproof bag and "
            "take it to a battery recycling drop-off"
        ),
        evidence_patterns=_rx(
            r"swollen",
            r"puffed",
            r"punctur\w*",
            r"\bbent\b",
            r"\bdent(?:ed)?\b",
            r"crush\w*",
            r"leak(?:s|ed|ing)?",
            r"ruptur\w*",
            r"vent(?:ed|ing)?",
        ),
        context_patterns=LITHIUM_CONTEXT,
    ),
    HazardRule(
        name="CRT or flyback",
        instead=(
            "do not open a CRT or touch a flyback; the tube holds a lethal "
            "charge even unplugged and needs a trained person to bleed it "
            "before it can be recapped, so take the whole unit to e-waste"
        ),
        type_patterns=_rx(r"\bcrt\b", r"cathode.?ray", r"flyback"),
        evidence_patterns=_rx(r"\bcrt?s?\b", r"cathode.?ray", r"\bflyback\b"),
    ),
    HazardRule(
        name="microwave HV capacitor or magnetron",
        instead=(
            "do not discharge, probe or open it; a microwave high-voltage "
            "capacitor can hold a lethal charge and its bleeder fails "
            "silently, so leave the assembly intact and take the unit to e-waste"
        ),
        type_patterns=_rx(r"microwave", r"magnetron", r"\bhv[ _-]?cap\w*"),
        evidence_patterns=_rx(
            r"microwave",
            r"\bmagnetron\b",
            r"high[- ]voltage capacitor",
            r"\bhv capacitor\b",
        ),
    ),
    HazardRule(
        name="mains primary side",
        instead=(
            "do not open or probe the mains primary side even unplugged; the "
            "bulk caps hold line voltage for days, so leave it to a qualified "
            "person or recycle the whole board"
        ),
        type_patterns=_rx(
            r"power[_ -]?supply",
            r"\bpsu\b",
            r"ac[_ -]?(?:adapter|brick|input|side)",
            r"ac[-/]dc",
            r"mains",
        ),
        evidence_patterns=_rx(
            r"\bmains\b",
            r"primary side",
            r"line[- ]?side",
            r"line voltage",
            r"\b1[01][05]\s*v\.?\s*ac\b",
            r"2[34]0\s*v\s*ac\b",
        ),
    ),
)


def _normalize(text: str) -> str:
    """Lowercase and split snake_case/kebab_case tokens into words, so the
    word boundaries in the rule patterns behave as written."""
    return text.lower().replace("_", " ").replace("-", " ")


def find_hazard(
    part_type: str = "",
    evidence: str = "",
    extra: str = "",
) -> Hazard | None:
    """Match one reading's fields (plus a measurement note) against the rules.

    Returns the first hazard that matches, or ``None`` when the part looks
    touchable. Rules are evaluated in table order, most specific first.
    """
    ptype = _normalize(part_type)
    hay = f"{_normalize(evidence)}\n{_normalize(extra)}"
    for rule in HAZARD_RULES:
        for pattern in rule.type_patterns:
            if pattern.search(ptype):
                return rule.as_hazard(pattern)
        if rule.evidence_patterns:
            context_ok = rule.context_patterns is None or any(
                p.search(hay) or p.search(ptype)
                for p in rule.context_patterns
            )
            if context_ok:
                for pattern in rule.evidence_patterns:
                    if pattern.search(hay):
                        return rule.as_hazard(pattern)
    return None


def find_hazard_in_readings(
    readings: Sequence[Reading],
    extra: str = "",
) -> Hazard | None:
    """Scan every reading (and an optional measurement note) for hazards."""
    for reading in readings:
        hazard = find_hazard(
            part_type=reading.part_type,
            evidence=reading.evidence,
        )
        if hazard is not None:
            return hazard
    return find_hazard(evidence=extra) if extra else None
