"""The Salvage verify-or-abstain gate.

Public surface: ``decide(part, readings, measurement=None) -> Verdict``
plus the hazard and check tables it is built from. Later briefs (readers,
measure, web) import from here and never redo the rules.
"""

from salvage.gate.checks import (
    CHECKS,
    FAMILY_PATTERNS,
    Check,
    consistency_problems,
    find_check,
)
from salvage.gate.decide import (
    MIN_AGREEMENT,
    MIN_READERS,
    decide,
)
from salvage.gate.hazards import (
    HAZARD_RULES,
    Hazard,
    HazardRule,
    find_hazard,
    find_hazard_in_readings,
)

__all__ = [
    "CHECKS",
    "Check",
    "FAMILY_PATTERNS",
    "HAZARD_RULES",
    "Hazard",
    "HazardRule",
    "MIN_AGREEMENT",
    "MIN_READERS",
    "consistency_problems",
    "decide",
    "find_check",
    "find_hazard",
    "find_hazard_in_readings",
]
