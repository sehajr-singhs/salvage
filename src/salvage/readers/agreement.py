"""Brief 03: agreement across the readers.

:func:`agreement` computes, over a list of Readings, the share of
readings that agree on part_type *and* on the part_number family —
the independent-evidence signal the gate (04) and the eval harness (07)
use to decide whether a part is worth believing at all.

Two deliberate conservatism rules, both on the "never more confident
because evidence is missing" principle:

- part_types are compared after light normalization only (same words,
  lowercase, whitespace -> underscore). A reader that says ``bldc_motor``
  and one that says ``BLDC motor`` agree; a reader that says ``esc`` does
  not.
- the part_number family is the number flattened to a single lowercase
  alphanumeric run (``IRF 540-N`` -> ``irf540n``). Suffix variants that
  render differently — ``IRF540`` vs ``IRF540N`` — count as *different*
  families, so ambiguous markings lower agreement rather than raise it.
- a reading with no legible part_number (family ``None``) does not
  agree with a family someone else did read; it agrees only when nobody
  could read a number at all.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

from salvage.schemas import Reading

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def part_number_family(part_number: str | None) -> str | None:
    """The part number flattened to one lowercase alphanumeric run."""
    if not part_number:
        return None
    cleaned = _NON_ALNUM.sub("", part_number.lower())
    return cleaned or None


def _norm_type(part_type: str) -> str:
    """Same words, lowercase, any whitespace runs -> underscores."""
    return "_".join(part_type.split()).lower()


def _mode(counter: Counter[str | None]) -> str | None:
    """Most common value; count ties break to the smallest, None last.

    Deterministic so the same readings always produce the same verdict.
    """
    return min(counter.items(), key=lambda kv: (-kv[1], kv[0] or "", kv[0] is None))[0]


def agreement(readings: Sequence[Reading]) -> float:
    """The share of readings that agree on part_type and part_number family."""
    if not readings:
        return 0.0
    type_mode = _mode(Counter(_norm_type(r.part_type) for r in readings))
    family_mode = _mode(Counter(part_number_family(r.part_number) for r in readings))
    agreeing = sum(
        1
        for reading in readings
        if _norm_type(reading.part_type) == type_mode
        and part_number_family(reading.part_number) == family_mode
    )
    return agreeing / len(readings)
