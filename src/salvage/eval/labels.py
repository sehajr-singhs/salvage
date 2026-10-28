"""Labels for the eval harness: one row per part, one split per device.

The label file is a tiny CSV, ``data/labels/parts.csv``:

    device_id,part_id,true_type,true_part_number,true_condition,measured_value,hazard,split

Rules (roadmap section 5, brief 07):
- one row per labeled part; ``true_*`` columns are the human ground truth,
  written before looking at what the models say,
- ``true_condition`` is one of works/dead/unknown,
- ``measured_value`` is the free-text multimeter reading the human took
  (empty when none was taken) — this is what feeds the D3 resolution metric,
- ``hazard`` is true/false,
- ``split`` is dev or test, and **the split is by device, never by part**:
  every part of one device is in the same split, so nothing leaks between
  development and the held-out test.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

VALID_CONDITIONS = ("works", "dead", "unknown")
VALID_SPLITS = ("dev", "test")

COLUMNS = (
    "device_id",
    "part_id",
    "true_type",
    "true_part_number",
    "true_condition",
    "measured_value",
    "hazard",
    "split",
)

_TRUE_VALUES = {"true", "1", "yes", "y"}


@dataclass(frozen=True)
class LabelRow:
    """One labeled part: the truth about it, and which split it belongs to."""

    device_id: str
    part_id: str
    true_type: str
    true_part_number: str | None
    true_condition: str
    measured_value: str | None
    hazard: bool
    split: str

    @property
    def is_test(self) -> bool:
        return self.split == "test"


def _clean(value: str | None) -> str | None:
    """An empty CSV cell is None, not the empty string."""
    if value is None:
        return None
    value = value.strip()
    return value or None


def _clean_bool(value: str | None) -> bool:
    return _clean(value) is not None and _clean(value).lower() in _TRUE_VALUES  # type: ignore[union-attr]


def load_labels(path: str | Path) -> list[LabelRow]:
    """Parse a labels CSV, validating every row strictly.

    Raises ``ValueError`` naming the first row that is malformed: conditions
    and splits must be from the fixed vocabularies, and none of device_id,
    part_id, true_type or split may be missing. bad cells keep the harness
    honest rather than silently dropping parts.
    """
    rows: list[LabelRow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = tuple(reader.fieldnames or ())[0 : len(COLUMNS)]
        if fields != COLUMNS:
            raise ValueError(
                f"{path}: label header must be exactly {','.join(COLUMNS)}, "
                f"got {','.join(fields)}"
            )
        for number, record in enumerate(reader, start=2):
            line = f"{path} line {number}"
            device_id = _clean(record.get("device_id"))
            part_id = _clean(record.get("part_id"))
            true_type = _clean(record.get("true_type"))
            split = _clean(record.get("split"))
            condition = _clean(record.get("true_condition"))
            if not (device_id and part_id and true_type and split):
                raise ValueError(f"{line}: empty required cell")
            if condition not in VALID_CONDITIONS:
                raise ValueError(
                    f"{line}: true_condition must be one of "
                    f"{'/'.join(VALID_CONDITIONS)}, got {condition!r}"
                )
            if split not in VALID_SPLITS:
                raise ValueError(
                    f"{line}: split must be one of {'/'.join(VALID_SPLITS)}, "
                    f"got {split!r}"
                )
            rows.append(
                LabelRow(
                    device_id=device_id,
                    part_id=part_id,
                    true_type=true_type,
                    true_part_number=_clean(record.get("true_part_number")),
                    true_condition=condition,
                    measured_value=_clean(record.get("measured_value")),
                    hazard=_clean_bool(record.get("hazard")),
                    split=split,
                )
            )
    if not rows:
        raise ValueError(f"{path}: no label rows")
    return rows


def check_device_split(rows: list[LabelRow]) -> list[str]:
    """Device ids whose parts sit in more than one split — these must be fixed.

    The split is by device so nothing leaks between dev and test; a device
    that appears in both would leak, so the harness refuses to score until
    the list it returns here is empty.
    """
    seen: dict[str, set[str]] = {}
    for row in rows:
        seen.setdefault(row.device_id, set()).add(row.split)
    return sorted(d for d, splits in seen.items() if len(splits) > 1)
