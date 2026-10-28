"""The eval harness runner (brief 07): ``python -m salvage.eval.run --split dev|test``.

Reads the label file, builds Readings, runs the verify-or-abstain gate over
them at several strictness levels (the coverage-error curve), runs the
baseline comparator (trusting one reader's own confidence), re-runs the gate
with the labeled multimeter reading as the measurement (the D3 resolution
pass), computes D2's AUROCs, and writes ``reports/results.md`` with plots.

Offline path, per the README rule that no keys exist yet: the Readings come
from a deterministic mock reader seeded per part (``--seed``), so every run
is reproducible and costs nothing. When a real reader package exists and
keys arrive, the mock block is the single swap point — nothing downstream
of ``build_readings`` knows where readings came from.

The test split refuses to run without ``--i-mean-it``: dev numbers are for
iterating, the held-out test is run once, on purpose, late (roadmap section 5).
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

from salvage.eval import report
from salvage.eval.labels import LabelRow, check_device_split, load_labels
from salvage.eval.metrics import (
    baseline_confidence_curve,
    coverage_error_curve,
    disagreement_auroc,
    disagreement_scores,
    measurement_resolution_counts,
    roc_curve,
    wrong_verified_ids,
)
# The gate package re-exports the ``decide`` function as an attribute, which
# shadows the submodule name, so resolve the module itself explicitly: that is
# the object whose MIN_AGREEMENT the strictness sweep patches.
GATE_DECIDE = importlib.import_module("salvage.gate.decide")
decide = GATE_DECIDE.decide
from salvage.schemas import Condition, Part, Reading

DEFAULT_LABELS = "data/labels/parts.csv"
DEFAULT_OUT = "reports"
DEFAULT_THRESHOLDS = (1.0, 0.9, 0.8, 0.67)

READER_MODELS = ("muse-spark-1.3", "muse-spark-open", "free-pool-contributor")

# Plausible confusions, used when the mock reader misidentifies a part.
DECOY_TYPES = {
    "motor": "fan",
    "fan": "motor",
    "capacitor": "resistor",
    "resistor": "capacitor",
    "battery": "capacitor",
    "connector": "switch",
}
DEFAULT_DECOY = "connector"

# Evidence a mock reader reports about a hazardous part: keyed by words that
# appear in labeled true_type values, matched in order against the normalized
# type. The text must trip the matching rule in gate/hazards.py, which is how
# the HAZARD state shows up in the offline numbers.
_HAZARD_EVIDENCE: tuple[tuple[str, str], ...] = (
    ("lipo", "the cell looks swollen, straight to recycling"),
    ("cell", "the cell looks swollen, straight to recycling"),
    ("battery", "the cell looks swollen, straight to recycling"),
    ("mains", "mains primary side visible, board not openable"),
    ("power supply", "mains primary side visible, board not openable"),
    ("psu", "mains primary side visible, board not openable"),
    ("crt", "the crt tube holds a lethal charge, take the unit to e-waste"),
    ("flyback", "the flyback holds a lethal charge, take the unit to e-waste"),
    ("microwave", "microwave high-voltage assembly, leave it sealed"),
    ("magnetron", "microwave high-voltage assembly, leave it sealed"),
)


def _draw_proofs_wrong(model_rng: random.Random, true_pn: str | None) -> str | None:
    """A wrong part-number transcription: same lead-in, mangled tail."""
    if true_pn is None:
        return None
    return f"{true_pn[:2].upper()}-{model_rng.randint(10, 99):02d}"


def build_readings(rows: Sequence[LabelRow], seed: int) -> list[Reading]:
    """Deterministic mock readers, one reading per model per labeled part.

    Seeded by ``seed`` and the part id, so reruns reproduce exactly. Every
    6th part or so is "hard", in one of three flavors:

    - ``confident-wrong``: all readers agree on the same wrong family at
      high confidence — the hardest failure, D1's only real risk,
    - ``uncertain``: everyone guesses a decoy type and shrugs — cheap
      abstentions, exactly what the gate should refuse to verify,
    - ``conflicted``: readers split on the family — the raw material the
      D2 disagreement signal is built on.

    Easy parts read the true family with the occasional reader mistake and
    mostly-reliable part numbers, so the gate has honest work to do.
    """
    readings: list[Reading] = []
    for row in rows:
        part_rng = random.Random(f"{seed}:{row.part_id}")
        true_pn = row.true_part_number
        hazard_text = next(
            (text for key, text in _HAZARD_EVIDENCE if key in row.true_type.lower()),
            None,
        )

        if row.hazard:
            # The label row says refuse; pick the evidence text the gate's
            # hazard rules will catch, keyed on the labeled type.
            flavor = "hazard"
        elif part_rng.random() < 0.18:
            flavor = "confident-wrong" if part_rng.random() < 0.4 else (
                "uncertain" if part_rng.random() < 0.5 else "conflicted"
            )
        else:
            flavor = "easy"

        # One shared condition profile per part for easy parts: readers are
        # mostly unanimous, with occasional useful hedging. Truth-unknown
        # parts claim nothing about the condition, like a good reader would.
        if row.true_condition == "works":
            profile_roll = part_rng.random()
            base_condition = (
                "works" if profile_roll < 0.78 else "unknown" if profile_roll < 0.88 else "mixed"
            )
        else:
            base_condition = (
                "dead" if row.true_condition == "dead" and part_rng.random() < 0.82 else "unknown"
            )

        for model_index, model_id in enumerate(READER_MODELS):
            reader_rng = random.Random(f"{seed}:{row.part_id}:{model_id}")

            if flavor == "hazard":
                readings.append(
                    Reading(
                        part_id=row.part_id,
                        model_id=model_id,
                        part_type=row.true_type,
                        part_number=true_pn,
                        condition=Condition.UNKNOWN,
                        confidence=round(0.6 + 0.3 * reader_rng.random(), 2),
                        evidence=hazard_text or "",
                        key_rating=None,
                    )
                )
                continue

            if flavor == "confident-wrong":
                right_type = False
                condition = {
                    "works": Condition.WORKS,
                    "dead": Condition.DEAD,
                    "unknown": Condition.UNKNOWN,
                }[row.true_condition]
                confidence = 0.78 + 0.17 * reader_rng.random()
            elif flavor == "uncertain":
                right_type = False
                condition = Condition.UNKNOWN
                confidence = 0.4 + 0.25 * reader_rng.random()
            elif flavor == "conflicted":
                right_type = reader_rng.random() < 0.5
                condition = (
                    (Condition.WORKS, Condition.UNKNOWN)[reader_rng.random() < 0.6]
                    if row.true_condition in ("works", "unknown")
                    else Condition.UNKNOWN
                )
                confidence = 0.45 + 0.35 * reader_rng.random()
            else:  # easy
                right_type = reader_rng.random() < 0.92
                condition = {
                    "works": Condition.WORKS,
                    "dead": Condition.DEAD,
                    "unknown": Condition.UNKNOWN,
                }.get(base_condition, Condition.UNKNOWN)
                if base_condition == "mixed" and model_index == 0:
                    condition = Condition.DEAD
                elif condition is Condition.WORKS and reader_rng.random() < 0.1:
                    condition = Condition.UNKNOWN  # occasional honest hedge
                confidence = 0.55 + 0.4 * reader_rng.random()

            if right_type:
                type_claim = row.true_type
                part_number = true_pn if reader_rng.random() < 0.97 else (
                    _draw_proofs_wrong(reader_rng, true_pn)
                )
            else:
                type_claim = DECOY_TYPES.get(row.true_type.lower(), DEFAULT_DECOY)
                part_number = _draw_proofs_wrong(reader_rng, true_pn)

            readings.append(
                Reading(
                    part_id=row.part_id,
                    model_id=model_id,
                    part_type=type_claim,
                    part_number=part_number,
                    condition=condition,
                    confidence=round(min(confidence, 1.0), 2),
                    evidence=(
                        # Part numbers live in the structured field, not in
                        # free evidence text: a marking like "CR-50" read as
                        # prose must never be scanned by hazard word rules.
                        f"read the part at high zoom; reporting the family "
                        f"{type_claim} and condition {condition.value}"
                    ),
                    key_rating=None,
                )
            )
    return readings


@contextmanager
def _strictness(value: float):
    """Sweep the gate's agreement threshold without editing the gate module."""
    original = GATE_DECIDE.MIN_AGREEMENT
    GATE_DECIDE.MIN_AGREEMENT = value
    try:
        yield
    finally:
        GATE_DECIDE.MIN_AGREEMENT = original


def _part_for(row: LabelRow) -> Part:
    return Part(
        id=row.part_id,
        device_id=row.device_id,
        crop_path=f"crop/{row.part_id}.png",
        bbox=(0.0, 0.0, 10.0, 10.0),
        mask_path=f"mask/{row.part_id}.png",
    )


def run_split(
    rows: Sequence[LabelRow],
    *,
    seed: int,
    thresholds: Sequence[float],
) -> report.EvalResult:
    """The full metric pass over one split; nothing here touches the network."""
    readings = build_readings(rows, seed)
    parts = [_part_for(row) for row in rows]
    by_id = {row.part_id: row for row in rows}

    curve_sets: dict[float, list] = {}
    operating = None
    for threshold in thresholds:
        with _strictness(threshold):
            verdicts = [decide(part, readings) for part in parts]
        curve_sets[threshold] = verdicts
        operating = verdicts

    # D3: supply each labeled part's true measured value, taken before the
    # models were ever asked anything.
    after_verdicts = [
        decide(part, readings, measurement=by_id[part.id].measured_value)
        for part in parts
    ]
    resolvable, resolved = measurement_resolution_counts(operating or [], after_verdicts)

    scores = disagreement_scores(readings, rows)
    aurocs = disagreement_auroc(readings, rows)

    return report.EvalResult(
        split=rows[0].split if rows else "",
        labels_path=DEFAULT_LABELS,
        seed=seed,
        thresholds=list(thresholds),
        n_parts=len(rows),
        n_devices=len({row.device_id for row in rows}),
        curve=coverage_error_curve(curve_sets, rows),
        baseline=baseline_confidence_curve(readings, rows, thresholds),
        operating_threshold=thresholds[-1] if thresholds else None,
        wrong_ids=wrong_verified_ids(operating or [], rows),
        aurocs=aurocs,
        roc=roc_curve([score.score for score in scores], [score.wrong for score in scores]),
        resolvable=resolvable,
        resolved=resolved,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="salvage.eval.run",
        description="Salvage evaluation harness: D1 coverage/error, D2 AUROC, D3 measurement resolution.",
    )
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="path to the labels CSV")
    parser.add_argument("--out-dir", default=DEFAULT_OUT, help="where results.md and PNGs land")
    parser.add_argument(
        "--thresholds",
        default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
        help="comma-separated gate agreement thresholds to sweep",
    )
    parser.add_argument("--seed", type=int, default=7, help="mock-reader seed (reproducible runs)")
    parser.add_argument(
        "--i-mean-it",
        action="store_true",
        help="required to run the held-out test split, which is run once",
    )
    args = parser.parse_args(argv)

    if args.split == "test" and not args.i_mean_it:
        print(
            "refusing the held-out test split without --i-mean-it: dev first, "
            "test once, and only when the numbers are meant to be final",
            file=sys.stderr,
        )
        return 2

    try:
        thresholds = tuple(float(t) for t in str(args.thresholds).split(",") if t.strip())
    except ValueError:
        print(f"--thresholds must be comma-separated floats, got {args.thresholds!r}", file=sys.stderr)
        return 2
    if not thresholds:
        print("no thresholds to sweep", file=sys.stderr)
        return 2

    rows = load_labels(args.labels)
    leaking = check_device_split(rows)
    if leaking:
        print(
            "devices in both splits (would leak): " + ", ".join(leaking),
            file=sys.stderr,
        )
        return 3
    rows = [row for row in rows if row.split == args.split]
    if not rows:
        print(f"no labeled parts in split {args.split!r}", file=sys.stderr)
        return 3

    result = run_split(rows, seed=args.seed, thresholds=thresholds)
    path = report.write_report(result, args.out_dir)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
