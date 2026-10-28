"""Metrics for the eval harness (brief 07).

Every number here is deliberately plain and hand-checkable:

``wrong_verified_count``
    How many VERIFIED verdicts state something the ground-truth label
    contradicts (a wrong part type, or a dead/works claim that is backwards).
    The D1 headline is ``0`` of these.

``coverage_error_curve``
    Coverage (share of labeled parts the gate answers VERIFIED) versus the
    number of wrong VERIFIED labels, swept over gate strictness. The caller
    supplies verdict sets produced at several strictness levels as
    ``{threshold: [Verdict, ...]}``; a bare list of verdicts is a single
    operating point.

``disagreement_auroc``
    How well "the readers disagree" predicts a wrong ID (AUROC), compared
    against each single reader's own confidence on the same task (D2). A
    predictor with no positives or no negatives on the data has no signal,
    and is reported as 0.5 by convention.

``measurement_resolution_rate``
    The share of parts left NEEDS_MEASUREMENT or ABSTAIN that the gate turns
    into VERIFIED once one free-text meter reading is supplied (D3).

Type/condition matching rules, stated once so tests can mirror them:

- the verdict's claimed type is the token set of its ``label`` (with the
  trailing ``"(dead)"`` marker stripped); the claim matches truth when every
  token of the normalized ``true_type`` appears in it, so a label like
  ``"brushless motor 2205"`` matches ``true_type`` ``"brushless motor"`` but
  does not match ``"fan"``.
- the verdict's claimed condition is ``dead`` exactly when the label carries
  the ``(dead)`` marker, otherwise ``works``. It can only be wrong when the
  truth is works or dead; ``unknown`` truth contradicts nothing.
- the same token rule applies to a single reader's ``part_type`` for the
  baseline curve, with the reader's own ``condition`` claim compared the
  same way (UNKNOWN truth is never wrong, UNKNOWN reading is not wrong
  either — it identifies the type correctly or not, and claims nothing
  about the condition).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from salvage.eval.labels import LabelRow
from salvage.schemas import Condition, Reading, State, Verdict

_DEAD_MARKER = re.compile(r"\(\s*dead\s*\)\s*$")
_SPLIT = re.compile(r"[\s_\-/]+")


def _tokens(text: str) -> frozenset[str]:
    """Lowercase and split snake/kebab-case text into its word tokens."""
    return frozenset(t for t in _SPLIT.split(text.lower()) if t)


def _truth_type_ok(claimed: str, label: LabelRow) -> bool:
    return _tokens(label.true_type) <= _tokens(claimed)


def _dead_claim(claimed: str) -> bool:
    return _DEAD_MARKER.search(claimed) is not None


def _condition_claim_ok(claimed: str, label: LabelRow) -> bool:
    if label.true_condition not in ("works", "dead"):
        return True  # unknown truth can never be contradicted
    return (label.true_condition == "dead") == _dead_claim(claimed)


def verdict_is_wrong(verdict: Verdict, label: LabelRow) -> bool:
    """Does a VERIFIED verdict state something the label contradicts?"""
    if verdict.state is not State.VERIFIED:
        return False
    claimed = _DEAD_MARKER.sub("", verdict.label).strip()
    return not (
        _truth_type_ok(claimed, label) and _condition_claim_ok(verdict.label, label)
    )


def reading_is_wrong(reading: Reading, label: LabelRow) -> bool:
    """Would this single reader's claim, if trusted, be wrong?

    Used by the baseline curve (thresholding one model's own confidence),
    which is what verify-or-abstain has to beat (D1).
    """
    claimed = f"{reading.part_type} {reading.part_number or ''}"
    if not _truth_type_ok(claimed, label):
        return True
    if reading.condition in (Condition.WORKS, Condition.DEAD):
        if label.true_condition in ("works", "dead"):
            if (reading.condition is Condition.DEAD) != (label.true_condition == "dead"):
                return True
    return False


def wrong_verified_ids(verdicts: Sequence[Verdict], labels: Sequence[LabelRow]) -> list[str]:
    """The part_ids whose VERIFIED verdict is contradicted by the labels."""
    truth = {row.part_id: row for row in labels}
    return sorted(
        verdict.part_id
        for verdict in verdicts
        if verdict.part_id in truth and verdict_is_wrong(verdict, truth[verdict.part_id])
    )


def wrong_verified_count(verdicts: Sequence[Verdict], labels: Sequence[LabelRow]) -> int:
    """How many VERIFIED verdicts the ground truth contradicts (D1 headline)."""
    return len(wrong_verified_ids(verdicts, labels))


@dataclass(frozen=True)
class Point:
    """One (coverage, error) point of a coverage-versus-error curve."""

    threshold: float
    """The strictness the verdict set was produced at, or 0.0 for one point."""
    n_parts: int
    covered: int
    """Parts the gate answered VERIFIED (everything else is deferred/refused)."""
    wrong: int
    """Wrong VERIFIED labels among those covered."""

    @property
    def coverage(self) -> float:
        return self.covered / self.n_parts if self.n_parts else 0.0

    @property
    def wrong_rate(self) -> float:
        return self.wrong / self.covered if self.covered else 0.0


def evaluate_point(
    verdicts: Sequence[Verdict],
    labels: Sequence[LabelRow],
    *,
    threshold: float = 0.0,
) -> Point:
    """One operating point: coverage and wrong-VERIFIED count for ``verdicts``.

    Verdicts for parts with no label row are ignored (the labels are the
    denominator); VERIFIED is the only state counted as covered, because a
    NEEDS_MEASUREMENT / ABSTAIN / HAZARD part is exactly the one the gate did
    *not* claim something checkable about.
    """
    truth = {row.part_id for row in labels}
    covered = sum(
        1 for verdict in verdicts if verdict.part_id in truth and verdict.state is State.VERIFIED
    )
    return Point(
        threshold=threshold,
        n_parts=len(labels),
        covered=covered,
        wrong=wrong_verified_count(verdicts, labels),
    )


def coverage_error_curve(
    verdicts: Mapping[float, Sequence[Verdict]] | Sequence[Verdict],
    labels: Sequence[LabelRow],
) -> list[Point]:
    """Coverage-versus-error over gate strictness levels, or one point.

    ``verdicts`` is either ``{threshold: [Verdict, ...]}`` — one verdict set
    per strictness level the caller ran the gate at — or a bare list of
    verdicts, treated as the single operating point threshold 0.0. Points are
    returned ordered by threshold, loosest first.
    """
    if isinstance(verdicts, Mapping):
        return [
            evaluate_point(set_, labels, threshold=float(t))
            for t, set_ in sorted(verdicts.items())
        ]
    return [evaluate_point(verdicts, labels, threshold=0.0)]


# --- D1 baseline: thresholding one reader's own confidence -------------------


def _best_reading(readings: Sequence[Reading], part_id: str, model_id: str) -> Reading | None:
    own = [r for r in readings if r.part_id == part_id and r.model_id == model_id]
    if not own:
        return None
    return max(own, key=lambda r: r.confidence)


def baseline_confidence_curve(
    readings: Sequence[Reading],
    labels: Sequence[LabelRow],
    thresholds: Sequence[float],
    model_id: str | None = None,
) -> dict[str, list[Point]]:
    """Coverage/error when one reader is trusted above a confidence cutoff.

    For each reader model (or just ``model_id``), a part is covered at
    threshold ``t`` when that model's most confident reading for the part has
    ``confidence >= t``; the covered claim is that reading's type and
    condition. This is the comparator the gate's curve has to dominate.
    """
    truth = {row.part_id: row for row in labels}
    model_ids = [model_id] if model_id is not None else sorted({r.model_id for r in readings})
    result: dict[str, list[Point]] = {}
    for mid in model_ids:
        by_part = {
            pid: _best_reading(readings, pid, mid)
            for pid in truth
            if _best_reading(readings, pid, mid) is not None
        }
        points = []
        for t in thresholds:
            covered = {pid: r for pid, r in by_part.items() if r is not None and r.confidence >= t}
            wrong = sum(
                1 for pid, r in covered.items() if reading_is_wrong(r, truth[pid])
            )
            points.append(
                Point(
                    threshold=float(t),
                    n_parts=len(labels),
                    covered=len(covered),
                    wrong=wrong,
                )
            )
        result[mid] = points
    return result


# --- D2: the disagreement signal --------------------------------------------


def _normalize_pn(value: str | None) -> str:
    return value.strip().lower() if value else ""


@dataclass(frozen=True)
class PartScore:
    """One part's disagreement score and whether the readers got its ID wrong."""

    part_id: str
    score: float
    """0.0 when all readers agree, up to 1.0 when they all disagree."""
    wrong: bool
    """The readers' consensus part_number contradicts the labeled truth."""


def disagreement_scores(
    readings: Sequence[Reading],
    labels: Sequence[LabelRow],
) -> list[PartScore]:
    """Per part: how much the readers disagree, and whether the consensus ID is wrong.

    A part whose label has no true_part_number, or that no reader read, is
    skipped: there is no ID to be right or wrong about. Disagreement is the
    share of part_numbers that differ from the readers' modal one; when no
    reader produced a part_number at all it falls back to part_type, still
    measuring whether the readers tell one story or two.
    """
    truth = {row.part_id: row for row in labels if row.true_part_number}
    by_part: dict[str, list[Reading]] = {}
    for reading in readings:
        if reading.part_id in truth:
            by_part.setdefault(reading.part_id, []).append(reading)

    scores: list[PartScore] = []
    for part_id, own in sorted(by_part.items()):
        numbers = [r.part_number for r in own if r.part_number]
        values = numbers or [r.part_type for r in own]
        modal, _count = Counter(values).most_common(1)[0]
        differing = sum(1 for value in values if value != modal)
        score = 0.0 if len(values) <= 1 else differing / len(values)
        wrong = not numbers or _normalize_pn(modal) != _normalize_pn(truth[part_id].true_part_number)
        scores.append(PartScore(part_id=part_id, score=score, wrong=wrong))
    return scores


def auroc(scores: Sequence[float], positives: Sequence[bool]) -> float:
    """AUROC of ``scores`` for predicting ``positives`` (Mann-Whitney U).

    Tied scores count half a concordant pair. When everything is positive or
    everything is negative there is no signal to rank, and the result is 0.5
    instead of an undefined number.
    """
    n_pos = sum(1 for p in positives if p)
    n_neg = len(positives) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = sorted(range(len(scores)), key=lambda i: (scores[i], not positives[i]))
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        tied = scores[order[i]]
        while j + 1 < len(order) and scores[order[j + 1]] == tied:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    rank_sum = sum(rank for rank, is_pos in zip(ranks, positives) if is_pos)
    u = rank_sum - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def roc_curve(scores: Sequence[float], positives: Sequence[bool]) -> list[tuple[float, float]]:
    """ROC points (false-positive rate, true-positive rate) for plotting.

    Sweeps the decision threshold from above the top score down: (0, 0),
    one point per distinct score, then (1, 1) — skipped when the sweep
    already ended exactly there, so a perfect separator gives four points,
    not five.
    """
    thresholds = sorted(set(scores), reverse=True)
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for t in thresholds:
        called = [s >= t for s in scores]
        tp = sum(1 for c, p in zip(called, positives) if c and p)
        fp = sum(1 for c, p in zip(called, positives) if c and not p)
        fn = sum(1 for c, p in zip(called, positives) if not c and p)
        tn = sum(1 for c, p in zip(called, positives) if not c and not p)
        points.append((fp / (fp + tn) if fp + tn else 0.0, tp / (tp + fn) if tp + fn else 0.0))
    if points[-1] != (1.0, 1.0):
        points.append((1.0, 1.0))
    return points


def disagreement_auroc(
    readings: Sequence[Reading],
    labels: Sequence[LabelRow],
) -> dict[str, float]:
    """AUROC of each predictor of a wrong ID (D2).

    Returns ``{"disagreement": a}`` plus one ``"confidence:<model_id>"`` entry
    per reader model, where the score is that model's confidence on the part.
    The claim under test: the cross-model disagreement signal predicts error
    better than any single model's own confidence.
    """
    parts = disagreement_scores(readings, labels)
    if not parts:
        return {"disagreement": 0.5}
    result = {
        "disagreement": auroc([p.score for p in parts], [p.wrong for p in parts]),
    }
    model_ids = sorted({r.model_id for r in readings})
    for mid in model_ids:
        usable = [
            (
                sum(r.confidence for r in readings_for if r.model_id == mid)
                / max(1, sum(1 for r in readings_for if r.model_id == mid)),
                p.wrong,
            )
            for p in parts
            for readings_for in (readings_by_part(p.part_id, readings),)
            if any(r.model_id == mid for r in readings_for)
        ]
        if not usable:
            continue
        result[f"confidence:{mid}"] = auroc(
            [score for score, _wrong in usable], [wrong for _score, wrong in usable]
        )
    return result


def readings_by_part(part_id: str, readings: Sequence[Reading]) -> list[Reading]:
    return [r for r in readings if r.part_id == part_id]


# --- D3: one measurement resolves abstentions --------------------------------


def measurement_resolution_counts(
    before: Sequence[Verdict],
    after: Sequence[Verdict],
) -> tuple[int, int]:
    """(resolvable parts, parts resolved to VERIFIED by one measurement).

    Resolvable means the gate left the part NEEDS_MEASUREMENT or ABSTAIN
    before the measurement and has a verdict for it after. Hazard parts are
    refused either way and never inflate the number.
    """
    after_state = {verdict.part_id: verdict.state for verdict in after}
    resolvable = 0
    resolved = 0
    for verdict in before:
        if verdict.state not in (State.ABSTAIN, State.NEEDS_MEASUREMENT):
            continue
        after_this = after_state.get(verdict.part_id)
        if after_this is None:
            continue
        resolvable += 1
        if after_this is State.VERIFIED:
            resolved += 1
    return resolvable, resolved


def measurement_resolution_rate(before: Sequence[Verdict], after: Sequence[Verdict]) -> float:
    """Share of ABSTAIN/NEEDS_MEASUREMENT parts that one meter reading verifies (D3).

    An empty denominator is 0.0, never NaN: nothing resolved.
    """
    resolvable, resolved = measurement_resolution_counts(before, after)
    return resolved / resolvable if resolvable else 0.0
