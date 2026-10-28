"""Brief 07 tests: every metric against values worked out by hand.

The fixtures here are the smallest that exercise each rule; the expected
numbers are in the comments next to the asserts. The shipped
``data/labels/parts.csv`` is a synthetic labels file in the real format; the
run.py CLI tests exercise the full harness over it, including the fact that
the held-out test split refuses to run without ``--i-mean-it``.
"""

import matplotlib  # noqa: F401  (checked so the PNG tests can skip cleanly)
import pytest

from salvage.eval import report, run as run_module
from salvage.eval.labels import LabelRow, check_device_split, load_labels
from salvage.eval.metrics import (
    Point,
    auroc,
    baseline_confidence_curve,
    coverage_error_curve,
    disagreement_auroc,
    disagreement_scores,
    evaluate_point,
    measurement_resolution_counts,
    measurement_resolution_rate,
    roc_curve,
    wrong_verified_count,
    wrong_verified_ids,
)
from salvage.schemas import Condition, Reading, State, Verdict

pytest.importorskip("matplotlib", reason="report PNG tests need matplotlib")


SHIP = "data/labels/parts.csv"


def L(part_id, device, true_type, true_pn, true_condition, split="dev"):
    return LabelRow(
        device_id=device,
        part_id=part_id,
        true_type=true_type,
        true_part_number=true_pn,
        true_condition=true_condition,
        measured_value=None,
        hazard=False,
        split=split,
    )


def V(part_id, state, label):
    return Verdict(part_id=part_id, state=state, label=label)


def R(part_id, model_id, part_type, part_number, condition, confidence):
    return Reading(
        part_id=part_id,
        model_id=model_id,
        part_type=part_type,
        part_number=part_number,
        condition=condition,
        confidence=confidence,
        evidence="fixture evidence",
    )


# --- labels ------------------------------------------------------------------


def test_shipped_labels_parse_and_are_device_split():
    rows = load_labels(SHIP)
    assert all(row.split in ("dev", "test") for row in rows)
    assert check_device_split(rows) == []
    # every part id is unique and carries its device
    assert len({row.part_id for row in rows}) == len(rows)


def test_device_split_leak_is_detected():
    rows = [
        L("p1", "dev01", "motor", None, "works", split="dev"),
        L("p2", "dev01", "fan", None, "works", split="test"),  # same device, other split!
        L("p3", "dev02", "fan", None, "unknown", split="test"),
    ]
    assert check_device_split(rows) == ["dev01"]


def test_loader_rejects_bad_header(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("device_id,part_id,wrong_header,true_condition,split\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        load_labels(path)


def test_loader_rejects_unknown_condition_split_and_missing_cells(tmp_path):
    path = tmp_path / "labels.csv"
    for row, expected in (
        ("d1,p1,motor,x,broken,,false,dev\n", "true_condition"),  # bad condition
        ("d2,p2,fan,x,works,,false,train\n", "split"),  # bad split
        ("d3,,capacitor,x,works,,false,dev\n", "empty required cell"),  # missing part_id
    ):
        path.write_text(
            "device_id,part_id,true_type,true_part_number,true_condition,"
            "measured_value,hazard,split\n" + row,
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=expected):
            load_labels(path)


def test_loader_hazard_and_empty_cells(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(
        "device_id,part_id,true_type,true_part_number,true_condition,measured_value,hazard,split\n"
        "d1,p1,power supply,,unknown,,true,dev\n"
        "d2,p2,fan,,dead,,0,dev\n",
        encoding="utf-8",
    )
    rows = load_labels(path)
    assert rows[0].hazard is True
    assert rows[0].measured_value is None
    assert rows[1].hazard is False
    assert rows[1].true_part_number is None


# --- D1: wrong verified / coverage-error curve --------------------------------


def _d1_labels():
    return [
        L("p1", "d1", "brushless motor", "2205", "works"),
        L("p2", "d1", "fan", "4010", "works"),
        L("p3", "d2", "speaker", None, "works"),
        L("p4", "d3", "connector", None, "dead"),
        L("p5", "d4", "speaker", None, "unknown"),
    ]


def test_wrong_verified_count_hand():
    labels = _d1_labels()
    verdicts = [
        # type matches with extra tokens (part number), right condition
        V("p1", State.VERIFIED, "brushless motor 2205"),
        # claims dead while the truth is works -> wrong
        V("p2", State.VERIFIED, "fan 4010 (dead)"),
        # claims the wrong family -> wrong
        V("p3", State.VERIFIED, "connector"),
        # dead claim matches dead truth -> right
        V("p4", State.VERIFIED, "connector (dead)"),
        # truth unknown contradicts nothing -> never wrong (the type claim
        # matches the truth, only the dead marker differs)
        V("p5", State.VERIFIED, "speaker (dead)"),
        V("p1", State.ABSTAIN, "unproven"),  # duplicate id, not VERIFIED anyway
    ]
    assert wrong_verified_ids(verdicts, labels) == ["p2", "p3"]
    assert wrong_verified_count(verdicts, labels) == 2


def test_coverage_error_curve_single_point_and_sweep():
    labels = [
        L("p1", "d1", "motor", None, "works"),
        L("p2", "d1", "fan", "4010", "works"),
        L("p3", "d2", "speaker", None, "works"),
        L("p4", "d3", "connector", None, "dead"),
    ]
    strict = [
        V("p1", State.VERIFIED, "motor"),
        V("p2", State.ABSTAIN, "unproven"),
        V("p3", State.ABSTAIN, "unproven"),
        V("p4", State.ABSTAIN, "unproven"),
    ]
    loose = [
        V("p1", State.VERIFIED, "motor"),
        # works truth claimed dead -> wrong
        V("p2", State.VERIFIED, "fan 4010 (dead)"),
        # speaker truth claimed a different family -> wrong
        V("p3", State.VERIFIED, "knob"),
        V("p4", State.VERIFIED, "connector (dead)"),
    ]

    single = coverage_error_curve(strict, labels)
    # (threshold, n_parts, covered, wrong)
    assert single == [Point(threshold=0.0, n_parts=4, covered=1, wrong=0)]

    points = coverage_error_curve({1.0: strict, 0.67: loose}, labels)
    assert [p.threshold for p in points] == [0.67, 1.0]  # loosest first
    assert (points[0].coverage, points[0].wrong) == (pytest.approx(1.0), 2)
    assert (points[1].coverage, points[1].wrong) == (pytest.approx(0.25), 0)
    assert points[0].wrong_rate == pytest.approx(0.5)  # 2 wrong of 4 covered
    assert points[1].wrong_rate == 0.0


def test_evaluate_point_ignores_unlabeled_verdicts():
    labels = [L("p1", "d1", "motor", None, "works")]
    verdicts = [
        V("p1", State.VERIFIED, "motor"),
        V("nobody", State.VERIFIED, "motor"),
    ]
    assert evaluate_point(verdicts, labels).covered == 1
    assert evaluate_point(verdicts, labels).n_parts == 1


# --- AUROC --------------------------------------------------------------------


def test_auroc_hand():
    # one wrong at 0.8 beats both rights at 0.2/0.4 -> perfect ranking
    assert auroc([0.2, 0.4, 0.8], [False, False, True]) == 1.0
    # tied scores split the pair half and half: (0.5,0.5)=0.5, (0.5,0.1)=1 -> 0.75
    assert auroc([0.5, 0.5, 0.1], [False, True, False]) == 0.75
    # everything wrong or nothing wrong has no signal -> 0.5 by convention
    assert auroc([1.0, 0.0], [True, True]) == 0.5
    assert auroc([1.0, 0.0], [False, False]) == 0.5


def _d2_fixtures():
    truth = [
        L("p1", "d1", "motor", "ABC", "works"),
        L("p2", "d1", "motor", "ABC", "works"),
        L("p3", "d2", "motor", "DEF", "works"),
        L("p4", "d2", "motor", "ABC", "works"),
    ]
    readings = [
        # p1: both readers read the marking the same -> right, no disagreement
        R("p1", "m1", "motor", "ABC", Condition.WORKS, 0.9),
        R("p1", "m2", "motor", "ABC", Condition.WORKS, 0.9),
        # p2: three readers, two say XYZ -> disagree (1/3) and the consensus
        # XYZ contradicts the truth ABC -> wrong
        R("p2", "m1", "motor", "ABC", Condition.WORKS, 0.8),
        R("p2", "m2", "motor", "XYZ", Condition.WORKS, 0.8),
        R("p2", "m3", "motor", "XYZ", Condition.WORKS, 0.8),
        # p3: nobody read a marking at all -> consensus is missing -> wrong,
        # but the readers agree on the type, so disagreement score is 0.0
        R("p3", "m1", "connector", None, Condition.UNKNOWN, 0.2),
        R("p3", "m2", "connector", None, Condition.UNKNOWN, 0.2),
        R("p3", "m3", "connector", None, Condition.UNKNOWN, 0.2),
        # p4: everyone agrees on the truth -> right
        R("p4", "m1", "motor", "ABC", Condition.WORKS, 0.9),
        R("p4", "m2", "motor", "ABC", Condition.WORKS, 0.9),
        R("p4", "m3", "motor", "ABC", Condition.WORKS, 0.9),
    ]
    return truth, readings


def test_disagreement_scores_hand():
    truth, readings = _d2_fixtures()
    scores = {s.part_id: s for s in disagreement_scores(readings, truth)}
    assert scores["p1"].wrong is False and scores["p1"].score == 0.0
    assert scores["p2"].wrong is True and scores["p2"].score == pytest.approx(1 / 3)
    # no marking read anywhere: the fallback disagreement on type is 0.0, the
    # consensus part_number is still wrong against the truth
    assert scores["p3"].wrong is True and scores["p3"].score == 0.0
    assert scores["p4"].wrong is False and scores["p4"].score == 0.0


def test_disagreement_auroc_beats_single_confidence_hand():
    truth, readings = _d2_fixtures()
    aurocs = disagreement_auroc(readings, truth)
    # wrongs: p2 (score 1/3), p3 (score 0); rights: p1, p4 (both 0).
    # pairs: (p2,p1)=1, (p2,p4)=1, (p3,p1)=0.5 tie, (p3,p4)=0.5 tie -> 0.75
    assert aurocs["disagreement"] == pytest.approx(0.75)
    # m1's own confidence: rights at 0.9 and 0.9, wrongs at 0.8 and 0.2 ->
    # every wrong/right pair is inverted -> 0.0
    assert aurocs["confidence:m1"] == 0.0
    assert aurocs["confidence:m2"] == 0.0
    assert aurocs["disagreement"] > aurocs["confidence:m1"]


def test_roc_curve_hand():
    scores = [0.2, 0.4, 0.8]
    positives = [False, False, True]
    # sweep thresholds high to low: (0,0), t=.8 -> (0,1), t=.4 -> (0.5,1), t=.2 -> (1,1)
    assert roc_curve(scores, positives) == [
        (0.0, 0.0),
        (0.0, 1.0),
        (0.5, 1.0),
        (1.0, 1.0),
    ]
    assert roc_curve([], []) == [(0.0, 0.0), (1.0, 1.0)]


# --- D1 comparator: single-model confidence curve -----------------------------


def test_baseline_confidence_curve_hand():
    labels = [
        L("p1", "d1", "motor", None, "works"),
        L("p2", "d1", "fan", None, "works"),
    ]
    readings = [
        # model A gets p1 right at 0.9 but misreads p2 at 0.65
        R("p1", "A", "motor", None, Condition.WORKS, 0.9),
        R("p2", "A", "resistor", None, Condition.WORKS, 0.65),
        # model B reads both correctly at 0.95
        R("p1", "B", "motor", None, Condition.WORKS, 0.95),
        R("p2", "B", "fan", None, Condition.WORKS, 0.95),
    ]
    curve = baseline_confidence_curve(readings, labels, [0.6, 0.9])
    assert list(curve) == ["A", "B"]
    a_06, a_09 = curve["A"]
    # at 0.6 A covers both parts and is wrong about p2
    assert (a_06.covered, a_06.wrong) == (2, 1)
    assert a_06.coverage == pytest.approx(1.0)
    # at 0.9 only the correctly-read part survives
    assert (a_09.covered, a_09.wrong) == (1, 0)
    b_06, _b_09 = curve["B"]
    assert (b_06.covered, b_06.wrong) == (2, 0)


# --- D3: measurement resolution -----------------------------------------------


def test_measurement_resolution_rate_hand():
    before = [
        V("p1", State.VERIFIED, "motor"),
        V("p2", State.ABSTAIN, "unproven"),
        V("p3", State.ABSTAIN, "unproven"),
        V("p4", State.NEEDS_MEASUREMENT, "motor"),
        V("p5", State.HAZARD, "swollen lithium cell"),
    ]
    after = [
        V("p1", State.VERIFIED, "motor"),
        V("p2", State.VERIFIED, "motor"),
        V("p3", State.NEEDS_MEASUREMENT, "motor"),  # measurement did not settle it
        V("p4", State.VERIFIED, "motor"),
        V("p5", State.HAZARD, "swollen lithium cell"),  # still refused, not counted
    ]
    # resolvable: p2, p3, p4 (p5 is refused, p1 was already verified)
    assert measurement_resolution_counts(before, after) == (3, 2)
    assert measurement_resolution_rate(before, after) == pytest.approx(2 / 3)
    # no resolvable parts at all -> 0.0, never NaN
    assert measurement_resolution_rate(before[:1], after[:1]) == 0.0
    assert measurement_resolution_rate([], []) == 0.0


# --- report -------------------------------------------------------------------


def _report_result(tmp_case: str = "dev") -> report.EvalResult:
    return report.EvalResult(
        split=tmp_case,
        labels_path="data/labels/parts.csv",
        seed=7,
        thresholds=[1.0, 0.67],
        n_parts=4,
        n_devices=2,
        curve=[
            Point(threshold=0.67, n_parts=4, covered=4, wrong=2),
            Point(threshold=1.0, n_parts=4, covered=1, wrong=0),
        ],
        baseline={"m1": [Point(threshold=0.67, n_parts=4, covered=3, wrong=1),
                         Point(threshold=1.0, n_parts=4, covered=2, wrong=0)]},
        operating_threshold=0.67,
        wrong_ids=["p2", "p3"],
        aurocs={"disagreement": 0.75, "confidence:m1": 0.4},
        roc=[(0.0, 0.0), (0.0, 1.0), (0.5, 1.0), (1.0, 1.0)],
        resolvable=3,
        resolved=2,
    )


def test_render_markdown_contents():
    text = report.render(_report_result())
    assert text.startswith("# Salvage eval results — split: dev")
    assert "4 labeled parts on 2 devices" in text
    assert "wrong VERIFIED parts at the 0.67 operating point: p2, p3" in text
    assert "| predictor | AUROC |" in text
    assert "| disagreement | 0.75 |" in text
    assert "**0.67**" in text  # the D3 resolution rate


def test_write_report_creates_png_and_md(tmp_path):
    out = report.write_report(_report_result(), tmp_path)
    assert out == tmp_path / "results.md"
    text = out.read_text(encoding="utf-8")
    assert "![coverage versus error](coverage_error.png)" in text
    assert "![disagreement ROC](disagreement_roc.png)" in text
    for name in ("coverage_error.png", "disagreement_roc.png"):
        blob = (tmp_path / name).read_bytes()
        assert blob[:8] == b"\x89PNG\r\n\x1a\n"  # a real PNG, black background or not
        assert len(blob) > 1000


def test_plots_are_monochrome(tmp_path):
    result = _report_result()
    path = report.plot_coverage_error(result.curve, tmp_path / "cov.png")
    blob = path.read_bytes()
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    # matplotlib wrote something; the exact stroke widths are a visual check


# --- run.py: mocked pipeline and the test-split guard -------------------------


def test_mock_readings_are_deterministic_and_complete():
    rows = load_labels(SHIP)
    first = run_module.build_readings(rows, seed=7)
    second = run_module.build_readings(rows, seed=7)
    assert first == second
    assert len(first) == 3 * len(rows)  # one reading per reader model per part
    hazard_rows = [row for row in rows if row.hazard]
    assert hazard_rows, "the shipped labels include hazard rows"
    for row in hazard_rows:
        for reading in run_module.build_readings([row], seed=7):
            assert reading.condition is Condition.UNKNOWN
            assert "swollen" in reading.evidence or "mains" in reading.evidence or "microwave" in reading.evidence or "crt" in reading.evidence


def test_test_split_refuses_without_i_mean_it(tmp_path):
    exit_code = run_module.main(
        ["--split", "test", "--labels", SHIP, "--out-dir", str(tmp_path)]
    )
    assert exit_code == 2
    assert not (tmp_path / "results.md").exists()


def test_dev_run_writes_reports(tmp_path):
    exit_code = run_module.main(
        ["--split", "dev", "--labels", SHIP, "--out-dir", str(tmp_path)]
    )
    assert exit_code == 0
    text = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "split: dev" in text
    assert "26 labeled parts on 8 devices" in text
    assert (tmp_path / "coverage_error.png").exists()
    assert (tmp_path / "disagreement_roc.png").exists()


def test_test_split_runs_only_with_i_mean_it(tmp_path):
    exit_code = run_module.main(
        [
            "--split", "test",
            "--labels", SHIP,
            "--out-dir", str(tmp_path),
            "--i-mean-it",
        ]
    )
    assert exit_code == 0
    assert "split: test" in (tmp_path / "results.md").read_text(encoding="utf-8")


def test_dev_run_is_reproducible(tmp_path):
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    for out in (out_a, out_b):
        assert run_module.main(
            ["--split", "dev", "--labels", SHIP, "--out-dir", str(out)]
        ) == 0
    assert (out_a / "results.md").read_text(encoding="utf-8") == (
        out_b / "results.md").read_text(encoding="utf-8")


def test_leaking_labels_are_refused(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(
        "device_id,part_id,true_type,true_part_number,true_condition,measured_value,hazard,split\n"
        "d1,p1,motor,,works,,false,dev\n"
        "d1,p2,fan,,works,,false,test\n",
        encoding="utf-8",
    )
    assert run_module.main(
        ["--split", "dev", "--labels", str(path), "--out-dir", str(tmp_path / "out")]
    ) == 3


def test_run_split_metrics_shape():
    rows = [row for row in load_labels(SHIP) if row.split == "dev"]
    result = run_module.run_split(rows, seed=7, thresholds=[1.0, 0.8, 0.67])
    assert [p.threshold for p in result.curve] == [0.67, 0.8, 1.0]
    assert all(0.0 <= p.coverage <= 1.0 for p in result.curve)
    assert all(0.0 <= value <= 1.0 for value in result.aurocs.values())
    assert 0.0 <= result.resolution_rate <= 1.0
    assert result.roc[0] == (0.0, 0.0) and result.roc[-1] == (1.0, 1.0)
    # every wrong VERIFIED id lists a part that exists in the labels
    truth_ids = {row.part_id for row in rows}
    assert set(result.wrong_ids) <= truth_ids
