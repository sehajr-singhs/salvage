"""Report writer for the eval harness (brief 07).

Writes ``reports/results.md`` plus two black-and-white PNG plots:

- ``coverage_error.png`` — the gate's coverage-versus-error curve,
- ``disagreement_roc.png`` — the ROC of the disagreement predictor.

The plots follow the demo's look: white data lines on a black background,
hairline gray gridlines, no color anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from salvage.eval.metrics import Point

GRIDLINE = "#2A2A2A"
SECONDARY = "#8A8A8A"
WHITE = "#FFFFFF"
BLACK = "#000000"

RESULTS_MD = "results.md"
COVERAGE_PNG = "coverage_error.png"
ROC_PNG = "disagreement_roc.png"


@dataclass(frozen=True)
class EvalResult:
    """Everything one run computed, in the shape the report renders.

    Built by ``salvage.eval.run`` and consumed here; tests also build it by
    hand with small numbers to check the markdown.
    """

    split: str
    labels_path: str
    seed: int
    thresholds: list[float]
    n_parts: int
    n_devices: int
    curve: list[Point]
    """The gate's coverage-error points, loosest strictness first."""
    baseline: dict[str, list[Point]] = field(default_factory=dict)
    """Per reader model: its own-confidence coverage-error points."""
    operating_threshold: float | None = None
    """The strictness the wrong_ids below were counted at."""
    wrong_ids: list[str] = field(default_factory=list)
    """Parts whose VERIFIED claim is contradicted by the labels."""
    aurocs: dict[str, float] = field(default_factory=dict)
    roc: list[tuple[float, float]] = field(default_factory=list)
    resolvable: int = 0
    resolved: int = 0

    @property
    def resolution_rate(self) -> float:
        return self.resolved / self.resolvable if self.resolvable else 0.0


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def render(result: EvalResult) -> str:
    """The full results.md as text (the writer's input made testable)."""
    lines: list[str] = []
    add = lines.append
    add(f"# Salvage eval results — split: {result.split}")
    add("")
    add(
        f"{result.n_parts} labeled parts on {result.n_devices} devices; "
        f"labels `{result.labels_path}`; mock-reader seed {result.seed}; "
        f"gate strictness swept over {', '.join(_fmt(t) for t in result.thresholds)}."
    )
    add("")

    # D1
    add("## D1 — verify-or-abstain versus single-model confidence")
    add("")
    add("| min agreement | coverage | wrong VERIFIED | wrong rate |")
    add("|---|---|---|---|")
    for point in result.curve:
        add(
            f"| {_fmt(point.threshold)} "
            f"| {_fmt(point.coverage)} ({point.covered}/{point.n_parts}) "
            f"| {point.wrong} | {_fmt(point.wrong_rate)} |"
        )
    if result.wrong_ids:
        add("")
        add(
            f"wrong VERIFIED parts at the {result.operating_threshold} operating "
            f"point: {', '.join(result.wrong_ids)}"
        )
    else:
        add("")
        add("**0 wrong VERIFIED labels.**")
    add("")
    if result.baseline:
        add(
            "Comparator: trusting one reader above a confidence cutoff "
            "(each column: coverage / wrong VERIFIED)."
        )
        add("")
        models = list(result.baseline)
        add("| cutoff | " + " | ".join(models) + " |")
        add("|---" * (1 + len(models)) + "|")
        for row_index, threshold in enumerate(result.thresholds):
            cells = [
                _fmt(point.coverage) if point.wrong == 0
                else f"{_fmt(point.coverage)} / {point.wrong}"
                for point in (points[row_index] for points in result.baseline.values())
            ]
            add(f"| {_fmt(threshold)} | " + " | ".join(cells) + " |")
        add("")

    # D2
    add("## D2 — what predicts a wrong ID (AUROC)")
    add("")
    add("| predictor | AUROC |")
    add("|---|---|")
    for name, value in result.aurocs.items():
        add(f"| {name} | {_fmt(value)} |")
    best = max(result.aurocs, key=lambda name: result.aurocs[name], default=None)
    if best is not None and best == "disagreement" and len(result.aurocs) > 1:
        add("")
        add("Disagreement across readers beats every single reader's own confidence.")
    add("")

    # D3
    add("## D3 — one measurement resolves abstentions")
    add("")
    add(
        f"{result.resolvable} parts were left NEEDS_MEASUREMENT or ABSTAIN; "
        f"{result.resolved} of them reached VERIFIED once one meter reading was "
        f"supplied — resolution rate **{_fmt(result.resolution_rate)}**."
    )
    add("")

    # Plots
    add("## Plots")
    add("")
    add("![coverage versus error](%s)" % COVERAGE_PNG)
    add("")
    add("![disagreement ROC](%s)" % ROC_PNG)
    add("")
    return "\n".join(lines)


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover - environment message only
        raise ImportError(
            "eval plots need matplotlib (pip install matplotlib); the "
            "results.md tables are written without it"
        ) from exc


def _new_figure(plt):  # noqa: ANN001 - plt module, typed loosely on purpose
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=BLACK)
    ax.set_facecolor(BLACK)
    for spine in ax.spines.values():
        spine.set_color(WHITE)
        spine.set_linewidth(1.0)
    ax.grid(True, color=GRIDLINE, linewidth=0.6, linestyle="-")
    ax.tick_params(colors=WHITE, labelcolor=WHITE, length=3)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(WHITE)
    return fig, ax


def _save(fig, path: Path) -> None:  # noqa: ANN001
    fig.savefig(path, dpi=150, facecolor=BLACK, bbox_inches="tight")


def plot_coverage_error(points: list[Point], path: str | Path) -> Path:
    """The D1 curve: coverage on x, wrong VERIFIED labels on y."""
    plt = _matplotlib()
    fig, ax = _new_figure(plt)
    xs = [point.coverage for point in points]
    ys = [point.wrong for point in points]
    ax.plot(xs, ys, color=WHITE, linewidth=1.2, marker="o", markersize=3)
    ax.set_xlabel("coverage (share of parts VERIFIED)", color=WHITE)
    ax.set_ylabel("wrong VERIFIED labels", color=WHITE)
    path = Path(path)
    _save(fig, path)
    plt.close(fig)
    return path


def plot_roc(points: list[tuple[float, float]], path: str | Path, auroc_value: float) -> Path:
    """The D2 ROC of the disagreement predictor, with its AUROC in the corner."""
    plt = _matplotlib()
    fig, ax = _new_figure(plt)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot([0.0, 1.0], [0.0, 1.0], color=SECONDARY, linewidth=0.6)
    ax.plot(xs, ys, color=WHITE, linewidth=1.2)
    ax.text(
        0.97,
        0.04,
        f"AUROC {auroc_value:.2f}",
        color=SECONDARY,
        ha="right",
        fontsize=8,
        transform=ax.transAxes,
    )
    ax.set_xlabel("false positive rate", color=WHITE)
    ax.set_ylabel("true positive rate", color=WHITE)
    ax.set_aspect("equal")
    path = Path(path)
    _save(fig, path)
    plt.close(fig)
    return path


def write_report(result: EvalResult, out_dir: str | Path = "reports") -> Path:
    """Write results.md and both PNGs under ``out_dir``; returns the md path."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / RESULTS_MD).write_text(render(result), encoding="utf-8")
    if result.roc:
        auroc_value = result.aurocs.get("disagreement", 0.5)
        plot_roc(result.roc, directory / ROC_PNG, auroc_value)
    if result.curve:
        plot_coverage_error(result.curve, directory / COVERAGE_PNG)
    return directory / RESULTS_MD
