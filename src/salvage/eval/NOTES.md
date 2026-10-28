# NOTES.md — 07: evaluation harness

## What works

- **Label format** (`eval/labels.py` + `data/labels/parts.csv`): one row per
  labeled part with exactly the brief's columns
  (`device_id,part_id,true_type,true_part_number,true_condition,measured_value,hazard,split`).
  The loader validates every row strictly (condition ∈ works/dead/unknown,
  split ∈ dev/test, no empty required cells, hazard is true/false) and
  `check_device_split()` refuses to score when one device has parts in both
  splits — the roadmap's "split by device, never by part" leak guard. The
  CSV currently holds ~35 synthetic stand-in rows on 12 devices matching the
  real format exactly (see `data/labels/README.md`); real labeled devices
  replace that file in place.
- **Metrics** (`eval/metrics.py`), all plain and hand-checkable:
  - `wrong_verified_count` / `wrong_verified_ids`: VERIFIED verdicts whose
    claimed family (token-set rule vs `true_type`) or claimed condition (the
    `(dead)` marker) the ground truth contradicts. Unknown truth is never
    wrong; the rule set is documented at the top of the module and mirrored
    by the tests.
  - `coverage_error_curve(verdicts, labels)`: accepts `{threshold:
    [Verdict]}` (one gate run per agreement strictness) or a bare list
    (single operating point), returns `(coverage, wrong)` points — coverage
    counting VERIFIED only, since everything else is a deferred/refused
    claim.
  - `disagreement_auroc(readings, labels)`: AUROC of "share of readers
    disagreeing with the modal part number" against each single reader's
    confidence, for predicting a wrong ID; falls back to part-type
    disagreement when no reader produced a marking, and returns 0.5 when a
    predictor has no positives or negatives to rank.
  - `measurement_resolution_rate(before, after)`: share of
    NEEDS_MEASUREMENT/ABSTAIN parts the gate turns into VERIFIED once the
    labeled meter reading is supplied; HAZARD parts never inflate it.
  - `baseline_confidence_curve`: the D1 comparator — trust one reader above
    a confidence cutoff — so the gate's curve is compared against
    thresholding each reader's own confidence, per the brief.
- **Report** (`eval/report.py`): writes `reports/results.md` plus two PNGs
  (coverage-versus-error, disagreement ROC) in the demo's look: white data
  lines on black, hairline `#2A2A2A` gridlines, no color anywhere.
  matplotlib is imported lazily with a clear message, so the tables alone
  render without it.
- **Runner** (`eval/run.py`): `python -m salvage.eval.run --split dev|test`
  with `--labels/--out-dir/--thresholds/--seed`. Sweeps the gate's
  agreement threshold by patching `salvage.gate.decide.MIN_AGREEMENT`
  around each run of the real `decide()` (restored in a finally), feeds the
  labeled `measured_value` back through the gate for the D3 pass, and never
  touches the network. **`--split test` refuses to run without
  `--i-mean-it`** (exit code 2), per the "nobody peeks early" rule, and any
  device-split leak exits 3.
- **Offline mocked path**: readings come from `build_readings` — three
  deterministic readers (`muse-spark-1.3`, `muse-spark-open`,
  `free-pool-contributor`), seeded per part, no keys, reproducible (same
  seed ⇒ byte-identical `results.md`). About 18% of parts are hard in three
  flavors: confidently-wrong unanimous readers (D1's only real risk),
  shrugging unknowns (cheap abstentions), and split readers (D2 signal
  material). Hazard-flavored parts are keyed off the label row's `hazard`
  flag only, so verdict HAZARD states always match the labeled truth.
- Tests: `tests/test_eval.py`, 24 tests, every metric checked against
  values worked out by hand (AUROC 0.75/1.0/0.0/0.5 cases, coverage-error
  sweep, resolution 2/3, loader rejections, determinism, the test-split
  refusal, report contents and real PNG bytes). Whole repo: 191 passed.

## What doesn't work / known limits

- The mock readers are a stand-in for the real reader package (empty while
  the readers session runs): `build_readings` is the single swap point.
  Real readings arrive as `Reading` models and everything downstream is
  unchanged. Numbers in `reports/` are mock-data numbers, not research
  results.
- With this seed/dev split the gate's coverage is flat across the sweep
  (3/26 at every threshold): the parts the gate verifies all have unanimous
  agreements, so agreement-strictness never moves coverage. The machinery
  is tested; a live labeled set is needed for an interesting curve shape.
- The D3 rate on mock dev data is 8/21 ≈ 0.38. Parts it can't resolve:
  truth-unknown parts with no measurement (honest), works/dead reader
  contradictions, and single-reader part-number typos that trip the gate's
  family downgrade (one bad marking among three readers abstains the whole
  part — that's brief 04's strict rule, surfaced here by the numbers).
- `evaluate_point` ignores verdicts for unlabeled parts; parts the gate
  never answered still count in the coverage denominator. Deliberate.
- matplotlib warnings seen in `pytest` come from the segment session's
  tests, not from this package.

## What's stubbed

- Real model calls: none (no keys exist; the readers/gate remain the only
  paths to real readings, and they mock through `cache.py`).
- `data/labels/parts.csv` holds synthetic rows in the real format until the
  ~300 labeled parts from the roadmap's section 5 exist.
- D4 (fine-tuning) is out of this brief's scope entirely.
