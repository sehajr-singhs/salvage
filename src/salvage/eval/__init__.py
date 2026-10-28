"""Salvage evaluation harness (brief 07).

Public surface:
- ``salvage.eval.labels``: the label CSV format, loader, and device-split check.
- ``salvage.eval.metrics``: D1 coverage/error and wrong VERIFIED, D2
  disagreement AUROC, D3 measurement resolution, plus the single-model
  confidence baseline curve.
- ``salvage.eval.report``: ``reports/results.md`` and the black-and-white
  plots.
- ``salvage.eval.run``: the CLI — ``python -m salvage.eval.run
  --split dev|test``; the test split refuses to run without ``--i-mean-it``.
"""
