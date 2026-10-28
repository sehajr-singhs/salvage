"""Brief 08's command line: the one command, text in, verdicts out.

    python -m salvage scan photo.jpg --device drone1 [--measure]
    python -m salvage need "12 V motor with driver"

``scan`` runs the full 08 chain (:func:`salvage.pipeline.run_device`:
photo, segment, read, gate, inventory) and prints one line per part —
state first, then the gate's label, then its reasons and, for a part the
gate deferred, the exact meter ask. ``--measure`` then runs the bench
dialogue: the agent asks out loud for every resolvable part, the user
answers in text (voice is later, behind the same ``Prompter`` seam), and
the gate re-grades; state changes print as ``BEFORE -> AFTER``.

``need`` is the reverse search over the remembered inventory: it prints
which device to open and which part to pull, VERIFIED first, and falls
back to static household-source advice when nothing is remembered yet.

Exit codes: 0 on a scan (even a soft one with advice), 1 when nothing
could be scanned at all (``ok=False``). No key is ever read here —
providers come from ``salvage.config``, and with none set everything
runs offline on the shared mock reader and the gate's rules.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from salvage.inventory import Inventory
from salvage.measure.dialogue import ConsolePrompter
from salvage.pipeline import DeviceScan, run_device, run_dialogue
from salvage.schemas import State

DEFAULT_DB = "data/inventory.db"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="salvage",
        description=(
            "Salvage: scan a dead device, prove its parts, remember them, "
            "and find them again."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser(
        "scan",
        help="photo -> segment -> read -> gate -> inventory (one command)",
    )
    scan.add_argument("image", type=Path, help="path to the device photo")
    scan.add_argument("--device", required=True, help="device id, e.g. drone1")
    scan.add_argument(
        "--measure",
        action="store_true",
        help="run the measurement dialogue: ask for a meter reading on "
        "every part the gate deferred, then re-grade",
    )
    scan.add_argument(
        "--db", default=DEFAULT_DB, help=f"inventory SQLite file (default {DEFAULT_DB})"
    )
    scan.add_argument(
        "--data-root", default="data", help="where crops/masks/overlays are written"
    )
    scan.add_argument(
        "--no-remember",
        action="store_true",
        help="scan without writing the inventory",
    )

    need = sub.add_parser(
        "need",
        help='reverse search: "I need a 12 V motor with a driver"',
    )
    need.add_argument("text", help="what you need, in your words")
    need.add_argument("--db", default=DEFAULT_DB, help=f"inventory file (default {DEFAULT_DB})")

    return parser


def _state_tag(state: State) -> str:
    """Fixed-width state column so verdict lists read like a table."""
    return f"[{state.value}]"


def _print_verdicts(scan: DeviceScan) -> None:
    head = scan.device_id if scan.ok else f"{scan.device_id} (nothing scanned)"
    mode = scan.readings_mode if scan.ok else "n/a"
    print(f"\n{head} · {len(scan.parts)} part(s) · readings: {mode}")
    for verdict in scan.verdicts:
        print(f"{verdict.part_id}  {_state_tag(verdict.state)}  {verdict.label}")
        for reason in verdict.reasons:
            print(f"        - {reason}")
        if verdict.requested_measurement:
            print(f"        ? {verdict.requested_measurement}")


def cmd_scan(
    args: argparse.Namespace,
    *,
    prompter: Any | None = None,
    scan_backend: Any | None = None,
) -> int:
    """`scan`: the one command, then (optionally) the measurement dialogue."""
    inventory = None if args.no_remember else Inventory(args.db)
    try:
        scan = run_device(
            args.image,
            args.device,
            backend=scan_backend,
            data_root=args.data_root,
            inventory=inventory,
            cache_dir=None,
        )
    except FileNotFoundError as exc:
        print(f"scan failed: {exc}")
        return 1
    _print_verdicts(scan)

    if not scan.ok:
        print(f"\n{scan.message}")
        return 1

    deferred = sum(
        1
        for verdict in scan.verdicts
        if verdict.state in (State.NEEDS_MEASUREMENT, State.ABSTAIN)
    )
    print(f"\n{deferred} part(s) left unproven; hazard parts are refused outright.")

    if args.measure:
        updated, transitions = run_dialogue(
            scan, prompter or ConsolePrompter(), inventory=inventory
        )
        if not transitions:
            print("no measurement changed any verdict.")
        for transition in transitions:
            print(
                f"{transition.part_id}  "
                f"{_state_tag(transition.before)} -> {_state_tag(transition.after)}"
            )
        _print_verdicts(updated)
    return 0


def cmd_need(args: argparse.Namespace) -> int:
    """`need`: which device to open, which part to pull."""
    inventory = Inventory(args.db)
    matches = inventory.query(args.text, cache_dir=None)
    if not matches:
        print(f"\nnothing matches yet: {args.text!r} — scan a device first.")
        return 0
    print(f"\n{len(matches)} match(es) for {args.text!r}:")
    for match in matches:
        if match.source == "inventory":
            print(
                f"{_state_tag(match.state)}  {match.label}  "
                f"\u2192 device {match.device!r} (id {match.device_id}) part {match.part_id}"
            )
            print(f"        crop: {match.crop_path}")
        else:
            print(
                f"[common sources] nothing matched on the bench yet: open a "
                f"{match.device} \u2014 {', '.join(match.alternatives or [])}"
            )
        for reason in match.reasons:
            print(f"        - {reason}")
        if match.requested_measurement:
            print(f"        ? {match.requested_measurement}")
    return 0


def main(
    argv: list[str] | None = None,
    *,
    prompter: Any | None = None,
    scan_backend: Any | None = None,
) -> int:
    """Parse argv and dispatch; ``prompter``/``scan_backend`` are injection
    points for tests (console text input / auto-selection are the defaults).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "scan":
        return cmd_scan(args, prompter=prompter, scan_backend=scan_backend)
    return cmd_need(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
