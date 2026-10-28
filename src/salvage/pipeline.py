"""Brief 08: one chain — photo, segment, read, gate, inventory, measure.

:func:`run_device` is the one command the whole brief was built around::

    python -m salvage scan photo.jpg --device drone1

wires, by import only — never a reimplementation:

- 02 ``salvage.segment.segment.segment`` + ``overlay`` (parts, crops, masks,
  the white-outline teardown view);
- 03 ``salvage.readers.read_part`` when that subpackage ships; while it is
  still an empty stub the same offline mock reader the web layer uses
  (``salvage.web.pipeline.offline_mock_readings``) keeps every run
  deterministic and keyless — one mock implementation for the CLI and the
  web demo, so both show the same verdicts for the same photo;
- 04 ``salvage.gate.decide`` (plain code, no model);
- 05 ``salvage.inventory.Inventory.add`` (the parts library).

:func:`apply_measurements` re-runs the gate per part with a measurement
reply and :func:`run_dialogue` drives the ``measure`` package's ask loop
(text input first, voice later), so one NEEDS_MEASUREMENT or ABSTAIN part
can turn into a VERIFIED one with a single meter reading (the D3 beat).

Failure rules from 00-README: an unavailable backend is a soft ``ok=False``
scan pointing at the sample device, a failed reader yields no vote, and a
failed measurement changes nothing. Nothing here ever crashes on missing
keys — there are none on this machine.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from salvage.gate import decide
from salvage.inventory.models import Device
from salvage.measure.dialogue import (
    ConsolePrompter,
    Prompter,
    Transition,
    measure,
    state_transitions,
)
from salvage.schemas import Part, Reading, Verdict
from salvage.segment.overlay import overlay
from salvage.segment.segment import first_available_backend, segment

# 03's reader when it exists; the shared offline mock while it does not.
try:  # pragma: no cover - exercised the day the readers subpackage ships
    from salvage.readers import read_part as _default_reader  # type: ignore[attr-defined]

    READINGS_MODE_REAL = True
except ImportError:
    from salvage.web.pipeline import (  # type: ignore[no-redef]
        offline_mock_readings as _default_reader,
    )

    READINGS_MODE_REAL = False

READINGS_MODE = "real readers (03)" if READINGS_MODE_REAL else "offline mock reader"


ReadingsProvider = Any  # a callable Part -> list[Reading]; injected for tests


class DeviceScan(BaseModel):
    """Everything one photo produced: parts, voices, verdicts, artifacts.

    JSON-ready (pydantic), so the CLI, the web layer and tests all pass
    the same object around; ``measurement`` remembers the device-wide note
    this run graded with, so a per-part reply can be re-applied on top of
    it without losing the original context.
    """

    model_config = ConfigDict(extra="forbid")

    device_id: str
    ok: bool
    message: str | None = None
    device_name: str | None = None
    """The display name the device was remembered under."""
    measurement: str | None = None
    """The device-wide measurement note fed to every part this run, if any."""
    photo_path: str | None = None
    overlay_path: str | None = None
    parts: list[Part] = []
    readings: list[Reading] = []
    verdicts: list[Verdict] = []
    readings_mode: str = READINGS_MODE


def run_device(
    image_path: str | Path,
    device_id: str,
    *,
    backend: Any | None = None,
    data_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    readings_provider: ReadingsProvider | None = None,
    measurement: str | None = None,
    inventory: Any | None = None,
    device_name: str | None = None,
    remember: bool = True,
) -> DeviceScan:
    """Run one photo through 02 -> 03 -> 04 -> 05 and return the scan.

    ``backend`` follows the web layer's contract (an explicitly passed
    backend is respected as-is; ``None`` auto-selects the hosted SAM
    endpoint and then the Modal app, both available only with keys).
    ``measurement`` is the device-wide meter note re-run with — one photo
    can be scanned and then re-graded in the same command when the user
    already probed the device. ``inventory`` receives the scan when
    ``remember`` is set (the CLI builds the default SQLite store).
    """
    reader = readings_provider if readings_provider is not None else _default_reader

    root = Path(data_root) if data_root is not None else Path("data")

    parts: list[Part] = []
    if backend is not None:
        if backend.available():
            parts = segment(
                image_path,
                device_id,
                backend=backend,
                data_root=data_root,
                cache_dir=cache_dir,
            )
    else:
        shared = first_available_backend()
        if shared is not None:
            parts = segment(
                image_path,
                device_id,
                backend=shared,
                data_root=data_root,
                cache_dir=cache_dir,
            )

    if not parts:
        return DeviceScan(
            device_id=device_id,
            ok=False,
            message=(
                "no segmentation backend available (no model keys on this "
                "machine): use the sample device button"
            ),
        )

    scan_dir = root / "scan" / device_id
    scan_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = scan_dir / "overlay.png"
    overlay(image_path, parts, overlay_path)

    from PIL import Image

    copy_path = scan_dir / "photo.png"
    Image.open(image_path).convert("RGB").save(copy_path)

    readings: list[Reading] = []
    verdicts: list[Verdict] = []
    for part in parts:
        votes = reader(part)
        readings.extend(votes)
        verdicts.append(decide(part, votes, measurement))

    resolved_name = device_name or Path(image_path).stem
    if inventory is not None and remember:
        inventory.add(
            Device(
                id=device_id,
                name=resolved_name,
                source="scan",
            ),
            parts,
            verdicts,
            readings=readings,
        )

    return DeviceScan(
        device_id=device_id,
        ok=True,
        device_name=resolved_name,
        measurement=(measurement or None),
        photo_path=str(copy_path),
        overlay_path=str(overlay_path),
        parts=parts,
        readings=readings,
        verdicts=verdicts,
    )


def apply_measurements(
    scan: DeviceScan,
    answers: Mapping[str, str],
    *,
    inventory: Any | None = None,
    remember: bool = True,
) -> DeviceScan:
    """Re-run ``gate.decide`` per part with the dialogue's measurements.

    Each part is graded with ``answers[part_id]`` when the dialogue got a
    reply there, else with the original device-wide note, else with
    nothing — an unanswered part is never silently *more* confident than
    before. All other scan fields are carried over unchanged; an
    ``inventory`` (if given) is re-added so the parts library shows the
    new states (05 upserts by device, so the second add is an update).
    """
    votes_by_part = _votes_by_part(scan.parts, scan.readings)
    verdicts = [
        decide(part, votes_by_part.get(part.id, []), answers.get(part.id, scan.measurement))
        for part in scan.parts
    ]
    updated = DeviceScan(
        device_id=scan.device_id,
        ok=scan.ok,
        message=scan.message,
        device_name=scan.device_name,
        measurement=scan.measurement,
        photo_path=scan.photo_path,
        overlay_path=scan.overlay_path,
        parts=list(scan.parts),
        readings=list(scan.readings),
        verdicts=verdicts,
    )
    if inventory is not None and remember and scan.ok:
        inventory.add(
            Device(id=scan.device_id, name=scan.device_name or scan.device_id, source="scan"),
            scan.parts,
            verdicts,
            readings=scan.readings,
        )
    return updated


def run_dialogue(
    scan: DeviceScan,
    prompter: Prompter | None = None,
    *,
    inventory: Any | None = None,
    retries: int = 1,
) -> tuple[DeviceScan, list[Transition]]:
    """Ask the user about every resolvable part, then re-run the gate.

    Returns the re-graded scan and the before/after transitions (the list
    includes NEEDS_MEASUREMENT -> VERIFIED and ABSTAIN -> VERIFIED, the
    D3 beat: one meter reading settles the part). A part the user skipped
    or answered with gibberish twice keeps its verdict exactly as it was.
    """
    answers = measure(scan.verdicts, scan.readings, prompter, retries=retries)
    updated = apply_measurements(scan, answers, inventory=inventory)
    return updated, state_transitions(scan.verdicts, updated.verdicts)


def _votes_by_part(
    parts: list[Part], readings: list[Reading]
) -> dict[str, list[Reading]]:
    votes: dict[str, list[Reading]] = {part.id: [] for part in parts}
    for reading in readings:
        votes.setdefault(reading.part_id, []).append(reading)
    return votes
