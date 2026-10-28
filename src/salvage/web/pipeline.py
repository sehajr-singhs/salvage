"""Brief 06: the thin scan pipeline the web app serves.

One function, :func:`scan_device`, wires the sessions 02 to 05 by import —
never a reimplementation:

- ``salvage.segment.segment.segment`` (02) turns an uploaded photo into Parts
  with crops and masks on disk;
- ``salvage.readers.read_part`` (03) reads each part — the readers subpackage
  is still a stub, so an offline mock reader fills in instead; the pipeline
  prefers the real one the moment it exists;
- ``salvage.gate.decide`` (04) makes the final verify-or-abstain call;
- ``salvage.inventory.Inventory.add`` (05) remembers the scan.

Every backend and reader failure is failure-soft per the 00-README rules:
a failed segmentation yields an empty device, a failed reader simply does
not vote, and nothing here raises except a caller error (a missing image
reaches ``segment`` and raises its ``FileNotFoundError``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Sequence

from salvage.gate import decide
from salvage.inventory.models import Device
from salvage.schemas import Part, Reading, State, Verdict
from salvage.segment.segment import first_available_backend, segment
from salvage.segment.overlay import overlay

ReadingsProvider = Callable[[Part], list[Reading]]

# The three model ids voters behind the offline mock reader, so the gate
# still sees several independent voters rather than one duplicated voice.
_MOCK_READERS = ("mock-reader-open", "mock-reader-closed")

# Families the offline mock reader may claim, all inside the gate's check
# and consistency tables, so every verdict it produces has a real reason.
_MOCK_TYPES = (
    "bldc_motor",
    "dc_motor",
    "electrolytic_capacitor",
    "ceramic_capacitor",
    "cooling_fan",
    "lipo_cell",
    "21850_cell",
    "ribbon_connector",
    "heatsink",
    "integrated_circuit",
    "transformer",
    "relay",
)


def readers_available() -> bool:
    """Whether the real 03 reader module exists yet."""
    try:
        from salvage.readers import read_part  # noqa: F401

        return True
    except ImportError:
        return False


def offline_mock_readings(part: Part) -> list[Reading]:
    """Two agreeing, deterministic mock readings for one part.

    A sha256 of the device and part ids picks the part_type, so the same
    photo always produces the same verdict, and re-running costs nothing.
    The condition is unknown on purpose: without keys nothing has actually
    looked at the crop, so the gate should stay cautious and ask for a
    measurement rather than invent a verdict.
    """
    digest = hashlib.sha256(f"{part.device_id}\x00{part.id}".encode("utf-8")).digest()
    part_type = _MOCK_TYPES[digest[0] % len(_MOCK_TYPES)]
    evidence = "offline mock reader: no model keys on this machine"
    readings = []
    for index, model_id in enumerate(_MOCK_READERS):
        readings.append(
            Reading(
                part_id=part.id,
                model_id=model_id,
                part_type=part_type,
                part_number=None,
                key_rating=None,
                condition="unknown",
                confidence=0.5 + digest[1] / 255.0 * 0.1,
                evidence=(evidence if index == 0 else evidence + " (second opinion)"),
            )
        )
    return readings


def resolve_readings(provider: ReadingsProvider | None) -> ReadingsProvider:
    """The readings provider to use: injected, real 03, or the offline mock."""
    if provider is not None:
        return provider
    try:
        from salvage.readers import read_part  # type: ignore[no-redef]
    except ImportError:
        return offline_mock_readings
    return read_part  # type: ignore[no-any-return]


def scan_device(
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
) -> dict[str, Any]:
    """Run photo → parts → readings → gate for one device photo.

    Returns a JSON-ready scan dict with an overlay URL and one entry per
    part carrying the gate's verdict. When no segmentation backend can run
    (no keys, no Modal app), the result is an empty device with a pointer
    to the sample device — never an error page, never a partial guess.

    An explicitly passed ``backend`` is respected as-is: available or not,
    nothing else is auto-selected, so a hung remote can never stall a demo.
    Only a caller that passes no backend at all gets the auto-selection
    (hosted endpoint first, then the Modal app).
    """
    reader = resolve_readings(readings_provider)

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

    root = Path(data_root) if data_root is not None else Path("data")
    scan_dir = root / "scan" / device_id
    photo_url: str | None = None
    overlay_url: str | None = None

    if not parts:
        return {
            "ok": False,
            "device_id": device_id,
            "photo_url": None,
            "overlay_url": None,
            "parts": [],
            "readings_mode": "none",
            "message": (
                "no segmentation backend available (no model keys on this "
                "machine): use the sample device button"
            ),
            "hint": "sample",
        }

    scan_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = scan_dir / "overlay.png"
    overlay(image_path, parts, overlay_path)

    copy_path = scan_dir / "photo.png"
    from PIL import Image

    Image.open(image_path).convert("RGB").save(copy_path)

    readings: list[Reading] = []
    verdicts: list[Verdict] = []
    part_entries: list[dict[str, Any]] = []
    for part in parts:
        votes = reader(part)
        readings.extend(votes)
        verdict = decide(part, votes, measurement)
        verdicts.append(verdict)
        part_entries.append(
            {
                "id": part.id,
                "bbox": [round(v, 3) for v in part.bbox],
                "label": verdict.label,
                "state": verdict.state.value,
                "reasons": list(verdict.reasons),
                "requested_measurement": verdict.requested_measurement,
                "mask_url": _data_url(root, part.mask_path),
                "crop_url": _data_url(root, part.crop_path),
            }
        )

    if inventory is not None and remember:
        inventory.add(
            Device(
                id=device_id,
                name=device_name or Path(image_path).stem,
                source="upload",
            ),
            parts,
            verdicts,
            readings=readings,
        )

    return {
        "ok": True,
        "device_id": device_id,
        "photo_url": _data_url(root, copy_path),
        "overlay_url": _data_url(root, overlay_path),
        "parts": part_entries,
        "readings_mode": "real" if readers_available() else "offline mock reader",
        "message": None,
    }


def _data_url(root: Path, path: str | Path) -> str:
    """Turn a file path under ``root`` into a URL on the ``/data`` mount."""
    p = Path(path)
    rel = p.relative_to(root) if p.is_relative_to(root) else p
    return f"/data/{Path(rel).as_posix()}"


def states_in(entries: Sequence[dict[str, Any]]) -> set[str]:
    """The distinct gate states present in a scan's part entries."""
    return {entry["state"] for entry in entries}


def all_states_present(entries: Sequence[dict[str, Any]]) -> bool:
    """True when the scan demonstrates all four gate states (sample demo)."""
    return set(State) <= states_in(entries)
