"""Brief 02 core: photo of an opened device in, one Part per component out.

Pipeline:
1. send the image bytes and the fixed noun-phrase prompts through a backend
   (hosted SAM endpoint, SAM 3.1 on Modal, or an injected mock for tests);
2. clamp every box to the image, drop degenerate ones, and give every
   detection a mask (rectangle fallback if the backend returned boxes only);
3. remove overlaps with mask IoU above 0.8, keeping the larger detection;
4. write each mask PNG to ``<data_root>/masks/<device_id>/<part_id>.png`` and
   each crop with 10% padding to ``<data_root>/crops/<device_id>/<part_id>.png``;
5. return one :class:`~salvage.schemas.Part` per detection, ids ``p_001...``
   ordered top-to-bottom, left-to-right, so they are stable across reruns.

``data_root`` defaults to ``data`` under the current working directory, in the
same spirit as the ``cache/`` default in ``salvage.cache``; pass an absolute
path in tests and the web app.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from salvage.schemas import Part
from salvage.segment.backends import Detection, SegmentBackend
from salvage.segment.overlap import DEFAULT_IOU_THRESHOLD, no_overlap
from salvage.segment.prompts import PART_PROMPTS
from salvage.segment.rle import Mask, rle_decode, rect_mask

DEFAULT_DATA_ROOT = "data"
DEFAULT_PADDING = 0.10


def first_available_backend() -> SegmentBackend | None:
    """First backend that can run right now: hosted endpoint, then Modal."""
    from salvage.segment.backends import HostedSAMBackend, ModalSAMBackend

    for factory in (HostedSAMBackend, ModalSAMBackend):
        backend = factory()
        if backend.available():
            return backend
    return None


def segment(
    image_path: str | Path,
    device_id: str,
    *,
    backend: SegmentBackend | None = None,
    prompts: Sequence[str] | None = None,
    data_root: str | Path | None = None,
    padding: float = DEFAULT_PADDING,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    cache_dir: str | Path | None = None,
) -> list[Part]:
    """Segment one device photo into Parts with crops and masks on disk.

    No backend available, a failed backend call (None), or an empty result
    yields an empty list rather than a crash, per the 00-README failure rule.
    A missing image file is a caller bug and raises ``FileNotFoundError``.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"no such image: {path}")

    image = Image.open(path).convert("RGB")
    width, height = image.size
    detector = backend if backend is not None else first_available_backend()
    if detector is None:
        return []
    use_prompts = list(prompts) if prompts is not None else list(PART_PROMPTS)
    try:
        raw = detector.detect(
            path.read_bytes(), prompts=use_prompts, cache_dir=cache_dir
        )
    except Exception:  # noqa: BLE001 - a failed backend call returns [], never crashes
        raw = None
    detections = raw if raw is not None else []
    detections = _clean(detections, width, height)
    detections = no_overlap(detections, iou_threshold)
    detections.sort(key=_sort_key)

    root = Path(data_root) if data_root is not None else Path(DEFAULT_DATA_ROOT)
    crops_dir = root / "crops" / device_id
    masks_dir = root / "masks" / device_id
    crops_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    parts: list[Part] = []
    for index, detection in enumerate(detections, start=1):
        part_id = f"p_{index:03d}"
        mask = detection.mask if detection.mask is not None else rect_mask(
            detection.bbox, width, height
        )
        mask_path = masks_dir / f"{part_id}.png"
        _write_mask(mask, mask_path)
        crop_path = crops_dir / f"{part_id}.png"
        crop_box = padded_box(detection.bbox, width, height, padding)
        image.crop(crop_box).save(crop_path)
        parts.append(
            Part(
                id=part_id,
                device_id=device_id,
                crop_path=crop_path.relative_to(
                    Path.cwd()
                ).as_posix()
                if crop_path.is_absolute() and crop_path.is_relative_to(Path.cwd())
                else crop_path.as_posix(),
                bbox=tuple(float(v) for v in detection.bbox),  # type: ignore[arg-type]
                mask_path=mask_path.relative_to(Path.cwd()).as_posix()
                if mask_path.is_absolute() and mask_path.is_relative_to(Path.cwd())
                else mask_path.as_posix(),
            )
        )
    return parts


def _sort_key(detection: Detection) -> tuple[float, float]:
    x1, y1, _, _ = detection.bbox
    return (y1, x1)


def _clean(
    detections: list[Detection], width: int, height: int
) -> list[Detection]:
    """Clamp boxes to the image, drop degenerates, normalize masks."""
    kept: list[Detection] = []
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        x1 = min(max(float(x1), 0.0), float(width))
        y1 = min(max(float(y1), 0.0), float(height))
        x2 = min(max(float(x2), 0.0), float(width))
        y2 = min(max(float(y2), 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        mask = detection.mask
        # masks on a different canvas cannot be trusted; fall back to a rect
        if mask is not None and (mask.width != width or mask.height != height):
            mask = None
        if mask is None:
            mask = rect_mask((x1, y1, x2, y2), width, height)
        kept.append(
            Detection(
                prompt=detection.prompt,
                score=detection.score,
                bbox=(x1, y1, x2, y2),
                mask=mask,
            )
        )
    return kept


def _write_mask(mask: Mask, path: Path) -> None:
    decoded = rle_decode(mask)
    pixels = decoded.astype(np.uint8) * 255
    Image.fromarray(pixels, mode="L").save(path)


def padded_box(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    """The crop box: bbox expanded by ``padding`` on each side, then clamped.

    Padding is 10% of the box's own width and height on each side, so a part
    seen fully keeps some surrounding context; at image edges the box simply
    clamps, never goes out of bounds.
    """
    x1, y1, x2, y2 = bbox
    px = (x2 - x1) * padding
    py = (y2 - y1) * padding
    cx1 = min(max(int(round(x1 - px)), 0), width)
    cy1 = min(max(int(round(y1 - py)), 0), height)
    cx2 = min(max(int(round(x2 + px)), 0), width)
    cy2 = min(max(int(round(y2 + py)), 0), height)
    return (cx1, cy1, cx2, cy2)
