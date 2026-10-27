"""Overlap removal for detections, per brief 02.

Brief rule: "Remove overlapping masks with IoU above 0.8, keeping the larger
one." Implemented as a greedy pass over detections sorted by mask area
(largest first): a detection is kept only if its IoU against every already
kept detection is at or below the threshold, so from an overlapping pair the
larger mask survives. IoU is computed on the masks when both have one, and on
bounding boxes otherwise.
"""

from __future__ import annotations

from salvage.segment.backends import Detection
from salvage.segment.rle import Mask, bbox_iou, mask_iou, rle_area

DEFAULT_IOU_THRESHOLD = 0.8

THRESHOLD_NOTE = "remove a detection when its IoU with a kept one is above 0.8"


def detection_bounding_box(d: Detection) -> tuple[int, int, int, int]:
    return tuple(d.bbox)  # type: ignore[return-value]


def _area(d: Detection) -> int:
    """Detected-object area: the mask area when present, else the bbox area."""
    if d.mask is not None:
        return rle_area(d.mask)
    x1, y1, x2, y2 = d.bbox
    return max(int(round(x2 - x1)), 0) * max(int(round(y2 - y1)), 0)


def pair_iou(a: Detection, b: Detection) -> float:
    """Mask IoU when both masks share a canvas, bbox IoU otherwise."""
    if a.mask is not None and b.mask is not None:
        if (a.mask.width, a.mask.height) == (b.mask.width, b.mask.height):
            return mask_iou(a.mask, b.mask)
        return 0.0
    return bbox_iou(a.bbox, b.bbox)


def canvas_of(detections: list[Detection]) -> Mask | None:
    """The shared mask canvas of a detection list, or None if inconsistent."""
    for d in detections:
        if d.mask is not None:
            return d.mask
    return None


def no_overlap(
    detections: list[Detection], iou_threshold: float = DEFAULT_IOU_THRESHOLD
) -> list[Detection]:
    """Drop detections whose mask overlaps a larger kept one above the threshold."""
    kept: list[Detection] = []
    for candidate in sorted(detections, key=_area, reverse=True):
        if all(pair_iou(candidate, keeper) <= iou_threshold for keeper in kept):
            kept.append(candidate)
    return kept
