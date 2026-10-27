"""Run-length encoded masks, shared by every segmentation backend.

SAM returns dense boolean masks; on-disk mask files are PNG, but backend results
travel through ``salvage.cache.cached_call`` which stores JSON, so masks are
encoded as row-major run-length pairs ``runs = [start, length, ...]`` over a
``width x height`` boolean canvas. This module also holds the overlap math
(IoU) and the rectangle-mask helper used by the tests and the mock backend.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict


class Mask(BaseModel):
    """A binary mask as run-length encoding, row-major over width*height."""

    model_config = ConfigDict(extra="forbid")

    width: int
    height: int
    runs: list[int]

    @property
    def area(self) -> int:
        """Number of set pixels without decoding."""
        return sum(int(length) for _, length in zip(self.runs[0::2], self.runs[1::2]))


def rle_encode(mask: np.ndarray) -> Mask:
    """Encode a boolean HxW array as row-major RLE."""
    flat = np.asarray(mask, dtype=bool).ravel()
    boundaries = np.flatnonzero(flat[:-1] != flat[1:]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [flat.size]))
    runs: list[int] = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        if flat[start]:
            runs.extend((int(start), int(end - start)))
    height, width = mask.shape
    return Mask(width=int(width), height=int(height), runs=runs)


def rle_decode(mask: Mask) -> np.ndarray:
    """Decode an RLE mask back to a boolean HxW array."""
    flat = np.zeros(mask.width * mask.height, dtype=bool)
    run_iter = iter(mask.runs)
    for start, length in zip(run_iter, run_iter):
        flat[int(start) : int(start) + int(length)] = True
    return flat.reshape(mask.height, mask.width)


def rle_area(mask: Mask) -> int:
    return mask.area


def rect_mask(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> Mask:
    """A rectangular mask covering ``bbox`` (clamped) on a width x height canvas."""
    x1, y1, x2, y2 = bbox
    x1 = min(max(int(round(x1)), 0), width)
    y1 = min(max(int(round(y1)), 0), height)
    x2 = min(max(int(round(x2)), 0), width)
    y2 = min(max(int(round(y2)), 0), height)
    if x2 <= x1 or y2 <= y1:
        return Mask(width=width, height=height, runs=[])
    runs: list[int] = []
    for row in range(y1, y2):
        runs.extend((row * width + x1, x2 - x1))
    return Mask(width=width, height=height, runs=runs)


def mask_iou(a: Mask, b: Mask) -> float:
    """IoU of two masks. They must share the same canvas size."""
    if a.width != b.width or a.height != b.height:
        raise ValueError("IoU is only defined for masks on the same canvas")
    a_flat = np.asarray(rle_decode(a)).ravel()
    b_flat = np.asarray(rle_decode(b)).ravel()
    union = int(np.logical_or(a_flat, b_flat).sum())
    if union == 0:
        return 0.0
    intersection = int(np.logical_and(a_flat, b_flat).sum())
    return intersection / union


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    if union <= 0.0:
        return 0.0
    return inter / union
