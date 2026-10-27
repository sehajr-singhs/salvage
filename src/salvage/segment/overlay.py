"""House visual style for segmentation overlays, per brief 02.

Masks are drawn as 1 px white outlines on a darkened copy of the photo.
No fill. The outline is the mask's 4-neighbor edge, so it is exactly one
pixel wide at any resolution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from salvage.schemas import Part

DARKEN_FACTOR = 0.45
OUTLINE_RGB = (255, 255, 255)


def darken(image: Image.Image) -> Image.Image:
    """A darkened RGB copy of the photo, the background of every overlay."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) * DARKEN_FACTOR
    return Image.fromarray(pixels.astype(np.uint8), "RGB")


def mask_outline(mask: np.ndarray) -> np.ndarray:
    """The 1 px edge of a boolean mask (pixels with a non-set 4-neighbor)."""
    m = np.asarray(mask, dtype=bool)
    interior = m.copy()
    interior[1:, :] &= m[:-1, :]
    interior[:-1, :] &= m[1:, :]
    interior[:, 1:] &= m[:, :-1]
    interior[:, :-1] &= m[:, 1:]
    return m & ~interior


def overlay(
    image_path: str | Path,
    parts: list[Part],
    out_path: str | Path | None = None,
) -> Image.Image:
    """Render ``parts`` masks onto a darkened copy of ``image_path``.

    Mask files are read from each Part's ``mask_path`` as written by
    :func:`salvage.segment.segment.segment`. Saves to ``out_path`` when given
    and always returns the rendered image.
    """
    darkened = darken(Image.open(image_path))
    canvas = np.asarray(darkened, dtype=np.uint8).copy()
    for part in parts:
        mask_png = np.asarray(
            Image.open(part.mask_path).convert("L"), dtype=np.uint8
        )
        outline = mask_outline(mask_png > 127)
        canvas[outline] = OUTLINE_RGB
    result = Image.fromarray(canvas, "RGB")
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)
    return result
