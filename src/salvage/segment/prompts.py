"""Fixed prompt vocabulary for segmentation, per brief 02.

The 12 noun phrases below are the parts Salvage looks for in a device photo.
They are passed to SAM 3.1 as text ("concept") prompts. Keep this list exactly
as the brief fixes it; readers and the gate downstream expect these families.
"""

from __future__ import annotations

PART_PROMPTS: tuple[str, ...] = (
    "electrolytic capacitor",
    "dc motor",
    "lithium battery cell",
    "ribbon cable connector",
    "screw",
    "integrated circuit",
    "heatsink",
    "fan",
    "transformer",
    "inductor",
    "sensor module",
    "relay",
)
