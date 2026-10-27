"""Shared Salvage schemas: Part, Reading, Verdict, plus the State enum.

These are the contract every other Salvage module and session builds on.
Defined by brief 01; other briefs import from here and never redefine them.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class State(str, Enum):
    """Final gate states. A part is always in exactly one of these."""

    VERIFIED = "VERIFIED"
    NEEDS_MEASUREMENT = "NEEDS_MEASUREMENT"
    ABSTAIN = "ABSTAIN"
    HAZARD = "HAZARD"


class Condition(str, Enum):
    """Whether a part actually works, as determined by evidence."""

    WORKS = "works"
    DEAD = "dead"
    UNKNOWN = "unknown"


BBox = Annotated[tuple[float, float, float, float], Field(min_length=4, max_length=4)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class Part(BaseModel):
    """One segmented object inside one device photo/video frame."""

    model_config = ConfigDict(extra="forbid")

    id: str
    device_id: str
    crop_path: str
    bbox: BBox
    mask_path: str


class Reading(BaseModel):
    """What one reader model says about one part."""

    model_config = ConfigDict(extra="forbid")

    part_id: str
    model_id: str
    part_type: str
    part_number: str | None = None
    key_rating: str | None = None
    condition: Condition
    confidence: Confidence
    evidence: str


class Verdict(BaseModel):
    """The gate's final, provable answer about one part."""

    model_config = ConfigDict(extra="forbid")

    part_id: str
    state: State
    label: str
    reasons: list[str] = Field(default_factory=list)
    requested_measurement: str | None = None
