"""Inventory data models (brief 05): Device, and the Match that query returns.

The shared Part / Reading / Verdict contract stays in ``salvage.schemas``;
these two models belong to the inventory alone. A Match always says which
device to open (``device``) and, when one exists, the ``part_id`` to pull —
the brief's requirement. Fallback matches from ``common_sources`` have no
part yet (``part_id`` is None, ``state`` is None) and instead name the
household devices that usually contain the requested kind of part.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from salvage.schemas import State


class Device(BaseModel):
    """A dead device on the bench that was scanned into the inventory."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    source: str | None = None
    """Where it came from (drawer, neighbor, e-waste drop-off)."""

    notes: str | None = None


class Match(BaseModel):
    """One answer to a reverse-search request.

    ``source`` says where the answer came from: ``inventory`` (a real part in
    the SQLite store) or ``common_sources`` (static advice when nothing in
    the inventory matches).
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["inventory", "common_sources"]

    part_id: str | None = None
    part_type: str | None = None
    label: str | None = None
    state: State | None = None
    """VERIFIED / NEEDS_MEASUREMENT; None on a fallback match."""

    device_id: str | None = None
    device: str | None = None
    """The device to open (its name, or the household source on a fallback)."""

    device_source: str | None = None
    alternatives: list[str] = Field(default_factory=list)
    """Fallback only: other household devices that usually contain the part."""

    voltage: float | None = None
    """The part's own voltage rating in volts, when known."""

    current: float | None = None
    """The part's current rating in amps, when known."""

    interface: str | None = None
    """Space-joined interface tokens (i2c, spi, pwm, esc, driver, ...)."""

    crop_path: str | None = None
    requested_measurement: str | None = None
    """Carried through for NEEDS_MEASUREMENT matches: the bench can ask out loud."""

    reasons: list[str] = Field(default_factory=list)
    """The gate's reasons, kept so an answer is never unexplained."""
