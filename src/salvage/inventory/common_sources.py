"""Static fallback answers (brief 05): household devices that contain a part.

When the inventory has no part that matches a request, ``fallback_matches``
answers from this static table instead: which household devices usually
contain the requested kind of part, best first. Those Matches are advice,
not stock: they carry no part_id, no state, and no device_id, so a caller
can tell them apart from a real inventory hit and phrase its answer
accordingly (``"Nothing verified on your bench yet — but toy cars usually
have one."``).

Deliberately mains-safe, in step with the gate's HAZARD rules: no CRTs,
microwaves, or mains-primary power supplies appear here, so the table can
never point a person at a teardown the gate refuses. Where a listed device
could sit on a mains circuit, the entry names the low-voltage side only.
"""

from __future__ import annotations

from dataclasses import dataclass

from salvage.inventory.filters import Filters
from salvage.inventory.models import Match


@dataclass(frozen=True)
class CommonSource:
    """One kind of part and where it usually lives in household devices."""

    part_type: str
    """Canonical Salvage part-family name, e.g. ``"brushed_dc_motor"``."""

    keywords: tuple[str, ...]
    """Request words that ask for this kind of part."""

    voltage: float | None = None
    """Typical rating, volts. None when it genuinely varies by device."""

    interfaces: tuple[str, ...] = ()
    """Interface tokens this kind of part usually satisfies."""

    devices: tuple[str, ...] = ()
    """Household devices that usually contain it, best first."""

    note: str = ""
    """One honest caveat or safety line shown with the suggestion."""


COMMON_SOURCES: tuple[CommonSource, ...] = (
    CommonSource(
        part_type="brushed_dc_motor",
        keywords=("motor",),
        voltage=5.0,
        interfaces=("pwm",),
        devices=(
            "toy cars",
            "old tray-loading CD and DVD drives",
            "cordless handheld fans",
        ),
        note=(
            "the driver is usually a separate board; meter the two leads, "
            "expect a few ohms of winding resistance"
        ),
    ),
    CommonSource(
        part_type="bldc_motor",
        keywords=("bldc", "brushless"),
        voltage=11.1,
        interfaces=("esc", "pwm"),
        devices=("camera drones", "RC cars", "electric skateboards"),
        note="meter each phase pair; all three should read alike",
    ),
    CommonSource(
        part_type="esc",
        keywords=("esc", "driver"),
        interfaces=("esc",),
        devices=("camera drones", "RC cars", "robot vacuum wheel boards"),
        note=(
            "diode-test the MOSFET stage; the driver and motor usually go "
            "together, so pull both while the device is open"
        ),
    ),
    CommonSource(
        part_type="stepper_motor",
        keywords=("stepper",),
        voltage=12.0,
        devices=(
            "flatbed scanners",
            "inkjet printers",
            "old desktop 3D printers",
        ),
        note="pairs with a A4988/DRV8825-style driver board in the same unit",
    ),
    CommonSource(
        part_type="servo_motor",
        keywords=("servo",),
        voltage=5.0,
        interfaces=("servo", "pwm"),
        devices=("RC cars and planes", "robot arm kits", "camera gimbal mounts"),
        note="meter the pot across the signal arm mid-travel",
    ),
    CommonSource(
        part_type="cooling_fan",
        keywords=("fan", "cooling"),
        voltage=12.0,
        devices=("desktop PCs", "game consoles", "dead wifi routers"),
        note="power it briefly from its rated rail; a dead bearing clicks",
    ),
    CommonSource(
        part_type="lithium_cell",
        keywords=("battery", "cell", "lipo", "18650", "pack", "liion"),
        voltage=3.7,
        devices=(
            "old laptops (18650 packs)",
            "power banks",
            "cordless tool packs",
        ),
        note=(
            "never puncture, bend or short a cell, and never keep a swollen "
            "one — the gate refuses those; check open-circuit voltage"
        ),
    ),
    CommonSource(
        part_type="nimh_pack",
        keywords=("nimh",),
        voltage=1.2,
        devices=(
            "cordless home phones",
            "remote-control toys",
            "handheld vacuums",
        ),
        note="check each sub-cell's open-circuit voltage separately",
    ),
    CommonSource(
        part_type="buck_regulator",
        keywords=("regulator", "buck", "boost"),
        devices=(
            "car 12 V adapters (input side only)",
            "power banks",
            "router boards",
        ),
        note=(
            "take parts from the low-voltage side only; nothing listed here "
            "is wired to the wall outlet"
        ),
    ),
    CommonSource(
        part_type="imu_sensor",
        keywords=("sensor", "imu", "accelerometer", "gyro"),
        interfaces=("i2c", "spi"),
        devices=("dead smartphones", "motion game controllers", "camera drones"),
        note="board-level parts; probe the i2c/spi pads, not the pins",
    ),
    CommonSource(
        part_type="distance_sensor",
        keywords=("distance", "ultrasonic", "sonar", "lidar"),
        interfaces=("analog",),
        devices=("robot vacuums", "hobby robot kits", "car parking aid kits"),
        note="the ultrasonic heads are usually i2c-friendly breakouts",
    ),
    CommonSource(
        part_type="loudspeaker",
        keywords=("speaker", "audio", "buzzer"),
        voltage=8.0,
        devices=("bluetooth speakers", "alarm clocks", "talking toys"),
        note="meter the voice coil, expect a few ohms",
    ),
    CommonSource(
        part_type="microphone",
        keywords=("microphone", "mic"),
        devices=("old headsets", "laptops", "baby monitors"),
        note="electret capsules die quietly; test with a bias resistor",
    ),
    CommonSource(
        part_type="ribbon_connector",
        keywords=("connector", "ribbon", "cable"),
        devices=("old printers", "scanners", "internal DVD drive boards"),
        note="note the pin count and pitch before reusing a ribbon",
    ),
    CommonSource(
        part_type="led_backlight",
        keywords=("led", "backlight", "light"),
        voltage=3.0,
        devices=("dead LED monitors (output side only)", "flashlights", "indicator panels"),
        note="drive from a constant-current source, never a bare rail",
    ),
    CommonSource(
        part_type="camera_module",
        keywords=("camera", "webcam"),
        interfaces=("usb",),
        devices=("old smartphones", "laptops", "dash cams"),
        note="most boards expose MIPI, not USB, despite looking like a webcam",
    ),
    CommonSource(
        part_type="relay",
        keywords=("relay",),
        devices=("car fuse boxes (spares)", "old alarm systems (12 V side)"),
        note="meter the coil resistance, then check the contacts click",
    ),
)

_VOLT_MATCH_TOLERANCE = 2.0  # volts: close enough for household advice


def _source_score(source: CommonSource, parsed: Filters) -> tuple[int, int]:
    """(type tokens matched, voltage penalty) — higher first, lower second.

    A source matches when at least one requested type token is one of its
    keywords or inside its part_type. The voltage penalty is 0 when the
    source's typical rating is close to the (optional) requested voltage
    or either side is unspecified, else 1.
    """
    type_hit = any(
        token in source.keywords or token in _words(source.part_type)
        for token in parsed.type_tokens
    )
    score = sum(
        1
        for token in parsed.type_tokens
        if token in source.keywords or token in _words(source.part_type)
    )
    if not type_hit:
        return (0, 1)

    penalty = 0
    if parsed.voltage is not None and source.voltage is not None:
        if abs(source.voltage - parsed.voltage) > _VOLT_MATCH_TOLERANCE:
            penalty = 1
    return (score, penalty)


def _words(text: str) -> list[str]:
    return text.lower().replace("_", " ").split()


def fallback_matches(parsed: Filters) -> list[Match]:
    """Household advice Matches when the inventory itself has no hit.

    One Match per table row that speaks the requested part type, advice
    ordered: better type coverage and closer voltage first, then table
    order (stable). Unrequested interface tokens the source cannot
    satisfy are flagged in the reasons instead of hiding the suggestion.
    """
    if not parsed.type_tokens:
        return []

    matches: list[Match] = []
    for index, source in enumerate(COMMON_SOURCES):
        score, voltage_penalty = _source_score(source, parsed)
        if score == 0:
            continue
        reasons = [
            "nothing verified in your inventory matches this request yet",
            f"household advice: {source.part_type} parts usually live in "
            f"{source.devices[0]}",
        ]
        if voltage_penalty:
            reasons.append(
                f"this kind usually runs near {source.voltage:g} V and you "
                f"asked for {parsed.voltage:g} V; check the rating before wiring"
            )
        for token in parsed.interface_tokens:
            if token not in source.interfaces:
                reasons.append(
                    f"the requested {token} interface is not part of this "
                    "kind of part itself; it usually pairs with a separate "
                    "board that provides it"
                )
        if source.note:
            reasons.append(source.note)
        matches.append(
            Match(
                source="common_sources",
                part_id=None,
                part_type=source.part_type,
                label=source.part_type,
                state=None,
                device=source.devices[0],
                device_source="household advice",
                alternatives=list(source.devices[1:]),
                voltage=source.voltage,
                interface=" ".join(source.interfaces) or None,
                reasons=reasons,
            )
        )

    matches.sort(
        key=lambda match: (
            -_type_score(match, parsed),
            _voltage_penalty(match, parsed),
        )
    )
    return matches


def _type_score(match: Match, parsed: Filters) -> int:
    source = next(s for s in COMMON_SOURCES if s.part_type == match.part_type)
    return sum(
        1
        for token in parsed.type_tokens
        if token in source.keywords or token in _words(source.part_type)
    )


def _voltage_penalty(match: Match, parsed: Filters) -> int:
    if parsed.voltage is None or match.voltage is None:
        return 0
    return int(abs(match.voltage - parsed.voltage) > _VOLT_MATCH_TOLERANCE)
