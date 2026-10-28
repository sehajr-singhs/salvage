"""Build the cached sample-device bundle for the no-key demo.

Run once (``python -m salvage.web.samples_build``), commit the artifacts.
The demo story: a dead drone on the bench, already scanned, already gated.
Everything under ``web/static/samples/`` is a cache artifact — the sample
button never runs a model, hits the network, or touches a camera.

The bundle reuses the real pipeline honestly:
- the photo is a synthetic grayscale bench shot drawn with PIL;
- segmentation runs the real ``salvage.segment.segment.segment`` with a
  ``MockSAMBackend`` replaying hand-drawn detections (masks are real RLEs);
- the readings are the curated readings the swarm would produce for this
  device, and the verdicts come from the real ``salvage.gate.decide``,
  including one measurement ("8.2 ohms" on the motor leads) already taken
  on the bench — the D3 story in miniature;
- the overlay is the real ``salvage.segment.overlay`` (1 px white outlines).

States demonstrated: VERIFIED (x2), NEEDS_MEASUREMENT, ABSTAIN, HAZARD.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from salvage.gate import decide
from salvage.schemas import Condition, Part, Reading
from salvage.segment.backends import Detection, MockSAMBackend
from salvage.segment.overlay import overlay as render_overlay
from salvage.segment.rle import rle_encode
from salvage.segment.segment import segment

WIDTH, HEIGHT = 640, 480

# --- synthetic bench photo --------------------------------------------------


def _photo() -> Image.Image:
    """A grayscale mock shot of an opened dead drone on a black mat."""
    image = Image.new("RGB", (WIDTH, HEIGHT), (8, 8, 8))
    draw = ImageDraw.Draw(image)

    # The opened shell: an ESD mat with the frame half-folded onto it.
    draw.rectangle((16, 16, WIDTH - 16, HEIGHT - 16), fill=(20, 20, 20))
    draw.rectangle((16, 16, WIDTH - 16, HEIGHT - 16), outline=(48, 48, 48))

    # 1. BLDC motor bell (circle + shaft) with its marking visible.
    _circle(draw, (115, 170), 78, (58, 58, 58))
    _circle(draw, (115, 170), 60, (44, 44, 44))
    _circle(draw, (115, 170), 30, (78, 78, 78))
    _circle(draw, (115, 170), 12, (95, 95, 95))
    _text(draw, "2205 2300KV", (62, 212))

    # 2. Cooling fan: square frame, round hub, blades.
    draw.rounded_rectangle((230, 40, 370, 160), radius=8, fill=(38, 38, 38))
    _circle(draw, (300, 100), 46, (30, 30, 30))
    for angle in range(0, 360, 60):
        draw.pieslice((262, 62, 338, 138), angle - 18, angle + 18, fill=(52, 52, 52))
    _circle(draw, (300, 100), 18, (70, 70, 70))
    _text(draw, "4010", (278, 132))

    # 3. Electrolytic capacitor can with a rating stripe.
    draw.rounded_rectangle((480, 60, 560, 210), radius=10, fill=(40, 40, 40))
    draw.rectangle((486, 150, 554, 158), fill=(70, 70, 70))
    draw.rectangle((486, 120, 554, 126), fill=(62, 62, 62))
    _text(draw, "470uF 25V", (486, 180))

    # 4. Heatsink: a block with fins.
    draw.rectangle((400, 300, 600, 420), fill=(34, 34, 34))
    for x in range(412, 596, 18):
        draw.rectangle((x, 306, x + 8, 414), fill=(56, 56, 56))

    # 5. A swollen LiPo pouch: two wrapper layers, obvious bulge.
    draw.rounded_rectangle((60, 320, 250, 450), radius=6, fill=(36, 36, 36))
    draw.rounded_rectangle((66, 326, 244, 426), radius=6, outline=(60, 60, 60))
    _circle(draw, (170, 380), 52, (52, 52, 52))
    _text(draw, "3.7V 150mAh", (74, 428))

    return image


def _circle(draw: ImageDraw.ImageDraw, xy: tuple[int, int], r: int, fill: tuple) -> None:
    x, y = xy
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)


def _text(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int]) -> None:
    draw.text(xy, text, fill=(120, 120, 120))


def _mask_of(shapes: list[Image.Image]) -> Any:
    """Union of drawn shapes -> RLE Mask on the full canvas."""
    canvas = Image.new("L", (WIDTH, HEIGHT), 0)
    for shape in shapes:
        canvas.paste(shape.convert("L"), (0, 0), shape.convert("L"))
    return rle_encode(np.asarray(canvas) > 127)


# --- the parts the sample declares -----------------------------------------

B_MOTOR = (38, 92, 193, 248)
B_FAN = (230, 40, 370, 160)
B_CAP = (480, 60, 560, 210)
B_HEATSINK = (400, 300, 600, 420)
B_CELL = (60, 320, 250, 450)


def detections() -> list[Detection]:
    """Hand-drawn detections with real masks, in the prompts' vocabulary."""
    motor = Image.new("L", (WIDTH, HEIGHT), 0)
    md = ImageDraw.Draw(motor)
    md.rectangle((38, 92, 193, 248), fill=255)
    _circle(md, (115, 170), 78, 255)
    fan = Image.new("L", (WIDTH, HEIGHT), 0)
    fd = ImageDraw.Draw(fan)
    fd.rectangle(B_FAN, fill=255)
    cap = Image.new("L", (WIDTH, HEIGHT), 0)
    ImageDraw.Draw(cap).rectangle(B_CAP, fill=255)
    heatsink = Image.new("L", (WIDTH, HEIGHT), 0)
    ImageDraw.Draw(heatsink).rectangle(B_HEATSINK, fill=255)
    cell = Image.new("L", (WIDTH, HEIGHT), 0)
    cd = ImageDraw.Draw(cell)
    cd.rectangle(B_CELL, fill=255)
    _circle(cd, (170, 380), 58, 255)

    return [
        Detection(prompt="dc motor", score=0.91, bbox=B_MOTOR, mask=_mask_of([motor])),
        Detection(prompt="fan", score=0.88, bbox=B_FAN, mask=_mask_of([fan])),
        Detection(prompt="electrolytic capacitor", score=0.84, bbox=B_CAP, mask=_mask_of([cap])),
        Detection(prompt="heatsink", score=0.79, bbox=B_HEATSINK, mask=_mask_of([heatsink])),
        Detection(prompt="lithium battery cell", score=0.86, bbox=B_CELL, mask=_mask_of([cell])),
    ]


# --- curated readings per part (what the swarm says about this device) -----


def _reading(part_id: str, model: str, ptype: str, evidence: str, **kw: Any) -> Reading:
    return Reading(
        part_id=part_id,
        model_id=model,
        part_type=ptype,
        condition=Condition(kw.get("condition", "unknown")),
        part_number=kw.get("part_number"),
        key_rating=kw.get("key_rating"),
        confidence=kw.get("confidence", 0.82),
        evidence=evidence,
    )


def readings_by_key() -> dict[tuple[float, float, float, float], list[Reading]]:
    """Two independent Meta readers per part, keyed by the detection bbox."""
    motor_key, fan_key = B_MOTOR, B_FAN
    cap_key, sink_key, cell_key = B_CAP, B_HEATSINK, B_CELL
    return {
        # Measurement already taken on the bench: 8.2 ohms of winding
        # resistance, inside the gate's 0.5 to 20 ohm expectation -> VERIFIED.
        motor_key: [
            _reading(
                "p_motor_a", "muse-spark-1.3", "bldc_motor",
                "bell marked '2205, 2300KV'; esc wires and the pwm driver cable leave the base",
                part_number="2205 2300KV", key_rating="2300KV, 12 V", confidence=0.9,
            ),
            _reading(
                "p_motor_b", "muse-spark-open", "bldc_motor",
                "same 2205 bell, 2300KV marking; three-phase esc leads, pwm driver on board",
                part_number="2205 2300KV", key_rating="2300KV, 12 V", confidence=0.87,
            ),
        ],
        # Both readers say it works, but a fan still gets asked for a spin
        # test -> NEEDS_MEASUREMENT, the "it asks the physical world" beat.
        fan_key: [
            _reading(
                "p_fan_a", "muse-spark-1.3", "cooling_fan",
                "4010 frame marking reads cleanly; blades unbroken, hub intact",
                part_number="4010", key_rating="5 V 0.2 A", condition="works",
            ),
            _reading(
                "p_fan_b", "muse-spark-open", "cooling_fan",
                "4010 brushless fan; bearings look sound, no cracking in the blades",
                part_number="4010", key_rating="5 V 0.2 A", condition="works",
            ),
        ],
        # The two readers disagree about what this black cylinder is ->
        # agreement 0.5 < 0.67 -> ABSTAIN. The gate proves nothing it cannot.
        cap_key: [
            _reading(
                "p_cap_a", "muse-spark-1.3", "electrolytic_capacitor",
                "cylindrical can with a positive stripe, '470uF 25V' on the sleeve",
                key_rating="470uF 25V",
            ),
            _reading(
                "p_cap_b", "muse-spark-open", "integrated_circuit",
                "black cylinder could be a coin cell holder; the marking is out of focus",
            ),
        ],
        # Plain heatsink, no cheap check exists for it, readers agree it is
        # intact -> VERIFIED without a measurement.
        sink_key: [
            _reading(
                "p_sink_a", "muse-spark-1.3", "heatsink",
                "aluminum fins, no fusing or heat discoloration visible", condition="works",
            ),
            _reading(
                "p_sink_b", "muse-spark-open", "heatsink",
                "pressed-aluminum heatsink bolted over the driver, fins straight",
                condition="works",
            ),
        ],
        # Any reader flagging a swollen lithium cell -> HAZARD, and the gate
        # refuses the step in one vote.
        cell_key: [
            _reading(
                "p_cell_a", "muse-spark-1.3", "lipo_cell",
                "pouch is swollen and puffed along one edge", key_rating="3.7V",
            ),
            _reading(
                "p_cell_b", "muse-spark-open", "lipo_cell",
                "single-cell pouch with '3.7V 150mAh'; wrappers intact",
                key_rating="3.7V",
            ),
        ],
    }


def build(out_dir: str | Path = "web/static/samples") -> None:
    """Generate photo.png, overlay.png and sample.json into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    photo = _photo()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        photo_path = tmp_path / "photo.png"
        photo.save(photo_path)

        parts = segment(
            photo_path,
            "sample_drone",
            backend=MockSAMBackend(detections()),
            data_root=tmp_path,
        )
        assert parts, "segmentation produced no parts for the sample device"

        by_bbox = {part.bbox: part for part in parts}
        curated = readings_by_key()
        # Map (curated bbox -> part) with a tolerance, guarding against a
        # bbox roundtrip through detection cleaning.
        def part_for(key: tuple[float, float, float, float]) -> Part:
            near = min(
                by_bbox,
                key=lambda box: sum(abs(a - b) for a, b in zip(box, key)),
            )
            assert sum(abs(a - b) for a, b in zip(near, key)) < 4.0, (near, key)
            return by_bbox[near]

        measurement_note = "8.2 ohms across the two motor leads"
        readings: list[Reading] = []
        entries: list[dict[str, Any]] = []
        verdicts = []
        for index, (key, votes) in enumerate(curated.items(), start=1):
            part = part_for(key)
            fixed = [vote.model_copy(update={"part_id": part.id}) for vote in votes]
            readings.extend(fixed)
            verdict = decide(
                part,
                fixed,
                measurement_note if key == B_MOTOR else None,
            )
            verdicts.append(verdict)
            entries.append(
                {
                    "id": part.id,
                    "n": index,
                    "bbox": [round(v, 1) for v in part.bbox],
                    "name": {
                        B_MOTOR: "BLDC 2205",
                        B_FAN: "FAN 4010",
                        B_CAP: "CAP 470uF 25V",
                        B_HEATSINK: "HEATSINK",
                        B_CELL: "LIPO 3.7V CELL",
                    }[key],
                    "label": verdict.label,
                    "state": verdict.state.value,
                    "reasons": list(verdict.reasons),
                    "requested_measurement": verdict.requested_measurement,
                }
            )

        render_overlay(photo_path, parts, out / "overlay.png")
        part_json = [part.model_dump(mode="json") for part in parts]
        readings_json = [reading.model_dump(mode="json") for reading in readings]
        verdicts_json = [verdict.model_dump(mode="json") for verdict in verdicts]

    photo.save(out / "photo.png")

    bundle = {
        "device": {
            "id": "sample_drone",
            "name": "DEAD DRONE",
            "source": "junk drawer",
            "notes": "quad that stopped arming after a wet field day",
        },
        "note": (
            "cached scan: precomputed offline, no keys, no camera. The motor "
            "verdict includes the bench measurement '8.2 ohms' already taken."
        ),
        "photo": "photo.png",
        "overlay": "overlay.png",
        "size": [WIDTH, HEIGHT],
        "measurement_note": measurement_note,
        "parts": entries,
        "inventory_seed": {
            "device": {
                "id": "sample_drone",
                "name": "DEAD DRONE",
                "source": "junk drawer",
                "notes": "quad that stopped arming after a wet field day",
            },
            "parts": part_json,
            "verdicts": verdicts_json,
            "readings": readings_json,
        },
    }

    (out / "sample.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    states = sorted(entry["state"] for entry in entries)
    print(f"sample bundle written to {out} (states: {states})")


if __name__ == "__main__":
    build()
