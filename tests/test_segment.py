"""Tests for brief 02: segmentation and crops.

Everything runs offline: the model backend is a deterministic mock, the
fixture image is generated on disk, and no network call is ever made.
"""

from __future__ import annotations

import ast
import base64
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from salvage.schemas import Part
from salvage.segment.backends import (
    Detection,
    HostedSAMBackend,
    MockSAMBackend,
    build_payload,
    split_payload,
)
from salvage.segment.overlay import darken, mask_outline, overlay
from salvage.segment.prompts import PART_PROMPTS
from salvage.segment.rle import (
    bbox_iou,
    mask_iou,
    rect_mask,
    rle_area,
    rle_decode,
    rle_encode,
)
from salvage.segment.segment import padded_box, segment

WIDTH, HEIGHT = 320, 240


@pytest.fixture()
def fixture_image(tmp_path: Path) -> Path:
    """A deterministic RGB device photo: gray plate with darker components."""
    rng = np.random.default_rng(7)
    base = np.full((HEIGHT, WIDTH, 3), 150, dtype=np.uint8)
    base[40:100, 30:130] = 90  # a component
    base[140:200, 200:300] = 95  # another one
    base += rng.integers(0, 8, size=base.shape).astype(np.uint8)
    path = tmp_path / "device.png"
    Image.fromarray(base, "RGB").save(path)
    return path


def det(
    bbox: tuple[float, float, float, float],
    prompt: str = "electrolytic capacitor",
    score: float = 0.9,
    mask: bool = True,
) -> Detection:
    """A Detection with a rectangle mask on the standard test canvas."""
    return Detection(
        prompt=prompt,
        score=score,
        bbox=bbox,
        mask=rect_mask(bbox, WIDTH, HEIGHT) if mask else None,
    )


# --- rle / iou helpers -----------------------------------------------------


def test_rle_roundtrip_and_area() -> None:
    raw = np.zeros((6, 8), dtype=bool)
    raw[1:4, 2:6] = True
    mask = rle_encode(raw)
    assert rle_area(mask) == 12
    assert mask.width == 8 and mask.height == 6
    assert (rle_decode(mask) == raw).all()


def test_rect_mask_matches_bbox() -> None:
    mask = rect_mask((10.0, 12.0, 40.0, 22.0), WIDTH, HEIGHT)
    pixels = rle_decode(mask)
    assert pixels[12, 10] and not pixels[12, 9] and not pixels[11, 10]
    assert rle_area(mask) == 30 * 10


def test_mask_iou_and_bbox_iou() -> None:
    a = rect_mask((0.0, 0.0, 100.0, 100.0), WIDTH, HEIGHT)
    b = rect_mask((0.0, 0.0, 110.0, 110.0), WIDTH, HEIGHT)
    assert abs(mask_iou(a, b) - 10000.0 / 12100.0) < 1e-3
    assert abs(bbox_iou((0, 0, 100, 100), (0, 0, 110, 110)) - 10000.0 / 12100.0) < 1e-3
    assert mask_iou(a, a) == 1.0
    assert bbox_iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


# --- the "done when" scenario -----------------------------------------------


def test_segment_with_mock_backend_returns_parts_with_files(
    fixture_image: Path, tmp_path: Path
) -> None:
    detections = [
        det((26.0, 36.0, 134.0, 104.0), prompt="integrated circuit"),
        det((196.0, 136.0, 304.0, 204.0), prompt="heatsink"),
    ]
    data_root = tmp_path / "data"
    parts = segment(
        fixture_image,
        "dev001",
        backend=MockSAMBackend(detections),
        data_root=data_root,
        cache_dir=tmp_path / "cache",
    )
    assert len(parts) == 2
    assert all(isinstance(p, Part) for p in parts)
    for part in parts:
        # crop and mask files exist on disk (brief: crops in data/crops/<id>/)
        crop = Path(part.crop_path)
        mask_file = Path(part.mask_path)
        assert crop.is_file()
        assert mask_file.is_file()
        assert (crop.parent == data_root / "crops" / "dev001") or crop.parent.as_posix().endswith(
            "data/crops/dev001"
        )
        assert part.mask_path.replace("\\", "/").endswith("data/masks/dev001/p_001.png") or part.mask_path.replace("\\", "/").endswith("data/masks/dev001/p_002.png")
        # the bbox sits inside the image
        x1, y1, x2, y2 = part.bbox
        assert 0 <= x1 <= x2 <= WIDTH
        assert 0 <= y1 <= y2 <= HEIGHT
        # 10% padding: the crop is at least the bbox plus padded margins
        with Image.open(crop) as crop_img:
            w = x2 - x1
            h = y2 - y1
            assert crop_img.size[0] >= round(w * 1.2)
            assert crop_img.size[1] >= round(h * 1.2)
        # mask PNGs are single-channel binary images
        with Image.open(mask_file) as mask_png:
            assert mask_png.mode == "L"
            assert mask_png.size == (WIDTH, HEIGHT)
            raw = np.asarray(mask_png)
            assert set(np.unique(raw)) <= {0, 255}


def test_segment_ids_are_stable_top_to_bottom(fixture_image: Path, tmp_path: Path) -> None:
    detections = [
        det((196.0, 136.0, 304.0, 204.0)),  # bottom right
        det((26.0, 36.0, 134.0, 104.0)),  # top left
        det((150.0, 30.0, 190.0, 60.0)),  # top middle-left
    ]
    cache = tmp_path / "cache"
    first = segment(
        fixture_image,
        "dev001",
        backend=MockSAMBackend(detections),
        data_root=tmp_path / "data",
        cache_dir=cache,
    )
    second = segment(
        fixture_image,
        "dev001",
        backend=MockSAMBackend(detections),
        data_root=tmp_path / "data",
        cache_dir=cache,
    )
    assert [p.id for p in first] == ["p_001", "p_002", "p_003"]
    assert [p.id for p in first] == [p.id for p in second]
    # sorted by (y1, x1): top-most first, left wins on ties
    assert first[0].bbox[1] <= first[1].bbox[1]
    assert first[1].bbox[1]  # belongs to the overlapping sort order


def test_fixture_image(tmp_path: Path) -> None:
    """Recreate the fixture inline so module-level fixture generation is covered."""
    rng = np.random.default_rng(7)
    base = np.full((HEIGHT, WIDTH, 3), 150, dtype=np.uint8)
    path = tmp_path / "img.png"
    Image.fromarray(base + rng.integers(0, 8, size=(HEIGHT, WIDTH, 3), dtype=np.uint8), "RGB").save(path)
    assert path.is_file()


# --- prompts and backend plumbing -------------------------------------------


class RecordingBackend(MockSAMBackend):
    """Mock that records how it was called."""

    def __init__(self, detections: list[Detection]) -> None:
        super().__init__(detections)
        self.seen_prompts: list[str] | None = None

    def detect(self, image_bytes, *, prompts=None, cache_dir=None):
        self.seen_prompts = list(prompts) if prompts is not None else None
        return super().detect(image_bytes)


def test_segment_prompts_the_fixed_vocabulary(fixture_image, tmp_path) -> None:
    backend = RecordingBackend([det((0.0, 0.0, 10.0, 10.0))])
    segment(
        fixture_image,
        "dev001",
        backend=backend,
        data_root=tmp_path / "data",
        cache_dir=tmp_path / "cache",
    )
    assert backend.seen_prompts == list(PART_PROMPTS)
    assert "electrolytic capacitor" in PART_PROMPTS
    assert "ribbon cable connector" in PART_PROMPTS


def test_duplicate_prompt_removed_no_overlap() -> None:
    from salvage.segment.overlap import no_overlap

    big = det((0.0, 0.0, 110.0, 110.0), score=0.8)
    small = det((0.0, 0.0, 100.0, 100.0), score=0.95)
    kept = no_overlap([small, big])
    assert len(kept) == 1
    assert kept[0].bbox == (0.0, 0.0, 110.0, 110.0)  # larger mask survives
    # IoU just under the threshold stays (100x100 vs 112x112: IoU ~= 0.797)
    assert len(no_overlap([det((0.0, 0.0, 100.0, 100.0)), det((0.0, 0.0, 112.0, 112.0))])) == 2
    # IoU just above the threshold drops to the larger one (109x109: 0.842)
    assert len(no_overlap([det((0.0, 0.0, 100.0, 100.0)), det((0.0, 0.0, 109.0, 109.0))])) == 1


# --- validation and failure rules -------------------------------------------


def test_missing_image_raises(fixture_image: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        segment(
            fixture_image.with_name("nope.png"),
            "dev001",
            backend=MockSAMBackend([]),
            cache_dir=tmp_path / "cache",
        )


def test_failed_backend_returns_empty_not_crash(fixture_image: Path, tmp_path: Path) -> None:
    class FailingBackend:
        name = "failing"
        model_id = "fails"

        def available(self) -> bool:
            return True

        def detect(self, image_bytes, *, prompts=None, cache_dir=None):
            return None

    parts = segment(
        fixture_image,
        "dev001",
        backend=FailingBackend(),  # type: ignore[arg-type]
        data_root=tmp_path / "data",
        cache_dir=tmp_path / "cache",
    )
    assert parts == []


class ExplodingBackend:
    name = "exploding"
    model_id = "boom"

    def available(self) -> bool:
        return True

    def detect(self, image_bytes, *, prompts=None, cache_dir=None):
        raise RuntimeError("the GPU caught fire")

    def _boom(self) -> None:  # pragma: no cover
        pass


def test_raising_backend_returns_empty_not_crash(fixture_image: Path, tmp_path: Path) -> None:
    parts = segment(
        fixture_image,
        "dev001",
        backend=ExplodingBackend(),  # type: ignore[arg-type]
        data_root=tmp_path / "data",
        cache_dir=tmp_path / "cache",
    )
    assert parts == []


def test_no_backend_available_returns_empty(fixture_image: Path, tmp_path: Path, monkeypatch) -> None:
    import salvage.segment.segment as seg

    monkeypatch.setattr(seg, "first_available_backend", lambda: None)
    parts = segment(
        fixture_image,
        "dev001",
        data_root=tmp_path / "data",
        cache_dir=tmp_path / "cache",
    )
    assert parts == []


def test_off_canvas_and_degenerate_boxes_are_dropped(fixture_image: Path, tmp_path: Path) -> None:
    detections = [
        det((-20.0, -20.0, 150.0, 150.0)),  # clamped, becomes smaller
        det((0.0, 0.0, 1.0, 1.0)),  # kept (tiny but valid)
        Detection(prompt="fan", score=0.5, bbox=(0.0, 0.0, 40.0, 0.0)),  # degenerate
    ]
    parts = segment(
        fixture_image,
        "dev001",
        backend=MockSAMBackend(detections),
        data_root=tmp_path / "data",
        cache_dir=tmp_path / "cache",
    )
    assert len(parts) == 2
    for part in parts:
        x1, y1, x2, y2 = part.bbox
        assert 0 <= x1 <= x2 <= WIDTH and 0 <= y1 <= y2 <= HEIGHT


# --- crops, padding ----------------------------------------------------------


def test_padded_box_clamps_at_edges() -> None:
    box = padded_box((0.0, 0.0, 100.0, 100.0), WIDTH, HEIGHT, 0.10)
    assert box == (0, 0, 110, 110)
    box = padded_box((WIDTH - 50.0, HEIGHT - 40.0, float(WIDTH), float(HEIGHT)), WIDTH, HEIGHT, 0.10)
    assert box == (WIDTH - 55, HEIGHT - 44, WIDTH, HEIGHT)
    # never out of bounds
    for x in box:
        assert 0 <= x <= max(WIDTH, HEIGHT)


# --- hosted backend B (offline, via monkeypatched HTTP) -----------------------


API_URL = "https://api.meta.ai/v1/vision/segmentation"


def _api_response(detections: list[dict]) -> httpx.Response:
    # a request object must be attached or raise_for_status() raises RuntimeError
    return httpx.Response(
        200,
        json={"detections": detections},
        request=httpx.Request("POST", API_URL),
    )


def test_build_payload_roundtrip() -> None:
    payload = build_payload(["screw", "fan"], b"IMAGEBYTES")
    prompts, image = split_payload(payload)
    assert prompts == ["screw", "fan"]
    assert image == b"IMAGEBYTES"


def test_hosted_backend_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("META_API_KEY", raising=False)
    backend = HostedSAMBackend()
    assert not backend.available()
    assert backend.detect(b"img") is None


def test_hosted_backend_detect_and_cache_hit(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("META_API_KEY", "fake-key-for-tests")
    detection = det((10.0, 10.0, 60.0, 40.0)).model_dump(mode="json")
    payload = base64.b64encode(b"imgbytes").decode("ascii")
    calls = {"n": 0}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        assert json["model"] == "sam-3.1"
        assert json["image"] == payload
        assert headers["Authorization"].startswith("Bearer fake-key")
        assert timeout is not None  # every network call carries a timeout
        return _api_response([detection])

    monkeypatch.setattr(httpx, "post", fake_post)
    backend = HostedSAMBackend()
    cache_dir = tmp_path / "cache"
    first = backend.detect(b"imgbytes", cache_dir=cache_dir)
    second = backend.detect(b"imgbytes", cache_dir=cache_dir)
    assert first is not None and second is not None
    assert len(first) == 1 and first[0].prompt == "electrolytic capacitor"
    assert [d.model_dump(mode="json") for d in first] == [d.model_dump(mode="json") for d in second]
    assert calls["n"] == 1  # second call served from the sha256 cache


def test_hosted_backend_http_failure_returns_none(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("META_API_KEY", "fake-key-for-tests")

    def failing_post(*args, **kwargs):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(httpx, "post", failing_post)
    backend = HostedSAMBackend()
    assert backend.detect(b"imgbytes", cache_dir=tmp_path / "cache") is None
    assert backend.detect(b"other", cache_dir=tmp_path / "cache") is None


def test_hosted_backend_bad_payload_returns_none(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("META_API_KEY", "fake-key-for-tests")

    def fake_post(*args, **kwargs):
        return httpx.Response(200, json={"unexpected": "shape"})

    monkeypatch.setattr(httpx, "post", fake_post)
    backend = HostedSAMBackend()
    assert backend.detect(b"imgbytes", cache_dir=tmp_path / "cache") is None


# --- Modal app (Backend A): static checks only, modal is not installed -------

MODAL_APP = Path(__file__).resolve().parents[1] / "src" / "salvage" / "segment" / "modal_app.py"


def test_modal_app_is_valid_python_and_correct_repo_id() -> None:
    tree = ast.parse(MODAL_APP.read_text(encoding="utf-8"))
    module_text = MODAL_APP.read_text(encoding="utf-8")
    assert 'MODEL_ID = "facebook/sam3.1"' in module_text
    assert "facebook/sam3.1" in module_text  # exact gated HF repo id
    assert "class Sam3Detector" in module_text  # backends.py imports it
    contents = ast.dump(tree)
    assert "modal" in contents
    assert "detect" in [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]


# --- overlay, the house visual style ----------------------------------------


def test_overlay_darkens_and_outlines_once(fixture_image: Path, tmp_path: Path) -> None:
    detections = [det((26.0, 36.0, 134.0, 104.0))]
    parts = segment(
        fixture_image,
        "dev001",
        backend=MockSAMBackend(detections),
        data_root=tmp_path / "data",
        cache_dir=tmp_path / "cache",
    )
    out = tmp_path / "overlay.png"
    result = overlay(fixture_image, parts, out_path=out)
    assert out.is_file()

    original = np.asarray(Image.open(fixture_image).convert("RGB"), dtype=np.float32)
    rendered = np.asarray(result, dtype=np.uint8)
    # background is the darkened photo, no fill
    dark = np.asarray(darken(Image.open(fixture_image)), dtype=np.uint8)
    white = np.asarray(Image.fromarray(rendered)) == 255
    # every white pixel is exactly on the 1 px outline of the union of masks
    mask = np.asarray(Image.open(parts[0].mask_path)) > 127
    expected_outline = mask_outline(mask)
    assert np.array_equal(white.any(axis=-1), expected_outline)
    # interior pixel is darkened, not white-filled
    interior = (58, 70)  # strictly inside the mask (rows 36..104, cols 26..134)
    assert tuple(rendered[interior[0], interior[1]]) == tuple(dark[interior[0], interior[1]])


def test_darken_factor() -> None:
    image = Image.new("RGB", (4, 4), (100, 100, 200))
    dark = np.asarray(darken(image))
    assert dark.max() <= int(200 * 0.45) + 1
    assert dark[0, 0].tolist() == [45, 45, 90]
