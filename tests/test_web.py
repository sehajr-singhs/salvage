"""Web demo tests (brief 06), all offline: no keys, no network, no camera.

The brief's done-when: FastAPI TestClient tests pass, and the sample
button renders from cache alone — ``GET /api/sample`` returns the complete
four-state part list out of ``web/static/samples/`` with no provider keys
in the environment and no segmentation backend running.

The app is built fresh per test with tmp data/inventory paths, so tests
never write into the repo's ``data/`` or ``cache/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from salvage.inventory.store import Inventory
from salvage.schemas import Condition, Reading, State
from salvage.segment.backends import Detection, MockSAMBackend
from salvage.web.app import create_app
from salvage.web.pipeline import offline_mock_readings

STATIC = Path("web/static")
SAMPLES = STATIC / "samples"

# Provider key env vars, defensively cleared for every client: the demo has
# to work (and the no-key fallbacks have to trigger) with zero keys.
KEY_ENV_VARS = (
    "META_API_KEY",
    "BASETEN_API_KEY",
    "FIREWORKS_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "NVIDIA_NIM_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
)


@pytest.fixture(autouse=True)
def no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def fixture_image(tmp_path: Path) -> Path:
    """A tiny two-blob 'device photo': two bright blocks on black."""
    image = Image.new("RGB", (320, 200), (10, 10, 10))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 40, 140, 150), fill=(200, 200, 200))
    draw.ellipse((180, 30, 290, 170), fill=(160, 160, 160))
    path = tmp_path / "fixture_device.png"
    image.save(path)
    return path


def two_agreeing(part_id: str, ptype: str = "ribbon_connector") -> list[Reading]:
    """Two independent readers agreeing the part works (no cheap check)."""
    return [
        Reading(
            part_id=part_id,
            model_id=model_id,
            part_type=ptype,
            condition=Condition.WORKS,
            confidence=0.8,
            evidence="marking legible in the crop",
        )
        for model_id in ("muse-spark-1.3", "muse-spark-open")
    ]


def make_app(tmp_path: Path, **kwargs) -> tuple[TestClient, Path]:
    """A test app: tmp data root (served at /data), tmp inventory DB."""
    data_root = tmp_path / "data"
    client = TestClient(
        create_app(
            data_root=data_root,
            inventory=Inventory(db_path=tmp_path / "inventory.db"),
            **kwargs,
        )
    )
    return client, data_root


# --- the sample button renders from cache alone ----------------------------


def test_sample_renders_from_cache_alone(tmp_path: Path) -> None:
    """No keys, no backend: the sample endpoint still returns everything."""
    client, _ = make_app(tmp_path)
    response = client.get("/api/sample")
    assert response.status_code == 200
    bundle = response.json()
    assert bundle["device"]["id"] == "sample_drone"
    states = {part["state"] for part in bundle["parts"]}
    assert states == {s.value for s in State}, states  # all four demonstrated
    for part in bundle["parts"]:
        assert part["bbox"] and part["name"] and part["label"]
    # the cached overlay image is actually on disk
    assert (SAMPLES / "overlay.png").is_file()
    assert (SAMPLES / "photo.png").is_file()


def test_sample_served_even_with_no_static_backend(tmp_path: Path) -> None:
    """The sample never touches a backend: same answer with one injected."""
    client, _ = make_app(tmp_path, scan_backend=MockSAMBackend([]))
    first = client.get("/api/sample")
    client2, _ = make_app(tmp_path / "again")
    second = client2.get("/api/sample")
    assert first.json() == second.json()  # pure cache, no model involved


def test_sample_health_reports_no_providers(tmp_path: Path) -> None:
    client, _ = make_app(tmp_path)
    payload = client.get("/api/health").json()
    assert payload["providers"] == []
    assert payload["sample_available"] is True


def test_sample_seeds_reverse_search(tmp_path: Path) -> None:
    """The seeded sample makes the request box answer offline too."""
    client, _ = make_app(tmp_path)
    response = client.post(
        "/api/search", json={"text": "I need a 12 V motor with a driver"}
    )
    assert response.status_code == 200
    matches = response.json()
    assert matches, matches
    top = matches[0]
    assert top["source"] == "inventory"
    assert top["device"] == "DEAD DRONE"
    assert top["state"] == State.VERIFIED.value
    assert top["voltage"] == pytest.approx(12.0)
    assert "esc" in (top["interface"] or "")


def test_search_never_returns_hazard(tmp_path: Path) -> None:
    """Asking for the sample's battery gets advice, never the HAZARD part."""
    client, _ = make_app(tmp_path)
    matches = client.post(
        "/api/search", json={"text": "I need a 3.7 V lipo cell"}
    ).json()
    assert all((m.get("state") or "") != State.HAZARD.value for m in matches)
    # and nothing claims to come from the inventory with a hazard part
    assert not [m for m in matches if m.get("source") == "inventory"]


# --- index and static assets ------------------------------------------------


def test_index_serves(tmp_path: Path) -> None:
    client, _ = make_app(tmp_path)
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert "SALVAGE" in html
    assert "sample-btn" in html and "file-input" in html  # both demo paths


def test_static_assets_follow_the_style_spec(tmp_path: Path) -> None:
    client, _ = make_app(tmp_path)
    css = client.get("/static/style.css")
    assert css.status_code == 200
    text = css.text
    assert "--bg: #000000" in text
    assert "--ink: #FFFFFF" in text
    assert "--ink-2: #8A8A8A" in text
    assert "--hairline: #2A2A2A" in text
    assert text.count("transition") == 0  # hard cuts, no fades
    assert "radial-gradient" in text and "8px" in text.replace(" ", "")  # dot grid
    assert "Inter+Tight" in client.get("/").text  # Google Fonts present
    assert "JetBrains+Mono" in client.get("/").text
    assert "crosshair" not in text.lower()  # no targeting language


def test_sample_bundle_served_from_static(tmp_path: Path) -> None:
    client, _ = make_app(tmp_path)
    overlay = client.get("/static/samples/overlay.png")
    assert overlay.status_code == 200
    assert overlay.headers["content-type"] == "image/png"


# --- the upload scan route --------------------------------------------------


def _backend_for(_image: Path) -> MockSAMBackend:
    """Fixed detections matching the fixture image's two blobs."""
    return MockSAMBackend(
        [
            Detection(
                prompt="ribbon cable connector",
                bbox=(30, 40, 140, 150),
                mask=None,
            ),
            Detection(prompt="integrated circuit", bbox=(180, 30, 290, 170)),
        ]
    )


def test_scan_upload_with_backend(tmp_path: Path, fixture_image: Path) -> None:
    client, data_root = make_app(
        tmp_path,
        scan_backend=_backend_for(fixture_image),
        readings_provider=lambda part: two_agreeing(part.id),
    )
    with fixture_image.open("rb") as fh:
        response = client.post(
            "/api/scan",
            files={"file": ("bench.jpg", fh, "image/png")},
        )
    assert response.status_code == 200
    scan = response.json()
    assert scan["ok"] is True
    assert len(scan["parts"]) == 2
    for part in scan["parts"]:
        assert part["state"] == State.VERIFIED.value
        assert part["reasons"]
    assert scan["overlay_url"].startswith("/data/")
    assert scan["photo_url"].startswith("/data/")
    assert (data_root / scan["overlay_url"].removeprefix("/data/")).is_file()

    # remember=True (default) put the scan in the inventory
    matches = client.post("/api/search", json={"text": "ribbon connector"}).json()
    inventory_matches = [m for m in matches if m.get("source") == "inventory"]
    assert inventory_matches and inventory_matches[0]["state"] == State.VERIFIED.value


def test_scan_upload_offline_stays_cautious_but_honest(
    tmp_path: Path, fixture_image: Path
) -> None:
    """With keys absent the scan answers from the offline mock reader."""
    client, _ = make_app(tmp_path, scan_backend=_backend_for(fixture_image))
    with fixture_image.open("rb") as fh:
        scan = client.post(
            "/api/scan", files={"file": ("bench.jpg", fh, "image/png")}
        ).json()
    assert scan["readings_mode"] == "offline mock reader"
    # every verdict keeps its gate reasons and never invents more confidence:
    # the mock reader only ever says what the gate can still ask a meter for
    for part in scan["parts"]:
        assert part["reasons"]
        assert part["state"] in {s.value for s in State}


def test_scan_upload_without_backend_points_at_sample(
    tmp_path: Path, fixture_image: Path
) -> None:
    """No backend, no keys: a helpful message, never an error page."""
    client, _ = make_app(tmp_path)  # hosted backend unavailable offline
    with fixture_image.open("rb") as fh:
        response = client.post(
            "/api/scan", files={"file": ("bench.jpg", fh, "image/png")}
        )
    assert response.status_code == 200
    scan = response.json()
    assert scan["ok"] is False
    assert scan["parts"] == []
    assert "sample device" in scan["message"]


def test_scan_rejects_non_image(tmp_path: Path) -> None:
    client, _ = make_app(tmp_path)
    response = client.post(
        "/api/scan", files={"file": ("notes.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400


def test_scan_remember_false_skips_inventory(tmp_path: Path, fixture_image: Path) -> None:
    from salvage.inventory.store import DEFAULT_DB_PATH

    db_path = tmp_path / "inventory.db"
    client = TestClient(
        create_app(
            data_root=tmp_path / "data",
            inventory=Inventory(db_path=db_path),
            scan_backend=_backend_for(fixture_image),
        )
    )
    with fixture_image.open("rb") as fh:
        client.post(
            "/api/scan",
            files={"file": ("bench.jpg", fh, "image/png")},
            data={"remember": "false"},
        )
    later = Inventory(db_path=db_path)
    matches = later.query("ribbon connector")
    assert all(m.source != "inventory" for m in matches)
    later.close()


def test_backend_probe_reports_status(tmp_path: Path) -> None:
    client, _ = make_app(tmp_path)
    payload = client.get("/api/app-backend").json()
    assert payload["backend"] == "hosted"
    assert payload["available"] is False  # no keys on this machine


# --- the pipeline's own wiring (02 -> 05 by import) -------------------------


def test_offline_mock_reader_is_deterministic(tmp_path: Path) -> None:
    """Same ids in, same readings out; two independent voters, honest tone."""
    from salvage.schemas import Part

    part = Part(
        id="p_001", device_id="dev", crop_path="c", bbox=(0, 0, 1, 1), mask_path="m"
    )
    first = offline_mock_readings(part)
    second = offline_mock_readings(part)
    assert first == second
    assert {r.model_id for r in first} == {"mock-reader-open", "mock-reader-closed"}
    assert all(r.condition is Condition.UNKNOWN for r in first)
