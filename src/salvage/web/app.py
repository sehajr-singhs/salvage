"""Brief 06: the thin FastAPI layer over sessions 02 to 05.

Nothing here re-implements pipeline logic: scanning imports
``salvage.web.pipeline.scan_device`` (which imports 02, 03-by-mock, 04 and
05), and reverse search imports ``salvage.inventory.Inventory.query``.

The demo has to work with no keys and no camera: ``GET /api/sample`` serves
a precomputed scan straight out of ``web/static/samples/`` (committed cache
artifacts), which also seeds the app's inventory when it is empty so the
reverse-search box answers offline too.

Rules from 00-README kept here: every network call carries a timeout
inside the modules this app imports; failures return response JSON saying
so, not an error page; and no key is ever read here directly — providers
come from ``salvage.config``.
"""

from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from salvage.config import available_providers
from salvage.inventory.models import Device, Match
from salvage.inventory.store import Inventory
from salvage.schemas import Part, Reading, Verdict
from salvage.segment.backends import HostedSAMBackend
from salvage.web.pipeline import scan_device

DEFAULT_STATIC_DIR = Path("web/static")
DEFAULT_DATA_ROOT = Path("data")
SAMPLE_JSON = "sample.json"


class SearchRequest(BaseModel):
    """The body of ``POST /api/search``."""

    text: str


def sample_dir(static_dir: Path) -> Path:
    """Where the cached sample-device bundle lives inside ``static_dir``."""
    return Path(static_dir) / "samples"


def _slugs(text: str) -> str:
    """A filesystem-safe guess for a device id from an upload filename."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", Path(text).stem).strip("_").lower()
    return slug or "upload"


def load_sample(static_dir: Path) -> dict[str, Any] | None:
    """The parsed sample bundle, or None when nothing is cached on disk."""
    path = sample_dir(static_dir) / SAMPLE_JSON
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def seed_from_sample(inventory: Inventory, bundle: dict[str, Any]) -> int:
    """Seed the inventory from the sample bundle; count devices added.

    The bundle stores pydantic JSON of Device / Part / Verdict / Reading in
    ``inventory_seed`` exactly as ``Inventory.add`` expects them, so the
    reverse-search box works offline from cache alone. Seeding is idempotent:
    ``Inventory.add`` uses INSERT OR REPLACE under a fixed device id, and it
    only ever writes the sample device's own rows.
    """
    seed = bundle.get("inventory_seed")
    if not isinstance(seed, dict):
        return 0
    inventory.add(
        Device.model_validate(seed["device"]),
        [Part.model_validate(item) for item in seed.get("parts", [])],
        [Verdict.model_validate(item) for item in seed.get("verdicts", [])],
        readings=[Reading.model_validate(item) for item in seed.get("readings", [])],
    )
    return 1


def create_app(
    *,
    static_dir: Path | str | None = None,
    data_root: Path | str | None = None,
    cache_dir: Path | str | None = None,
    inventory: Inventory | None = None,
    scan_backend: Any | None = None,
    readings_provider: Any | None = None,
    seed_sample: bool = True,
) -> FastAPI:
    """Build the Salvage demo app; every dependency is injectable for tests.

    ``static_dir`` defaults to ``web/static`` under the current directory,
    matching the ``data``/``cache`` cwd-relative conventions other sessions
    use. ``data_root`` is served under ``/data`` so uploaded overlays and
    crops have URLs. ``scan_backend`` (``salvage.segment.backends`` protocol)
    and ``readings_provider`` exist so tests and downstream sessions can run
    the whole app with mocked models; with nothing injected the scan uses
    the hosted SAM endpoint only (available when ``META_API_KEY`` is set),
    never a surprise Modal call mid-demo.
    """
    static_dir_resolved = Path(static_dir).resolve() if static_dir else (
        Path.cwd() / DEFAULT_STATIC_DIR
    ).resolve()
    data_root_resolved = Path(data_root).resolve() if data_root else (
        Path.cwd() / DEFAULT_DATA_ROOT
    ).resolve()
    data_root_resolved.mkdir(parents=True, exist_ok=True)

    inventory = inventory if inventory is not None else Inventory()
    backend = scan_backend if scan_backend is not None else HostedSAMBackend()

    app = FastAPI(title="Salvage", version="0.1.0")
    app.state.inventory = inventory
    app.state.data_root = data_root_resolved
    app.state.static_dir = static_dir_resolved

    app.mount("/static", StaticFiles(directory=str(static_dir_resolved)), name="static")
    app.mount("/data", StaticFiles(directory=str(data_root_resolved)), name="data")

    sample = load_sample(static_dir_resolved)
    if seed_sample and sample is not None:
        seed_from_sample(inventory, sample)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir_resolved / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Providers with a key right now, plus whether a sample exists."""
        return {
            "providers": available_providers(),
            "sample_available": sample is not None,
        }

    @app.get("/api/sample")
    def sample_device() -> dict[str, Any]:
        """The cached sample device: the demo with no keys and no camera."""
        if sample is None:
            bundle_dir = sample_dir(static_dir_resolved) / SAMPLE_JSON
            raise HTTPException(
                status_code=503,
                detail=f"no sample bundle cached at {bundle_dir}",
            )
        return sample

    @app.post("/api/scan")
    async def scan_upload(
        file: UploadFile = File(...),
        device_id: str | None = Form(None),
        measurement: str | None = Form(None),
        remember: str = Form("true"),
    ) -> dict[str, Any]:
        """Scan one uploaded photo: segment, read, gate, remember."""
        image_path = data_root_resolved / "uploads" / (
            f"{device_id or _slugs(file.filename or 'upload')}.png"
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="empty upload")
        image_path.write_bytes(payload)

        device_id_resolved = (device_id or _slugs(file.filename or "upload")).strip()
        try:
            Image.open(io.BytesIO(payload)).verify()
        except Exception:  # noqa: BLE001 - a non-image upload is a client error
            image_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="upload is not an image")

        return scan_device(
            image_path,
            device_id_resolved,
            backend=backend,
            data_root=data_root_resolved,
            cache_dir=cache_dir or os.environ.get("SALVAGE_CACHE_DIR"),
            readings_provider=readings_provider,
            measurement=measurement,
            inventory=inventory,
            device_name=_slugs(file.filename or "upload"),
            remember=remember.lower() != "false",
        )

    @app.get("/api/app-backend")
    def which_backend() -> dict[str, Any]:
        """Which segmentation backend the scan route would use right now."""
        return {"backend": backend.name, "available": backend.available()}

    @app.post("/api/search")
    async def search(request: SearchRequest) -> list[dict[str, Any]]:
        """Reverse search: 'I need a 12 V motor with a driver'."""
        matches: list[Match] = inventory.query(
            request.text,
            cache_dir=cache_dir or os.environ.get("SALVAGE_CACHE_DIR"),
        )
        return [match.model_dump(mode="json") for match in matches]

    return app


_lazy: dict[str, FastAPI] = {}


def get_app() -> FastAPI:
    """The default app over the cwd-relative paths, built at most once.

    Built lazily so importing this module never writes ``data/inventory.db``
    on disk as a side effect; uvicorn runs it with
    ``uvicorn salvage.web.app:get_app`` (a factory) or the ``app`` attribute.
    """
    if "app" not in _lazy:
        _lazy["app"] = create_app()
    return _lazy["app"]


def __getattr__(name: str):  # PEP 562 module attribute
    if name == "app":
        return get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
