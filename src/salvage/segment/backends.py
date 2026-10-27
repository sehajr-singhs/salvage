"""Segmentation backends: hosted SAM endpoint (B), Modal SAM 3.1 (A), mock.

Every backend implements the same small protocol:

    available() -> bool                            can it run right now?
    detect(image_bytes, ...) -> list[Detection] | None   masks + boxes, or None

Rules from 00-README kept here:
- model calls go through ``salvage.cache.cached_call`` (sha256 of model id +
  payload), so re-running the same image costs nothing;
- every network call carries a timeout, and any failure returns ``None``
  instead of raising or crashing.

Cache payload format: ``b"<header json>\\x00<image bytes>"`` where the header
is ``{"prompts": [...]}``, so the sha256 cache key covers both the prompt list
and the exact image bytes. Results are plain JSON (list of Detection dicts);
on a cache hit they are re-validated into ``Detection`` objects.
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import os
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol, Sequence

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from salvage import cache
from salvage.config import get_provider
from salvage.segment.prompts import PART_PROMPTS
from salvage.segment.rle import Mask, bbox_iou, mask_iou

HTTP_TIMEOUT = httpx.Timeout(90.0, connect=10.0)
MODAL_TIMEOUT_S = 300.0


# --- Detection model -----------------------------------------------------


class Detection(BaseModel):
    """One detected instance: a prompt label, a score, a box, and an RLE mask.

    ``bbox`` is pixel coordinates ``(x1, y1, x2, y2)`` in the source image.
    ``mask`` may be ``None`` from a backend that only returns boxes; the core
    then falls back to a rectangle mask so every Part still gets a mask file.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float]
    mask: Mask | None = None


# --- cache payload helpers ------------------------------------------------


def build_payload(prompts: Sequence[str], image_bytes: bytes) -> bytes:
    """Combine the prompt list and the image into one cached payload."""
    header = json.dumps({"prompts": list(prompts)}).encode("utf-8")
    return header + b"\x00" + image_bytes


def split_payload(payload: bytes) -> tuple[list[str], bytes]:
    """Inverse of :func:`build_payload`."""
    header, _, image_bytes = payload.partition(b"\x00")
    prompts = json.loads(header.decode("utf-8"))["prompts"]
    return list(prompts), image_bytes


def _hydrate(result: Any) -> list[Detection] | None:
    """Turn a cache hit (list of JSON dicts) back into Detection objects."""
    if result is None:
        return None
    try:
        return [Detection.model_validate(item) for item in result]
    except (ValidationError, TypeError):
        return None


def cached_detection(
    backend: "SegmentBackend",
    prompts: Sequence[str],
    image_bytes: bytes,
    call: Any,
    cache_dir: str | os.PathLike[str] | None = None,
) -> list[Detection] | None:
    """Shared plumbing: ``cached_call`` + rehydration for one backend.

    ``call(payload_bytes)`` runs the real network (with its own timeout) and
    returns a JSON-safe list of Detection dicts, or ``None`` on failure.
    """
    payload = build_payload(prompts, image_bytes)
    return _hydrate(
        cache.cached_call(backend.model_id, payload, call, cache_dir=cache_dir)
    )


# --- backend protocol ------------------------------------------------------


class SegmentBackend(Protocol):
    """The protocol every segmentation backend satisfies."""

    name: str
    model_id: str

    def available(self) -> bool: ...

    def detect(
        self,
        image_bytes: bytes,
        *,
        prompts: Sequence[str] | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> list[Detection] | None: ...


# --- Backend B: hosted SAM endpoint ---------------------------------------


class HostedSAMBackend:
    """Backend B: a hosted SAM endpoint, enabled only when a key is configured.

    Uses the ``meta-api`` provider from ``src/salvage/config.py``, whose
    ``vision`` model is ``sam-3.1``. It is available exactly when
    ``META_API_KEY`` is set in the environment (.env), and every call is a
    timed HTTP POST wrapped in the sha256 cache.

    Endpoint guess: ``POST {base_url}/vision/segmentation`` with a JSON body
    ``{"model", "text_prompts", "image" (base64), "return_masks": "rle"}`` and
    a response ``{"detections": [Detection...]}``. There is no live key on
    this laptop today, so the exact route is unverified until a key exists;
    any HTTP error, timeout, or payload that fails validation returns None.
    """

    name = "hosted"

    def __init__(self, provider_name: str = "meta-api") -> None:
        self._provider = get_provider(provider_name)
        self.model_id = self._provider.models["vision"]

    def available(self) -> bool:
        return self._provider.is_available()

    def detect(
        self,
        image_bytes: bytes,
        *,
        prompts: Sequence[str] | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> list[Detection] | None:
        use_prompts = list(prompts) if prompts is not None else list(PART_PROMPTS)
        if not self.available():
            return None
        provider = self._provider
        text_prompts = use_prompts

        def call(payload: bytes) -> list[dict[str, Any]] | None:
            _, image = split_payload(payload)
            url = f"{provider.base_url.rstrip('/')}/vision/segmentation"
            body = {
                "model": self.model_id,
                "text_prompts": text_prompts,
                "image": base64.b64encode(image).decode("ascii"),
                "return_masks": "rle",
            }
            try:
                response = httpx.post(
                    url,
                    json=body,
                    headers={"Authorization": f"Bearer {provider.api_key()}"},
                    timeout=HTTP_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
            except Exception:  # noqa: BLE001 - failures return None, never crash
                return None
            items = (data or {}).get("detections") if isinstance(data, dict) else None
            if not isinstance(items, list):
                return None
            try:
                return [
                    Detection.model_validate(item).model_dump(mode="json")
                    for item in items
                ]
            except ValidationError:
                return None

        return cached_detection(self, use_prompts, image_bytes, call, cache_dir=cache_dir)


# --- Backend A: SAM 3.1 open weights on Modal ------------------------------


class ModalSAMBackend:
    """Backend A: SAM 3.1 open weights (``facebook/sam3.1``) on a Modal GPU.

    The weights are public but gated and licensed under Meta's SAM License
    (see ``modal_app.py``); access is requested on the HF model card and the
    token reaches the Modal box through a Modal secret. Called through the
    app in ``salvage/segment/modal_app.py`` (register once with
    ``modal deploy``), wrapped in the same sha256 cache, with a wall-clock
    timeout enforced on this side so a hung remote call cannot block forever.
    """

    name = "modal"
    model_id = "facebook/sam3.1"

    def available(self) -> bool:
        try:
            import modal  # noqa: F401  (check only; the call imports it lazily)
        except ImportError:
            return False
        return True

    def detect(
        self,
        image_bytes: bytes,
        *,
        prompts: Sequence[str] | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> list[Detection] | None:
        use_prompts = list(prompts) if prompts is not None else list(PART_PROMPTS)

        def call(payload: bytes) -> list[dict[str, Any]] | None:
            raw = self._remote(payload)
            if raw is None:
                return None
            try:
                data = json.loads(raw)
                items = data.get("detections")
                if not isinstance(items, list):
                    return None
                return [
                    Detection.model_validate(item).model_dump(mode="json")
                    for item in items
                ]
            except (ValidationError, TypeError, ValueError):
                return None

        return cached_detection(self, use_prompts, image_bytes, call, cache_dir=cache_dir)

    def _remote(self, payload: bytes) -> bytes | None:
        try:
            from salvage.segment.modal_app import Sam3Detector  # imports modal lazily

            runner = Sam3Detector()
        except Exception:  # noqa: BLE001 - missing modal or token: return None
            return None
        pool = ThreadPoolExecutor(max_workers=1)
        future: Future[bytes] | None = None
        try:
            future = pool.submit(runner.detect.remote, payload)
            return future.result(timeout=MODAL_TIMEOUT_S)
        except Exception:  # noqa: BLE001 - timeout / remote failure: return None
            return None
        finally:
            if future is not None:
                future.cancel()
            pool.shutdown(wait=False)


# --- Mock backend (tests) ---------------------------------------------------


class MockSAMBackend:
    """Deterministic stand-in for tests and offline demos.

    Replays a fixed list of Detection objects passed at construction time,
    without any network or GPU. Not part of the auto-selected backends in
    :mod:`salvage.segment.segment`; tests construct it explicitly.
    """

    name = "mock"
    model_id = "mock-sam"

    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections

    def available(self) -> bool:
        return True

    def detect(
        self,
        image_bytes: bytes,
        *,
        prompts: Sequence[str] | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> list[Detection] | None:
        return list(self._detections)
