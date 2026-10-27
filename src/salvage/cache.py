"""Model call cache.

Every model call in Salvage goes through this module. Responses are cached by
sha256 of the model id and the exact input payload, so re-running a device
check costs nothing, offline or online.

Files live under ``cache/<sha256>.json`` (overridable with the
``SALVAGE_CACHE_DIR`` env var, or the ``cache_dir`` keyword, so tests and the
sample-device UI can point at their own store).

Rules from 00-README, kept here:
- Rerunning must cost nothing.
- A failure returns ``None``, never a crash.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

DEFAULT_CACHE_DIR = "cache"


def cache_root() -> Path:
    """The directory cached responses live in (created lazily on use)."""
    return Path(os.environ.get("SALVAGE_CACHE_DIR", DEFAULT_CACHE_DIR))


def payload_digest(model_id: str, payload_bytes: bytes) -> str:
    """sha256 of the model id and the exact input payload."""
    hasher = hashlib.sha256()
    hasher.update(model_id.encode("utf-8"))
    hasher.update(b"\x00")  # separate model id from payload unambiguously
    hasher.update(payload_bytes)
    return hasher.hexdigest()


def _jsonable(result: Any) -> Any:
    """Coerce a call result into something json.dumps can serialize."""
    if result is None or isinstance(result, (bool, int, float, str)):
        return result
    if isinstance(result, (list, tuple)):
        return [_jsonable(item) for item in result]
    if isinstance(result, dict):
        return {str(key): _jsonable(value) for key, value in result.items()}
    dump = getattr(result, "model_dump", None)
    if callable(dump):
        return _jsonable(dump(mode="json"))
    return str(result)


def _load(path: Path) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _store(path: Path, result: Any) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(_jsonable(result), fh, indent=2, sort_keys=True)
        tmp.replace(path)  # atomic-enough: readers never see half a file
        return True
    except OSError:
        return False


def cached_call(
    model_id: str,
    payload_bytes: bytes,
    fn: Callable[[bytes], Any],
    *,
    cache_dir: str | os.PathLike[str] | None = None,
) -> Any | None:
    """Call ``fn(payload_bytes)`` for ``model_id``, served from cache when possible.

    The cache key is the sha256 of the model id and the payload, so the same
    question asked of the same model is answered from disk for free.

    ``fn`` should make the real network call with a timeout. If ``fn`` raises,
    times out, or returns ``None``, ``cached_call`` returns ``None`` and caches
    nothing, per the README rule that failures never crash and never get
    cached. A corrupt or unreadable cache entry is treated as a miss.
    """
    root = Path(cache_dir) if cache_dir is not None else cache_root()
    path = root / f"{payload_digest(model_id, payload_bytes)}.json"

    cached = _load(path)
    if cached is not None:
        return cached

    try:
        result = fn(payload_bytes)
    except Exception:  # noqa: BLE001 - a failed model call returns None, never crashes
        return None

    if result is None:
        return None  # a failed call is not a valid answer; don't cache it

    _store(path, result)
    return result
