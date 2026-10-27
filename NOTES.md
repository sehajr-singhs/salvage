# NOTES.md — 01: repo skeleton and shared schemas

## What works

- `pydantic` schemas in `src/salvage/schemas.py` exactly per the 00-README contract:
  `Part` (id, device_id, crop_path, bbox, mask_path), `Reading` (part_id, model_id,
  part_type, part_number, key_rating, condition, confidence 0..1, evidence),
  `Verdict` (part_id, state, label, reasons, requested_measurement), plus a
  `State` enum (VERIFIED / NEEDS_MEASUREMENT / ABSTAIN / HAZARD) and a
  `Condition` enum (works / dead / unknown). JSON round-trip is tested for all
  three models and all four states.
- `src/salvage/config.py`: loads `.env` with `python-dotenv` and exposes
  `PROVIDERS`, one entry per provider, each with `name`, OpenAI-compatible
  `base_url`, `api_key_env`, `model_id`, `enabled`. The meta API provider also
  carries a role→model map (vision=Sam 3.1, foreman=Muse Wonder, reader=Muse
  Spark 1.3, ears=Muse Voice Transcribe). `available_providers()` returns only
  providers that are enabled and have a key in the environment.
- `src/salvage/cache.py`: `cached_call(model_id, payload_bytes, fn)`. Stores JSON
  under `cache/<sha256(model_id + payload)>.json`, returns the cached copy when
  it exists, writes atomically, and never caches or crashes on failure. A
  failed call raises→returns `None` and leaves the cache untouched; a corrupt
  cache file is treated as a miss and refetched. Cache root is overridable via
  `SALVAGE_CACHE_DIR` or the `cache_dir` keyword (used by every test; no
  network, no keys needed).
- Empty subpackages with `__init__.py`: `segment/`, `readers/`, `gate/`,
  `inventory/`, `measure/`, `eval/`, `web/` — stubs for sessions 02–07.
- `.env.example` lists all 8 provider key names with empty values;
  `.gitignore` covers `.env`, `cache/`, `data/raw/`, `__pycache__/`, `.venv/`,
  `.freebuff/`, plus `TASK.md`/`RESULT.md` session scratch files.
- Tests: 28 passed (`python -m pytest`), all offline.

## What doesn't work / known limits

- Real HTTP calls don't exist yet: no keys do, and no session has built a
  caller. 02–07 must put timeouts in their `fn` and return `None` on failure.
- `cached_call` returns the *stored JSON copy* on a cache hit (a dict) but the
  caller's original object on a fresh call. Callers should re-hydrate pydantic
  results (`Reading.model_validate(hit)`) after a hit if they need a model.
  Bytes payloads are fine: payloads may be any bytes (e.g. image bytes); it's
  the *result* that must be JSON-serializable. `cache._jsonable` falls
  back to `str()` for anything exotic rather than crashing.
- A globally installed `typeguard 4.1.2` fails to import on this Python 3.14
  (`ast.Str` removed) and breaks pytest auto-loading; disabled via
  `addopts = "-p no:typeguard"` in `pyproject.toml`. It is not a Salvage dep.

## What's stubbed

- Nothing in the 01 scope. All subpackages are empty placeholders by design,
  to be filled by their owning briefs.
