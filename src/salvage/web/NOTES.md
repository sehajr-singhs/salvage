# NOTES — brief 06: black-and-white web UI

## What works

- `src/salvage/web/app.py`: thin FastAPI layer. Routes: `GET /` (the page),
  `GET /api/sample` (cached sample device), `POST /api/scan` (multipart photo
  → segment → read → gate → inventory), `POST /api/search` (reverse search),
  `GET /api/health`, `GET /api/app-backend`. Static assets are served from
  `web/static/` at `/static`; scan artifacts (overlays, crops, masks, uploads)
  from the data root at `/data`. Everything injectable by `create_app()`:
  static dir, data root, cache dir, `Inventory`, a `salvage.segment`-protocol
  backend, and a readings provider — which is how tests and session 08 can run
  the whole app with mocked models.
- `src/salvage/web/pipeline.py`: `scan_device(...)` wires sessions 02→05 by
  import only — `salvage.segment.segment.segment`, `salvage.segment.overlay`,
  `salvage.gate.decide`, `salvage.inventory.Inventory.add`. No reimplementation,
  no schema copies. An explicitly passed backend is respected as-is (available
  or not), so a demo can never stall on an auto-selected remote; the app injects
  the hosted SAM endpoint (usable the moment `META_API_KEY` exists), never a
  surprise Modal call. Failure-soft per 00-README: no backend/no parts returns
  an ok:false message pointing at the sample, not an error page.
- Offline reader fallback: `salvage.readers` (03) is still a stub, so while it
  is, a deterministic sha256-keyed mock reader produces two honest "unknown
  condition" votes per part. The gate therefore stays cautious (mostly
  NEEDS_MEASUREMENT with the real check instructions). `resolve_readings()`
  prefers the real `salvage.readers.read_part` the moment session 03 ships —
  no code change needed in the web layer.
- `src/salvage/web/samples_build.py`: regenerates the committed sample bundle
  (`web/static/samples/`: photo.png, overlay.png, sample.json) with
  `python -m salvage.web.samples_build`. The bundle is produced by the real
  pipeline with hand-drawn detections and the curated swarm readings the gate
  actually judges: BLDC 2205 VERIFIED (bench measurement "8.2 ohms" already
  taken — the D3 beat), FAN 4010 NEEDS_MEASUREMENT (spin test), CAP 470uF 25V
  ABSTAIN (readers disagree), HEATSINK VERIFIED, swollen LIPO HAZARD. The
  sample also seeds the app's inventory on startup (idempotent, sample-device
  rows only), so the reverse-search box answers offline too.
- `web/static/`: `index.html` + `style.css` + `app.js` per the strict spec:
  bg #000, text #FFF, secondary #8A8A8A, hairlines #2A2A2A; one muted amber
  for HAZARD only; Inter Tight headings (uppercase, tracked) and JetBrains
  Mono from Google Fonts; faint 8 px dot grid; masks as 1 px white outlines
  (rendered by 02's overlay) with SVG leader lines and callouts like
  `M1 · BLDC 2205 · VERIFIED`; verdict chips per part with reasons and the
  plain-English meter ask; reverse-search box listing device + part to pull;
  zero CSS transitions (hard cuts); phone-width layout at ≤640 px where the
  callouts yield to the part list. No crosshairs, no "target" language, no
  logos.
- `tests/test_web.py`: 15 offline tests — the brief's done-when (TestClient
  passes; the sample renders with no keys and no backend) plus style-spec
  assertions (palette vars in the CSS, zero `transition`s, Google Fonts
  linked, no "crosshair"), a full mocked-backend scan writing real overlay /
  mask / crop files, the no-backend path, non-image rejection, remember=false,
  seed-integrity of the search ("12 V motor with a driver" → the drone's
  VERIFIED motor), and HAZARD never surfacing in search.
- Suite: `python -m pytest` → 167 passed (152 pre-existing + 15 new), all
  offline, nothing written into the repo's `data/` or `cache/`.

## What doesn't (yet)

- No live model call behind the UI: no keys exist. `/api/scan` without keys
  uses the mock reader; the hosted SAM endpoint path is unverified until a
  key exists (same caveat as session 02's backends).
- Fonts load from Google Fonts at runtime; offline the page falls back to
  system grotesk/monospace. Nothing lines up worse for it — layout is width-
  driven, not font-driven.
- Voice input and the live measurement dialogue are session 08's (`measure/`
  + `salvage/pipeline.py`); the UI currently reflects the verdicts it is
  served, and the meter ask appears verbatim as text.

## Stubbed / assumed

- `fastapi`, `python-multipart`, `pillow`, `httpx` are required by this brief
  but `pyproject.toml` is not in 06's owned folders, so they were installed
  into the environment and left out of the manifest on purpose. Session 08
  should add `fastapi>=0.110`, `python-multipart`, (and `uvicorn` to serve)
  to `[project.dependencies]` when it wires the full command.
- `data/` (inventory.db, uploads, scan artifacts) is created at runtime next
  to `cache/`; consider adding `data/` to `.gitignore` in session 08 — only
  `data/raw/` is ignored today (that edit touches a shared file, so it was
  not made here).
- The endpoints consume module version 1 surfaces of 02–05 only. If session
  08 renames anything, there is exactly one import site per session in
  `pipeline.py`/`app.py` to patch.
