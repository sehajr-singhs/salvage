# NOTES.md — 08: wire it together
*Lives in `measure/` (the folder this brief owns) and covers the whole 08
build: `pipeline.py`, `measure/`, `__main__.py`, `tests/test_pipeline.py`.*

## What works

- **The one chain — `salvage/pipeline.py:run_device(image_path, device_id)`** —
  wires 02 → 03 → 04 → 05 by import only, no reimplementation:
  `segment.segment` + `segment.overlay` (02, parts + crops + masks + the
  white-outline teardown view), one readings provider per part (03), the
  gate `decide` (04), and `Inventory.add` (05). It returns a JSON-ready
  `DeviceScan` (pydantic: parts, readings, verdicts, artifact paths, the
  display name the device was remembered under, and the device-wide
  measurement note the run graded with). A missing image is the one raise
  (`FileNotFoundError`, 02's contract); everything else is failure-soft —
  no segmentation backend available (no keys, no Modal app) is an
  `ok=False` scan pointing at the sample device, never a crash.
- **Reader selection:** the real `salvage.readers.read_part` (03) is the
  default the moment a provider key exists; with none, the package raises
  ImportError (03's keyless guard) and the pipeline falls back to the
  shared offline mock reader (`web.pipeline.offline_mock_readings`), so one
  mock implementation serves the CLI, the web demo and this pipeline and
  every run is keyless and deterministic. An injected
  `readings_provider=` overrides both — the seam tests use — and
  `DeviceScan.readings_mode` reports which happened ("real readers (03)",
  "offline mock reader", or "injected readings provider").
- **The measurement dialogue — `measure/dialogue.py`** (the D3 beat):
  * `parse_measurement(text)` parses one free-text meter reply into a unit-
    typed value ("8.2 ohms", "1.8k", "3.7 V", "spins" / "didn't spin",
    "beeps", "open circuit"), defaulting a bare number to ohms — the most
    common bench check — and returns `None` for gibberish. The phrase it
    hands back is exactly what the gate's own parsers accept, so dialogue
    and gate can never disagree.
  * `asks(verdicts, readings)`: every NEEDS_MEASUREMENT verdict asked with
    the gate's own instruction, an ABSTAIN asked only when the readers'
    consensus type still has a cheap check in `gate.checks`, and HAZARD
    never — no measurement makes a swollen cell safe.
  * `measure()` runs the ask loop (ask → parse → one retry → polite skip;
    skipped parts never change verdicts), `ConsolePrompter` takes terminal
    text (EOF/KeyboardInterrupt = skip, never crash, per 00-README), and
    `ScriptPrompter` scripts replies for tests and offline demos. Voice is
    later: any Muse Voice Transcribe client plugs in behind the two-method
    `Prompter` protocol without another change.
  * `state_transitions(before, after)` lists every part whose state moved,
    as `(part_id, before, after)`.
- **`apply_measurements(scan, answers)` / `run_dialogue(scan, prompter)`**
  re-run `gate.decide` per part (with a per-part answer, the original
  device-wide note, or nothing — an unanswered part is never silently more
  confident) and re-add the device to the inventory, so 05's upsert shows
  the new states. Carried over unchanged: parts, readings, artifact paths,
  messages, and the `readings_mode` label.
- **The CLI — `python -m salvage`:**
  * `scan photo.jpg --device drone1 [--measure]` runs `run_device`, prints
    one verdict line per part (state, label, reasons, the meter ask) and,
    with `--measure`, runs the dialogue and prints each state change as
    `BEFORE -> AFTER`. Exit 1 only when nothing could be scanned.
  * `need "12 V motor with driver"` is 05's reverse search: which device
    to open, which part to pull, VERIFIED first, HAZARD/ABSTAIN never, and
    the common-sources fallback when nothing is remembered yet.
  * `prompter` / `scan_backend` / `readings_provider` are injection points
    so tests (and offline demos) can run the full CLI without keys.
- **`tests/test_pipeline.py`** — 9 offline tests, the brief's done-when
  verbatim: photo → segment → read → gate → inventory on a fixture with
  mocked models, **including one ABSTAIN turning into VERIFIED after a
  simulated measurement** (the motor that abstained on an "open circuit"
  contradiction verifies once the user reads 8.2 ohms), plus the fan spin
  test, the unresolvable reader-contradiction ABSTAIN, a HAZARD part that
  is never asked, retry/skip behaviour, artifact files on disk, inventory
  upsert and search, both CLI commands end-to-end, and JSON round-trip of
  the scan. Whole suite: `pytest -q` → **232 passed**, all offline.

## What doesn't work / known limits

- No keys on this machine: the real segment/read call paths are exercised
  only through the mocks their briefs shipped (02/03 notes carry the same
  caveat). The chain itself is 08's real code — only the network is faked.
- The CLI's `scan` cannot take a measurement **before** the dialogue (no
  `--measurement` flag), so the initial scan always grades with no note;
  the device-wide pre-probe path is reachable through `run_device(...,
  measurement=...)` and tested there.
- Inventory reverse search sees only what 05's filter vocabulary accepts
  (`parse_filters`): "a motor for a 12 V pump" sets a voltage filter that
  a KV-only reading cannot match, so the demo query is "a motor". Fixing
  that belongs to 05's parser, not the wiring.
- A consensus-type tie inside `asks()` (ABSTAIN, no check record, two
  types at one vote each) resolves by reader order — deterministic for a
  given scan, but arbitrary in principle.

## Fixed while wiring (interface mismatches, no schema changes)

- `__main__.py` gained the `readings_provider` injection point (`main` →
  `cmd_scan` → `run_device`); before, the CLI could not run a scripted
  reader offline.
- `run_device` now labels an injected provider honestly in
  `readings_mode`, and `apply_measurements` carries that label through —
  previously the second (post-dialogue) scan reset to the keyless default
  label.
- Leftover cross-brief cleanups 06's NOTES flagged for 08:
  `pyproject.toml` now lists `fastapi`, `python-multipart`, `uvicorn`
  (installed in the venv all along), and `.gitignore` covers the runtime
  `data/` artifacts (`scan/`, `crops/`, `masks/`, `uploads/`,
  `inventory.db`) while `data/labels/` stays tracked.

## What's stubbed

- Voice input: the `Prompter` seam exists, no transcribe client — by design
  ("text input first, voice later").
- The foreman (Muse Wonder planning the fan-out instead of the hard-coded
  chain) is the roadmap's step beyond this brief; the chain here is the
  deterministic wiring whose order a foreman would one day emit.
- Nothing else: every brief's interface (02–07) is consumed as shipped,
  and schemas from 01 were not changed anywhere.
