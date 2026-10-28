# Salvage

Salvage is a teardown agent for dead electronics. It finds the parts inside a broken device, works out which ones still work, and refuses to call a part good until the evidence says so. When it isn't sure, it asks for one cheap measurement instead of guessing.

Site: https://sehajr-singhs.github.io/salvage/

## How it works

1. **Segment.** SAM 3.1 outlines every part in a photo and crops each one.
2. **Read.** Each crop goes to several models in parallel (Muse Spark, Muse Spark Open, Muse Glimmer, and a non-Meta model). Each one returns the part type, markings, rating, condition and the evidence it actually sees.
3. **Gate.** Plain code, no model, decides one of four states in strict order: `HAZARD`, `ABSTAIN`, `NEEDS_MEASUREMENT`, `VERIFIED`.
4. **Measure.** For parts a meter can settle (motor windings, cell voltage, fan spin, fuse continuity, capacitor ESR), it asks for the reading and the expected range, then re-runs the gate.
5. **Inventory.** Verified parts are stored and searchable ("12 V motor with driver"). Hazards and unknowns are never returned.

## Run it

```sh
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install -e .
python -m pytest -q              # 232 tests, fully offline
python -m uvicorn salvage.web.app:get_app --factory   # open http://127.0.0.1:8000 and press "Sample device"
python -m salvage scan photo.jpg --device drone1 --measure
python -m salvage need "12 V motor with driver"
```

Copy `.env.example` to `.env` and add provider keys to use live models. Without keys, every model call falls back to mocked or cached answers, so everything above still runs.

## Layout

| Folder | What's in it |
|---|---|
| `src/salvage/segment` | SAM backends (hosted API, open weights on Modal, mock), crops, overlays |
| `src/salvage/readers` | parallel multi-model readers, shared prompt, agreement signal |
| `src/salvage/gate` | the deterministic gate: hazard rules, cheap checks, consistency checks |
| `src/salvage/measure` | the measurement dialogue |
| `src/salvage/inventory` | SQLite inventory and reverse search |
| `src/salvage/web` | FastAPI app and the black-and-white front end (`web/static`) |
| `src/salvage/eval` | coverage/error curves, disagreement AUROC, measurement resolution rate |

## Status

Built end to end and tested offline. Live model runs haven't happened yet: no API keys are connected, and the model IDs in `src/salvage/config.py` still need to be checked against the providers' live model lists.
