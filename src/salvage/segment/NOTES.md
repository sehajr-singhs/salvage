# NOTES — brief 02: segmentation and crops

## What works

- `segment(image_path, device_id) -> list[Part]` in `segment.py`: detects per
  component, clamps boxes to the image, drops degenerate detections (empty
  boxes, masks on a different canvas), removes overlaps with mask IoU above
  0.8 keeping the larger mask, writes masks to
  `data/masks/<device_id>/<part_id>.png` and 10%-padded crops to
  `data/crops/<device_id>/<part_id>.png`, and returns Parts with ids
  `p_001…` sorted top-to-bottom, left-to-right (stable across reruns).
  A missing image raises `FileNotFoundError`; everything downstream is
  failure-soft per the 00-README rules (failed/absent backend -> `[]`,
  never a crash).
- `prompts.py` fixes the 12 noun phrases exactly as the brief lists them.
- `backends.py`: Backend B is a hosted SAM endpoint (`meta-api` provider,
  vision model `sam-3.1`), enabled only when `META_API_KEY` is in `.env`;
  every call is a timed HTTP POST routed through `salvage.cache` (sha256 of
  model id + payload, prompt list and image bytes both in the key), so
  re-running one device costs nothing. HTTP errors, timeouts, and bad
  payloads all return `None`, and failures are never cached.
- Backend A: `modal_app.py` deploys SAM 3.1 open weights on Modal
  (`modal deploy src/salvage/segment/modal_app.py`). Model card checked:
  repo id `facebook/sam3.1` (checkpoint-only repo, "manual" gated, Meta's
  custom SAM License — request access once on the card). Backend A's wall
  clock is guarded by a ThreadPool timeout on the calling side too.
- `overlap.py` greedy pass (largest mask first) and `rle.py` (RLE, IoU,
  rectangle fallback) — unit tested directly.
- `overlay.py` draws each mask as a 1 px white outline (4-neighbor edge) on
  a 0.45-darkened copy of the photo, no fill: the house visual style.
- `tests/test_segment.py`: 22 tests, all offline on a generated fixture
  image with a mock backend, including the brief's "done when": parts whose
  crop files exist and whose bboxes sit inside the image.

## What doesn't (yet)

- No live model call has ever been made: there are no keys on this laptop.
  The hosted endpoint's exact route (`POST {base_url}/vision/segmentation`,
  base64 image, RLE masks back) is a guess documented in `backends.py`
  and must be verified against the real API when a key exists.
- `modal_app.py` has never been deployed (no Modal run here); it is a
  static-check-only artifact (`ast.parse` + repo id asserted in tests).
  The SAM 3.1 inference API it uses (`build_sam3_image_model` +
  `Sam3Processor.set_image` / `set_text_prompt`) is copied from the sam3
  repo README and may need friends when it actually runs on a GPU.

## Stubbed / assumed

- Score normalization on the Modal side: raw scores outside [0, 1] get a
  sigmoid; the noise floor that drops detections is hardcoded at 0.05.
- Hosted endpoint response shape assumed `{"detections": [...]}` with
  Detection-shaped items; anything else validates to `None` and the
  pipeline returns `[]`.
- Masks from a backend that returns only boxes become rectangle masks.
- `data/` root defaults to `./data` (cwd-relative), matching the `cache/`
  convention; tests and callers can pass `data_root` and `cache_dir`.
