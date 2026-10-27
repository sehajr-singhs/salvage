"""Backend A: SAM 3.1 open weights served on Modal.

The laptop has no GPU, so the open-weight checkpoint runs on Modal:

    modal deploy src/salvage/segment/modal_app.py

The app loads ``facebook/sam3.1`` (imported and run through Meta's sam3 code
at https://github.com/facebookresearch/sam3). The model card
(https://huggingface.co/facebook/sam3.1) hosts the checkpoints only; they are
gated ("manual" access) under Meta's custom SAM License ("license: other",
LICENSE file on the card), so access must be requested on the card once and
the Hugging Face token must reach the Modal box. Deploy with:

    modal secret create huggingface HF_TOKEN=hf_...

and reference that secret below; the token is never written into this repo.
No real key lives anywhere here.

Wire format (mirrors ``salvage.segment.backends.build_payload`` so it can be
swapped straight into ``ModalSAMBackend``):

input:  ``b"<header json>\\x00<image bytes>"`` with header ``{"prompts": [...]}``
output: ``b"{\\"detections\\": [...]}\""``

Each detection is ``{"prompt", "score", "bbox" [x1, y1, x2, y2], "mask":
{"width", "height", "runs"}}`` where the mask is row-major run-length pairs
``[start, length, ...]`` over the full image canvas, matching
``salvage.segment.rle``. Products of raw model tensors are JSON-safe numbers
only, so the sha256 cache in ``salvage.cache`` can store the response.
"""

from __future__ import annotations

import io
import json

import modal

MODEL_ID = "facebook/sam3.1"  # exact HF repo id, gated, Meta's SAM License
GITHUB_REPO = "https://github.com/facebookresearch/sam3"

app = modal.App("salvage-sam31")

sam_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    # CUDA 12.8 wheels, per the sam3 installation instructions (torch >= 2.7).
    .pip_install(
        "torch==2.10.0",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("pillow", "numpy")
    .run_commands(
        f"git clone {GITHUB_REPO} /srv/sam3", "pip install -e /srv/sam3"
    )
)


def _clamp_score(value: float) -> float:
    """Squash a raw detection score into [0, 1] (sigmoid outside the range)."""
    value = float(value)
    if value < 0.0 or value > 1.0:
        # typical logits: sigmoid keeps ordering and lands in [0, 1]
        return 1.0 / (1.0 + pow(2.0, -value))
    return value


def _rle_runs(bool_rows) -> list[int]:
    """Row-major run-length pairs [start, length, ...] for a 2D boolean array."""
    flat = bool_rows.ravel().tolist()
    runs: list[int] = []
    start: int | None = None
    for index, pixel in enumerate(flat):
        if pixel and start is None:
            start = index
        elif not pixel and start is not None:
            runs.extend((start, index - start))
            start = None
    if start is not None:
        runs.extend((start, len(flat) - start))
    return runs


@app.cls(
    image=sam_image,
    gpu="L40S",  # 848M params + SAM2-style decoder fits well below 48 GB
    secrets=[
        modal.Secret.from_name(
            "huggingface"
        )  # create: modal secret create huggingface HF_TOKEN=hf_...
    ],
    timeout=300,
    scaledown_window=300,
)
class Sam3Detector:
    """Text-prompted concept segmentation: image + noun phrases -> masks."""

    @modal.enter()
    def load(self) -> None:
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        torch.manual_seed(0)
        torch.set_grad_enabled(False)
        self._processor = Sam3Processor(build_sam3_image_model())

    @modal.method()
    def detect(self, payload: bytes) -> bytes:
        import numpy as np
        from PIL import Image
        from sam3.model_builder import build_sam3_image_model  # noqa: F401

        header, _, image_bytes = payload.partition(b"\x00")
        prompts = json.loads(header.decode("utf-8"))["prompts"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size

        state = self._processor.set_image(image)
        detections: list[dict[str, object]] = []
        for prompt in prompts:
            output = self._processor.set_text_prompt(state=state, prompt=prompt)
            masks, boxes, scores = (
                output["masks"],
                output["boxes"],
                output["scores"],
            )
            for index in range(len(scores)):
                score = _clamp_score(scores[index].item())
                if score < 0.05:  # noise floor: below this the mask is junk
                    continue
                x1, y1, x2, y2 = (
                    float(v) for v in boxes[index].reshape(-1).tolist()
                )
                x1, x2 = min(max(x1, 0.0), float(width)), min(
                    max(x2, 0.0), float(width)
                )
                y1, y2 = min(max(y1, 0.0), float(height)), min(
                    max(y2, 0.0), float(height)
                )
                if x2 <= x1 or y2 <= y1:
                    continue  # degenerate box: nothing to crop
                mask = np.asarray(masks[index], dtype=bool)
                mask = mask.reshape(mask.shape[-2], mask.shape[-1])
                if mask.shape != (height, width):
                    continue  # off-canvas mask: cannot be trusted
                detections.append(
                    {
                        "prompt": prompt,
                        "score": score,
                        "bbox": [x1, y1, x2, y2],
                        "mask": {
                            "width": width,
                            "height": height,
                            "runs": _rle_runs(mask),
                        },
                    }
                )
        return json.dumps({"detections": detections}).encode("utf-8")
