"""Stage 2: generate one Stable Diffusion image per scene."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

from . import config

log = logging.getLogger(__name__)

_STYLE_SUFFIX = ", clean modern digital illustration, high quality, vivid colors, sharp focus"
_NEGATIVE = "text, watermark, logo, signature, blurry, low quality, deformed, ugly"


class ImageGenerator:
    """Wraps a diffusers text-to-image pipeline; auto-picks CUDA or CPU."""

    def __init__(self, backend_key: str):
        import torch
        from diffusers import AutoPipelineForText2Image

        label, model_id, self.steps, self.guidance = config.IMAGE_MODELS[backend_key]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("loading image model %s on %s", model_id, self.device)

        # bfloat16 on CPU: halves RAM versus fp32 on this memory-constrained machine
        dtype = torch.float16 if self.device == "cuda" else torch.bfloat16
        # fp16 variant weights: loading fp32 then casting needs a ~5 GB transient
        # peak that exceeds this machine's commit limit. safetensors only, since
        # .bin checkpoints get read fully into RAM.
        kwargs = dict(torch_dtype=dtype, safety_checker=None, use_safetensors=True)
        try:
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                model_id, variant="fp16", **kwargs)
        except (OSError, ValueError):
            log.info("no fp16 variant for %s, loading default weights", model_id)
            self.pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)
        self.pipe.enable_attention_slicing()
        # Fully on GPU: system RAM is the scarce resource on this machine, and
        # fp16 SD-class models (~2.5 GB) fit the 4 GB card outright. CPU offload
        # would pin the weights in RAM and trigger MemoryErrors.
        self.pipe.to(self.device)

    def generate(self, prompt: str, orientation: str, seed: int, out_path: Path) -> Path:
        import torch

        width, height = config.GEN_SIZES[orientation]
        generator = torch.Generator("cpu").manual_seed(seed)
        image = self.pipe(
            prompt=prompt + _STYLE_SUFFIX,
            negative_prompt=_NEGATIVE if self.guidance > 1.0 else None,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

        # upscale toward output resolution; ffmpeg scales the rest for Ken Burns
        out_w, out_h = config.ORIENTATIONS[orientation]
        image = image.resize((out_w, out_h))
        image.save(out_path)
        return out_path

    def close(self):
        import torch

        del self.pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
