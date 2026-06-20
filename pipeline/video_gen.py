"""Stage 2: generate one video clip per scene."""

from __future__ import annotations

import gc
import hashlib
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from . import config

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from PIL import Image

# quality boosters appended to every prompt (the per-scene visual style comes
# from the script's video_prompt itself). Keep these descriptive, not stylistic.
_STYLE_SUFFIX = (
    " Smooth natural motion, coherent consistent subject, sharp focus, highly "
    "detailed, professional cinematography, volumetric lighting, depth of field, "
    "high quality, 4k."
)

_NEGATIVE_PROMPT = (
    "worst quality, low quality, blurry, out of focus, low resolution, pixelated, "
    "jpeg artifacts, deformed, distorted, disfigured, extra limbs, glitch, flicker, "
    "jitter, stutter, watermark, signature, text, caption, subtitles, logo, border, "
    "static, still image, duplicate frames"
)


def _profile_settings(profile_key: str) -> dict:
    try:
        return config.RUNTIME_PROFILES[profile_key]
    except KeyError as e:
        raise ValueError(f"unknown runtime profile: {profile_key}") from e


def _seed_colors(text: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return max(40, digest[0]), max(40, digest[1]), max(40, digest[2])


def _run_ffmpeg(args: list[str]):
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr[-3000:]}")


class DebugVideoGenerator:
    """Synthetic scene clips so the full pipeline can be validated locally."""

    def __init__(self, profile_key: str):
        self.profile = _profile_settings(profile_key)

    def generate(self, prompt: str, motion_hint: str, orientation: str,
                 seed: int, out_path: Path) -> Path:
        width, height = self.profile["video_size"][orientation]
        fps = self.profile["fps"]
        frames = self.profile["num_frames"]
        duration_s = frames / fps
        r, g, b = _seed_colors(f"{seed}:{prompt}:{motion_hint}")
        draw = (
            f"drawgrid=width=64:height=64:thickness=2:color=white@0.20,"
            f"drawbox=x='mod(t*{40 + seed % 30}\\,{max(width - 80, 1)})':"
            f"y={height//4}:w=80:h=80:color=white@0.45:t=fill"
        )
        source = (
            f"color=c=0x{r:02X}{g:02X}{b:02X}:s={width}x{height}:r={fps}:d={duration_s:.3f},"
            f"format=yuv420p,{draw}"
        )
        _run_ffmpeg(["-f", "lavfi", "-i", source, "-t", f"{duration_s:.3f}", str(out_path)])
        return out_path

    def close(self):
        pass


class DiffusersTextToVideoGenerator:
    """Text-to-video backend for stronger GPUs / cloud machines."""

    def __init__(self, backend_key: str, profile_key: str):
        import torch
        from diffusers import DiffusionPipeline

        label, _kind, model_id = config.VIDEO_MODELS[backend_key]
        self.profile = _profile_settings(profile_key)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self.model_id = model_id
        log.info("loading video model %s on %s (profile %s)", label, self.device, profile_key)

        # preflight VRAM guard: below the profile's floor a run dies mid-decode
        # with an opaque CUDA OOM, so fail early with an actionable message.
        min_free = float(self.profile.get("min_free_vram_gb", 3.0)) * 1024 ** 3
        if self.device == "cuda":
            free, total = torch.cuda.mem_get_info()
            if free < min_free:
                raise RuntimeError(
                    f"'{label}' on profile '{profile_key}' needs ~{min_free / 1e9:.1f} GB "
                    f"free VRAM but only {free / 1e9:.1f} GB of {total / 1e9:.1f} GB is free. "
                    f"Pick a lighter runtime profile (e.g. 'cloud_default'/'local_lowmem'), "
                    f"the 'debug-video' backend, or a larger GPU."
                )

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        # don't force safetensors: some open T2V checkpoints only ship .bin.
        # diffusers auto-selects safetensors when both are present.
        self.pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        if self.device == "cuda":
            if self.profile["enable_cpu_offload"]:
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to("cuda")
            # slicing/tiling trade speed for VRAM - only on memory-tight profiles.
            # Big-GPU profiles keep diffusers' default (SDPA) attention for speed.
            if self.profile.get("low_vram_opts", True):
                self.pipe.enable_attention_slicing()
                if hasattr(self.pipe, "enable_vae_slicing"):
                    self.pipe.enable_vae_slicing()
                if hasattr(self.pipe, "enable_vae_tiling"):
                    self.pipe.enable_vae_tiling()
        else:
            self.pipe.to("cpu")

    def generate(self, prompt: str, motion_hint: str, orientation: str,
                 seed: int, out_path: Path) -> Path:
        import inspect

        width, height = self.profile["video_size"][orientation]
        fps = self.profile["fps"]
        num_frames = self.profile["num_frames"]
        generator = self._torch.Generator(device="cpu").manual_seed(seed)
        full_prompt = f"{prompt}. Motion direction: {motion_hint}.{_STYLE_SUFFIX}"

        accepted = inspect.signature(self.pipe.__call__).parameters
        kwargs = dict(
            prompt=full_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=self.profile["num_inference_steps"],
            guidance_scale=self.profile["guidance_scale"],
            generator=generator,
            output_type="pil",  # default is "np"; _frames_to_mp4 wants PIL images
        )
        if "negative_prompt" in accepted:
            kwargs["negative_prompt"] = _NEGATIVE_PROMPT
        # decode_chunk_size only exists on frame-by-frame decoders (e.g. SVD);
        # passing it to a UNet T2V pipeline raises TypeError.
        if "decode_chunk_size" in accepted:
            kwargs["decode_chunk_size"] = self.profile["decode_chunk_size"]

        result = self.pipe(**kwargs)
        frames = getattr(result, "frames", None)
        if frames is None or len(frames) == 0 or len(frames[0]) == 0:
            raise RuntimeError("video model returned no frames")
        self._frames_to_mp4(frames[0], fps, out_path)
        return out_path

    @staticmethod
    def _frames_to_mp4(frames: list["Image.Image"], fps: int, out_path: Path):
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for i, frame in enumerate(frames):
                if not isinstance(frame, Image.Image):
                    arr = np.asarray(frame)
                    if arr.dtype != np.uint8:  # diffusers float frames are 0..1
                        arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
                    frame = Image.fromarray(arr)
                frame.save(tmp / f"frame_{i:03d}.png")
            _run_ffmpeg([
                "-framerate", str(fps),
                "-i", str(tmp / "frame_%03d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(out_path),
            ])

    def close(self):
        del self.pipe
        gc.collect()
        if self.device == "cuda":
            self._torch.cuda.empty_cache()


def make_video_generator(backend_key: str, profile_key: str):
    _label, kind, _model_id = config.VIDEO_MODELS[backend_key]
    if kind == "debug":
        return DebugVideoGenerator(profile_key)
    return DiffusersTextToVideoGenerator(backend_key, profile_key)


def validate_backend(backend_key: str, profile_key: str) -> dict:
    gen = make_video_generator(backend_key, profile_key)
    try:
        return {
            "backend": backend_key,
            "profile": profile_key,
            "status": "ok",
            "device": getattr(gen, "device", "cpu"),
        }
    finally:
        gen.close()
