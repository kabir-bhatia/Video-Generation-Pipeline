"""Registries of swappable model backends for each pipeline stage."""

# ---------------------------------------------------------------- script
# key -> (label, backend kind, model id / None)
SCRIPT_MODELS = {
    "qwen2.5-0.5b": ("Qwen2.5-0.5B-Instruct (local, default)", "hf", "Qwen/Qwen2.5-0.5B-Instruct"),
    "qwen2.5-1.5b": ("Qwen2.5-1.5B-Instruct (local, better, needs more RAM)", "hf", "Qwen/Qwen2.5-1.5B-Instruct"),
    "ollama":       ("Ollama (best quality, needs ollama serve running)", "ollama", None),
    "template":     ("Template fallback (no LLM, placeholder text)", "template", None),
}
DEFAULT_SCRIPT_MODEL = "qwen2.5-0.5b"

# ---------------------------------------------------------------- scene video
# key -> (label, backend kind, model id / None)
VIDEO_MODELS = {
    "ltx-video": (
        "LTX-Video (real T2V, fast, best on >=12 GB GPUs - default)",
        "diffusers_t2v",
        "Lightricks/LTX-Video",
    ),
    "zeroscope-576w": (
        "ZeroScope 576w text-to-video (lighter, lower quality)",
        "diffusers_t2v",
        "cerspense/zeroscope_v2_576w",
    ),
    "debug-video": (
        "Debug video (synthetic scene clips, no GPU - local validation)",
        "debug",
        None,
    ),
}
DEFAULT_VIDEO_MODEL = "ltx-video"

# ---------------------------------------------------------------- tts
# key -> (label, supported languages)
# Hindi is temporarily removed from the active pipeline (English-only release);
# the MMS backend can do Hindi and will be re-enabled when Hindi returns.
TTS_MODELS = {
    "mms":    ("Meta MMS-TTS (English, default)", ("en",)),
    "kokoro": ("Kokoro-82M (English, best quality)", ("en",)),
}
DEFAULT_TTS_MODEL = "mms"

LANGUAGES = {"en": "English"}

# ---------------------------------------------------------------- video
ORIENTATIONS = {
    "horizontal": (1280, 720),
    "vertical": (720, 1280),
}
# Notes on the knobs below:
#   video_size       - (w, h) per orientation; many T2V models need each side
#                      divisible by 32 (LTX-Video does).
#   num_frames       - LTX-Video requires (num_frames - 1) % 8 == 0 (e.g. 81, 97).
#   min_free_vram_gb - preflight guard: refuse to load if less is free, so tiny
#                      GPUs fail fast with a clear message instead of an OOM.
#   low_vram_opts    - enable attention/VAE slicing + tiling (saves VRAM, slower).
#                      Off on big GPUs for speed.
RUNTIME_PROFILES = {
    "cloud_high": {
        "label": "Cloud high quality (>=16 GB GPU, e.g. L4/A10/A100)",
        "video_size": {"horizontal": (704, 480), "vertical": (480, 704)},
        "num_frames": 81,
        "fps": 24,
        "num_inference_steps": 35,
        "guidance_scale": 3.0,
        "enable_cpu_offload": False,
        "low_vram_opts": False,
        "min_free_vram_gb": 10.0,
        "decode_chunk_size": 8,
    },
    "cloud_default": {
        "label": "Cloud default (~8-12 GB GPU)",
        "video_size": {"horizontal": (576, 320), "vertical": (320, 576)},
        "num_frames": 49,
        "fps": 16,
        "num_inference_steps": 30,
        "guidance_scale": 5.0,
        "enable_cpu_offload": True,
        "low_vram_opts": True,
        "min_free_vram_gb": 6.0,
        "decode_chunk_size": 8,
    },
    "local_lowmem": {
        "label": "Local low-memory (small GPU / CPU)",
        "video_size": {"horizontal": (384, 224), "vertical": (224, 384)},
        "num_frames": 25,
        "fps": 8,
        "num_inference_steps": 12,
        "guidance_scale": 5.0,
        "enable_cpu_offload": True,
        "low_vram_opts": True,
        "min_free_vram_gb": 3.0,
        "decode_chunk_size": 4,
    },
}
DEFAULT_RUNTIME_PROFILE = "cloud_high"

FPS = 30
CROSSFADE_S = 0.8          # crossfade length between slides
AUDIO_LEAD_S = 0.6         # silence before narration starts on each slide
AUDIO_TAIL_S = 0.8         # silence after narration ends on each slide
MIN_SLIDE_S = 3.0

# approximate speaking rate, words per second, used to size the script
WORDS_PER_SECOND = {"en": 2.4}
SECONDS_PER_SCENE = 12     # target average scene length -> scene count
MAX_SCENES = 24            # cap so a 4 min video stays generatable in reasonable time
