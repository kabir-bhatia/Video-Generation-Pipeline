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
    "debug-video": (
        "Debug video (synthetic scene clips for local validation)",
        "debug",
        None,
    ),
    "zeroscope-576w": (
        "ZeroScope 576w text-to-video (cloud / higher VRAM)",
        "diffusers_t2v",
        "cerspense/zeroscope_v2_576w",
    ),
}
DEFAULT_VIDEO_MODEL = "debug-video"

# ---------------------------------------------------------------- tts
# key -> (label, supported languages)
TTS_MODELS = {
    "mms":    ("Meta MMS-TTS (English + Hindi, default)", ("en", "hi")),
    "kokoro": ("Kokoro-82M (English only, best quality)", ("en",)),
}
DEFAULT_TTS_MODEL = "mms"

LANGUAGES = {"en": "English", "hi": "Hindi"}

# ---------------------------------------------------------------- video
ORIENTATIONS = {
    "horizontal": (1280, 720),
    "vertical": (720, 1280),
}
RUNTIME_PROFILES = {
    "local_lowmem": {
        "label": "Local low-memory",
        "video_size": {"horizontal": (384, 224), "vertical": (224, 384)},
        "num_frames": 16,
        "fps": 8,
        "num_inference_steps": 12,
        "guidance_scale": 7.0,
        "enable_cpu_offload": True,
        "decode_chunk_size": 4,
    },
    "cloud_default": {
        "label": "Cloud default",
        "video_size": {"horizontal": (576, 320), "vertical": (320, 576)},
        "num_frames": 24,
        "fps": 8,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "enable_cpu_offload": False,
        "decode_chunk_size": 8,
    },
}
DEFAULT_RUNTIME_PROFILE = "local_lowmem"

FPS = 30
CROSSFADE_S = 0.8          # crossfade length between slides
AUDIO_LEAD_S = 0.6         # silence before narration starts on each slide
AUDIO_TAIL_S = 0.8         # silence after narration ends on each slide
MIN_SLIDE_S = 3.0

# approximate speaking rate, words per second, used to size the script
WORDS_PER_SECOND = {"en": 2.4, "hi": 2.0}
SECONDS_PER_SCENE = 11     # target average slide length -> scene count
