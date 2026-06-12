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

# ---------------------------------------------------------------- images
# key -> (label, diffusers model id, steps, guidance)
IMAGE_MODELS = {
    "sd-turbo":      ("SD-Turbo (fast, default)", "stabilityai/sd-turbo", 2, 0.0),
    "dreamshaper-8": ("DreamShaper 8 (prettier, slower)", "Lykon/dreamshaper-8", 25, 7.0),
    "sd-1.5":        ("Stable Diffusion 1.5 (classic)", "stable-diffusion-v1-5/stable-diffusion-v1-5", 25, 7.5),
}
DEFAULT_IMAGE_MODEL = "sd-turbo"

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
# generation size fed to Stable Diffusion (multiples of 8, near training res)
GEN_SIZES = {
    "horizontal": (768, 448),
    "vertical": (448, 768),
}

FPS = 30
CROSSFADE_S = 0.8          # crossfade length between slides
AUDIO_LEAD_S = 0.6         # silence before narration starts on each slide
AUDIO_TAIL_S = 0.8         # silence after narration ends on each slide
MIN_SLIDE_S = 3.0

# approximate speaking rate, words per second, used to size the script
WORDS_PER_SECOND = {"en": 2.4, "hi": 2.0}
SECONDS_PER_SCENE = 11     # target average slide length -> scene count
