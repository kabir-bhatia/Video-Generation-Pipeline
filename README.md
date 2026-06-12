# Topic → Explainer Video Pipeline

Type a topic ("Explain inflation in 60 seconds"), get a finished slideshow-style
explainer video: AI-generated images with slow zoom/pan motion (Ken Burns),
crossfade transitions, key-term text overlays, and an English **or Hindi**
voiceover. Everything runs locally on open-source models — no API keys.

## Requirements

- **Python 3.10+**
- **ffmpeg on PATH** — for Hindi text overlays it must be built with
  `libharfbuzz`/`libfribidi` (the [gyan.dev full build](https://www.gyan.dev/ffmpeg/builds/)
  on Windows and most Linux distro packages include this)
- **GPU optional** — an NVIDIA card with ~4 GB VRAM makes image generation ~1 s
  per image; CPU-only works too, just slower
- Disk: first run downloads ~6 GB of models (more if you enable the alternatives)

## Quick start

```bash
python -m venv .venv                     # or: uv venv
.venv/Scripts/activate                   # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python app.py                            # opens the web UI in your browser
```

For an NVIDIA GPU, install the CUDA build of torch **before**
`pip install -r requirements.txt`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Or skip the UI and use the command line:

```bash
python cli.py "Explain inflation in 60 seconds" --duration 60 --language en
python cli.py "ब्लैक होल क्या है?" --duration 90 --language hi --orientation vertical
```

Outputs land in `outputs/<timestamp>_<topic>/final.mp4`, alongside the
script JSON, per-scene images, and narration WAVs. Models are cached in
`models_cache/` inside the project (set `HF_HOME` before launching to use a
shared Hugging Face cache instead).

## Pipeline stages & swappable models

| Stage | Default | Alternatives |
|---|---|---|
| Script | Qwen2.5-0.5B-Instruct (local) | Qwen2.5-1.5B (better, needs more RAM), any **Ollama** model, template fallback |
| Images | SD-Turbo (1-4 steps, fast) | DreamShaper 8, Stable Diffusion 1.5 |
| Voice | Meta MMS-TTS (English + Hindi) | Kokoro-82M (English only, `pip install kokoro`) |
| Assembly | ffmpeg (Ken Burns + xfade) | — |

Pick alternatives in the UI under **Advanced: swap models**, or with CLI flags.
To add a new model, register it in [pipeline/config.py](pipeline/config.py).

For noticeably better scripts, install [Ollama](https://ollama.com), pull a model
(`ollama pull llama3.2`), keep `ollama serve` running, and choose the Ollama
script backend.

### Options

- **Duration**: 30 / 60 / 90 / 120 / 180 s (target — actual length follows the narration)
- **Orientation**: horizontal 1280×720 or vertical 720×1280
- **Language**: English or Hindi (script narration *and* voiceover switch together)

### How Hindi works

Small local LLMs write incoherent Hindi directly, so with local script models the
pipeline runs: topic → English (NLLB-200) → English script (Qwen) → Hindi
narration (NLLB-200) → MMS-Hindi voice. Ollama models write Hindi natively and
skip the translation steps. Hindi overlays need a Devanagari font (see Fonts).

## Fonts

Overlay fonts are auto-detected (Segoe UI / Nirmala on Windows, DejaVu /
Noto Sans Devanagari on Linux, Helvetica / Kohinoor on macOS). On Linux,
`sudo apt install fonts-dejavu fonts-noto` covers both languages. To use a
specific font, set `VIDEO_PIPELINE_FONT_EN` / `VIDEO_PIPELINE_FONT_HI` to a
`.ttf`/`.ttc` path.

## Low-memory machines

The pipeline is built to survive on ~4-6 GB of free RAM:

- models load in bfloat16/float16 (fp16 weight variants where available)
- the script LLM and translator each run in their own subprocess so their
  memory returns to the OS before Stable Diffusion loads
- the script model only goes to the GPU if it actually fits in free VRAM
- a torch↔numpy conversion self-check runs at startup and fails fast with a
  fix suggestion instead of crashing mid-pipeline

If you still hit `MemoryError` / "paging file too small", close other apps —
image-model loading is the peak.

## How it works

1. **Script** — an LLM converts the topic into N scenes (≈11 s each) as JSON:
   narration text, an English image prompt, and an optional key term. If the
   model returns fewer scenes than asked, the pipeline requests continuations
   until the target count is met.
2. **Images** — Stable Diffusion renders one image per scene at the target
   aspect ratio.
3. **Voiceover** — TTS narrates each scene to its own WAV (Hindi text is
   romanized via `uroman` for MMS).
4. **Assembly** — ffmpeg gives each slide Ken Burns motion (alternating
   zoom in / zoom out / pan), composites a fading key-term label rendered via
   drawtext, chains slides with `xfade` crossfades, and lays each narration WAV
   at its slide's start time. Slide length adapts to its narration's duration.

## Model licenses

The pipeline code downloads models with their own licenses — check they fit
your use: Qwen2.5 (Apache-2.0), MMS-TTS (CC-BY-NC 4.0), NLLB-200 (CC-BY-NC 4.0),
SD-Turbo (Stability AI Community License), SD 1.5 / DreamShaper (CreativeML
OpenRAIL-M), Kokoro (Apache-2.0). The non-commercial terms on MMS/NLLB matter
if you plan to publish generated videos commercially.
