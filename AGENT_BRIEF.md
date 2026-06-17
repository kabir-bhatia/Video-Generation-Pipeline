# Agent Brief: Topic → Explainer Video Pipeline

> **Purpose of this file**: Give a coding agent (or new developer) a complete mental model
> of this project so they can continue development without needing to re-read every source file.
> Last updated: 2026-06-17.

---

## 1. What the project does

Takes a text topic (e.g. `"Explain inflation in 60 seconds"`) and produces a finished
`.mp4` slideshow-style explainer video:

- **AI-generated images** per scene (Stable Diffusion)
- **Ken Burns motion** on each still (slow zoom in/out, pan left/right via ffmpeg `zoompan`)
- **Crossfade transitions** between scenes (ffmpeg `xfade`)
- **Voiceover narration** via TTS, placed per-scene on the timeline
- **Key-term text overlays** (bottom-center label box via ffmpeg `drawtext`)
- **English or Hindi** language throughout: narration + voice + overlay switch together

Everything runs locally with open-source models — no API keys needed.

---

## 2. Repo layout

```
video_pipeline/
├── app.py                  # Gradio web UI entry point
├── cli.py                  # argparse CLI entry point
├── requirements.txt
├── AGENT_BRIEF.md          # ← you are here
├── README.md               # public-facing quick-start docs
├── pipeline/
│   ├── __init__.py         # sets HF_HOME + HF_HUB_DISABLE_XET before any imports
│   ├── config.py           # all model registries + timing constants
│   ├── run.py              # pipeline orchestrator (the main entry point from code)
│   ├── script_gen.py       # Stage 1: LLM script generation + NLLB translation
│   ├── script_cli.py       # subprocess wrapper for Stage 1 (memory isolation)
│   ├── image_gen.py        # Stage 2: Stable Diffusion image generation
│   ├── tts.py              # Stage 3: TTS voiceover (MMS-TTS + Kokoro)
│   ├── text_overlay.py     # render key-term overlay PNGs via ffmpeg drawtext
│   └── assemble.py         # Stage 4: ffmpeg assembly (Ken Burns + xfade + audio mix)
├── outputs/                # gitignored; each run → timestamped subdirectory
├── models_cache/           # gitignored; HF model weights land here (not C:)
└── tests/
    └── memcheck.py         # Windows-only debug helper (print RAM / commit stats)
```

---

## 3. Pipeline stages

```
topic string
     │
     ▼
┌──────────────────────────────────────────────────────┐
│ Stage 1: Script generation  (subprocess)             │
│   • Devanagari topic? → NLLB hi→en translation first │
│   • LLM generates N scenes as JSON                   │
│   • Hindi mode + HF backend? → NLLB en→hi afterwards │
│   Output: script.json                                │
└──────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│ Stage 2: Image generation  (in-process, GPU/CPU)     │
│   • One SD image per scene from image_prompt field   │
│   • Saved to outputs/<run>/images/scene_NN.png       │
└──────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│ Stage 3: TTS voiceover  (in-process)                 │
│   • One WAV per scene from narration field           │
│   • Hindi text → uroman romanization for MMS-TTS     │
│   • Saved to outputs/<run>/audio/scene_NN.wav        │
└──────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│ Stage 4: Assembly  (ffmpeg subprocesses)             │
│   • Render overlay PNG per scene (key_term label)    │
│   • Render Ken Burns clip per scene (zoompan)        │
│   • Crossfade all clips together (xfade)             │
│   • adelay each WAV to its start time, amix          │
│   • Mux video + audio → final.mp4                    │
└──────────────────────────────────────────────────────┘
     │
     ▼
outputs/<run>/final.mp4
```

---

## 4. Script JSON format

Everything downstream is driven by this structure:

```json
{
  "title": "What is Inflation?",
  "scenes": [
    {
      "narration": "Spoken voiceover text for this slide.",
      "image_prompt": "English SD prompt describing the scene image.",
      "key_term": "Inflation"
    },
    ...
  ]
}
```

- `narration` — language-aware (English or Hindi Devanagari)
- `image_prompt` — **always English** regardless of language (SD was trained on English)
- `key_term` — short on-screen label, language-aware; `""` means no overlay

---

## 5. Model registry (`pipeline/config.py`)

All swappable backends are registered here. To add a new model, add an entry to the
appropriate dict — the UI and CLI pick it up automatically.

### Script models
```python
SCRIPT_MODELS = {
    "qwen2.5-0.5b":  ("...", "hf",       "Qwen/Qwen2.5-0.5B-Instruct"),
    "qwen2.5-1.5b":  ("...", "hf",       "Qwen/Qwen2.5-1.5B-Instruct"),
    "ollama":         ("...", "ollama",   None),
    "template":       ("...", "template", None),
}
DEFAULT_SCRIPT_MODEL = "qwen2.5-0.5b"
```
Kind `"hf"` → `_HFChat` class (transformers).  
Kind `"ollama"` → `_OllamaChat` (REST to `localhost:11434`).  
Kind `"template"` → `_generate_template()` (no model, placeholder text — good for testing).

### Image models
```python
IMAGE_MODELS = {
    "sd-turbo":      ("...", "stabilityai/sd-turbo",                              2, 0.0),
    "dreamshaper-8": ("...", "Lykon/dreamshaper-8",                              25, 7.0),
    "sd-1.5":        ("...", "stable-diffusion-v1-5/stable-diffusion-v1-5",      25, 7.5),
}
DEFAULT_IMAGE_MODEL = "sd-turbo"
```
Tuple: `(label, HF model id, inference steps, guidance scale)`.

### TTS models
```python
TTS_MODELS = {
    "mms":    ("...", ("en", "hi")),   # Meta MMS-TTS
    "kokoro": ("...", ("en",)),        # Kokoro-82M (pip install kokoro)
}
DEFAULT_TTS_MODEL = "mms"
```

### Timing constants
```python
FPS = 30
CROSSFADE_S = 0.8        # xfade length between slides
AUDIO_LEAD_S = 0.6       # silence before narration on each slide
AUDIO_TAIL_S = 0.8       # silence after narration
MIN_SLIDE_S = 3.0        # minimum slide duration
WORDS_PER_SECOND = {"en": 2.4, "hi": 2.0}
SECONDS_PER_SCENE = 11   # → n_scenes = round(duration_s / 11)
```

### Orientations & generation sizes
```python
ORIENTATIONS = {"horizontal": (1280, 720), "vertical": (720, 1280)}
GEN_SIZES    = {"horizontal": (768, 448),  "vertical": (448, 768)}
```
SD generates at `GEN_SIZES` (multiples of 8, close to training resolution); PIL upscales
to `ORIENTATIONS`; ffmpeg handles the rest in the Ken Burns step.

---

## 6. Key design decisions and why

### 6a. Subprocess isolation for Stage 1
`script_cli.py` runs as a **child process** (not in-process) for every script/translate
call. This is intentional:

- The LLM (Qwen 0.5B–1.5B) and the NLLB translator each need ~1-2 GB of RAM.
- If they run in-process, Python's allocator retains the memory even after `del model`,
  preventing Stable Diffusion from loading.
- A subprocess gives back RAM fully to the OS when it exits.
- `CUDA_VISIBLE_DEVICES=""` is set for these subprocesses — a second CUDA-torch process
  doubles the GPU DLL commit reservation and can exhaust the Windows paging file.

### 6b. Two-pass Hindi: English → LLM → NLLB
Small local LLMs (Qwen 0.5B, 1.5B) write incoherent Hindi when asked directly.
Solution: always generate the script in English with an English-capable LLM, then
translate `narration` and `key_term` to Hindi with NLLB-200-distilled-600M.

`needs_translation(language, backend_key)` returns `True` for `hi` + `hf` backends.
Ollama models (llama3.2 etc.) write Hindi natively and bypass this.

### 6c. VRAM measurement before GPU placement (script LLM)
```python
need = sum(p.numel() * p.element_size() for p in model.parameters())
free, _ = torch.cuda.mem_get_info()
if need * 1.25 < free:
    model = model.to("cuda", dtype=torch.float16)
```
`try/except` for OOM is **not used** — a CUDA OOM mid-`.to()` poisons the CUDA context
for all subsequent calls in the same process. Measure first, decide, then move.

### 6d. fp16 variant + safetensors for SD
```python
kwargs = dict(torch_dtype=dtype, safety_checker=None, use_safetensors=True)
try:
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, variant="fp16", **kwargs)
except (OSError, ValueError):
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)
```
Loading fp32 weights and then casting requires a transient 5 GB peak that exceeds the
commit limit on machines with 8 GB RAM + small pagefile. The `fp16` variant downloads
weights already in fp16, and `use_safetensors=True` avoids reading `.bin` files into
RAM all at once.

`pipe.to(self.device)` is used (not `enable_model_cpu_offload`) because system RAM is
scarcer than VRAM on the target machine — offload would pin weights in RAM.

### 6e. HF_HOME on E: (not C:)
`pipeline/__init__.py` sets `HF_HOME` to `<project_root>/models_cache/` before any
import that could trigger `huggingface_hub`. If `gradio` is imported first, it freezes
HF_HOME to `~/.cache/huggingface` which may be on a nearly-full drive.

**Critical import order** in `app.py` and `cli.py`:
```python
from pipeline import config, run_pipeline   # sets HF_HOME
import gradio as gr                          # NOW gradio imports huggingface_hub
```

### 6f. HF_HUB_DISABLE_XET=1
The xet downloader (a Rust process) can abort the whole Python process when its memory
allocations fail on low-RAM machines. Setting this env var forces the plain HTTP
downloader, which is slightly slower but survives under memory pressure.

### 6g. Devanagari font via ffmpeg drawtext (not PIL)
PIL on Windows (the standard wheel) does not include Raqm (HarfBuzz + FriBiDi), so
it cannot shape complex scripts like Devanagari. ffmpeg's `drawtext` filter uses
`libharfbuzz`/`libfribidi` internally and handles it correctly.

The overlay is rendered as a full-frame RGBA PNG (transparent background) and then
composited over the video in the same ffmpeg filter chain as the Ken Burns effect.

### 6h. Continuation top-up loop
Qwen-0.5B frequently returns fewer scenes than requested. `_generate_llm()` loops up to
3 times, sending `_CONTINUE_PROMPT` with the topic, title, and last narration line as
context, then appending the extra scenes.

---

## 7. File-by-file reference

### `pipeline/__init__.py`
Sets env vars before anything else. The only public symbol it exports is `run_pipeline`.
```python
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "models_cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from .run import run_pipeline
```

---

### `pipeline/run.py` — orchestrator

**`run_pipeline(topic, duration_s, orientation, language, script_model, image_model, tts_model, ollama_model, seed, progress)`**

Returns `{"video": Path, "script": dict, "workdir": Path}`.

The `progress` callback is `(msg: str, frac: float | None)` — used by both the
Gradio `gr.Progress()` wrapper and optionally for CLI logging.

**`_run_stage_subprocess(op, in_path, out_path)`** — runs `python -m pipeline.script_cli`
as a child process with `CUDA_VISIBLE_DEVICES=""`. Raises `RuntimeError` on non-zero
exit or missing output file.

**`_run_script_stage()`** — orchestrates the three possible script subprocesses:
1. `topic_to_en` (only if Hindi + Devanagari detected in topic)
2. `generate`
3. `translate` (only if `needs_translation()` is True)

**`_check_torch_numpy()`** — runs at the start of Stage 2 (lazily, after subprocesses
finish). Tests `torch.from_numpy(np.ones(...)).numpy()` — fails fast with an actionable
error message if the numpy C ABI doesn't match the torch build.

---

### `pipeline/script_gen.py` — Stage 1

**`generate_script(topic, language, duration_s, backend_key, ollama_model)`**

Entry point. Computes `n_scenes` and `words_per_scene` from duration, then dispatches
to template / `_HFChat` / `_OllamaChat`.

**`_HFChat`** — loads the model in bfloat16, measures VRAM, conditionally moves to GPU
in float16. `ask()` uses `apply_chat_template(..., return_dict=True)` (transformers v5
API — returns a dict, not a tensor). `close()` deletes the model and calls
`cuda.empty_cache()`.

**`_OllamaChat`** — HTTP POST to `localhost:11434/api/chat` with `format: "json"`.

**`_generate_llm(chat, topic, language, n_scenes, words_per_scene)`** — up to 2 parse
retries, then up to 3 continuation top-up loops.

**`translate_to_hindi(script)`** — NLLB en→hi. Translates sentence-by-sentence (NLLB
quality degrades on long inputs). Image prompts are NOT translated.

**`translate_topic_to_english(data)`** — NLLB hi→en for the topic string only.

**`needs_translation(language, backend_key)`** — `True` iff `language == "hi"` and the
backend kind is `"hf"`.

---

### `pipeline/script_cli.py` — subprocess entry

```
python -m pipeline.script_cli generate    args.json   out.json
python -m pipeline.script_cli translate   in.json     out.json
python -m pipeline.script_cli topic_to_en topic.json  topic.json
```

Reads JSON from `in_path`, calls the appropriate `script_gen` function, writes JSON to
`out_path`. Called by `_run_stage_subprocess()` in `run.py`.

---

### `pipeline/image_gen.py` — Stage 2

**`ImageGenerator(backend_key)`** — loads the diffusers pipeline; tries fp16 variant +
safetensors first; enables attention slicing; calls `.to(device)`.

**`generate(prompt, orientation, seed, out_path)`** — generates at `GEN_SIZES`,
appends `_STYLE_SUFFIX` to prompt, uses `negative_prompt` only when `guidance > 1.0`
(SD-Turbo uses guidance 0, which ignores negative prompts). PIL-upscales to
`ORIENTATIONS` before saving.

**`close()`** — `del self.pipe`, `gc.collect()`, `cuda.empty_cache()`. Called in a
`finally` block in `run.py` so VRAM is freed before TTS loads.

---

### `pipeline/tts.py` — Stage 3

**`MMSTTS(language)`** — loads `facebook/mms-tts-eng` or `facebook/mms-tts-hin`
(`VitsModel` + `AutoTokenizer`). If the tokenizer has `is_uroman=True`, instantiates
`uroman.Uroman()` for romanization.

**`MMSTTS.synthesize(text, out_path)`** — splits on sentence boundaries (`.!?।`),
synthesizes each sentence, concatenates with 250 ms silence pauses, normalizes peak
amplitude to 0.9, writes WAV via `soundfile`. Returns duration in seconds.

**`KokoroTTS`** — optional; requires `pip install kokoro`. English only.
`KPipeline(lang_code="a")` with voice `"af_heart"`.

**`make_tts(backend_key, language)`** — factory; returns `KokoroTTS` or `MMSTTS`.

---

### `pipeline/text_overlay.py`

**`_find_font(language)`** — checks `VIDEO_PIPELINE_FONT_EN` / `VIDEO_PIPELINE_FONT_HI`
env vars first, then probes platform candidate lists (Windows/Linux/macOS paths).
Caches result in module-level `_resolved` dict. Raises a helpful `RuntimeError` naming
the env var if nothing is found.

**`render_overlay(text, video_w, video_h, language, out_path)`** — writes text to a
temp `.txt` file (Unicode-safe), runs ffmpeg with a `lavfi color=black@0.0` source and
`drawtext` filter to produce a full-frame RGBA PNG with a semi-transparent box
(`boxcolor=0x0A0A14@0.65`) and white text at bottom-center.

Font size: `max(28, video_w // 22)`. Bottom margin: `video_h // 12`.

---

### `pipeline/assemble.py` — Stage 4

**`_ken_burns(motion, frames)`** — returns a `zoompan` filter string. Four motions
cycle per scene: `zoom_in`, `zoom_out`, `pan_right`, `pan_left`. Zoom range is 1.0–1.12.
Pre-scales image to 2× output size before zoompan to avoid jitter artifacts.

**`render_scene_clip(image, overlay, duration_s, orientation, motion, out_path)`** — 
combines Ken Burns + optional overlay compositing into a single ffmpeg filter_complex.
Overlay fades in at 0.5 s, fades out before the crossfade.

**`crossfade_concat(clips, durations, out_path)`** — chains all clips with
`xfade=transition=fade`. Computes xfade offsets from cumulative durations minus
`CROSSFADE_S` per boundary.

**`mix_audio(wavs, starts_s, total_s, out_path)`** — uses `adelay` + `amix` to place
each narration WAV at its absolute start time on one combined audio track. All streams
are resampled to 24 kHz mono first.

**`mux(video, audio, out_path)`** — stream-copies the video, encodes audio as AAC 160k.

**`assemble(scene_images, scene_overlays, scene_wavs, audio_durations, orientation, workdir, progress)`** — 
top-level function. Computes per-slide durations accounting for audio lead/tail + crossfade
overlap. Returns `workdir/final.mp4`.

Slide duration formula:
```
lead  = AUDIO_LEAD_S + (CROSSFADE_S/2 if not first)
tail  = AUDIO_TAIL_S + (CROSSFADE_S/2 if not last)
dur   = max(lead + audio_d + tail, MIN_SLIDE_S, 2*CROSSFADE_S + 0.2)
```

Audio start time = `sum(prev durations - CROSSFADE_S) + lead` for that slide.

---

### `app.py` — Gradio UI

**Import order is critical** (see §6e above): `pipeline` before `gradio`.

UI components: topic textbox, duration dropdown (30/60/90/120/180), orientation radio
(Horizontal/Vertical), language radio (English/Hindi), Advanced accordion with model
dropdowns.

`generate()` callback validates TTS–language compatibility, calls `run_pipeline()`, returns
`(str(video_path), script_dict)` to a `gr.Video` + `gr.JSON` pair.

---

### `cli.py` — argparse CLI

```bash
python cli.py "Explain inflation in 60 seconds" \
    --duration 60 --language en \
    --script-model qwen2.5-0.5b \
    --image-model sd-turbo \
    --tts-model mms \
    --seed 42
```

All flags mirror the `run_pipeline()` signature exactly.

---

## 8. Output directory structure

```
outputs/20240617-143022_explain-inflation-in-60-seco/
├── script_args.json          # args passed to the generate subprocess
├── script.json               # final scene list (possibly translated)
├── topic.json                # (only present for Devanagari topics)
├── images/
│   ├── scene_00.png          # SD-generated image
│   ├── overlay_00.png        # key-term label PNG (transparent RGBA)
│   ├── scene_01.png
│   └── ...
├── audio/
│   ├── scene_00.wav
│   ├── scene_01.wav
│   └── ...
├── clips/
│   ├── scene_00.mp4          # Ken Burns clip (silent)
│   └── ...
├── video_silent.mp4          # all clips crossfaded, no audio
├── voiceover.wav             # all WAVs mixed onto one timeline
└── final.mp4                 # ← the deliverable
```

---

## 9. Environment variables

| Variable | Effect | Default |
|---|---|---|
| `HF_HOME` | Where HF models are cached | `<project>/models_cache/` |
| `HF_HUB_DISABLE_XET` | Disable the xet Rust downloader | `1` (set in `__init__.py`) |
| `CUDA_VISIBLE_DEVICES` | Set to `""` in script subprocesses | (unset in main process) |
| `VIDEO_PIPELINE_FONT_EN` | Path to .ttf/.ttc for EN overlays | auto-detected |
| `VIDEO_PIPELINE_FONT_HI` | Path to .ttf/.ttc for HI overlays (needs Devanagari) | auto-detected |

---

## 10. Dependencies and external requirements

### Python packages (`requirements.txt`)
- `torch` — auto-installs CPU version from PyPI; pre-install CUDA build for GPU
- `diffusers` — SD pipeline
- `transformers` — LLM + NLLB + MMS-TTS (VitsModel)
- `accelerate` — required by some diffusers loading paths
- `gradio` — web UI
- `soundfile` — WAV read/write
- `numpy>=2.1,<3` — torch↔numpy bridge; a conversion self-check validates compatibility
- `pillow` — image upscaling after SD generation
- `requests` — Ollama REST calls
- `uroman` — Hindi romanization for MMS-TTS
- `kokoro` (optional) — higher quality English TTS

### System requirement
- **ffmpeg on PATH**, built with `libharfbuzz` and `libfribidi` (needed for Devanagari
  overlay rendering). On Windows, the [gyan.dev full build](https://www.gyan.dev/ffmpeg/builds/)
  includes these. On Linux, the distro package usually includes them.

### Ollama (optional, for best script quality)
- Install Ollama, run `ollama pull llama3.2`, keep `ollama serve` running, select the
  Ollama backend in the UI or with `--script-model ollama`.

---

## 11. Known constraints and gotchas

1. **Small LLMs return fewer scenes than requested.** The top-up loop in `_generate_llm`
   handles this up to 3 extra rounds, but very short videos may still get fewer scenes.
   Using the `template` backend or Ollama avoids this entirely.

2. **Hindi quality with HF backends.** NLLB translation is good but not native-quality.
   For the best Hindi output, use Ollama with a multilingual model.

3. **RAM is the bottleneck, not VRAM.** On machines with ≥16 GB RAM this is a non-issue.
   On 8 GB machines the subprocess isolation and fp16 loading are load-bearing — don't
   collapse stages into one process.

4. **NLLB sentence-by-sentence translation.** Long narration sentences are split on
   `.!?।` before being fed to NLLB because quality degrades on long inputs.

5. **SD-Turbo ignores negative prompts.** `guidance_scale=0.0` means CFG is disabled;
   `negative_prompt` is silently ignored. This is intentional for speed.

6. **ffmpeg path escaping.** Windows paths with backslashes and colons need special
   escaping for `drawtext` option values. `_ff_escape_path()` handles this.

7. **transformers v5 API.** `apply_chat_template` returns a dict, not a tensor, when
   `return_dict=True`. The `**inputs` splat and `inputs["input_ids"].shape[1]` for
   `prompt_len` are correct for v5. Do not revert to the v4 pattern.

---

## 12. How to extend

### Add a new script backend
1. Add an entry to `SCRIPT_MODELS` in `config.py` with a new kind string (e.g. `"api"`).
2. Create a backend class with `.ask(user_prompt, creative) -> str` and `.close()`.
3. Add a branch in `generate_script()` to instantiate it.

### Add a new image model
Add an entry to `IMAGE_MODELS`: `(label, hf_model_id, steps, guidance)`. No other
changes needed — `ImageGenerator` uses `AutoPipelineForText2Image.from_pretrained`.

### Add a new TTS backend
1. Add to `TTS_MODELS` in `config.py`.
2. Create a class with `.synthesize(text, out_path) -> float` (duration in seconds).
3. Add a branch in `make_tts()`.

### Add a new language
1. Add to `LANGUAGES` and `WORDS_PER_SECOND` in `config.py`.
2. Add NLLB language codes to `script_gen.py` translation functions.
3. Add font candidates to `text_overlay.py`.
4. Add a romanization path in `tts.py` if the TTS tokenizer needs it.

### Change video style
- Ken Burns parameters: edit `_ken_burns()` in `assemble.py`.
- Transition style: `xfade=transition=fade` in `crossfade_concat()` — see ffmpeg docs
  for other transition names (wipeleft, circleopen, etc.).
- Overlay style: edit `drawtext` filter in `text_overlay.py`; box color, font size,
  position are all in `render_overlay()`.

---

## 13. Running the pipeline

### Web UI
```bash
python app.py
```
Opens browser at `http://127.0.0.1:7860`.

### CLI
```bash
python cli.py "What is quantum entanglement?" --duration 60 --language en
python cli.py "ब्लैक होल क्या है?" --duration 90 --language hi --orientation vertical
```

### From Python
```python
from pipeline import run_pipeline
result = run_pipeline(
    topic="Explain photosynthesis",
    duration_s=60,
    orientation="horizontal",
    language="en",
)
print(result["video"])   # Path to final.mp4
print(result["script"])  # dict with title + scenes
```

### Test without downloading models (template backend)
```bash
python cli.py "Test topic" --duration 30 --script-model template
```
Uses placeholder narration, still generates images and audio with default models.

---

## 14. GitHub

Public repo: **https://github.com/kabir-bhatia/Video-Generation-Pipeline**  
Branch: `main`  
Git user: `kabir-bhatia`
