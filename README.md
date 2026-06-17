# Topic -> Explainer Video Pipeline

Type a topic, get a finished explainer video with actual generated scene clips,
crossfades, key-term overlays, and an English or Hindi voiceover. The pipeline
now supports real scene-video generation plus a lightweight debug-video backend
for local validation on constrained machines.

## Requirements

- Python 3.10+
- ffmpeg on PATH
- GPU optional for debug/local validation, strongly recommended for real text-to-video
- Windows/Linux/macOS font with Devanagari support if you want Hindi overlays

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
python app.py
```

CLI example:

```bash
python cli.py "Explain inflation in 60 seconds" --duration 60 --language en
python cli.py "Explain inflation in 60 seconds" --video-model zeroscope-576w --runtime-profile cloud_default
```

## Pipeline stages

1. Script generation:
   The LLM produces scenes with `narration`, `video_prompt`, `motion_hint`, and `key_term`.
2. Scene-video generation:
   A video backend renders one mp4 clip per scene. `debug-video` is intended for local smoke tests. `zeroscope-576w` is the real text-to-video backend for stronger GPUs/cloud.
3. Voiceover:
   MMS-TTS handles English and Hindi, and Kokoro remains available for English.
4. Assembly:
   ffmpeg normalizes each scene clip, overlays the key term, crossfades scenes, mixes narration, and writes `final.mp4`.

## Runtime profiles

- `local_lowmem`: low-resolution, fewer frames, CPU offload enabled. Best for your 4 GB VRAM laptop and component checks.
- `cloud_default`: larger frames and more inference steps for stronger GPUs.

## Local validation

For a constrained laptop, validate the pipeline without a full heavy run:

```bash
python -m pipeline.validate
python tests/test_assembly.py
```

That checks component loading plus a debug-backend clip assembly path. Once a larger GPU is provisioned, switch to a real video backend and `cloud_default`.
