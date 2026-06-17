"""Orchestrator: topic in, finished explainer video out."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from . import assemble, config, text_overlay, video_gen

log = logging.getLogger(__name__)

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"


def _check_torch_numpy():
    """Fail fast if the installed numpy can't interoperate with torch.

    The torch<->numpy bridge breaks silently when numpy's C ABI changes
    underneath a prebuilt torch, so test the actual conversion path instead
    of trusting version numbers alone.
    """
    import numpy as np
    import torch

    try:
        roundtrip = torch.from_numpy(np.ones(2, dtype=np.float32)).numpy()
        assert roundtrip.sum() == 2.0
    except Exception as e:
        raise RuntimeError(
            f"numpy {np.__version__} is incompatible with this torch build "
            f"({torch.__version__}). Downgrade numpy (e.g. pip install 'numpy<2.5') "
            f"or reinstall a matching torch."
        ) from e


def _run_stage_subprocess(op: str, in_path: Path, out_path: Path) -> dict:
    """Run one script-stage op in a child process (see script_cli docstring)."""
    import subprocess
    import sys

    import os

    # CPU-only for these small models: a second CUDA-torch process would
    # double-commit GPU DLL reservations and exhaust the paging file
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [sys.executable, "-m", "pipeline.script_cli", op, str(in_path), str(out_path)],
        capture_output=True, text=True, env=env,
        cwd=Path(__file__).resolve().parent.parent)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"script stage '{op}' failed (exit {proc.returncode}):\n"
                           f"{proc.stderr[-2000:]}")
    return json.loads(out_path.read_text(encoding="utf-8"))


def _run_script_stage(topic: str, language: str, duration_s: int,
                      script_model: str, ollama_model: str, workdir: Path,
                      report=None) -> dict:
    from . import script_gen

    translate = script_gen.needs_translation(language, script_model)

    if translate and re.search(r"[ऀ-ॿ]", topic):
        # the English script model can't read a Devanagari topic
        if report:
            report("Translating topic to English...", 0.03)
        topic_path = workdir / "topic.json"
        topic_path.write_text(json.dumps({"text": topic}), encoding="utf-8")
        topic = _run_stage_subprocess("topic_to_en", topic_path, topic_path)["text"]
        log.info("topic translated for the script model: %s", topic)

    args_path = workdir / "script_args.json"
    args_path.write_text(json.dumps({
        "topic": topic,
        "language": "en" if translate else language,
        "duration_s": duration_s,
        "backend_key": script_model, "ollama_model": ollama_model,
    }), encoding="utf-8")

    script = _run_stage_subprocess("generate", args_path, workdir / "script.json")
    if translate:
        if report:
            report("Translating script to Hindi...", 0.10)
        script = _run_stage_subprocess("translate", workdir / "script.json",
                                       workdir / "script.json")
    return script


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "video"


def run_pipeline(
    topic: str,
    duration_s: int = 60,
    orientation: str = "horizontal",
    language: str = "en",
    script_model: str = config.DEFAULT_SCRIPT_MODEL,
    video_model: str = config.DEFAULT_VIDEO_MODEL,
    tts_model: str = config.DEFAULT_TTS_MODEL,
    runtime_profile: str = config.DEFAULT_RUNTIME_PROFILE,
    ollama_model: str = "llama3.2",
    seed: int = 42,
    progress=None,
) -> dict:
    """Returns {"video": Path, "script": dict, "workdir": Path}."""

    def report(msg: str, frac: float | None = None):
        log.info(msg)
        if progress:
            progress(msg, frac)

    if orientation not in config.ORIENTATIONS:
        raise ValueError(f"orientation must be one of {list(config.ORIENTATIONS)}")
    if language not in config.LANGUAGES:
        raise ValueError(f"language must be one of {list(config.LANGUAGES)}")
    if runtime_profile not in config.RUNTIME_PROFILES:
        raise ValueError(f"runtime_profile must be one of {list(config.RUNTIME_PROFILES)}")
    duration_s = max(20, min(180, int(duration_s)))

    workdir = OUTPUT_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}_{_slug(topic)}"
    (workdir / "videos").mkdir(parents=True)
    (workdir / "audio").mkdir()

    # ---- 1. script (subprocess: returns the LLM's memory to the OS) -------
    report("Writing script...", 0.02)
    script = _run_script_stage(topic, language, duration_s, script_model,
                               ollama_model, workdir, report)
    scenes = script["scenes"]
    n = len(scenes)
    report(f"Script ready: {n} scenes", 0.15)

    # ---- 2. scene videos -------------------------------------------------
    # first point where this process itself needs torch; check lazily so the
    # parent doesn't hold torch's memory while the script subprocess runs
    _check_torch_numpy()
    gen = None
    try:
        gen = video_gen.make_video_generator(video_model, runtime_profile)
        scene_videos = []
        for i, scene in enumerate(scenes):
            report(f"Generating video scene {i + 1}/{n}", 0.15 + 0.40 * i / n)
            scene_videos.append(
                gen.generate(
                    scene["video_prompt"],
                    scene.get("motion_hint", ""),
                    orientation,
                    seed + i,
                    workdir / "videos" / f"scene_{i:02d}.mp4",
                )
            )
    finally:
        if gen is not None:
            gen.close()  # free VRAM/RAM before TTS loads

    # ---- 3. voiceover ----------------------------------------------------
    report("Loading voice model...", 0.58)
    from . import tts

    voice = tts.make_tts(tts_model, language)
    wavs, audio_durs = [], []
    for i, scene in enumerate(scenes):
        report(f"Narrating scene {i + 1}/{n}", 0.58 + 0.22 * i / n)
        wav = workdir / "audio" / f"scene_{i:02d}.wav"
        audio_durs.append(voice.synthesize(scene["narration"], wav))
        wavs.append(wav)

    # ---- 4. overlays + assembly ------------------------------------------
    w, h = config.ORIENTATIONS[orientation]
    overlays: list[Path | None] = []
    for i, scene in enumerate(scenes):
        term = scene.get("key_term", "").strip()
        overlays.append(
            text_overlay.render_overlay(term, w, h, language,
                                        workdir / "videos" / f"overlay_{i:02d}.png")
            if term else None)

    report("Assembling video...", 0.82)
    final = assemble.assemble(scene_videos, overlays, wavs, audio_durs, orientation,
                              workdir,
                              progress=lambda m: report(m, 0.85))
    report("Done", 1.0)
    return {"video": final, "script": script, "workdir": workdir}
