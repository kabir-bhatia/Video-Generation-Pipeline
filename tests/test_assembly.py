"""Smoke-test overlay rendering and clip-first assembly with synthetic videos."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

from pipeline import assemble, config, text_overlay
from pipeline.video_gen import make_video_generator

workdir = Path(__file__).parent / "tmp"
(workdir / "clips").mkdir(parents=True, exist_ok=True)

terms = ["Inflation", "मुद्रास्फीति", ""]
scene_videos, overlays, wavs, durs = [], [], [], []

gen = make_video_generator("debug-video", "local_lowmem")
try:
    for i, term in enumerate(terms):
        clip = workdir / f"scene_{i}.mp4"
        gen.generate(f"Scene {i} about inflation", "gentle camera drift", "horizontal", i + 1, clip)
        scene_videos.append(clip)

        w, h = config.ORIENTATIONS["horizontal"]
        overlays.append(
            text_overlay.render_overlay(
                term, w, h, "hi" if i == 1 else "en", workdir / f"ov_{i}.png"
            ) if term else None
        )

        sr, dur = 24000, 2.0 + i
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        tone = (0.3 * np.sin(2 * np.pi * (300 + 100 * i) * t)).astype(np.float32)
        wav = workdir / f"a_{i}.wav"
        sf.write(wav, tone, sr)
        wavs.append(wav)
        durs.append(dur)
finally:
    gen.close()

final = assemble.assemble(scene_videos, overlays, wavs, durs, "horizontal", workdir)
print("OK:", final, final.stat().st_size, "bytes")
