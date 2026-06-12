"""Smoke-test overlays + ffmpeg assembly with synthetic assets (no AI models)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw

from pipeline import assemble, config, text_overlay

workdir = Path(__file__).parent / "tmp"
(workdir / "clips").mkdir(parents=True, exist_ok=True)

W, H = config.ORIENTATIONS["horizontal"]
colors = [(180, 60, 60), (60, 140, 80), (60, 80, 180)]
terms = ["Inflation", "मुद्रास्फीति", ""]

images, overlays, wavs, durs = [], [], [], []
for i, (color, term) in enumerate(zip(colors, terms)):
    img = Image.new("RGB", (W, H), color)
    d = ImageDraw.Draw(img)
    for x in range(0, W, 80):  # grid so motion is visible
        d.line([(x, 0), (x, H)], fill=(255, 255, 255), width=2)
    p = workdir / f"img_{i}.png"
    img.save(p)
    images.append(p)

    overlays.append(
        text_overlay.render_overlay(term, W, H, "hi" if i == 1 else "en",
                                    workdir / f"ov_{i}.png") if term else None)

    sr, dur = 24000, 2.0 + i
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * (300 + 100 * i) * t)).astype(np.float32)
    wp = workdir / f"a_{i}.wav"
    sf.write(wp, tone, sr)
    wavs.append(wp)
    durs.append(dur)

final = assemble.assemble(images, overlays, wavs, durs, "horizontal", workdir)
print("OK:", final, final.stat().st_size, "bytes")
