"""Component-level validation helpers for constrained machines."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from . import assemble, config, text_overlay, tts, video_gen


def run_component_checks(video_model: str = config.DEFAULT_VIDEO_MODEL,
                         runtime_profile: str = config.DEFAULT_RUNTIME_PROFILE,
                         tts_model: str = config.DEFAULT_TTS_MODEL,
                         language: str = "en") -> dict:
    report = {
        "video_backend": None,
        "tts": None,
        "overlay": None,
        "assembly": None,
    }

    report["video_backend"] = video_gen.validate_backend(video_model, runtime_profile)

    voice = tts.make_tts(tts_model, language)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        wav = tmp / "sample.wav"
        duration = voice.synthesize("This is a short validation line.", wav)
        report["tts"] = {"status": "ok", "sample_rate_seconds": round(duration, 3)}

        w, h = config.ORIENTATIONS["horizontal"]
        overlay = text_overlay.render_overlay("Validation", w, h, language, tmp / "overlay.png")
        report["overlay"] = {"status": "ok", "path": str(overlay)}

        scene_clip = tmp / "scene.mp4"
        gen = video_gen.make_video_generator("debug-video", "local_lowmem")
        try:
            gen.generate("Validation scene", "gentle camera drift", "horizontal", 7, scene_clip)
        finally:
            gen.close()
        rendered = assemble.render_scene_clip(
            scene_clip, overlay, 3.2, "horizontal", tmp / "rendered.mp4"
        )
        report["assembly"] = {"status": "ok", "path": str(rendered)}

    return report


if __name__ == "__main__":
    print(json.dumps(run_component_checks(), indent=2))
