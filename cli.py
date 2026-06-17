"""Command-line entry point, e.g.:

python cli.py "Explain inflation in 60 seconds" --duration 60 --language en
"""

import argparse
import logging

from pipeline import config, run_pipeline


def main():
    p = argparse.ArgumentParser(description="Topic -> explainer video")
    p.add_argument("topic")
    p.add_argument("--duration", type=int, default=60, help="target seconds (30-180)")
    p.add_argument("--orientation", choices=list(config.ORIENTATIONS),
                   default="horizontal")
    p.add_argument("--language", choices=list(config.LANGUAGES), default="en")
    p.add_argument("--script-model", choices=list(config.SCRIPT_MODELS),
                   default=config.DEFAULT_SCRIPT_MODEL)
    p.add_argument("--video-model", choices=list(config.VIDEO_MODELS),
                   default=config.DEFAULT_VIDEO_MODEL)
    p.add_argument("--runtime-profile", choices=list(config.RUNTIME_PROFILES),
                   default=config.DEFAULT_RUNTIME_PROFILE)
    p.add_argument("--tts-model", choices=list(config.TTS_MODELS),
                   default=config.DEFAULT_TTS_MODEL)
    p.add_argument("--ollama-model", default="llama3.2")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    result = run_pipeline(
        topic=args.topic, duration_s=args.duration, orientation=args.orientation,
        language=args.language, script_model=args.script_model,
        video_model=args.video_model, tts_model=args.tts_model,
        runtime_profile=args.runtime_profile,
        ollama_model=args.ollama_model, seed=args.seed,
    )
    print(f"\nDone: {result['video']}")


if __name__ == "__main__":
    main()
