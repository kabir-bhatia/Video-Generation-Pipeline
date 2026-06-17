"""Topic-to-video explainer pipeline.

Stages: script generation -> scene-video generation -> TTS voiceover -> ffmpeg assembly.
Every stage has swappable open-source backends (see pipeline.config).
"""

import os
from pathlib import Path

# Keep model downloads inside the project (self-contained, easy to clean up).
# Set HF_HOME yourself before launching to use a shared cache instead.
# Must be set before transformers/diffusers are imported anywhere.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "models_cache"))
# the xet downloader (Rust) can abort the whole process when its allocations
# fail on low-RAM machines; the plain HTTP downloader is slightly slower but frugal
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from .run import run_pipeline  # noqa: E402,F401
