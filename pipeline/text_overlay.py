"""Render key-term text overlays as transparent full-frame PNGs.

Uses ffmpeg drawtext (libharfbuzz + libfribidi) rather than PIL because the
bundled shaper renders complex scripts like Devanagari correctly, which the
Pillow wheels (no Raqm) do not.

Fonts are auto-detected per platform; override with the environment variables
VIDEO_PIPELINE_FONT_EN / VIDEO_PIPELINE_FONT_HI (path to a .ttf/.ttc file).
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

# bold/regular sans for Latin, plus a Devanagari-capable font for Hindi
_FONT_CANDIDATES = {
    "en": [
        # Windows
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
    ],
    "hi": [
        # Windows (Nirmala UI ships with Windows 8+)
        r"C:\Windows\Fonts\Nirmala.ttc",
        r"C:\Windows\Fonts\mangal.ttf",
        # Linux (fonts-noto / google-noto packages)
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/google-noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/KohinoorDevanagari.ttc",
        "/System/Library/Fonts/Supplemental/DevanagariMT.ttc",
    ],
}
_ENV_OVERRIDES = {"en": "VIDEO_PIPELINE_FONT_EN", "hi": "VIDEO_PIPELINE_FONT_HI"}
_resolved: dict[str, str] = {}


def _find_font(language: str) -> str:
    if language in _resolved:
        return _resolved[language]

    env_var = _ENV_OVERRIDES[language]
    override = os.environ.get(env_var, "")
    candidates = ([override] if override else []) + _FONT_CANDIDATES[language]
    for path in candidates:
        if Path(path).is_file():
            log.info("text overlays (%s) use font %s", language, path)
            _resolved[language] = path
            return path
    raise RuntimeError(
        f"No suitable font found for '{language}' text overlays. "
        f"Set the {env_var} environment variable to a .ttf/.ttc font file"
        + (" with Devanagari support (e.g. Noto Sans Devanagari)."
           if language == "hi" else "."))


def _ff_escape_path(p: str) -> str:
    """Escape a path for use inside a drawtext option value."""
    return p.replace("\\", "/").replace(":", r"\:")


def render_overlay(text: str, video_w: int, video_h: int,
                   language: str, out_path: Path) -> Path:
    """Bottom-centered label box on a transparent frame-sized PNG."""
    textfile = out_path.with_suffix(".txt")
    textfile.write_text(text, encoding="utf-8")

    fontsize = max(28, video_w // 22)
    margin = video_h // 12
    drawtext = (
        f"drawtext=textfile='{_ff_escape_path(str(textfile))}'"
        f":fontfile='{_ff_escape_path(_find_font(language))}'"
        f":fontsize={fontsize}:fontcolor=white"
        f":box=1:boxcolor=0x0A0A14@0.65:boxborderw={fontsize // 2}"
        f":x=(w-text_w)/2:y=h-text_h-{margin}"
    )
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", f"color=c=black@0.0:s={video_w}x{video_h},format=rgba",
           "-vf", drawtext, "-frames:v", "1", str(out_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"overlay render failed:\n{proc.stderr[-2000:]}")
    return out_path
