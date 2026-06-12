"""Stage 4: assemble images + voiceover into the final video with ffmpeg.

Per scene: still image -> Ken Burns motion (zoompan) + fading key-term overlay.
Scenes are joined with xfade crossfades; narration WAVs are placed on the
timeline with adelay/amix and muxed in at the end.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from . import config

log = logging.getLogger(__name__)

_MOTIONS = ["zoom_in", "zoom_out", "pan_right", "pan_left"]


def _run_ffmpeg(args: list[str]):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr[-3000:]}")


def _ken_burns(motion: str, frames: int) -> str:
    """zoompan filter expression for one slide."""
    center_x = "(iw-iw/zoom)/2"
    center_y = "(ih-ih/zoom)/2"
    if motion == "zoom_in":
        z, x, y = f"min(1+0.12*on/{frames},1.12)", center_x, center_y
    elif motion == "zoom_out":
        z, x, y = f"max(1.12-0.12*on/{frames},1.0)", center_x, center_y
    elif motion == "pan_right":
        z, x, y = "1.08", f"(iw-iw/zoom)*on/{frames}", center_y
    else:  # pan_left
        z, x, y = "1.08", f"(iw-iw/zoom)*(1-on/{frames})", center_y
    return f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:fps={config.FPS}"


def render_scene_clip(image: Path, overlay: Path | None, duration_s: float,
                      orientation: str, motion: str, out_path: Path) -> Path:
    w, h = config.ORIENTATIONS[orientation]
    frames = round(duration_s * config.FPS)
    # pre-scale large so zoompan's per-frame crop doesn't jitter
    bg = (f"[0:v]scale={w * 2}:{h * 2}:flags=lanczos,"
          f"{_ken_burns(motion, frames)}:s={w}x{h},format=yuv420p[bg]")

    args: list[str] = ["-i", str(image)]
    if overlay is not None:
        fade_out_at = max(1.2, duration_s - config.CROSSFADE_S - 0.6)
        args += ["-loop", "1", "-framerate", str(config.FPS), "-i", str(overlay)]
        fc = (f"{bg};"
              f"[1:v]format=rgba,"
              f"fade=t=in:st=0.5:d=0.4:alpha=1,"
              f"fade=t=out:st={fade_out_at:.2f}:d=0.4:alpha=1[txt];"
              f"[bg][txt]overlay=0:0:shortest=1[v]")
    else:
        fc = f"{bg.replace('[bg]', '[v]')}"

    args += ["-filter_complex", fc, "-map", "[v]", "-frames:v", str(frames),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
             "-pix_fmt", "yuv420p", str(out_path)]
    _run_ffmpeg(args)
    return out_path


def crossfade_concat(clips: list[Path], durations: list[float], out_path: Path) -> Path:
    """Chain all scene clips with xfade crossfades into one silent video."""
    if len(clips) == 1:
        out_path.write_bytes(clips[0].read_bytes())
        return out_path

    fade = config.CROSSFADE_S
    args: list[str] = []
    for c in clips:
        args += ["-i", str(c)]

    parts, offset, prev = [], 0.0, "0:v"
    for i in range(1, len(clips)):
        offset += durations[i - 1] - fade
        label = f"x{i}" if i < len(clips) - 1 else "v"
        parts.append(f"[{prev}][{i}:v]xfade=transition=fade:"
                     f"duration={fade}:offset={offset:.3f}[{label}]")
        prev = label
    fc = ";".join(parts)

    args += ["-filter_complex", fc, "-map", "[v]",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
             "-pix_fmt", "yuv420p", str(out_path)]
    _run_ffmpeg(args)
    return out_path


def mix_audio(wavs: list[Path], starts_s: list[float], total_s: float,
              out_path: Path) -> Path:
    """Place each narration WAV at its slide's start time on one track."""
    args: list[str] = []
    for wav in wavs:
        args += ["-i", str(wav)]

    parts, labels = [], []
    for i, start in enumerate(starts_s):
        ms = round(start * 1000)
        parts.append(f"[{i}:a]aresample=24000,aformat=channel_layouts=mono,"
                     f"adelay={ms}:all=1[a{i}]")
        labels.append(f"[a{i}]")
    parts.append(f"{''.join(labels)}amix=inputs={len(wavs)}:normalize=0:"
                 f"dropout_transition=0,apad[mix]")
    fc = ";".join(parts)

    args += ["-filter_complex", fc, "-map", "[mix]", "-t", f"{total_s:.3f}",
             "-c:a", "pcm_s16le", str(out_path)]
    _run_ffmpeg(args)
    return out_path


def mux(video: Path, audio: Path, out_path: Path) -> Path:
    _run_ffmpeg(["-i", str(video), "-i", str(audio),
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                 "-shortest", str(out_path)])
    return out_path


def assemble(scene_images: list[Path], scene_overlays: list[Path | None],
             scene_wavs: list[Path], audio_durations: list[float],
             orientation: str, workdir: Path,
             progress=None) -> Path:
    """Full assembly. Returns the final mp4 path."""
    n = len(scene_images)
    fade = config.CROSSFADE_S

    # slide timing around each narration segment
    durations, leads = [], []
    for i, audio_d in enumerate(audio_durations):
        lead = config.AUDIO_LEAD_S + (fade / 2 if i > 0 else 0)
        tail = config.AUDIO_TAIL_S + (fade / 2 if i < n - 1 else 0)
        durations.append(max(lead + audio_d + tail, config.MIN_SLIDE_S, 2 * fade + 0.2))
        leads.append(lead)

    clips_dir = workdir / "clips"
    clips_dir.mkdir(exist_ok=True)
    clips = []
    for i in range(n):
        if progress:
            progress(f"Rendering slide {i + 1}/{n}")
        clips.append(render_scene_clip(
            scene_images[i], scene_overlays[i], durations[i], orientation,
            _MOTIONS[i % len(_MOTIONS)], clips_dir / f"scene_{i:02d}.mp4"))

    if progress:
        progress("Crossfading slides together")
    silent = crossfade_concat(clips, durations, workdir / "video_silent.mp4")

    starts, t = [], 0.0
    for i in range(n):
        starts.append(t + leads[i])
        t += durations[i] - fade
    total = sum(durations) - (n - 1) * fade

    if progress:
        progress("Mixing voiceover")
    mix = mix_audio(scene_wavs, starts, total, workdir / "voiceover.wav")
    return mux(silent, mix, workdir / "final.mp4")
