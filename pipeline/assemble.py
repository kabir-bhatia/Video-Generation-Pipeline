"""Stage 4: assemble scene clips + voiceover into the final video with ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

from . import config


def _run_ffmpeg(args: list[str]):
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr[-3000:]}")


def _make_boomerang(scene_video: Path, out_path: Path) -> Path:
    """Forward + reversed copy so a short clip can fill a longer scene without the
    jarring hard cut-back that `-stream_loop` alone produces. The motion ping-pongs
    seamlessly, which reads far better than a visible loop seam."""
    _run_ffmpeg([
        "-i", str(scene_video),
        "-filter_complex", "[0:v]split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1[v]",
        "-map", "[v]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an",
        str(out_path),
    ])
    return out_path


def render_scene_clip(scene_video: Path, overlay: Path | None, duration_s: float,
                      orientation: str, out_path: Path) -> Path:
    w, h = config.ORIENTATIONS[orientation]
    fps = config.FPS

    # ping-pong source hides repetition when the clip is shorter than the scene
    boomerang = _make_boomerang(scene_video, out_path.with_name(out_path.stem + "_bm.mp4"))
    args: list[str] = ["-stream_loop", "-1", "-i", str(boomerang)]

    # subtle slow zoom (Ken Burns) adds life and further masks any residual loop
    zoom = (
        f"zoompan=z='min(zoom+0.0007,1.12)':d=1:fps={fps}:s={w}x{h}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    )
    base = (
        f"[0:v]fps={fps},scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:setsar=1,{zoom},"
        f"trim=duration={duration_s:.3f},setpts=N/({fps}*TB)[base]"
    )
    if overlay is not None:
        fade_out_at = max(1.2, duration_s - config.CROSSFADE_S - 0.6)
        args += ["-loop", "1", "-framerate", str(fps), "-i", str(overlay)]
        fc = (
            f"{base};"
            f"[1:v]format=rgba,"
            f"fade=t=in:st=0.5:d=0.4:alpha=1,"
            f"fade=t=out:st={fade_out_at:.2f}:d=0.4:alpha=1[txt];"
            f"[base][txt]overlay=0:0:shortest=1,format=yuv420p[v]"
        )
    else:
        fc = f"{base};[base]format=yuv420p[v]"

    args += [
        "-filter_complex", fc,
        "-map", "[v]",
        "-t", f"{duration_s:.3f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "19",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    _run_ffmpeg(args)
    return out_path


def crossfade_concat(clips: list[Path], durations: list[float], out_path: Path) -> Path:
    """Chain all scene clips with xfade crossfades into one silent video."""
    if len(clips) == 1:
        out_path.write_bytes(clips[0].read_bytes())
        return out_path

    fade = config.CROSSFADE_S
    args: list[str] = []
    for clip in clips:
        args += ["-i", str(clip)]

    parts, offset, prev = [], 0.0, "0:v"
    for i in range(1, len(clips)):
        offset += durations[i - 1] - fade
        label = f"x{i}" if i < len(clips) - 1 else "v"
        parts.append(
            f"[{prev}][{i}:v]xfade=transition=fade:duration={fade}:offset={offset:.3f}[{label}]"
        )
        prev = label
    args += [
        "-filter_complex", ";".join(parts),
        "-map", "[v]",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "19",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    _run_ffmpeg(args)
    return out_path


def mix_audio(wavs: list[Path], starts_s: list[float], total_s: float,
              out_path: Path) -> Path:
    args: list[str] = []
    for wav in wavs:
        args += ["-i", str(wav)]

    parts, labels = [], []
    for i, start in enumerate(starts_s):
        ms = round(start * 1000)
        parts.append(
            f"[{i}:a]aresample=24000,aformat=channel_layouts=mono,adelay={ms}:all=1[a{i}]"
        )
        labels.append(f"[a{i}]")
    parts.append(
        f"{''.join(labels)}amix=inputs={len(wavs)}:normalize=0:dropout_transition=0,apad[mix]"
    )

    args += [
        "-filter_complex", ";".join(parts),
        "-map", "[mix]",
        "-t", f"{total_s:.3f}",
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    _run_ffmpeg(args)
    return out_path


def mux(video: Path, audio: Path, out_path: Path) -> Path:
    _run_ffmpeg([
        "-i", str(video),
        "-i", str(audio),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "160k",
        "-shortest",
        str(out_path),
    ])
    return out_path


def assemble(scene_videos: list[Path], scene_overlays: list[Path | None],
             scene_wavs: list[Path], audio_durations: list[float],
             orientation: str, workdir: Path, progress=None) -> Path:
    """Full assembly. Returns the final mp4 path."""
    n = len(scene_videos)
    fade = config.CROSSFADE_S

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
            progress(f"Rendering scene clip {i + 1}/{n}")
        clips.append(
            render_scene_clip(
                scene_videos[i],
                scene_overlays[i],
                durations[i],
                orientation,
                clips_dir / f"scene_{i:02d}.mp4",
            )
        )

    if progress:
        progress("Crossfading scene clips")
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
