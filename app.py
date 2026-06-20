"""Gradio UI: topic + settings in, scene-video explainer out."""

import atexit
import logging
import os
import shutil
import signal

# pipeline must be imported before gradio: gradio pulls in huggingface_hub,
# which freezes its cache location (HF_HOME) at import time - pipeline points
# it to E: (C: is nearly full)
from pipeline import config, run_pipeline
from pipeline.run import OUTPUT_ROOT

import gradio as gr

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

_SCRIPT_CHOICES = [(label, key) for key, (label, *_rest) in config.SCRIPT_MODELS.items()]
_VIDEO_CHOICES = [(label, key) for key, (label, *_rest) in config.VIDEO_MODELS.items()]
_TTS_CHOICES = [(label, key) for key, (label, _langs) in config.TTS_MODELS.items()]
_PROFILE_CHOICES = [
    (profile["label"], key) for key, profile in config.RUNTIME_PROFILES.items()
]


def generate(topic, duration, orientation, script_model, video_model,
             runtime_profile, tts_model, ollama_model, progress=gr.Progress()):
    if not topic or not topic.strip():
        raise gr.Error("Please enter a topic.")

    def cb(msg, frac=None):
        progress(frac if frac is not None else 0.85, desc=msg)

    try:
        result = run_pipeline(
            topic=topic.strip(),
            duration_s=int(duration),
            orientation=orientation.lower(),
            language="en",
            script_model=script_model,
            video_model=video_model,
            tts_model=tts_model,
            runtime_profile=runtime_profile,
            ollama_model=ollama_model.strip() or "llama3.2",
            progress=cb,
        )
    except Exception as e:
        raise gr.Error(f"Pipeline failed: {e}") from e
    video_path = str(result["video"])
    return video_path, gr.update(value=video_path, visible=True), result["script"]


def clear_outputs():
    """Wipe all generated runs to save disk. Called on startup (guaranteed) and on
    graceful shutdown (best-effort) - generated videos must be downloaded to keep them."""
    log = logging.getLogger("app")
    try:
        shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        log.info("cleared outputs at %s", OUTPUT_ROOT)
    except Exception as e:  # never let cleanup crash the app
        log.warning("could not clear outputs: %s", e)


with gr.Blocks(title="Topic → Explainer Video") as demo:
    gr.Markdown("# 🎬 Topic → Explainer Video\n"
                "Type a topic, pick duration and orientation, and get a narrated video "
                "made from generated scene clips, overlays, and crossfades.\n\n"
                "**⚠️ Download your video before closing — outputs are cleared when the "
                "app restarts to save disk space.**")

    with gr.Row():
        with gr.Column(scale=1):
            topic = gr.Textbox(label="Topic",
                               placeholder='e.g. "Explain inflation in 60 seconds"')
            duration = gr.Slider(minimum=30, maximum=240, step=5, value=60,
                                 label="Target duration (seconds)")
            orientation = gr.Radio(["Horizontal", "Vertical"], value="Horizontal",
                                   label="Orientation")

            with gr.Accordion("Advanced: swap models", open=False):
                script_model = gr.Dropdown(_SCRIPT_CHOICES,
                                           value=config.DEFAULT_SCRIPT_MODEL,
                                           label="Script model")
                ollama_model = gr.Textbox(value="llama3.2",
                                          label="Ollama model name (if Ollama selected)")
                video_model = gr.Dropdown(_VIDEO_CHOICES,
                                          value=config.DEFAULT_VIDEO_MODEL,
                                          label="Scene video model")
                runtime_profile = gr.Dropdown(_PROFILE_CHOICES,
                                              value=config.DEFAULT_RUNTIME_PROFILE,
                                              label="Runtime profile")
                tts_model = gr.Dropdown(_TTS_CHOICES,
                                        value=config.DEFAULT_TTS_MODEL,
                                        label="Voice model")

            go = gr.Button("Generate video", variant="primary")

        with gr.Column(scale=1):
            video = gr.Video(label="Final video")
            download = gr.DownloadButton("⬇ Download MP4", visible=False)
            script_view = gr.JSON(label="Generated script")

    go.click(generate,
             inputs=[topic, duration, orientation,
                     script_model, video_model, runtime_profile, tts_model, ollama_model],
             outputs=[video, download, script_view])

if __name__ == "__main__":
    # Env-configurable so the same app.py works locally and on a headless VM:
    #   GRADIO_SERVER_NAME=0.0.0.0   bind on all interfaces (LAN/VM access)
    #   GRADIO_SHARE=1               create a temporary public gradio.live link
    #   GRADIO_SERVER_PORT=7860      port to listen on
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    share = os.environ.get("GRADIO_SHARE", "").lower() in ("1", "true", "yes")
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))

    # disk hygiene: guaranteed sweep on startup (runs no matter how the previous
    # instance died, incl. VM stop/SIGKILL), plus best-effort cleanup on a graceful exit.
    clear_outputs()
    atexit.register(clear_outputs)

    def _on_signal(signum, _frame):
        clear_outputs()
        raise SystemExit(0)

    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _on_signal)
        except (ValueError, OSError):
            pass  # not in main thread / unsupported platform

    demo.queue().launch(
        server_name=server_name,
        server_port=port,
        share=share,
        inbrowser=os.environ.get("GRADIO_INBROWSER", "").lower() in ("1", "true", "yes"),
    )
