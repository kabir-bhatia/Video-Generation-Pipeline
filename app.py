"""Gradio UI: topic + duration + orientation + language in, downloadable video out."""

import logging

# pipeline must be imported before gradio: gradio pulls in huggingface_hub,
# which freezes its cache location (HF_HOME) at import time - pipeline points
# it to E: (C: is nearly full)
from pipeline import config, run_pipeline

import gradio as gr

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

_SCRIPT_CHOICES = [(label, key) for key, (label, *_rest) in config.SCRIPT_MODELS.items()]
_IMAGE_CHOICES = [(label, key) for key, (label, *_rest) in config.IMAGE_MODELS.items()]
_TTS_CHOICES = [(label, key) for key, (label, _langs) in config.TTS_MODELS.items()]


def generate(topic, duration, orientation, language, script_model, image_model,
             tts_model, ollama_model, progress=gr.Progress()):
    if not topic or not topic.strip():
        raise gr.Error("Please enter a topic.")
    lang_key = "hi" if language == "Hindi" else "en"
    if lang_key not in config.TTS_MODELS[tts_model][1]:
        raise gr.Error(f"The '{tts_model}' voice doesn't support {language}. "
                       "Use 'mms' for Hindi.")

    def cb(msg, frac=None):
        progress(frac if frac is not None else 0.85, desc=msg)

    try:
        result = run_pipeline(
            topic=topic.strip(),
            duration_s=int(duration),
            orientation=orientation.lower(),
            language=lang_key,
            script_model=script_model,
            image_model=image_model,
            tts_model=tts_model,
            ollama_model=ollama_model.strip() or "llama3.2",
            progress=cb,
        )
    except Exception as e:
        raise gr.Error(f"Pipeline failed: {e}") from e
    return str(result["video"]), result["script"]


with gr.Blocks(title="Topic → Explainer Video") as demo:
    gr.Markdown("# 🎬 Topic → Explainer Video\n"
                "Type a topic, pick duration / orientation / language, and get a "
                "narrated slideshow video with AI images, motion, and crossfades.")

    with gr.Row():
        with gr.Column(scale=1):
            topic = gr.Textbox(label="Topic",
                               placeholder='e.g. "Explain inflation in 60 seconds"')
            duration = gr.Dropdown([30, 60, 90, 120, 180], value=60,
                                   label="Target duration (seconds)")
            orientation = gr.Radio(["Horizontal", "Vertical"], value="Horizontal",
                                   label="Orientation")
            language = gr.Radio(["English", "Hindi"], value="English",
                                label="Voiceover language")

            with gr.Accordion("Advanced: swap models", open=False):
                script_model = gr.Dropdown(_SCRIPT_CHOICES,
                                           value=config.DEFAULT_SCRIPT_MODEL,
                                           label="Script model")
                ollama_model = gr.Textbox(value="llama3.2",
                                          label="Ollama model name (if Ollama selected)")
                image_model = gr.Dropdown(_IMAGE_CHOICES,
                                          value=config.DEFAULT_IMAGE_MODEL,
                                          label="Image model")
                tts_model = gr.Dropdown(_TTS_CHOICES,
                                        value=config.DEFAULT_TTS_MODEL,
                                        label="Voice model")

            go = gr.Button("Generate video", variant="primary")

        with gr.Column(scale=1):
            video = gr.Video(label="Final video (right-click ⋮ to download)")
            script_view = gr.JSON(label="Generated script")

    go.click(generate,
             inputs=[topic, duration, orientation, language,
                     script_model, image_model, tts_model, ollama_model],
             outputs=[video, script_view])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)
