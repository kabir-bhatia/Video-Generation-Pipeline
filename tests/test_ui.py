"""Drive the running Gradio UI exactly as the browser button does."""

import sys

from gradio_client import Client

client = Client("http://127.0.0.1:7861", verbose=False)
result = client.predict(
    "Why is the sky blue?",   # topic
    30,                        # duration
    "Horizontal",              # orientation
    "English",                 # language
    "qwen2.5-0.5b",            # script model
    "debug-video",             # video model
    "local_lowmem",            # runtime profile
    "mms",                     # tts model
    "llama3.2",                # ollama model name
    api_name="/generate",
)
video, script = result
print("VIDEO:", video)
print("TITLE:", script.get("title"))
print("SCENES:", len(script.get("scenes", [])))
sys.exit(0 if video else 1)
