"""Stage 1: turn a topic into a structured scene-by-scene script.

A script is a dict:
    {"title": str,
     "visual_style": str,
     "scenes": [{"narration": str, "video_prompt": str,
                 "motion_hint": str, "key_term": str}, ...]}

narration    - spoken voiceover text (English)
visual_style - one global look (art style, palette, lighting, mood) reused in every
               scene so the whole video is visually coherent
video_prompt - detailed English text-to-video prompt for the scene; the global
               visual_style is prepended to it in _validate()
motion_hint  - English motion/camera hint for the scene clip
key_term     - short on-screen text overlay ("" to skip)
"""

from __future__ import annotations

import json
import logging
import re

from . import config

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are the art director and scriptwriter for short explainer videos. "
    "You think visually and write vivid, concrete scene descriptions for a "
    "text-to-video model. Reply with a single JSON object and nothing else."
)

_PROMPT = """Plan a short explainer video about: "{topic}"

First decide ONE consistent "visual_style" for the whole video and reuse that look in every
scene so the video is coherent. Write it as one sentence covering: art style (e.g. clean 3D
render, flat vector illustration, cinematic photorealistic, papercraft), color palette,
lighting, and overall mood. Never render text/words, logos, or watermarks in the frame, and
avoid close-up human faces.

Then write exactly {n_scenes} scenes. For each scene provide:
- "narration": about {words_per_scene} words of spoken voiceover. {lang_rule}
- "video_prompt": a vivid 40-70 word description of ONE concrete shot that fits the visual
  style. Name the main SUBJECT, the SETTING/background, and the ACTION (what happens). Be
  specific and visual - objects, colors, composition, depth. No text, captions, or logos.
- "motion_hint": a short camera or subject motion (e.g. "slow dolly-in", "gentle pan left",
  "subject rises upward", "particles drift past camera").
- "key_term": a 1-4 word on-screen label for the scene's main idea{key_term_lang}, or "".

Scene 1 hooks the viewer; the final scene gives a crisp takeaway.

Return JSON exactly in this shape:
{{"title": "...", "visual_style": "...", "scenes": [{{"narration": "...", "video_prompt": "...", "motion_hint": "...", "key_term": "..."}}]}}"""

# English-only pipeline for now; Hindi narration + translation will return later.
_LANG_RULES = {
    "en": "Write the narration in simple, conversational English.",
}
_KEY_TERM_LANG = {"en": ""}


def _build_prompt(topic: str, language: str, n_scenes: int, words_per_scene: int) -> str:
    return _PROMPT.format(
        topic=topic,
        n_scenes=n_scenes,
        words_per_scene=words_per_scene,
        lang_rule=_LANG_RULES[language],
        key_term_lang=_KEY_TERM_LANG[language],
    )


# ------------------------------------------------------------------ parsing

def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of LLM output, tolerating fences and chatter."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in model output")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = text[start : i + 1]
                blob = re.sub(r",\s*([}\]])", r"\1", blob)  # trailing commas
                return json.loads(blob)
    raise ValueError("unbalanced JSON in model output")


def _compose_prompt(visual_style: str, video_prompt: str) -> str:
    """Prepend the global visual style so every scene shares one coherent look."""
    base = video_prompt or "a clear concept illustration, dynamic composition"
    style = visual_style.strip().rstrip(".")
    if not style:
        return base
    # avoid doubling the style if the model already echoed it
    if base.lower().startswith(style.lower()[:24]):
        return base
    return f"{style}. {base}"


def _validate(script: dict, n_scenes: int, min_scenes: int = 2,
              visual_style: str = "") -> dict:
    scenes = script.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("script has no scenes")
    style = str(script.get("visual_style", "") or visual_style).strip()
    clean = []
    for s in scenes[:n_scenes]:
        narration = str(s.get("narration", "")).strip()
        if not narration:
            continue
        raw_prompt = str(s.get("video_prompt", "") or s.get("image_prompt", "")).strip()
        clean.append({
            "narration": narration,
            "video_prompt": _compose_prompt(style, raw_prompt),
            "motion_hint": str(s.get("motion_hint", "")).strip() or "slow gentle camera move",
            "key_term": str(s.get("key_term", "")).strip(),
        })
    if len(clean) < min_scenes:
        raise ValueError(f"script has fewer than {min_scenes} usable scenes")
    return {"title": str(script.get("title", "")).strip(),
            "visual_style": style, "scenes": clean}


# ------------------------------------------------------------------ backends
#
# A backend is anything with .ask(user_prompt, creative: bool) -> str and
# .close(). The orchestration in generate_script() may call .ask() several
# times (retries + continuation requests), so models stay loaded until then.

class _HFChat:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("loading script model %s", model_id)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # bfloat16/float16: halves memory, and this machine is constrained on
        # both RAM and VRAM. Decide the device by measuring, not by trying:
        # an OOM raised mid-move poisons the CUDA context for later calls.
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     torch_dtype=torch.bfloat16)
        self.device = "cpu"
        if torch.cuda.is_available():
            need = sum(p.numel() * p.element_size() for p in model.parameters())
            free, _total = torch.cuda.mem_get_info()
            if need * 1.25 < free:  # weights + KV cache + activations headroom
                self.device = "cuda"
                model = model.to("cuda", dtype=torch.float16)
            else:
                log.info("script model needs ~%.1f GB, only %.1f GB VRAM free - using CPU",
                         need / 1e9, free / 1e9)
        self.model = model

    def ask(self, user_prompt: str, creative: bool = False) -> str:
        messages = [{"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user_prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]
        out = self.model.generate(
            **inputs,
            max_new_tokens=2400,  # richer, longer per-scene video_prompts
            do_sample=creative,
            temperature=0.8 if creative else None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

    def close(self):
        # free the script model before scene-video generation loads
        # on a tight-memory machine
        import gc
        del self.model
        gc.collect()
        if self.device == "cuda":
            self._torch.cuda.empty_cache()


class _OllamaChat:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def ask(self, user_prompt: str, creative: bool = False) -> str:
        import requests

        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": self.model_name,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.8 if creative else 0.3},
                "messages": [{"role": "system", "content": _SYSTEM},
                             {"role": "user", "content": user_prompt}],
            },
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def close(self):
        pass


_CONTINUE_PROMPT = """The explainer video script about "{topic}" titled "{title}" currently ends with this scene:

"{last_narration}"

Write {n_more} MORE scenes that continue it (do not repeat earlier content; the
final one of these gives a crisp takeaway). Same rules as before: narration of
about {words_per_scene} words per scene. {lang_rule}
Keep the SAME visual style as the rest of the video: "{visual_style}". Each scene's
"video_prompt" is a vivid 40-70 word shot (subject, setting, action) in that style.
Return JSON exactly in this shape:
{{"scenes": [{{"narration": "...", "video_prompt": "...", "motion_hint": "...", "key_term": "..."}}]}}"""


def _generate_llm(chat, topic: str, language: str,
                  n_scenes: int, words_per_scene: int) -> dict:
    """Ask for the full script, then top up with continuation requests when the
    model returns fewer scenes than asked (small models routinely do)."""
    script = None
    last_err: Exception | None = None
    for attempt in range(2):
        text = chat.ask(_build_prompt(topic, language, n_scenes, words_per_scene),
                        creative=attempt > 0)
        try:
            script = _validate(_extract_json(text), n_scenes)
            break
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            log.warning("script parse attempt %d failed: %s", attempt + 1, e)
    if script is None:
        raise RuntimeError(f"script model produced unparseable output: {last_err}")

    for _ in range(3):
        missing = n_scenes - len(script["scenes"])
        if missing <= 0:
            break
        log.info("script has %d/%d scenes, requesting %d more",
                 len(script["scenes"]), n_scenes, missing)
        text = chat.ask(_CONTINUE_PROMPT.format(
            topic=topic, title=script["title"] or topic,
            last_narration=script["scenes"][-1]["narration"],
            n_more=missing, words_per_scene=words_per_scene,
            lang_rule=_LANG_RULES[language],
            visual_style=script.get("visual_style", "")), creative=True)
        try:
            extra = _validate(_extract_json(text), missing, min_scenes=1,
                              visual_style=script.get("visual_style", ""))
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("continuation parse failed: %s", e)
            continue
        script["scenes"].extend(extra["scenes"][:missing])

    return script


def _generate_template(topic: str, language: str, n_scenes: int) -> dict:
    """No-LLM fallback so the pipeline still runs end to end. Placeholder content."""
    log.warning("using template script backend - narration will be generic placeholder text")
    lines = [f"Today, let's understand: {topic}.",
             f"{topic} touches our everyday lives in important ways.",
             "A few simple principles explain how it works.",
             f"And that was a quick introduction to {topic}. Thanks for watching!"]
    terms = ["Introduction", "Why it matters", "How it works", "Takeaway"]
    visual_style = ("clean modern 3D illustration, vibrant saturated palette, soft studio "
                    "lighting, shallow depth of field, friendly explainer mood")
    scenes = []
    for i in range(min(n_scenes, len(lines))):
        scenes.append({
            "narration": lines[i],
            "video_prompt": _compose_prompt(
                visual_style,
                f"a clear conceptual scene illustrating {topic}, central subject with "
                f"supporting background elements, dynamic composition"),
            "motion_hint": "slow cinematic camera move with subtle subject motion",
            "key_term": terms[i],
        })
    return {"title": topic, "visual_style": visual_style, "scenes": scenes}


# ------------------------------------------------------------------ entry point

def generate_script(topic: str, language: str, duration_s: int,
                    backend_key: str, ollama_model: str = "llama3.2") -> dict:
    n_scenes = max(3, min(config.MAX_SCENES, round(duration_s / config.SECONDS_PER_SCENE)))
    total_words = duration_s * config.WORDS_PER_SECOND[language]
    words_per_scene = max(10, round(total_words / n_scenes))

    label, kind, model_id = config.SCRIPT_MODELS[backend_key]
    log.info("script backend=%s scenes=%d words/scene=%d", label, n_scenes, words_per_scene)

    if kind == "template":
        return _generate_template(topic, language, n_scenes)
    chat = _HFChat(model_id) if kind == "hf" else _OllamaChat(ollama_model)
    try:
        return _generate_llm(chat, topic, language, n_scenes, words_per_scene)
    finally:
        chat.close()
