"""Stage 1: turn a topic into a structured scene-by-scene script.

A script is a dict:
    {"title": str,
     "scenes": [{"narration": str, "video_prompt": str,
                 "motion_hint": str, "key_term": str}, ...]}

narration  - spoken voiceover text (English or Hindi)
video_prompt - English text-to-video prompt for the scene
motion_hint - English motion/camera hint for the scene clip
key_term   - short on-screen text overlay ("" to skip)
"""

from __future__ import annotations

import json
import logging
import re

from . import config

log = logging.getLogger(__name__)

_SYSTEM = (
    "You write scripts for short slideshow explainer videos. "
    "Reply with a single JSON object and nothing else."
)

_PROMPT = """Write the script for a short explainer video about: "{topic}"

Rules:
- Exactly {n_scenes} scenes.
- Each scene's "narration" is about {words_per_scene} words of spoken voiceover. {lang_rule}
- Scene 1 hooks the viewer; the last scene gives a crisp takeaway.
- Each scene's "video_prompt" is an English text-to-video prompt describing a single clear scene for that moment (style: clean modern digital illustration, no text in frame, no people's faces close-up).
- Each scene's "motion_hint" is a short English instruction for how the scene should move or how the camera should move.
- Each scene's "key_term" is a 1-4 word on-screen label for the scene's main idea{key_term_lang}, or "" if none fits.

Return JSON exactly in this shape:
{{"title": "...", "scenes": [{{"narration": "...", "video_prompt": "...", "motion_hint": "...", "key_term": "..."}}]}}"""

_LANG_RULES = {
    "en": "Write the narration in simple, conversational English.",
    "hi": "Write the narration in simple, natural Hindi (Devanagari script). "
          "Common English loanwords may stay in English.",
}
_KEY_TERM_LANG = {"en": "", "hi": " (in Hindi)"}


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


def _validate(script: dict, n_scenes: int, min_scenes: int = 2) -> dict:
    scenes = script.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("script has no scenes")
    clean = []
    for s in scenes[:n_scenes]:
        narration = str(s.get("narration", "")).strip()
        if not narration:
            continue
        clean.append({
            "narration": narration,
            "video_prompt": str(s.get("video_prompt", "") or s.get("image_prompt", "")).strip()
                            or "clean modern digital illustration, abstract concept, cinematic motion",
            "motion_hint": str(s.get("motion_hint", "")).strip() or "gentle forward motion",
            "key_term": str(s.get("key_term", "")).strip(),
        })
    if len(clean) < min_scenes:
        raise ValueError(f"script has fewer than {min_scenes} usable scenes")
    return {"title": str(script.get("title", "")).strip(), "scenes": clean}


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
            max_new_tokens=1800,
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
            lang_rule=_LANG_RULES[language]), creative=True)
        try:
            extra = _validate(_extract_json(text), missing, min_scenes=1)
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("continuation parse failed: %s", e)
            continue
        script["scenes"].extend(extra["scenes"][:missing])

    return script


def _generate_template(topic: str, language: str, n_scenes: int) -> dict:
    """No-LLM fallback so the pipeline still runs end to end. Placeholder content."""
    log.warning("using template script backend - narration will be generic placeholder text")
    if language == "hi":
        lines = [f"आज हम समझेंगे: {topic}।",
                 f"{topic} हमारी रोज़मर्रा की ज़िंदगी से जुड़ा एक अहम विषय है।",
                 "इसके पीछे कुछ आसान सिद्धांत काम करते हैं।",
                 f"तो यही था {topic} का एक छोटा परिचय। धन्यवाद!"]
        terms = ["परिचय", "महत्व", "सिद्धांत", "निष्कर्ष"]
    else:
        lines = [f"Today, let's understand: {topic}.",
                 f"{topic} touches our everyday lives in important ways.",
                 "A few simple principles explain how it works.",
                 f"And that was a quick introduction to {topic}. Thanks for watching!"]
        terms = ["Introduction", "Why it matters", "How it works", "Takeaway"]
    scenes = []
    for i in range(min(n_scenes, len(lines))):
        scenes.append({
            "narration": lines[i],
            "video_prompt": f"clean modern digital illustration about {topic}, "
                            f"concept art, vibrant colors, no text in frame",
            "motion_hint": "slow cinematic camera move with subtle subject motion",
            "key_term": terms[i],
        })
    return {"title": topic, "scenes": scenes}


# ------------------------------------------------------------------ translation

def _nllb_translator(src_lang: str, tgt_lang: str):
    """Returns tr(text) -> text using NLLB-200 (open-source MT model)."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_id = "facebook/nllb-200-distilled-600M"
    log.info("loading translator %s (%s -> %s)", model_id, src_lang, tgt_lang)
    tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang=src_lang)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == "cuda" else torch.bfloat16,
    ).to(device)
    tgt = tokenizer.convert_tokens_to_ids(tgt_lang)

    def tr(text: str) -> str:
        if not text.strip():
            return text
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=512).to(device)
        out = model.generate(**inputs, forced_bos_token_id=tgt, max_new_tokens=512)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    return tr


def translate_topic_to_english(data: dict) -> dict:
    """{"text": <hindi topic>} -> {"text": <english topic>} so the English
    script model actually understands what to write about."""
    tr = _nllb_translator("hin_Deva", "eng_Latn")
    return {"text": tr(data["text"])}


def translate_to_hindi(script: dict) -> dict:
    """Translate narration + key terms to Hindi with NLLB-200.

    Small local LLMs write incoherent Hindi directly; English generation
    followed by a dedicated open-source MT model gives far better scripts.
    Video prompts and motion hints stay in English for the video model.
    """
    tr = _nllb_translator("eng_Latn", "hin_Deva")

    for scene in script["scenes"]:
        # translate sentence by sentence - NLLB quality drops on long inputs
        sentences = re.split(r"(?<=[.!?])\s+", scene["narration"])
        scene["narration"] = " ".join(tr(s) for s in sentences if s.strip())
        scene["key_term"] = tr(scene["key_term"])
    script["title"] = tr(script.get("title", ""))
    return script


# ------------------------------------------------------------------ entry point

def generate_script(topic: str, language: str, duration_s: int,
                    backend_key: str, ollama_model: str = "llama3.2") -> dict:
    n_scenes = max(3, min(18, round(duration_s / config.SECONDS_PER_SCENE)))
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


def needs_translation(language: str, backend_key: str) -> bool:
    """Local HF models write incoherent Hindi: generate in English instead and
    translate afterwards (run as a separate stage so each model gets a fresh
    process). Ollama models (llama3.2 etc.) handle Hindi natively."""
    return language == "hi" and config.SCRIPT_MODELS[backend_key][1] == "hf"
