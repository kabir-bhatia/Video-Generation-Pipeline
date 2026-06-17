"""Stage 3: synthesize the voiceover, one WAV per scene.

Backends:
  mms    - Meta MMS-TTS (facebook/mms-tts-eng / facebook/mms-tts-hin). Open source,
           supports the English/Hindi toggle. Hindi text is romanized with uroman
           when the tokenizer requires it.
  kokoro - Kokoro-82M (English only, Apache-2.0). Optional: pip install kokoro
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+")  # । = devanagari danda
_PAUSE_S = 0.25


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip())]
    return [p for p in parts if p]


class MMSTTS:
    def __init__(self, language: str):
        import torch
        from transformers import AutoTokenizer, VitsModel

        model_id = {"en": "facebook/mms-tts-eng", "hi": "facebook/mms-tts-hin"}[language]
        log.info("loading TTS model %s", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = VitsModel.from_pretrained(model_id)
        forced_device = os.environ.get("VIDEO_PIPELINE_TTS_DEVICE", "").strip().lower()
        if forced_device not in {"", "cpu", "cuda"}:
            raise ValueError("VIDEO_PIPELINE_TTS_DEVICE must be 'cpu', 'cuda', or unset")
        self.device = forced_device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            log.warning("TTS requested cuda but CUDA is unavailable; falling back to cpu")
            self.device = "cpu"
        self.model.to(self.device)
        self.sample_rate = self.model.config.sampling_rate

        self.uroman = None
        if getattr(self.tokenizer, "is_uroman", False):
            import uroman
            self.uroman = uroman.Uroman()

    def _prep(self, text: str) -> str:
        if self.uroman is not None:
            text = str(self.uroman.romanize_string(text))
        return text

    def synthesize(self, text: str, out_path: Path) -> float:
        """Returns duration in seconds."""
        import torch

        chunks = []
        pause = np.zeros(int(_PAUSE_S * self.sample_rate), dtype=np.float32)
        for sentence in _split_sentences(text):
            inputs = self.tokenizer(self._prep(sentence), return_tensors="pt").to(self.device)
            with torch.no_grad():
                wav = self.model(**inputs).waveform[0].cpu().numpy().astype(np.float32)
            chunks.extend([wav, pause])
        if not chunks:
            chunks = [np.zeros(self.sample_rate, dtype=np.float32)]
        audio = np.concatenate(chunks)
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio * (0.9 / max(peak, 0.9))
        sf.write(out_path, audio, self.sample_rate)
        return len(audio) / self.sample_rate


class KokoroTTS:
    def __init__(self, language: str):
        if language != "en":
            raise ValueError("Kokoro backend supports English only - use MMS for Hindi")
        try:
            from kokoro import KPipeline
        except ImportError as e:
            raise RuntimeError(
                "Kokoro is not installed. Run: pip install kokoro soundfile"
            ) from e
        self.pipe = KPipeline(lang_code="a")  # American English
        self.sample_rate = 24000

    def synthesize(self, text: str, out_path: Path) -> float:
        chunks = []
        for _, _, audio in self.pipe(text, voice="af_heart"):
            chunks.append(np.asarray(audio, dtype=np.float32))
        audio = np.concatenate(chunks) if chunks else np.zeros(self.sample_rate, np.float32)
        sf.write(out_path, audio, self.sample_rate)
        return len(audio) / self.sample_rate


def make_tts(backend_key: str, language: str):
    if backend_key == "kokoro":
        return KokoroTTS(language)
    return MMSTTS(language)
