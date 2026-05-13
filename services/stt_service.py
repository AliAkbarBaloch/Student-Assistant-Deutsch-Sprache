"""
Speech-to-Text Service  —  Whisper large-v3-turbo (German)
===========================================================
Uses openai/whisper-large-v3-turbo — same accuracy as large-v3 but 3x faster.
Significantly better than whisper-base for German names and proper nouns.

To upgrade to Qwen3-ASR (state-of-the-art, Jan 2026), create a Python 3.12
venv and run: pip install qwen-asr
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import pipeline

_pipeline = None

# large-v3-turbo: fast + accurate — replaces whisper-base which was too weak for German names
_MODEL = "openai/whisper-large-v3-turbo"


def _get_pipeline():
    """Lazily load Whisper on first call (downloads ~1.5 GB on first use)."""
    global _pipeline
    if _pipeline is None:
        device = 0 if torch.cuda.is_available() else -1
        _pipeline = pipeline(
            "automatic-speech-recognition",
            model=_MODEL,
            device=device,
            chunk_length_s=30,
            generate_kwargs={"task": "transcribe", "language": "german"},
        )
    return _pipeline


def transcribe_german(audio: np.ndarray, sample_rate: int = 16_000) -> str:
    """
    Transcribe 16 kHz float32 audio to German text.

    Args:
        audio:       1-D float32 numpy array.
        sample_rate: Audio sample rate (default 16000 Hz, not used by pipeline).

    Returns:
        Transcribed German text. Returns "(unclear)" if empty.
    """
    pipe = _get_pipeline()
    result = pipe(audio)
    text: str = result.get("text", "").strip().strip(".,!?")
    return text if text else "(unclear)"
