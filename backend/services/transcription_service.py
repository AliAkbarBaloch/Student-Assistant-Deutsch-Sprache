# Legacy HuggingFace Whisper pipeline — English only, not used by the main app.
# Kept around in case you need free-form English transcription without BFA.
from __future__ import annotations

import numpy as np
import torch
from transformers import pipeline

WHISPER_MODEL = "openai/whisper-base"

_whisper_pipeline = None


def _get_pipeline():
    global _whisper_pipeline
    if _whisper_pipeline is None:
        device_index = 0 if torch.cuda.is_available() else -1
        _whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL,
            device=device_index,
            generate_kwargs={"task": "transcribe", "language": "english"},
        )
    return _whisper_pipeline


def transcribe(audio: np.ndarray) -> str:
    """Transcribe 16 kHz float32 audio to English text."""
    pipe = _get_pipeline()
    result = pipe(audio)
    raw_text: str = result.get("text", "")
    cleaned = raw_text.strip().strip(".,!?\"'").strip().lower()
    return cleaned if cleaned else "(unclear)"
