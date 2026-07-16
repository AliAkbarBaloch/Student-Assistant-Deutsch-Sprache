# Whisper STT for German — only used by pronunciation feedback.
# Voice chat STT is handled by Deepgram Nova-3 via LiveKit.
from __future__ import annotations

import numpy as np

_model = None
_MODEL_SIZE = "base"


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(_MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def transcribe_german(audio: np.ndarray, sample_rate: int = 16_000) -> str:  # noqa: ARG001
    """Transcribe 16 kHz float32 audio."""
    model = _get_model()
    # Force language to German since we assume the user only speaks German
    segments, info = model.transcribe(audio, language="de", beam_size=1, vad_filter=True)
    text = " ".join(seg.text for seg in segments).strip().strip(".,!?")
    
    if not text:
        return "(unclear)"
        
    return text
