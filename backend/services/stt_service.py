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
    """Transcribe 16 kHz float32 audio and prepend detected language if not German."""
    model = _get_model()
    # Let Whisper auto-detect the language instead of forcing "de"
    segments, info = model.transcribe(audio, beam_size=1, vad_filter=True)
    text = " ".join(seg.text for seg in segments).strip().strip(".,!?")
    
    if not text:
        return "(unclear)"
        
    # If a foreign language was detected, give the LLM a hint
    if info.language != "de":
        return f"[DETECTED_LANGUAGE: {info.language}] {text}"
        
    return text
