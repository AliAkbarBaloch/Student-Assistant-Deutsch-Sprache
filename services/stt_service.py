"""
Speech-to-Text Service  —  faster-whisper (German)
===================================================
Uses faster-whisper large-v3-turbo: same accuracy as openai-whisper
but ~4× faster and lower memory via CTranslate2 int8 quantization.
No PyTorch required on CPU.
"""
from __future__ import annotations

import numpy as np

_model = None
_MODEL_SIZE = "large-v3-turbo"


def _get_model():
    """Lazily load faster-whisper on first call."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        # int8 quantization — fast on CPU, good accuracy
        _model = WhisperModel(_MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def transcribe_german(audio: np.ndarray, sample_rate: int = 16_000) -> str:  # noqa: ARG001
    """
    Transcribe 16 kHz float32 audio to German text.

    Args:
        audio:       1-D float32 numpy array at 16 kHz.
        sample_rate: Ignored (faster-whisper handles resampling internally).

    Returns:
        Transcribed German text. Returns "(unclear)" if nothing recognized.
    """
    model = _get_model()
    segments, _ = model.transcribe(audio, language="de", beam_size=1, vad_filter=True)
    text = " ".join(seg.text for seg in segments).strip().strip(".,!?")
    return text if text else "(unclear)"
