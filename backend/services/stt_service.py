# Qwen3-ASR-1.7B STT for German — only used by pronunciation feedback.
# Voice chat STT is handled by Deepgram Nova-3 via LiveKit.
from __future__ import annotations

import numpy as np
import torch

_model = None
_MODEL_NAME = "Qwen/Qwen3-ASR-1.7B"

def _get_model():
    global _model
    if _model is None:
        from qwen_asr import Qwen3ASRModel
        _model = Qwen3ASRModel.from_pretrained(
            _MODEL_NAME,
            dtype=torch.bfloat16,
            device_map={"":"cpu"}
        )
    return _model

def transcribe_german(audio: np.ndarray, sample_rate: int = 16_000) -> str:
    """Transcribe 16 kHz float32 audio."""
    model = _get_model()
    # Force language to German since we assume the user only speaks German
    results = model.transcribe(
        audio=(audio, sample_rate),
        language="German"
    )
    
    if not results or not results[0].text:
        return "(unclear)"
        
    text = results[0].text.strip().strip(".,!?")
    if not text:
        return "(unclear)"
        
    return text
