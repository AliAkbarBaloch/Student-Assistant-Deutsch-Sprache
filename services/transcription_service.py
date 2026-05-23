"""
Transcription Service  —  Whisper ASR
======================================
Responsibility: when the user does not provide a target word, automatically
detect what word (or short phrase) was spoken in the audio using Whisper.

The detected text is then passed as the target_word to BFA for forced
alignment — so the whole pipeline still works without any user typing.

Why whisper-base?
  • ~74 MB — fast to download and cheap on CPU
  • Accurate enough for single words / short phrases
  • Already available via the transformers library (no extra install)

For longer phrases or noisy environments you can swap "openai/whisper-base"
for "openai/whisper-small" (~244 MB) by changing WHISPER_MODEL below.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import pipeline

# Change to "openai/whisper-small" for better accuracy on noisy/accented audio
WHISPER_MODEL = "openai/whisper-base"

# Module-level singleton — loaded once, reused for every request
_whisper_pipeline = None


def _get_pipeline():
    """
    Lazily load the Whisper ASR pipeline on the first call.
    Uses GPU if available, otherwise CPU.
    """
    global _whisper_pipeline
    if _whisper_pipeline is None:
        device_index = 0 if torch.cuda.is_available() else -1
        _whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL,
            device=device_index,
            # Return only the plain text (no timestamps needed here)
            generate_kwargs={"task": "transcribe", "language": "english"},
        )
    return _whisper_pipeline


def transcribe(audio: np.ndarray) -> str:
    """
    Transcribe a 16 kHz float32 audio array to text using Whisper.

    The result is cleaned to a single lowercase word or short phrase.
    Leading/trailing punctuation and filler words are stripped so the
    output is clean enough to pass directly to BFA as a target_word.

    Args:
        audio: 1-D float32 numpy array at 16 kHz.

    Returns:
        Transcribed text string, e.g. "butterfly".
        Returns "(unclear)" if Whisper produces an empty result.
    """
    pipe = _get_pipeline()
    result = pipe(audio)
    raw_text: str = result.get("text", "")

    # Strip punctuation and normalise whitespace
    cleaned = raw_text.strip().strip(".,!?\"'").strip().lower()

    return cleaned if cleaned else "(unclear)"
