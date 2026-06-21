# Phoneme-level pronunciation scoring via Bournemouth Forced Aligner (BFA).
# Forced alignment is used because we know the target word — it's far more accurate
# than free-running ASR for this use case.
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from bournemouth_aligner import PhonemeTimestampAligner
from phonemizer.backend import EspeakBackend

# Point phonemizer at the correct espeak-ng library path (Homebrew moves it)
_ESPEAK_CANDIDATES = [
    "/opt/homebrew/lib/libespeak-ng.dylib",          # Apple Silicon
    "/usr/local/lib/libespeak-ng.dylib",              # Intel Mac
    "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",   # Ubuntu/Debian
]
for _lib in _ESPEAK_CANDIDATES:
    if Path(_lib).exists():
        EspeakBackend.set_library(_lib)
        break

_aligner: PhonemeTimestampAligner | None = None


def _get_aligner() -> PhonemeTimestampAligner:
    global _aligner
    if _aligner is None:
        _aligner = PhonemeTimestampAligner(
            preset="en-us",
            device="auto",
            silence_anchors=3,  # helps with short single-word clips
        )
    return _aligner


def extract_phonemes(audio: np.ndarray, target_word: str) -> list[dict]:
    """Align the phonemes of target_word against the user's audio.

    Returns a list of dicts with keys: ipa, start_ms, end_ms, confidence, is_estimated.
    confidence < 0.25 usually means the phoneme was mispronounced.
    """
    aligner = _get_aligner()

    # Convert directly to a torch tensor — avoids the torchaudio/ffmpeg dependency
    # that BFA's load_audio() needs. Audio is already 16 kHz float32.
    audio_tensor: torch.Tensor = torch.from_numpy(audio).unsqueeze(0)

    result = aligner.process_sentence(target_word, audio_tensor)
    raw_phonemes: list[dict] = result["segments"][0]["phoneme_ts"]

    return [
        {
            "ipa":          p["ipa_label"],
            "start_ms":     round(p["start_ms"], 1),
            "end_ms":       round(p["end_ms"], 1),
            "confidence":   round(p["confidence"], 3),
            "is_estimated": p["is_estimated"],
        }
        for p in raw_phonemes
    ]
