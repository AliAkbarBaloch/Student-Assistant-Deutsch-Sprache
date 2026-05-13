"""
Phoneme Extraction Service  —  Bournemouth Forced Aligner (BFA v1.1.x)
========================================================================
Responsibility: align the target word's phonemes against the user's audio
and return per-phoneme timestamps + confidence scores.

Why BFA over wav2vec2-xlsr?
  • Forced alignment: because we KNOW the target word, BFA pins the correct
    phoneme sequence to the audio — much more accurate than free-running ASR.
  • ms-level timestamps: each phoneme gets a start_ms / end_ms pair.
  • Confidence scores: values close to 1.0 = correctly pronounced;
    values below 0.25 = likely mispronounced. The LLM uses these directly.
  • Smaller model (~50 MB vs ~300 MB), ~0.2 s per 10 s of audio on CPU.
  • 80+ languages via espeak-ng presets.

System prerequisites (install once):
  macOS:   brew install espeak-ng ffmpeg
  Linux:   sudo apt-get install espeak-ng ffmpeg

The model (~50 MB) is downloaded from HuggingFace on the first call and
cached locally — every subsequent call uses the cached copy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from bournemouth_aligner import PhonemeTimestampAligner
from phonemizer.backend import EspeakBackend

# ─────────────────────────────────────────────────────────────────────────────
# macOS Homebrew fix: phonemizer looks for "espeak" but Homebrew installs
# "espeak-ng" at a different path.  We point it to the correct .dylib so
# BFA's internal phonemizer can find it without a system-level symlink.
# ─────────────────────────────────────────────────────────────────────────────
_ESPEAK_CANDIDATES = [
    "/opt/homebrew/lib/libespeak-ng.dylib",    # Apple Silicon
    "/usr/local/lib/libespeak-ng.dylib",        # Intel Mac
    "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",  # Ubuntu/Debian
]
for _lib in _ESPEAK_CANDIDATES:
    if Path(_lib).exists():
        EspeakBackend.set_library(_lib)
        break

# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton — loaded once, reused for every request.
# ─────────────────────────────────────────────────────────────────────────────
_aligner: PhonemeTimestampAligner | None = None


def _get_aligner() -> PhonemeTimestampAligner:
    """
    Lazily initialise the BFA aligner on the first request.

    preset="en-us" selects the dedicated English model (highest accuracy).
    device="auto" uses MPS on Apple Silicon, CUDA on NVIDIA, else CPU.
    silence_anchors=3 improves accuracy on single-word clips by letting
    BFA anchor alignment to any detected silences around the word.
    """
    global _aligner
    if _aligner is None:
        _aligner = PhonemeTimestampAligner(
            preset="en-us",
            device="auto",
            silence_anchors=3,        # better boundary detection on short clips
        )
    return _aligner


def extract_phonemes(audio: np.ndarray, target_word: str) -> list[dict]:
    """
    Align the phonemes of target_word against the user's audio waveform.

    BFA is a forced aligner — it takes BOTH the audio AND the expected text,
    then finds exactly when each phoneme of that text occurs in the audio.
    Confidence scores reveal which sounds were clear (≥0.7) vs unclear (≤0.25).

    Args:
        audio:        1-D float32 numpy array at 16 kHz (already resampled by app.py).
        target_word:  The English word the user was trying to say, e.g. "beautiful".

    Returns:
        List of dicts, one per phoneme:
          {"ipa": "b", "start_ms": 33.6, "end_ms": 50.4,
           "confidence": 0.997, "is_estimated": False}
        confidence < 0.25 = likely mispronounced.
    """
    aligner = _get_aligner()

    # Convert the numpy array directly to a torch tensor — shape (1, T).
    # This bypasses BFA's load_audio() which requires torchaudio's ffmpeg
    # backend (not available on all macOS installs).
    # Our audio is already 16 kHz float32, which is exactly what BFA expects.
    audio_tensor: torch.Tensor = torch.from_numpy(audio).unsqueeze(0)

    # Run forced alignment — BFA phonemises target_word via espeak-ng,
    # then uses Viterbi decoding to pin each phoneme to the audio frames.
    result = aligner.process_sentence(target_word, audio_tensor)

    # Extract the per-phoneme timestamp entries from the first (only) segment
    raw_phonemes: list[dict] = result["segments"][0]["phoneme_ts"]

    # Reshape into a clean, serialisable list for the route handler and LLM
    phonemes = [
        {
            "ipa":          p["ipa_label"],
            "start_ms":     round(p["start_ms"], 1),
            "end_ms":       round(p["end_ms"], 1),
            "confidence":   round(p["confidence"], 3),
            "is_estimated": p["is_estimated"],
        }
        for p in raw_phonemes
    ]
    return phonemes
