# Qwen3-ASR-0.6B STT for German — only used by pronunciation feedback.
# Voice chat STT is handled by Deepgram Nova-3 via LiveKit.
#
# Uses mlx-audio on Apple Silicon for Metal GPU acceleration.
# 8-bit quantised model keeps memory under 1 GB and runs ~5-10× faster
# than the PyTorch CPU path.
from __future__ import annotations

import os
import tempfile

import numpy as np
import soundfile as sf

_model = None
_MODEL_NAME = "mlx-community/Qwen3-ASR-0.6B-8bit"


def _get_model():
    global _model
    if _model is None:
        from mlx_audio.stt import load
        _model = load(_MODEL_NAME)
    return _model


def transcribe_german(audio: np.ndarray, sample_rate: int = 16_000) -> str:
    """Transcribe 16 kHz float32 audio.

    Optimisations applied:
    - MLX Metal GPU acceleration (Apple Silicon) — ~5-10× faster than CPU
    - 8-bit quantised model — under 1 GB memory
    - Leading/trailing silence is trimmed so the model processes less audio
    - Audio is capped at 30 seconds to avoid excessive processing time
    """
    # ── Trim silence ──────────────────────────────────────────────────────
    try:
        import librosa
        trimmed, _ = librosa.effects.trim(audio, top_db=25)
        if len(trimmed) > 0:
            audio = trimmed
    except Exception:
        pass  # librosa not available or trim failed — use original

    # ── Cap at 30 seconds ─────────────────────────────────────────────────
    max_samples = sample_rate * 30
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    # ── Write to temp WAV (mlx-audio expects a file path) ─────────────────
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)  # close fd — soundfile opens its own handle
    try:
        sf.write(tmp_path, audio, sample_rate)

        # ── Transcribe via MLX Metal GPU ──────────────────────────────────
        model = _get_model()
        result = model.generate(tmp_path, language="German")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # ── Extract text ──────────────────────────────────────────────────────
    text = ""
    if hasattr(result, "text"):
        text = result.text
    elif isinstance(result, str):
        text = result
    elif isinstance(result, dict) and "text" in result:
        text = result["text"]

    text = text.strip().strip(".,!?")
    if not text:
        return "(unclear)"

    return text
