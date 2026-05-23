"""
Text-to-Speech Service  —  Edge TTS (German)
=============================================
Generates German speech audio using Microsoft Edge TTS.
Default voice: de-DE-KatjaNeural (warm, natural German female voice).
No API key required.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import edge_tts

# German female voice — warm and clear
_DEFAULT_VOICE = "de-DE-KatjaNeural"


async def generate_tts_audio(text: str, output_dir: Path, voice: str = _DEFAULT_VOICE) -> Path:
    """
    Convert text to speech and save as MP3.

    Args:
        text:       Text to speak (German).
        output_dir: Directory to save the MP3 file.
        voice:      Edge TTS voice name.

    Returns:
        Path to the generated MP3 file.
    """
    # Sanitise text for use in a filename prefix
    safe_prefix = "".join(c for c in text[:12].lower() if c.isalnum()) or "audio"
    filename = f"{safe_prefix}_{uuid.uuid4().hex[:8]}.mp3"
    output_path = output_dir / filename

    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(str(output_path))

    return output_path
