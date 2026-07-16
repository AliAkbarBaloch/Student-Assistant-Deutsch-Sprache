# Edge TTS for text chat — generates an MP3 and returns the path.
# Runs concurrently per sentence to reduce audio delay.
from __future__ import annotations

import uuid
from pathlib import Path

import edge_tts

_DEFAULT_VOICE = "de-DE-KatjaNeural"


async def generate_tts_audio(text: str, output_dir: Path, voice: str = _DEFAULT_VOICE) -> Path:
    """Generate speech for the given text and save it as an MP3."""
    safe_prefix = "".join(c for c in text[:12].lower() if c.isalnum()) or "audio"
    filename = f"{safe_prefix}_{uuid.uuid4().hex[:8]}.mp3"
    output_path = output_dir / filename

    communicate = edge_tts.Communicate(text=text, voice=voice, rate="-15%")
    await communicate.save(str(output_path))

    return output_path
