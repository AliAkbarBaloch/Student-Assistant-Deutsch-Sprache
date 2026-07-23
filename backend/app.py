from __future__ import annotations

# Must be set before ANY C-extension import to avoid macOS/Intel OMP errors
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS",        "1")
os.environ.setdefault("KMP_AFFINITY",           "disabled")
os.environ.setdefault("MKL_THREADING_LAYER",    "GNU")

import asyncio
import datetime
import json
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import librosa
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel
from sqlalchemy.orm import Session

import openai

from database import Call, CallMessage, Message, User, create_tables, get_db
from services import auth_service, llm_service, stt_service, tts_service

load_dotenv()
create_tables()

BASE_DIR            = Path(__file__).resolve().parent
BACKEND_STATIC_DIR  = BASE_DIR / "static"
FRONTEND_STATIC_DIR = BASE_DIR.parent / "frontend" / "static"
TTS_DIR             = BACKEND_STATIC_DIR / "tts"

TARGET_SR      = 16_000
MAX_FILE_BYTES = 20 * 1024 * 1024

TTS_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Deutsch Buddy — German AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount TTS media first (specific path), then frontend static files
app.mount("/static/tts", StaticFiles(directory=TTS_DIR), name="tts_media")
app.mount("/static", StaticFiles(directory=FRONTEND_STATIC_DIR), name="frontend_static")

@app.on_event("startup")
def startup_event():
    print("Preloading Qwen3-ASR model into memory...")
    try:
        stt_service._get_model()
        print("Qwen3-ASR loaded successfully.")
    except Exception as e:
        print(f"Failed to preload Qwen3-ASR: {e}")


def _get_user_from_header(authorization: Optional[str], db: Session) -> Optional[User]:
    """Pull the user from a Bearer JWT header. Returns None if missing or invalid."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token   = authorization.split(" ", 1)[1]
    payload = auth_service.decode_token(token)
    if not payload:
        return None
    return db.query(User).filter(User.id == int(payload["sub"])).first()


def _require_user(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)) -> User:
    """FastAPI dependency — raises 401 if the user is not authenticated."""
    user = _get_user_from_header(authorization, db)
    if not user:
        raise HTTPException(401, "Nicht angemeldet. Bitte zuerst einloggen.")
    return user


# Static / root

@app.get("/")
def root() -> FileResponse:
    return FileResponse(FRONTEND_STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# Auth routes

@app.post("/api/auth/register")
def register(
    email:    str = Form(...),
    password: str = Form(...),
    name:     str = Form(...),
    db: Session = Depends(get_db),
) -> dict:
    """Register a new user account."""
    if db.query(User).filter(User.email == email.lower().strip()).first():
        raise HTTPException(400, "Diese E-Mail ist bereits registriert.")

    user = User(
        email=email.lower().strip(),
        name=name.strip(),
        password_hash=auth_service.hash_password(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = auth_service.create_token(user.id, user.email)
    return {"token": token, "user": {"id": user.id, "name": user.name, "email": user.email, "avatar_url": user.avatar_url}}


@app.post("/api/auth/login")
def login(
    email:    str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
) -> dict:
    """Login with email + password, returns JWT."""
    user = db.query(User).filter(User.email == email.lower().strip()).first()
    if not user or not auth_service.verify_password(password, user.password_hash):
        raise HTTPException(401, "E-Mail oder Passwort ist falsch.")

    token = auth_service.create_token(user.id, user.email)
    return {"token": token, "user": {"id": user.id, "name": user.name, "email": user.email, "avatar_url": user.avatar_url}}


@app.get("/api/auth/me")
def me(current_user: User = Depends(_require_user)) -> dict:
    """Return the currently authenticated user's info."""
    return {"id": current_user.id, "name": current_user.name, "email": current_user.email, "avatar_url": current_user.avatar_url}


AVATARS_DIR = BACKEND_STATIC_DIR / "avatars"
AVATARS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.put("/api/auth/profile")
async def update_profile(
    name:   str = Form(...),
    avatar: Optional[UploadFile] = File(None),
    current_user: User = Depends(_require_user),
    db: Session = Depends(get_db),
) -> dict:
    """Update display name and/or avatar. Avatar saved to static/avatars/."""
    current_user.name = name.strip() or current_user.name

    if avatar and avatar.filename:
        if avatar.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(400, "Nur JPG, PNG oder WebP erlaubt.")

        ext      = Path(avatar.filename).suffix.lower() or ".jpg"
        filename = f"{current_user.id}{ext}"
        dest     = AVATARS_DIR / filename

        contents = await avatar.read()
        dest.write_bytes(contents)

        current_user.avatar_url = f"/static/avatars/{filename}"

    db.commit()
    db.refresh(current_user)

    return {
        "user": {
            "id":         current_user.id,
            "name":       current_user.name,
            "email":      current_user.email,
            "avatar_url": current_user.avatar_url,
        }
    }


# Conversation history

@app.get("/api/history")
def get_history(
    current_user: User = Depends(_require_user),
    db: Session = Depends(get_db),
) -> dict:
    """Return last 60 messages, oldest first."""
    messages = (
        db.query(Message)
        .filter(Message.user_id == current_user.id)
        .order_by(Message.created_at.desc())
        .limit(60)
        .all()
    )
    messages.reverse()   # oldest first

    return {
        "messages": [
            {
                "role":       m.role,
                "content_de": m.content_de,
                "content_en": m.content_en,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ]
    }


@app.delete("/api/history")
def clear_history(
    current_user: User = Depends(_require_user),
    db: Session = Depends(get_db),
) -> dict:
    """Delete all messages for the authenticated user."""
    db.query(Message).filter(Message.user_id == current_user.id).delete()
    db.commit()
    return {"detail": "Verlauf gelöscht."}


# Live-Anruf call transcripts

class CallMessageIn(BaseModel):
    role:    str
    content: str


class CallSaveRequest(BaseModel):
    messages:   list[CallMessageIn]
    started_at: Optional[str] = None
    ended_at:   Optional[str] = None


def _parse_iso(value: Optional[str]) -> Optional[datetime.datetime]:
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


@app.post("/api/calls")
def save_call(
    payload: CallSaveRequest,
    current_user: User = Depends(_require_user),
    db: Session = Depends(get_db),
) -> dict:
    """Persist a finished Live-Anruf transcript."""
    if not payload.messages:
        raise HTTPException(400, "Transkript ist leer.")

    now     = datetime.datetime.utcnow()
    call_id = str(uuid.uuid4())
    call = Call(
        id=call_id,
        user_id=current_user.id,
        started_at=_parse_iso(payload.started_at) or now,
        ended_at=_parse_iso(payload.ended_at) or now,
    )
    db.add(call)
    for m in payload.messages:
        if m.content.strip():
            db.add(CallMessage(call_id=call_id, role=m.role, content=m.content.strip()))
    db.commit()

    return {"call_id": call_id}


@app.get("/api/calls")
def list_calls(
    current_user: User = Depends(_require_user),
    db: Session = Depends(get_db),
) -> dict:
    """List the user's past Live-Anruf calls, most recent first."""
    calls = (
        db.query(Call)
        .filter(Call.user_id == current_user.id)
        .order_by(Call.started_at.desc())
        .all()
    )

    result = []
    for c in calls:
        first_user_msg = next((m.content for m in c.messages if m.role == "user"), "")
        duration = int((c.ended_at - c.started_at).total_seconds()) if c.ended_at else 0
        result.append({
            "id":               c.id,
            "started_at":       c.started_at.isoformat(),
            "ended_at":         c.ended_at.isoformat() if c.ended_at else None,
            "duration_seconds": max(duration, 0),
            "message_count":    len(c.messages),
            "preview":          first_user_msg[:120],
        })

    return {"calls": result}


@app.get("/api/calls/{call_id}")
def get_call(
    call_id: str,
    current_user: User = Depends(_require_user),
    db: Session = Depends(get_db),
) -> dict:
    """Return the full transcript for one Live-Anruf call."""
    call = (
        db.query(Call)
        .filter(Call.id == call_id, Call.user_id == current_user.id)
        .first()
    )
    if not call:
        raise HTTPException(404, "Anruf nicht gefunden.")

    return {
        "id":         call.id,
        "started_at": call.started_at.isoformat(),
        "ended_at":   call.ended_at.isoformat() if call.ended_at else None,
        "messages": [
            {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
            for m in call.messages
        ],
    }


# LiveKit token — used by both voice buttons

@app.post("/api/livekit-token")
async def livekit_token(
    level:         str = Form("B1"),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> dict:
    """Issue a LiveKit token and dispatch the voice agent to the room."""
    lk_key    = os.getenv("LIVEKIT_API_KEY")
    lk_secret = os.getenv("LIVEKIT_API_SECRET")
    lk_url    = os.getenv("LIVEKIT_URL")
    if not lk_key or not lk_secret or not lk_url:
        raise HTTPException(503, "LiveKit ist nicht konfiguriert.")

    from livekit import api as lk_api

    user     = _get_user_from_header(authorization, db)
    identity = f"user-{user.id}" if user else f"anon-{uuid.uuid4().hex[:8]}"
    name     = user.name if user else "Gast"
    room     = f"buddy-{identity}"
    cefr     = level.upper() if level.upper() in ("A1", "A2", "B1", "B2") else "B1"

    token = (
        lk_api.AccessToken(lk_key, lk_secret)
        .with_identity(identity)
        .with_name(name)
        .with_grants(lk_api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )

    # Dispatch the Buddy agent to this room with the user's CEFR level
    async with lk_api.LiveKitAPI(lk_url, lk_key, lk_secret) as lk:
        await lk.agent_dispatch.create_dispatch(
            lk_api.CreateAgentDispatchRequest(
                agent_name="deutsch-buddy",
                room=room,
                metadata=json.dumps({"level": cefr}),
            )
        )

    return {"token": token, "url": lk_url, "room": room}


# Chat helpers

def _save_turn(db: Session, user_id: int, user_text: str, ai_de: str, ai_en: str) -> None:
    """Persist one conversation turn (user message + assistant reply) to the DB."""
    db.add(Message(user_id=user_id, role="user",      content_de=user_text, content_en=""))
    db.add(Message(user_id=user_id, role="assistant", content_de=ai_de,    content_en=ai_en))
    db.commit()


def _build_history(history_json: str) -> list[dict]:
    """Parse the history JSON string sent by the frontend."""
    try:
        parsed = json.loads(history_json)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


# Chat endpoints


@app.post("/api/chat-text-stream")
async def chat_text_stream(
    message:       str = Form(...),
    history:       str = Form("[]"),
    level:         str = Form("B1"),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    Streaming text chat — SSE endpoint.
    Streams LLM tokens immediately; kicks off TTS in the background each time a
    sentence boundary is detected.  By the time the last token arrives, most
    audio is already generated, so playback starts with minimal delay.

    Events:
      {"type": "token", "text": "..."}   — one LLM token (streamed live)
      {"type": "audio", "tts_url": "..."} — one TTS chunk ready to play (in order)
      {"type": "done"}                   — all events sent
      {"type": "error", "message": "..."} — something went wrong
    """
    user_text = message.strip()
    if not user_text:
        raise HTTPException(400, "Nachricht darf nicht leer sein.")

    cefr = level.upper() if level.upper() in ("A1", "A2", "B1", "B2") else "B1"
    messages, model = llm_service.build_streaming_messages(
        user_text, _build_history(history), level=cefr
    )

    api_key  = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_API_BASE", "https://llms.innkube.fim.uni-passau.de/v1")

    current_user = _get_user_from_header(authorization, db)

    async def _tts_chunk(text: str) -> str:
        path = await tts_service.generate_tts_audio(text, TTS_DIR)
        return f"/static/tts/{path.name}"

    async def event_stream():
        full_text  = ""
        tts_buffer = ""
        tts_tasks: list[asyncio.Task] = []

        async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        try:
            stream = await async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=300,
                stream=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            async for chunk in stream:
                delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                if delta:
                    full_text  += delta
                    tts_buffer += delta
                    yield f"data: {json.dumps({'type': 'token', 'text': delta})}\n\n"

                    # When a sentence ends and we have enough text, start TTS in
                    # the background — it generates while the LLM keeps streaming.
                    stripped = tts_buffer.strip()
                    if len(stripped) >= 20 and stripped[-1] in ".!?":
                        tts_tasks.append(asyncio.create_task(_tts_chunk(stripped)))
                        tts_buffer = ""

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        if not full_text.strip():
            return

        # Flush any remaining text that didn't end with punctuation
        if tts_buffer.strip():
            tts_tasks.append(asyncio.create_task(_tts_chunk(tts_buffer.strip())))

        # Yield audio URLs in order — most tasks are already done by now
        for task in tts_tasks:
            tts_url = await task
            yield f"data: {json.dumps({'type': 'audio', 'tts_url': tts_url})}\n\n"

        if current_user:
            _save_turn(db, current_user.id, user_text, full_text.strip(), "")

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


#
# Pronunciation feedback endpoint
#

@app.post("/api/pronunciation-feedback")
async def pronunciation_feedback(
    audio:       Optional[UploadFile] = File(None),
    target_text: str = Form(""),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Pronunciation feedback endpoint.
    Accepts an optional MP3/WAV/WebM audio file. If audio is missing, uses target_text to test the pipeline.
    """
    if audio is not None and audio.filename:
        try:
            audio_array = await _read_audio(audio)
            # Offload heavy CPU-bound Qwen3-ASR inference to a thread so it
            # doesn't block the async event loop (was causing infinite spinner).
            transcribed = await asyncio.to_thread(
                stt_service.transcribe_german, audio_array, TARGET_SR
            )
        except Exception as e:
            raise HTTPException(500, f"Error processing audio: {e}")
    else:
        transcribed = target_text

    if not transcribed or not transcribed.strip():
        raise HTTPException(400, "Kein Deutsch erkannt. Bitte erneut versuchen.")

    try:
        # Offload synchronous OpenAI SDK call to a thread as well.
        feedback = await asyncio.to_thread(
            llm_service.pronunciation_feedback, transcribed, target_text
        )
    except (openai.APIError, openai.APIConnectionError, RuntimeError) as e:
        raise HTTPException(503, f"AI service temporarily unavailable: {e}")
    return feedback


#
# Audio decoding helper
#

async def _read_audio(upload: UploadFile) -> np.ndarray:
    """
    Decode uploaded audio to 16 kHz mono float32 numpy array.
    Stage 1 — soundfile (WAV/FLAC/OGG, fast path).
    Stage 2 — PyAV (WebM/Opus/MP3 from browser — no temp files, no deprecated audioread).
    """
    data = await upload.read()
    if not data:
        raise HTTPException(400, "Audio file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(400, "Audio exceeds 20 MB limit.")

    audio: np.ndarray | None = None
    sr:    int | None        = None

    # Stage 1: soundfile handles WAV/FLAC/OGG instantly
    try:
        audio, sr = sf.read(BytesIO(data), dtype="float32", always_2d=False)
    except Exception:
        pass

    # Stage 2: PyAV handles WebM/Opus/MP3 recorded by browsers
    if audio is None:
        try:
            import av
            container = av.open(BytesIO(data))
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=TARGET_SR)
            chunks: list[np.ndarray] = []
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray()[0])
            for out in resampler.resample(None):  # flush remaining samples
                chunks.append(out.to_ndarray()[0])
            container.close()
            if not chunks:
                raise ValueError("No audio frames decoded")
            audio = np.concatenate(chunks).astype(np.float32)
            sr = TARGET_SR  # already resampled by PyAV
        except Exception as exc:
            raise HTTPException(400, "Cannot decode audio. Use WAV, WebM, or MP3.") from exc

    if audio.ndim > 1:
        axis  = 0 if audio.shape[0] < audio.shape[-1] else 1
        audio = np.mean(audio, axis=axis)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    return np.asarray(audio, dtype=np.float32)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
