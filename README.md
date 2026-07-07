# Deutsch Buddy

An AI-powered German language learning app with ultra-low-latency voice conversation practice. Speak or type in German, and Buddy responds with natural German speech — correcting your mistakes gently and adapting to your CEFR level (A1 → B2).

## How It Works

```
Voice (Sprechen / Live-Anruf)
  WebRTC mic  →  Deepgram Nova-3 (STT, real-time)  →  LLM  →  Cartesia Sonic-3 (TTS streaming)
  Sprechen also shows transcriptions as chat messages; Live-Anruf is voice-only.

Text Chat  (streaming pipeline)
  Text input  →  LLM token stream (SSE)  →  text appears live in ~0.3 s
                  ↳ sentence complete? → Edge TTS starts concurrently
  Audio plays immediately after last token (TTS was generating in parallel)
```

### Models

| Component | Model | Notes |
|---|---|---|
| Voice STT | Deepgram Nova-3 | Real-time, German, via LiveKit inference |
| Voice TTS | Cartesia Sonic-3 | Streaming, German, via LiveKit inference |
| Voice LLM | `qwen36-35b` | Professor's API, `enable_thinking: false` for speed |
| Text LLM | `gemma4-31b-it` | Professor's OpenAI-compatible API |
| Text TTS | Microsoft Edge TTS `de-DE-KatjaNeural` | Free, no key needed |
| VAD | Silero | Turn detection for the voice agent |

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.12 | [python.org](https://www.python.org/downloads/) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org/) |
| LiveKit account | — | [livekit.io](https://livekit.io) — free tier works |

> ffmpeg is no longer required. PyAV handles all audio decoding.

---

## Backend Setup

### 1. Clone and enter the project

```bash
git clone <repo-url>
cd deutsche_buddy
```

### 2. Create a Python 3.12 virtual environment

> **Important:** Use Python 3.12 explicitly. Python 3.13/3.14 is not yet supported by the AI packages.

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create the `.env` file

Create a file named `.env` in the project root:

```env
# Professor's OpenAI-compatible API
PROF_API_KEY=your_api_key_here
PROF_API_BASE=https://llms.innkube.fim.uni-passau.de/v1

# LLM models
PROF_MODEL=gemma4-31b-it          # text chat
VOICE_MODEL=qwen36-35b            # voice agent (faster, lower latency)

# JWT signing key — use any random string of 32+ characters
SECRET_KEY=deutsch-buddy-secret-key-2026-secure

# LiveKit credentials (get these from your LiveKit project dashboard)
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
```

**Available models on the professor's API:**
- `gemma4-31b-it`
- `qwen36-35b`
- `qwen3-next-80b-a3b-instruct`
- `qwen35-397b`

---

## Running the App

Three terminals are required:

**Terminal 1 — FastAPI backend:**
```bash
source .venv/bin/activate
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — LiveKit voice agent:**
```bash
source .venv/bin/activate
python services/livekit_agent.py start
```

**Terminal 3 — Frontend dev server:**
```bash
cd frontend
npm install       # first time only
npm run dev
```

Then open **http://localhost:5173** in your browser.

> The Vite dev server proxies `/api` and `/static` to the backend on port 8000. All three processes must be running for voice features to work.

---

## First-Time Use

1. Open **http://localhost:5173**
2. Click **Register** and create an account
3. Select your German level (A1 / A2 / B1 / B2) from the navbar
4. **Text chat** — type a German message in the input bar and press Enter
5. **Sprechen (mic button)** — click to start, speak in German, click again to stop. Your speech and Buddy's reply both appear as chat messages. Ultra-low latency (~1–2 s).
6. **Live-Anruf (phone button)** — continuous voice conversation, voice-only (no text in chat). Same low-latency pipeline.

---

## Features

- **Ultra-low-latency voice** — both voice buttons use LiveKit WebRTC + Deepgram STT + Cartesia TTS streaming (~1–2 s round-trip vs ~5–8 s before)
- **Streaming text chat** — LLM tokens stream to the browser via SSE; first word appears in ~0.3 s (ChatGPT-style)
- **Concurrent TTS** — Edge TTS generates each sentence in the background while the LLM is still writing the next one; audio starts with near-zero delay after the last token
- **Sprechen shows chat transcriptions** — your speech and Buddy's response appear as text messages when using the mic button
- **Live-Anruf is voice-only** — continuous call mode with no chat clutter
- **CEFR-adaptive vocabulary** — responses use only words appropriate to your level
- **Gentle error correction** — wrong grammar is corrected by example, not criticism
- **Always German** — Buddy responds in German even if you write in English
- **Conversation history** — saved per user account in SQLite
- **Pronunciation feedback** — dedicated page to analyse your German pronunciation with a score and tips

---

## Project Structure

```
deutsche_buddy/
├── app.py                   # FastAPI — all endpoints, including /api/livekit-token
├── database.py              # SQLAlchemy models (User, Message)
├── requirements.txt
├── .env                     # secrets and config (never commit)
│
├── services/
│   ├── livekit_agent.py     # LiveKit voice agent (Deepgram STT + LLM + Cartesia TTS)
│   ├── stt_service.py       # faster-whisper STT (used by pronunciation feedback only)
│   ├── llm_service.py       # OpenAI-compatible LLM client (text chat)
│   ├── tts_service.py       # Edge TTS (text chat audio)
│   ├── auth_service.py      # JWT + bcrypt
│   ├── vocab_service.py     # CEFR vocabulary grounding
│   └── phoneme_service.py   # pronunciation analysis
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/      # auth, chat, profile, feedback UI
│   │   ├── contexts/        # Auth, Level, Theme global state
│   │   ├── hooks/           # useChat, useLiveKitCall
│   │   └── services/        # API client (api.ts)
│   ├── package.json
│   └── vite.config.ts
│
└── static/                  # served by FastAPI
    ├── index.html           # built SPA (after npm run build)
    ├── tts/                 # generated Edge TTS audio (text chat)
    └── avatars/             # user profile pictures
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/register` | Create account |
| POST | `/api/auth/login` | Login, returns JWT |
| GET | `/api/auth/me` | Current user info |
| PUT | `/api/auth/profile` | Update name & avatar |
| POST | `/api/chat` | Voice: STT → LLM → TTS (legacy HTTP, unused by main UI) |
| POST | `/api/chat-text` | Text: LLM → TTS (non-streaming fallback) |
| POST | `/api/chat-text-stream` | Text: LLM token stream + concurrent TTS (SSE, used by main UI) |
| POST | `/api/livekit-token` | Issue LiveKit room token + dispatch voice agent |
| GET | `/api/history` | Last 60 messages |
| DELETE | `/api/history` | Clear conversation |
| POST | `/api/pronunciation-feedback` | Pronunciation score + tips |
| GET | `/api/health` | Server status |

---

## What Changed (vs. original)

### Added
- `services/livekit_agent.py` — standalone LiveKit voice agent process
- `POST /api/livekit-token` endpoint — creates a room token and dispatches the agent with the user's CEFR level
- `POST /api/chat-text-stream` endpoint — SSE streaming with concurrent TTS (`AsyncOpenAI`, `asyncio.create_task`)
- `useLiveKitCall` hook (`frontend/src/hooks/useLiveKitCall.ts`) — manages the WebRTC room, transcription events, and audio playback
- `sendTextMessageStream()` in `api.ts` — SSE reader with `onToken` + `onAudio` callbacks
- `sendTextStream()` in `useChat.ts` — streaming chat with sequential audio chain
- LiveKit Python packages: `livekit`, `livekit-agents`, `livekit-plugins-openai`, `livekit-plugins-silero`
- `livekit-client` npm package (frontend)
- Chat transcription for Sprechen — `RoomEvent.TranscriptionReceived` with 1.2 s debounce to merge agent chunks

### Removed
- Old VAD + MediaRecorder live-call pipeline (replaced by LiveKit WebRTC)
- `useVAD` hook (no longer needed for Live-Anruf)
- Microsoft Edge TTS for voice responses (replaced by Cartesia Sonic-3 via LiveKit)
- faster-whisper for voice STT (replaced by Deepgram Nova-3 via LiveKit); Whisper is still used for pronunciation feedback

### Changed
- Both **Sprechen** and **Live-Anruf** buttons now share the same LiveKit pipeline
- Sprechen shows speech-to-text transcriptions and Buddy's replies in chat; Live-Anruf is voice-only
- Text chat now streams via SSE — first token in ~0.3 s, audio starts immediately after last token
- TTS for text chat generates sentence-by-sentence concurrently with the LLM stream (no sequential wait)
- faster-whisper `beam_size` reduced from 5 → 1 with `vad_filter=True` for the HTTP pronunciation path

---

## Production Build

```bash
cd frontend
npm run build
```

Outputs the SPA to `../static/react/`. The FastAPI server at **http://localhost:8000** then serves the full app — the frontend dev server is not needed. The LiveKit agent (`Terminal 2`) must still run separately.

---

## Troubleshooting

**Voice buttons don't connect**
→ Make sure the LiveKit agent is running (`python services/livekit_agent.py start`) and all three `LIVEKIT_*` env vars are set in `.env`.

**`pip install` fails / builds from source**
→ Confirm you are using Python 3.12:
```bash
python --version   # must show 3.12.x
```

**503 on text chat**
→ Check the uvicorn terminal for `[LLM]` log lines. Update `PROF_MODEL` in `.env` to one of the available models listed above.

**JWT `InsecureKeyLengthWarning`**
→ Your `SECRET_KEY` is shorter than 32 characters. Use a longer random string.

**Database reset**
```bash
rm deutsch_buddy.db
```
The database is recreated automatically on next server start.
