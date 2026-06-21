# Deutsch Buddy — AI German Learning Assistant

An AI-powered German language learning app with real-time voice conversation, text chat, and pronunciation feedback.

---

## Project Structure

```
student-assistant-deutsch-sprache/
├── backend/          ← FastAPI server (REST API + database)
├── frontend/         ← React + TypeScript + Vite (UI)
├── livekit_agent/    ← LiveKit real-time voice agent (runs separately)
└── .env              ← Shared environment variables (API keys)
```

---

## Prerequisites

- **Python 3.11+** — download from https://www.python.org/downloads/
- **Node.js 18+** — download from https://nodejs.org/
- A `.env` file at the project root (see [Environment Variables](#environment-variables))

---

## Environment Variables

Create a `.env` file in the **project root** with the following keys:

```env
# Professor's OpenAI-compatible LLM API
PROF_API_KEY=your_key_here
PROF_API_BASE=https://llms.innkube.fim.uni-passau.de/v1

# JWT secret for user authentication
SECRET_KEY=your_secret_key_here

# LiveKit credentials (required for voice calls)
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_key
LIVEKIT_API_SECRET=your_livekit_secret
```

---

## 1. Backend

The Flask/FastAPI server handles REST API, authentication, text chat, STT, TTS, and the database.

### Setup

```bash
cd backend

# Create virtual environment (requires Python 3.11+)
python3 -m venv .venv

# Activate — Mac/Linux:
source .venv/bin/activate
# Activate — Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

The backend runs at **http://localhost:8000**

### Key files

```
backend/
├── app.py            ← FastAPI entry point + all API routes
├── database.py       ← SQLAlchemy models (User, Message) + SQLite setup
├── requirements.txt  ← All Python dependencies
├── generate_pptx.py  ← Utility: generate PowerPoint from vocab data
├── static/           ← Served static files (avatars, TTS audio, legacy UI)
└── services/
    ├── auth_service.py          ← JWT creation + bcrypt password hashing
    ├── llm_service.py           ← LLM chat + streaming + pronunciation feedback
    ├── stt_service.py           ← Faster-Whisper German STT (offline, CPU)
    ├── tts_service.py           ← Microsoft Edge TTS (de-DE-KatjaNeural)
    ├── vocab_service.py         ← CEFR vocabulary CSV loader
    ├── phoneme_service.py       ← Phoneme-level pronunciation scoring
    └── transcription_service.py ← Legacy Whisper-base (unused by main app)
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/auth/register` | Register a new user |
| POST | `/api/auth/login` | Login, returns JWT |
| GET | `/api/auth/me` | Get current user info |
| PUT | `/api/auth/profile` | Update name / avatar |
| GET | `/api/history` | Fetch last 60 chat messages |
| DELETE | `/api/history` | Clear chat history |
| POST | `/api/chat-text` | Text chat (LLM → TTS) |
| POST | `/api/chat-text-stream` | Streaming text chat (SSE) |
| POST | `/api/chat` | Voice chat (STT → LLM → TTS) |
| POST | `/api/pronunciation-feedback` | Pronunciation analysis |
| POST | `/api/livekit-token` | Issue LiveKit token + dispatch agent |

---

## 2. Frontend

The React + TypeScript + Vite user interface.

### Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server (proxies API calls to backend on :8000)
npm run dev
```

The frontend dev server runs at **http://localhost:5173**

### Build for production

```bash
cd frontend
npm run build
# Built files appear in backend/static/react/ (served by the backend)
```

### Key files

```
frontend/
├── index.html
├── vite.config.ts
├── tailwind.config.js
└── src/
    ├── App.tsx
    ├── types.ts
    ├── components/
    │   ├── auth/        ← Login / Register page
    │   ├── chat/        ← Main chat UI, voice controls, message bubbles
    │   ├── feedback/    ← Pronunciation feedback page
    │   └── profile/     ← User profile page
    ├── contexts/        ← Auth, Level, Theme React contexts
    ├── hooks/
    │   ├── useChat.ts       ← Text chat logic
    │   ├── useLiveKitCall.ts ← LiveKit voice call integration
    │   └── useVAD.ts        ← Voice activity detection
    └── services/
        └── api.ts           ← All API calls to the backend
```

---

## 3. LiveKit Agent

The real-time voice agent. It connects to LiveKit Cloud and handles the live voice call feature.
Pipeline: **WebRTC audio → Deepgram STT → LLM → Cartesia TTS**

This must be running **at the same time as the backend** for voice calls to work.

### Setup

```bash
cd livekit_agent

# Reuse the backend virtual environment — Mac/Linux:
source ../backend/.venv/bin/activate
# Windows:
# ..\backend\.venv\Scripts\activate

# Start the agent (connects to LiveKit Cloud automatically)
python livekit_agent.py start
```

### Key files

```
livekit_agent/
├── livekit_agent.py  ← Agent entry point (DeutschBuddy class + session setup)
└── requirements.txt  ← LiveKit-specific Python dependencies
```

---

## Running Everything Together

Open **3 separate terminals**:

**Mac/Linux:**
```bash
# Terminal 1 — Backend
cd backend && source .venv/bin/activate && python app.py

# Terminal 2 — Frontend
cd frontend && npm run dev

# Terminal 3 — LiveKit Agent
cd livekit_agent && source ../backend/.venv/bin/activate && python livekit_agent.py start
```

**Windows:**
```bash
# Terminal 1 — Backend
cd backend && .venv\Scripts\activate && python app.py

# Terminal 2 — Frontend
cd frontend && npm run dev

# Terminal 3 — LiveKit Agent
cd livekit_agent && ..\backend\.venv\Scripts\activate && python livekit_agent.py start
```

Then open **http://localhost:5173** in your browser.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, SQLAlchemy, SQLite |
| Frontend | React, TypeScript, Vite, Tailwind CSS |
| Voice Agent | LiveKit Agents, Deepgram Nova-3 STT, Cartesia Sonic-3 TTS |
| LLM | OpenAI-compatible API (Qwen / Gemma via Professor's server) |
| STT (text chat) | faster-whisper large-v3-turbo (offline, CPU) |
| TTS (text chat) | Microsoft Edge TTS (de-DE-KatjaNeural, free) |
| Auth | JWT (HS256) + bcrypt |
