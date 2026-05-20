# Deutsch Buddy

An AI-powered German language learning app with voice conversation practice. Speak or type in German, and Buddy responds with natural German speech — correcting your mistakes gently and adapting to your CEFR level (A1 → B2).

## How It Works

```
User Voice  →  faster-whisper (STT)  →  LLM (German response)  →  Edge TTS  →  User hears reply
User Text   →                            LLM (German response)  →  Edge TTS  →  User hears reply
```

- **STT**: faster-whisper `large-v3-turbo` (German, int8, CPU-ready)
- **Voice LLM**: `qwen36-35b` via professor's OpenAI-compatible API
- **Text LLM**: `gemma4-31b-it` via professor's OpenAI-compatible API
- **TTS**: Microsoft Edge TTS — `de-DE-KatjaNeural` (free, no API key)

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.12 | [python.org](https://www.python.org/downloads/) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org/) |
| ffmpeg | any | `brew install ffmpeg` (macOS) |

---

## Backend Setup

### 1. Clone the repo and enter the project folder

```bash
git clone <repo-url>
cd deutsche_buddy
```

### 2. Create a Python 3.12 virtual environment

> **Important:** Use Python 3.12 explicitly. Python 3.14 (the macOS default on some systems) is not yet supported by the AI packages.

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

> First run downloads the faster-whisper model (~1.5 GB). Subsequent runs load it from cache.

### 4. Create the `.env` file

Create a file named `.env` in the project root (same folder as `app.py`):

```env
# Professor's OpenAI-compatible API
PROF_API_KEY=your_api_key_here
PROF_API_BASE=https://llms.innkube.fim.uni-passau.de/v1

# LLM models
PROF_MODEL=gemma4-31b-it          # used for text chat
VOICE_MODEL=qwen36-35b            # used for voice chat (faster)

# JWT signing key — use any random string of 32+ characters
SECRET_KEY=deutsch-buddy-secret-key-2026-secure

# macOS fix for OpenMP conflicts (do not remove)
KMP_DUPLICATE_LIB_OK=TRUE
OMP_NUM_THREADS=1
KMP_AFFINITY=disabled
MKL_THREADING_LAYER=GNU
```

**Available models on the professor's API:**
- `gemma4-31b-it`
- `qwen36-35b`
- `qwen3-next-80b-a3b-instruct`
- `qwen35-397b`

### 5. Start the backend

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Backend runs at **http://localhost:8000**

---

## Frontend Setup

### 1. Install Node dependencies

```bash
cd frontend
npm install
```

### 2. Run the development server

```bash
npm run dev
```

Frontend runs at **http://localhost:5173**

> The Vite dev server automatically proxies `/api` and `/static` requests to the backend on port 8000. Both servers must be running.

### 3. Build for production (optional)

```bash
npm run build
```

This outputs the built SPA to `../static/react/`. After building, the backend at **http://localhost:8000** serves the full app without needing the Vite dev server.

---

## Running the App

You need two terminals:

**Terminal 1 — Backend:**
```bash
source .venv/bin/activate
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend (dev mode):**
```bash
cd frontend
npm run dev
```

Then open **http://localhost:5173** in your browser.

---

## First-Time Use

1. Open **http://localhost:5173**
2. Click **Register** and create an account
3. Select your German level (A1 / A2 / B1 / B2) from the navbar
4. **Text chat**: type a German message in the input bar and press Enter
5. **Voice chat**: click the microphone button (Sprechen), speak, click again to stop — Buddy transcribes, replies in German, and reads the reply aloud
6. **Live call**: click the phone button for continuous voice conversation

---

## Features

- **CEFR-adaptive vocabulary** — responses use only words appropriate to your level
- **Gentle error correction** — wrong grammar is corrected by example, not criticism
- **Always German** — Buddy responds in German even if you write in English or another language
- **Conversation history** — saved per user account in SQLite
- **Pronunciation feedback** — dedicated page to analyse your German pronunciation with a score and tips

---

## Project Structure

```
deutsche_buddy/
├── app.py                   # FastAPI — all endpoints
├── database.py              # SQLAlchemy models (User, Message)
├── requirements.txt
├── .env                     # secrets and config (not committed)
│
├── services/
│   ├── stt_service.py       # faster-whisper STT
│   ├── llm_service.py       # OpenAI-compatible LLM client
│   ├── tts_service.py       # Edge TTS
│   ├── auth_service.py      # JWT + bcrypt
│   ├── vocab_service.py     # CEFR vocabulary grounding
│   └── phoneme_service.py   # pronunciation analysis
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/      # auth, chat, profile, feedback UI
│   │   ├── contexts/        # Auth, Level, Theme global state
│   │   ├── hooks/           # useChat, useVAD
│   │   └── services/        # API client
│   ├── package.json
│   └── vite.config.ts
│
└── static/                  # served by FastAPI
    ├── index.html           # built SPA (after npm run build)
    ├── tts/                 # generated audio files
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
| POST | `/api/chat` | Voice: STT → LLM → TTS |
| POST | `/api/chat-text` | Text: LLM → TTS |
| GET | `/api/history` | Last 60 messages |
| DELETE | `/api/history` | Clear conversation |
| POST | `/api/pronunciation-feedback` | Pronunciation score + tips |
| GET | `/api/health` | Server status |

---

## Troubleshooting

**`pip install` fails / builds from source**
→ Make sure you are using Python 3.12, not 3.13 or 3.14:
```bash
python --version   # must show 3.12.x
```

**503 on voice chat**
→ Check the uvicorn terminal for `[voice LLM]` log lines showing which model failed. Update `VOICE_MODEL` in `.env` to one of the allowed models listed above.

**`PySoundFile failed` warning**
→ Install ffmpeg: `brew install ffmpeg`. The app still works via PyAV fallback, but ffmpeg makes it cleaner.

**JWT `InsecureKeyLengthWarning`**
→ Your `SECRET_KEY` in `.env` is shorter than 32 characters. Use a longer key.

**Database reset**
```bash
rm deutsch_buddy.db
```
The database is recreated automatically on next server start.
