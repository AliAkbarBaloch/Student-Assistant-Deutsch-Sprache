# Deutsch Lernbot — CEFR Vocabulary-Controlled German Chatbot

A German language learning chatbot that enforces strict vocabulary constraints based on CEFR proficiency levels (A1 / A2 / B1 / B2). The model is physically prevented from generating words outside the selected level's vocabulary via a custom `LogitsProcessor` applied at every generation step.

---

## Model

### DiscoResearch/Llama3-German-8B

| Property | Details |
|---|---|
| **Model ID** | `DiscoResearch/Llama3-German-8B` |
| **Base model** | Meta Llama 3 8B |
| **Parameters** | 8 billion |
| **Fine-tuning** | Continued pre-training on a large German text corpus by DiscoResearch |
| **Language** | German (primary), some English retained |
| **Architecture** | Transformer decoder (Llama 3 architecture), 32 layers, 32 attention heads, 8192 hidden dim |
| **Context length** | 8 192 tokens (standard), 32 768 tokens (32k variant) |
| **Tokenizer** | SentencePiece / tiktoken (Llama 3 vocabulary, 128 256 tokens) |
| **HuggingFace** | https://huggingface.co/DiscoResearch/Llama3-German-8B |
| **License** | Meta Llama 3 Community License |

**Why this model?**
Llama 3 8B fine-tuned on German text produces fluent, grammatically correct German. Because it is open-weights and runs locally, the generation pipeline can be intercepted at the logit level to apply hard vocabulary constraints — something not possible with closed API models.

---

## Controlled Text Generation

The core research problem is **constrained text generation**: ensuring every token the model outputs belongs to a pre-defined vocabulary set.

### How it works

1. **Vocabulary loading** — `dictionary_a1a2b1_onlystems.csv` provides word stems grouped by CEFR level. Levels are cumulative:
   - A1 → 603 stems
   - A2 → 2 022 stems (A1 ∪ A2)
   - B1 → 5 557 stems (A1 ∪ A2 ∪ B1)
   - B2 → no restriction

2. **Token whitelist construction** — At startup, for each level every stem is combined with ~40 common German inflection suffixes (`-en`, `-er`, `-ung`, `-lich`, `-keit`, `-ig`, …) to produce a large set of valid word forms. All forms are tokenised with the model's own BPE tokeniser; the resulting token IDs form the **allowed set**.

3. **`VocabLogitsProcessor`** — A custom `transformers.LogitsProcessor` that runs at every generation step. It sets the logit of every token **not** in the allowed set to `-∞`, making those tokens impossible to sample regardless of temperature or top-p. Punctuation, digits, whitespace, and all special tokens are always permitted.

4. **System prompt** — A short system prompt tells the model it is a German tutor for the selected level, reinforcing the behavioural intent (though the hard constraint comes from the logit mask, not the prompt).

---

## Requirements

### Hardware

| Device | Minimum RAM/VRAM | Speed |
|---|---|---|
| NVIDIA GPU (CUDA) | 16 GB VRAM (fp16) | Fast (~10–50 tokens/s) |
| Apple Silicon (MPS) | 16 GB unified memory | Medium (~2–10 tokens/s) |
| CPU fallback | 16 GB RAM (fp16) | Slow (~0.5–2 tokens/s) |

> **Note:** The application auto-detects CUDA → CPU (fp16). MPS is skipped because
> Apple Silicon's MPS backend does not support bfloat16 and most consumer Macs
> have less than 16 GB of GPU-addressable memory.

### Software

- Python **3.12** (PyTorch does not yet support Python 3.14)
- See `requirements.txt` for all Python packages

---

## Installation

```bash
# 1. Clone / download the project
cd "Applied AI Lab"

# 2. Create a virtual environment with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Enable 4-bit quantisation — CUDA + Linux only
pip install bitsandbytes
```

---

## Running the Application

```bash
# Start the server (model weights download automatically on first run, ~16 GB)
python app.py
```

The server starts at **http://localhost:8000**

Expected startup output:
```
  A1: 603 cumulative stems
  A2: 2,022 cumulative stems
  B1: 5,557 cumulative stems

Loading tokenizer and model: DiscoResearch/Llama3-German-8B
  Using device: CPU  |  dtype: torch.float16
  Model loaded on CPU

INFO: Uvicorn running on http://0.0.0.0:8000
```

The **token cache** for each level is built on first use (a few seconds), then cached in memory for the rest of the session.

---

## Usage

1. Open **http://localhost:8000** in your browser
2. Select a CEFR level using the buttons at the top:
   - **A1 — Anfänger** — only 603 A1-level word stems allowed
   - **A2 — Grundstufe** — 2 022 stems (A1 + A2)
   - **B1 — Mittelstufe** — 5 557 stems (A1 + A2 + B1)
   - **B2 — Obere Mittelstufe** — no vocabulary restriction
3. Type a message in German and press **Enter** or the send button
4. The bot responds using only vocabulary from the selected level

Switching levels resets the conversation history.

---

## Project Structure

```
Applied AI Lab/
├── app.py                          # FastAPI backend
├── index.html                      # Single-file chat frontend
├── requirements.txt                # Python dependencies
├── dictionary_a1a2b1_onlystems.csv # CEFR vocabulary stems (A1/A2/B1)
├── german_vocabulary_dataset.json  # Additional vocabulary dataset
└── README.md                       # This file
```

---

## Stopping the Server

```bash
lsof -ti :8000 | xargs kill -9
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the chat UI |
| `POST` | `/chat` | Streaming chat endpoint (SSE) |
| `GET` | `/vocab_info` | Returns stem count per level |
| `GET` | `/debug-stream` | Test SSE without the model (5 tokens, 1 s apart) |

### POST `/chat` — request body

```json
{
  "message": "Wie geht es dir?",
  "level": "A1",
  "history": [
    {"role": "user", "content": "Hallo"},
    {"role": "assistant", "content": "Hallo! Wie heißt du?"}
  ]
}
```

### POST `/chat` — streaming response (Server-Sent Events)

```
data: {"token": "Gut"}
data: {"token": ", danke"}
data: {"token": "!"}
data: {"done": true, "full": "Gut, danke!"}
```
