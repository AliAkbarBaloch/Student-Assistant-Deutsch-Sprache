# German Pronunciation Scorer (MVP)

This project compares two German speech recordings:

- **Reference audio** (good pronunciation)
- **Learner audio** (to evaluate)

It runs two open-source speech models:

1. **Whisper (`openai/whisper-small`)** for German transcription
2. **Wav2Vec2 eSpeak (`facebook/wav2vec2-xlsr-53-espeak-cv-ft`)** for phoneme-like decoding

Then it computes pronunciation metrics and returns user feedback.

## What It Does

- Transcribes both audio files to German text
- Extracts phoneme token sequences from both audio files
- Calculates:
  - Phoneme Error Rate (PER)
  - Text similarity
  - Overall pronunciation score
- Returns verdict and actionable feedback

## Project Structure

- `app.py` FastAPI backend and model inference logic
- `static/index.html` one-page UI
- `static/style.css` UI styling
- `static/script.js` frontend logic
- `requirements.txt` Python dependencies

## Quick Start

### 1. Create and activate a virtual environment

```bash
cd /Users/mustafa786/Desktop/phoneme_scorer
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start the app

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open in browser

- http://127.0.0.1:8000

## Notes

- First analysis call can be slow because model weights are downloaded and loaded.
- Recommended audio formats: WAV, FLAC, OGG.
- Keep recordings short (single sentence or short phrase) for faster response.
- Record in a quiet environment for better scoring quality.

## Limitations (Current MVP)

- Phoneme mismatch highlighting is token-level and simple.
- Accuracy depends on ASR quality and recording conditions.
- This does **not** yet provide forced alignment by word boundary.

## Next Improvements

- Add per-word timing and forced alignment (for exact error localization).
- Add confidence intervals and pronunciation heatmap.
- Add history of attempts and progress tracking.
- Add optional predefined sentence prompts for practice.
