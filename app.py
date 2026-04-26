"""
German Vocabulary-Controlled Chatbot
Uses DiscoResearch/Llama3-German-8B with a custom LogitsProcessor that
restricts generation to tokens derived from the CEFR-level vocabulary stems.
"""

import asyncio
import csv
import json
import os
import re
import signal
import sys
import threading
from pathlib import Path
from typing import Optional


def _force_exit(sig, frame):
    print("\nInterrupted — shutting down.", flush=True)
    os._exit(0)

signal.signal(signal.SIGINT, _force_exit)
signal.signal(signal.SIGTERM, _force_exit)

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    TextIteratorStreamer,
)
# Silence the "pad_token_id not set" warning at import time
import transformers
transformers.logging.set_verbosity_error()

app = FastAPI()
BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Vocabulary loading  (cumulative: each level includes all levels below it)
# ---------------------------------------------------------------------------

def load_vocabulary() -> dict[str, Optional[list[str]]]:
    stems: dict[str, set] = {"A1": set(), "A2": set(), "B1": set()}
    with open(BASE_DIR / "dictionary_a1a2b1_onlystems.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lvl = row["level"].strip()
            if lvl in stems:
                stems[lvl].add(row["stem"].strip().lower())

    cumulative = {
        "A1": sorted(stems["A1"]),
        "A2": sorted(stems["A1"] | stems["A2"]),
        "B1": sorted(stems["A1"] | stems["A2"] | stems["B1"]),
        "B2": None,  # no restriction
    }
    for lvl, lst in cumulative.items():
        if lst is not None:
            print(f"  {lvl}: {len(lst):,} cumulative stems")
    return cumulative


VOCAB: dict[str, Optional[list]] = load_vocabulary()

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_ID = "DiscoResearch/Llama3-German-8B"

print(f"\nLoading tokenizer and model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Device / dtype selection:
#   CUDA  → float16  (fastest, full GPU acceleration)
#   MPS   → float16  (Apple Silicon GPU; MPS does NOT support bfloat16)
#   CPU   → float32  (fallback; bfloat16/float16 give no benefit on CPU)
# An 8B model in float16 needs ~16 GB.
# CUDA:  use float16 on GPU VRAM (fast).
# MPS:   Apple Silicon GPUs typically have 8 GB shared memory — not enough
#        for 8B float16.  Fall through to CPU.
# CPU:   use float16 to halve RAM vs float32 (~16 GB instead of ~32 GB).
if torch.cuda.is_available():
    _DEVICE = "cuda"
    _DTYPE  = torch.float16
else:
    _DEVICE = "cpu"
    _DTYPE  = torch.float16   # float16 on CPU = half the RAM of float32

print(f"  Using device: {_DEVICE.upper()}  |  dtype: {_DTYPE}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=_DTYPE,
    device_map=_DEVICE,
    low_cpu_mem_usage=True,
)
print(f"  Model loaded on {_DEVICE.upper()}\n")

model.eval()

# ---------------------------------------------------------------------------
# Constrained generation: pre-compute allowed token IDs per level
# ---------------------------------------------------------------------------

# Common German inflection suffixes appended to each stem so that inflected
# forms are covered by the token-level filter.
_ENDINGS = [
    "", "e", "en", "er", "es", "em", "et", "est", "st", "n", "t", "s",
    "te", "ten", "ter", "tes", "tem", "ste", "sten", "ster", "stes", "stem",
    "ung", "ungen",
    "lich", "liche", "lichen", "licher", "liches", "lichem",
    "keit", "keiten", "heit", "heiten",
    "isch", "ische", "ischen", "ischer", "isches", "ischem",
    "ig", "ige", "igen", "iger", "iges", "igem",
    "bar", "bare", "baren", "barer", "bares", "barem",
    "los", "lose", "losen", "loser", "loses", "losem",
    "voll", "volle", "vollen", "voller", "volles", "vollem",
    "sam", "same", "samen",
    "haft", "hafte", "haften",
]

_TOKEN_CACHE: dict[str, set] = {}


def _build_allowed_ids(stems: list[str]) -> set[int]:
    """
    Build the set of token IDs that may appear when generating words whose
    stems are in `stems`.  Always includes: special tokens, whitespace,
    punctuation, and digit tokens.
    """
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    allowed: set[int] = set()

    # 1. Always allow: special / non-alphabetic tokens
    for token, tid in vocab.items():
        # Strip common BPE space-markers before checking
        clean = token.replace("▁", "").replace("Ġ", "").replace("Ċ", "").strip()
        if not clean:
            allowed.add(tid)
            continue
        # Allow punctuation, digits, whitespace, and mixed tokens starting
        # with a non-letter (e.g. "," / "." / "0–9" / newline tokens).
        if not clean[0].isalpha():
            allowed.add(tid)

    # 2. Explicitly add all special-token IDs the model uses
    for special_id in [
        tokenizer.eos_token_id,
        tokenizer.bos_token_id,
        tokenizer.pad_token_id,
    ]:
        if special_id is not None:
            allowed.add(special_id)

    # Also add any <|...|> control tokens Llama 3 uses
    for name in tokenizer.special_tokens_map.values():
        if isinstance(name, str):
            tid = tokenizer.convert_tokens_to_ids(name)
            if tid is not None and 0 <= tid < vocab_size:
                allowed.add(tid)
        elif isinstance(name, list):
            for n in name:
                tid = tokenizer.convert_tokens_to_ids(n)
                if tid is not None and 0 <= tid < vocab_size:
                    allowed.add(tid)

    # 3. Tokenise every (stem + ending) combination and collect all sub-tokens
    word_forms: set[str] = set()
    for stem in stems:
        for ending in _ENDINGS:
            w = stem + ending
            word_forms.add(w)
            word_forms.add(w.capitalize())

    for word in word_forms:
        # Llama 3 / sentencepiece treats a leading space as "start-of-word";
        # tokenise both variants to catch all sub-token representations.
        for prefix in ("", " "):
            ids = tokenizer.encode(prefix + word, add_special_tokens=False)
            allowed.update(ids)

    print(f"    Token cache built: {len(allowed):,} allowed IDs "
          f"(vocab size {vocab_size:,})")
    return allowed


def get_allowed_ids(level: str) -> Optional[set[int]]:
    if level == "B2":
        return None
    if level not in _TOKEN_CACHE:
        print(f"  Building token cache for level {level}…")
        _TOKEN_CACHE[level] = _build_allowed_ids(VOCAB[level])
    return _TOKEN_CACHE[level]


class VocabLogitsProcessor(LogitsProcessor):
    """Sets logits of disallowed tokens to -inf at every generation step."""

    def __init__(self, allowed_ids: set[int], vocab_size: int):
        # Boolean mask: True where a token is *disallowed*
        mask = torch.ones(vocab_size, dtype=torch.bool)
        for tid in allowed_ids:
            if 0 <= tid < vocab_size:
                mask[tid] = False
        self._disallowed = mask  # shape (vocab_size,)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        mask = self._disallowed.to(scores.device)
        scores[:, mask] = float("-inf")
        return scores


# ---------------------------------------------------------------------------
# System-prompt builder
# ---------------------------------------------------------------------------

_LEVEL_DESC = {
    "A1": "Anfänger (A1-Niveau)",
    "A2": "Grundstufe (A2-Niveau)",
    "B1": "Mittelstufe (B1-Niveau)",
    "B2": "Obere Mittelstufe (B2-Niveau)",
}


def build_system_prompt(level: str) -> str:
    if level == "B2":
        return (
            "Du bist ein freundlicher Deutschlehrer und Gesprächspartner. "
            "Antworte immer auf Deutsch in klaren, verständlichen Sätzen."
        )

    stems = VOCAB[level]
    # Include the full stem list in the prompt so the model has explicit guidance
    stem_str = ", ".join(stems)
    return (
        f"Du bist ein freundlicher Deutschlehrer für {_LEVEL_DESC[level]}-Lernende. "
        f"STRIKTE REGEL: Verwende AUSSCHLIESSLICH Wörter, deren Grundform (Stamm) "
        f"in der folgenden erlaubten Wortliste vorkommt. "
        f"Verwende KEIN einziges Wort, das nicht aus diesen Stämmen ableitbar ist. "
        f"Halte Sätze kurz und einfach. Antworte immer auf Deutsch.\n\n"
        f"Erlaubte Wortstämme: {stem_str}"
    )


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    level: str = "A1"
    history: list = []  # list of {"role": ..., "content": ...} dicts


@app.post("/chat")
async def chat(req: ChatRequest):
    level = req.level.upper()
    if level not in ("A1", "A2", "B1", "B2"):
        return {"error": "level must be A1, A2, B1, or B2"}

    messages = [{"role": "system", "content": build_system_prompt(level)}]
    for turn in req.history[-8:]:  # keep last 8 turns as context
        messages.append(turn)
    messages.append({"role": "user", "content": req.message})

    # Tokenise with the model's chat template
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(_DEVICE)
    except Exception:
        # Fallback: manual ChatML-style format when no chat_template is set
        prompt = ""
        for m in messages:
            role = m["role"].capitalize()
            prompt += f"{role}: {m['content']}\n"
        prompt += "Assistant:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(_DEVICE)

    attention_mask = torch.ones_like(input_ids)

    # Build logits processor list
    processors = []
    allowed_ids = get_allowed_ids(level)
    if allowed_ids:
        vocab_size = len(tokenizer.get_vocab())
        processors.append(VocabLogitsProcessor(allowed_ids, vocab_size))

    streamer = TextIteratorStreamer(
        tokenizer, skip_special_tokens=True, skip_prompt=True
    )

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.15,
        streamer=streamer,
    )
    if processors:
        gen_kwargs["logits_processor"] = processors

    # Capture any exception from the generation thread so we can forward it.
    gen_error: list = []

    def run_generation():
        try:
            model.generate(**gen_kwargs)
        except Exception as exc:
            gen_error.append(exc)
            # Put a sentinel so the streamer iterator unblocks
            streamer.text_queue.put(None)

    thread = threading.Thread(target=run_generation, daemon=True)
    thread.start()

    # Sentinel used to signal end-of-stream without relying on StopIteration.
    # In Python 3.7+ StopIteration raised inside run_in_executor is silently
    # converted to RuntimeError, so catching it directly is unreliable.
    _END = object()

    def _next_chunk(it):
        try:
            return next(it)
        except StopIteration:
            return _END

    async def event_stream():
        full = ""
        loop = asyncio.get_event_loop()
        streamer_iter = iter(streamer)
        try:
            while True:
                chunk = await loop.run_in_executor(None, _next_chunk, streamer_iter)
                if chunk is _END:
                    break
                print(f"[stream] token: {repr(chunk)}", flush=True)
                full += chunk
                yield f"data: {json.dumps({'token': chunk})}\n\n"

            if gen_error:
                err_msg = str(gen_error[0])
                print(f"[stream] generation error: {err_msg}", flush=True)
                yield f"data: {json.dumps({'error': err_msg})}\n\n"
            else:
                print(f"[stream] done, total chars: {len(full)}", flush=True)
                yield f"data: {json.dumps({'done': True, 'full': full})}\n\n"
        except Exception as exc:
            print(f"[stream] unexpected error: {exc}", flush=True)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.get("/debug-stream")
async def debug_stream():
    """Streams 5 test tokens 1 s apart — verifies SSE works without the model."""
    async def gen():
        for i in range(1, 6):
            await asyncio.sleep(1)
            yield f"data: {json.dumps({'token': f'token{i} '})}\n\n"
        yield f"data: {json.dumps({'done': True, 'full': 'token1 token2 token3 token4 token5 '})}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.get("/vocab_info")
async def vocab_info():
    return {
        level: len(stems) if stems else "unlimited"
        for level, stems in VOCAB.items()
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
