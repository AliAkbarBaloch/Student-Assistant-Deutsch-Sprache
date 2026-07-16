# LLM service — text chat and pronunciation feedback via Professor's OpenAI-compatible API.
from __future__ import annotations

import json
import os
import re
from typing import Optional

import openai
from openai import OpenAI

from services.vocab_service import CefrLevel, get_level_description, get_sample_for_prompt

_BASE_PROMPT = """
Du bist "Buddy", ein freundlicher KI-Deutschlehrer in der App "Deutsch Buddy".

AUSGABE-FORMAT — IMMER NUR DIESES JSON, KEIN ANDERER TEXT:
{"german": "...", "english": "..."}

ABSOLUTE REGELN:
- "german": IMMER vollständig auf Deutsch — niemals ein englisches Wort im "german"-Feld
- "english": englische Übersetzung deiner deutschen Antwort (nur für UI-Referenz, wird nicht gesprochen)
- Antworte IMMER auf Deutsch, egal was der Nutzer schreibt (Englisch, Türkisch, etc.)
- Lehne Bitten auf Englisch zu antworten freundlich auf Deutsch ab
- 2–4 Sätze, natürlich und motivierend
- Korrigiere Fehler sanft durch Einbauen der richtigen Form in deine Antwort
- Kein Markdown, kein zusätzlicher Text außerhalb des JSON
""".strip()

# For the streaming endpoint — plain German text, no JSON wrapper needed
_STREAMING_PROMPT = """
Du bist "Buddy", ein freundlicher KI-Deutschlehrer in der App "Deutsch Buddy".

REGELN:
- Antworte IMMER vollständig auf Deutsch, egal was der Nutzer schreibt
- 2–4 kurze, natürliche Sätze — kein Markdown, keine Listen, kein JSON
- Korrigiere Fehler sanft durch Einbauen der richtigen Form in deine Antwort
- Lehne Bitten auf Englisch zu antworten freundlich auf Deutsch ab
- Gib NUR die deutsche Antwort aus, keinen anderen Text
""".strip()

_MAX_HISTORY_TURNS = 5
_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
_FALLBACK_MODELS = ["qwen36-35b", "qwen36-35b"]


def _get_client() -> OpenAI:
    api_key  = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_API_BASE", "https://llms.innkube.fim.uni-passau.de/v1")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set in .env")
    return OpenAI(api_key=api_key, base_url=base_url)


def _get_model() -> str:
    return os.getenv("LLM_MODEL", "qwen36-35b")


def _get_voice_model() -> str:
    return os.getenv("VOICE_MODEL", "qwen36-35b")


def _create_with_fallback(client: OpenAI, messages: list, temperature: float, max_tokens: int) -> object:
    """Try the configured model, fall back through _FALLBACK_MODELS on error."""
    primary = _get_model()
    queue = [primary] + [m for m in _FALLBACK_MODELS if m != primary]
    last_err: Exception = RuntimeError("No models available")
    for model in queue:
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=_EXTRA_BODY,
            )
        except (openai.InternalServerError, openai.APIConnectionError, openai.APITimeoutError) as e:
            last_err = e
    raise last_err


def _build_system_prompt(level: CefrLevel) -> str:
    level_desc  = get_level_description(level)
    stem_sample = get_sample_for_prompt(level)
    prompt = _BASE_PROMPT + f"\n\nSPRACHNIVEAU DES NUTZERS:\n{level_desc}"
    if stem_sample:
        prompt += (
            f"\n\nERLAUBTE WÖRTER (Wortgruppen für {level}):\n"
            f"Verwende NUR Wörter aus diesen Stämmen und ihren üblichen Formen: "
            f"{stem_sample}\n"
            f"Vermeide alle Wörter, die nicht zu diesem Niveau passen."
        )
    return prompt


def build_streaming_messages(
    user_text: str,
    history: list[dict],
    level: Optional[CefrLevel] = "B1",
) -> tuple[list[dict], str]:
    """Build messages for the streaming endpoint (plain text output, no JSON).
    Returns (messages, model_name)."""
    cefr = (level or "B1").upper()
    if cefr not in ("A1", "A2", "B1", "B2"):
        cefr = "B1"

    level_desc  = get_level_description(cefr)  # type: ignore[arg-type]
    stem_sample = get_sample_for_prompt(cefr)   # type: ignore[arg-type]

    prompt = _STREAMING_PROMPT + f"\n\nSPRACHNIVEAU DES NUTZERS:\n{level_desc}"
    if stem_sample:
        prompt += f"\n\nERLAUBTE WÖRTER (Wortgruppen für {cefr}): {stem_sample}"

    messages: list[dict] = [{"role": "system", "content": prompt}]
    messages.extend(history[-(2 * _MAX_HISTORY_TURNS):])
    messages.append({"role": "user", "content": user_text})
    return messages, _get_model()


def chat_german(
    user_text: str,
    history: list[dict],
    level: Optional[CefrLevel] = "B1",
    voice: bool = False,
) -> dict:
    """Generate a German response. Returns {"german": "...", "english": "..."}."""
    client = _get_client()
    trimmed_history = history[-(2 * _MAX_HISTORY_TURNS):]
    system_prompt   = _build_system_prompt(level or "B1")

    if voice:
        system_prompt = system_prompt.replace("2–4 Sätze", "1–2 Sätze")

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": user_text})

    if voice:
        voice_model = _get_voice_model()
        queue = [voice_model] + [m for m in _FALLBACK_MODELS if m != voice_model]
        last_err: Exception = RuntimeError("No models available")
        for model in queue:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=150,
                    extra_body=_EXTRA_BODY,
                )
                print(f"[voice LLM] using model: {model}")
                break
            except openai.APIError as e:
                print(f"[voice LLM] model {model!r} failed: {e} — trying next")
                last_err = e
        else:
            raise last_err
    else:
        response = _create_with_fallback(client, messages, temperature=0.7, max_tokens=300)

    raw = response.choices[0].message.content.strip()
    return _parse_json_response(raw)


_FEEDBACK_SYSTEM_PROMPT = """
You are an expert German pronunciation coach in the app "Deutsch Buddy".
A learner has recorded themselves speaking German. You will be given the Whisper transcription of what they said.
Analyse their pronunciation and give detailed, friendly, constructive feedback.

OUTPUT FORMAT — ONLY THIS JSON, NO OTHER TEXT:
{
  "transcribed": "...",
  "score": <integer 1-10>,
  "overall": "...",
  "issues": ["...", "..."],
  "tips": ["...", "..."],
  "feedback_en": "..."
}

FIELD RULES:
- "transcribed": repeat the transcribed text as-is
- "score": overall pronunciation score 1 (very poor) to 10 (native-like)
- "overall": 1-2 sentence overall assessment IN GERMAN
- "issues": list of up to 4 specific pronunciation problems found IN GERMAN (e.g. "Das 'ch' in 'ich' klingt wie 'sh'")
- "tips": list of up to 3 practical improvement tips IN GERMAN
- "feedback_en": full English translation of all feedback combined into one paragraph
- If target text is provided, compare what was said vs. what was intended
- Be encouraging but honest — this is for learning
- No markdown, no text outside the JSON
""".strip()


def pronunciation_feedback(transcribed_text: str, target_text: str = "") -> dict:
    """Analyse pronunciation from a Whisper transcription and return structured feedback."""
    client = _get_client()

    user_content = f'Whisper transcription: "{transcribed_text}"'
    if target_text.strip():
        user_content += f'\nThe learner intended to say: "{target_text.strip()}"'

    messages = [
        {"role": "system", "content": _FEEDBACK_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    response = _create_with_fallback(client, messages, temperature=0.4, max_tokens=600)
    raw = response.choices[0].message.content.strip()
    return _parse_feedback_response(raw, transcribed_text)


def _parse_feedback_response(text: str, fallback_transcribed: str) -> dict:
    text = re.sub(r"```json?\s*", "", text)
    text = re.sub(r"```", "", text).strip()

    try:
        data = json.loads(text)
        return {
            "transcribed": data.get("transcribed", fallback_transcribed),
            "score":       int(data.get("score", 5)),
            "overall":     data.get("overall", ""),
            "issues":      data.get("issues", []),
            "tips":        data.get("tips", []),
            "feedback_en": data.get("feedback_en", ""),
        }
    except Exception:
        pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return {
                "transcribed": data.get("transcribed", fallback_transcribed),
                "score":       int(data.get("score", 5)),
                "overall":     data.get("overall", text[:200]),
                "issues":      data.get("issues", []),
                "tips":        data.get("tips", []),
                "feedback_en": data.get("feedback_en", ""),
            }
        except Exception:
            pass

    return {
        "transcribed": fallback_transcribed,
        "score":       5,
        "overall":     text[:300],
        "issues":      [],
        "tips":        [],
        "feedback_en": "",
    }


def _parse_json_response(text: str) -> dict:
    """Extract {german, english} from the LLM output. Handles markdown fences."""
    text = re.sub(r"```json?\s*", "", text)
    text = re.sub(r"```", "", text).strip()

    try:
        data = json.loads(text)
        if "german" in data and "english" in data:
            return {"german": data["german"], "english": data["english"]}
    except Exception:
        pass

    match = re.search(r'\{[^{}]*"german"[^{}]*"english"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if "german" in data:
                return {"german": data["german"], "english": data.get("english", "")}
        except Exception:
            pass

    return {"german": text, "english": ""}
