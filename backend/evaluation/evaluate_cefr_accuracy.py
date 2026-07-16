from __future__ import annotations

import argparse
import csv as csv_module
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# -- Path bootstrap: allow `from services.x import y` when run from backend/ --
_BACKEND      = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND.parent
sys.path.insert(0, str(_BACKEND))

load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_BACKEND / ".env")  # also try backend/.env, whichever exists

from services.llm_service import chat_german  # noqa: E402

# -- Config -------------------------------------------------------------------
_CSV_PATH    = _BACKEND / "services" / "dictionary_a1a2b1_onlystems.csv"
_OUTPUT_DIR  = Path(os.getenv("EVAL_OUTPUT_DIR", str(Path(__file__).parent / "output")))
_API_DELAY   = float(os.getenv("EVAL_API_DELAY_SEC", "1.0"))
_MAX_RETRIES = int(os.getenv("EVAL_MAX_RETRIES", "3"))

_LEVELS: list[str] = ["A1", "A2", "B1", "B2"]
_LEVEL_RANK: dict[str, int] = {"A1": 0, "A2": 1, "B1": 2, "B2": 3}
_PROMPTS_PER_LEVEL = 20

# Common German function words - always allowed, skipped from scoring so the
# analysis focuses on actual vocabulary choice ("main vocabulary") rather than
# grammar glue that appears at every level regardless of difficulty.
_STOPWORDS: set[str] = {
    # articles / determiners
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
    "einer", "eines", "kein", "keine", "keinen", "keinem", "keiner",
    # pronouns
    "ich", "du", "er", "sie", "es", "wir", "ihr", "mich", "dich", "sich",
    "uns", "euch", "mir", "dir", "ihm", "ihn", "ihnen", "mein", "meine",
    "meinen", "meiner", "meinem", "dein", "deine", "deinen", "deiner",
    "sein", "seine", "seinen", "seiner", "unser", "unsere", "euer", "eure",
    "man", "wer", "was", "wen", "wem",
    # conjunctions / connectors
    "und", "aber", "oder", "denn", "sondern", "dass", "weil", "wenn", "als",
    "ob", "damit", "obwohl", "waehrend", "während", "bevor", "nachdem",
    "sowie", "doch", "also", "jedoch",
    # prepositions
    "in", "an", "auf", "aus", "bei", "mit", "nach", "seit", "von", "zu",
    "zur", "zum", "fuer", "für", "gegen", "ohne", "um", "durch", "ueber",
    "über", "unter", "vor", "hinter", "neben", "zwischen",
    # common auxiliary / modal verb forms
    "bin", "bist", "ist", "sind", "seid", "war", "warst", "waren", "wart",
    "habe", "hast", "hat", "haben", "habt", "hatte", "hattest", "hatten",
    "werde", "wirst", "wird", "werden", "werdet",
    "kann", "kannst", "koennen", "können", "koennt", "könnt", "konnte",
    "muss", "musst", "muessen", "müssen", "muesst", "müsst", "musste",
    "will", "willst", "wollen", "wollt", "wollte",
    "soll", "sollst", "sollen", "sollt", "sollte",
    "darf", "darfst", "duerfen", "dürfen", "duerft", "dürft", "durfte",
    "mag", "magst", "moegen", "mögen", "moegt", "mögt",
    "moechte", "möchte", "moechtest", "möchtest", "moechten", "möchten",
    # adverbs / fillers / question words that appear at every level
    "nicht", "auch", "noch", "schon", "nur", "sehr", "so", "dann", "hier",
    "da", "jetzt", "immer", "wieder", "mal", "ja", "nein", "bitte", "danke",
    "wie", "was", "wo", "wann", "warum", "wieso", "welche", "welcher",
    "welches", "alle", "alles", "etwas", "nichts", "viel", "viele", "mehr",
    "am", "im", "beim", "vom", "ans", "aufs", "ins",
}


# ==============================================================================
# Test prompts - 20 per CEFR level, 80 total
# ==============================================================================
TEST_PROMPTS: dict[str, list[str]] = {
    "A1": [
        "Hallo! Wie heißt du?",
        "Was ist das?",
        "Ich heiße Anna. Und du?",
        "Wie alt bist du?",
        "Wo wohnst du?",
        "Ich wohne in Berlin.",
        "Was magst du essen?",
        "Wie viele Geschwister hast du?",
        "Was ist deine Lieblingsfarbe?",
        "Das ist ein Hund.",
        "Guten Morgen! Wie geht es dir?",
        "Was machst du?",
        "Ich bin müde.",
        "Danke schön!",
        "Ich mag Äpfel.",
        "Wie spät ist es?",
        "Ich trinke gerne Wasser.",
        "Hast du eine Katze?",
        "Wie ist dein Name?",
        "Ich komme aus Spanien.",
    ],
    "A2": [
        "Wie komme ich zum Bahnhof?",
        "Was kostet das?",
        "Ich möchte ein Kilo Äpfel, bitte.",
        "Wann fährt der nächste Bus?",
        "Wie ist das Wetter heute?",
        "Was hast du gestern gemacht?",
        "Ich gehe gerne ins Kino.",
        "Was machst du in deiner Freizeit?",
        "Ich suche ein Zimmer.",
        "Können Sie mir helfen?",
        "Um wie viel Uhr beginnt der Kurs?",
        "Ich lerne seit drei Monaten Deutsch.",
        "Was isst du zum Frühstück?",
        "Welchen Beruf hat dein Vater?",
        "Ich habe Kopfschmerzen.",
        "Wie war dein Wochenende?",
        "Ich möchte einen Tisch für zwei Personen reservieren.",
        "Wo kann ich Briefmarken kaufen?",
        "Was für Hobbys hast du?",
        "Wie oft gehst du einkaufen?",
    ],
    "B1": [
        "Was denkst du über das Leben in einer Großstadt?",
        "Erzähl mir von deinem letzten Urlaub.",
        "Was sind deine Pläne für die Zukunft?",
        "Warum lernst du Deutsch?",
        "Wie war deine Kindheit?",
        "Was gefällt dir an Deutschland?",
        "Beschreibe deinen Alltag.",
        "Was machst du, wenn du gestresst bist?",
        "Welche Musik hörst du gerne und warum?",
        "Wie findest du die Deutschen?",
        "Was würdest du mit einer Million Euro machen?",
        "Was sind die Unterschiede zwischen deiner Heimat und Deutschland?",
        "Wie wichtig ist Gesundheit für dich?",
        "Was hast du letztes Wochenende gemacht?",
        "Erzähl mir über einen Film, den du magst.",
        "Wie stellst du dir dein Leben in zehn Jahren vor?",
        "Was war die größte Herausforderung in deinem Leben?",
        "Wie gehst du mit Konflikten um?",
        "Was bedeutet Erfolg für dich?",
        "Welche Rolle spielt Freundschaft in deinem Leben?",
    ],
    "B2": [
        "Was sind die Vor- und Nachteile der sozialen Medien?",
        "Wie beeinflusst die Globalisierung unsere Gesellschaft?",
        "Was bedeutet Freiheit für dich?",
        "Welche Rolle spielt Sprache in der Identitätsbildung?",
        "Diskutiere die Auswirkungen des Klimawandels.",
        "Wie hat sich die Arbeitswelt durch die Digitalisierung verändert?",
        "Was denkst du über Minimalismus als Lebensphilosophie?",
        "Inwiefern prägt die Kultur eines Landes seinen Humor?",
        "Welche ethischen Fragen stellt die künstliche Intelligenz?",
        "Wie sollte man mit historischer Schuld umgehen?",
        "Was sind die psychologischen Auswirkungen von Social Media auf Jugendliche?",
        "Diskutiere das Konzept der Work-Life-Balance.",
        "Wie könnte eine gerechte Gesellschaft aussehen?",
        "Was hältst du von bedingungslosem Grundeinkommen?",
        "Erkläre, warum Mehrsprachigkeit von Vorteil ist.",
        "Wie beeinflusst die Wirtschaft politische Entscheidungen?",
        "Was sind die Herausforderungen der modernen Bildung?",
        "Wie verändert künstliche Intelligenz den Arbeitsmarkt?",
        "Welche Verantwortung tragen Unternehmen für die Umwelt?",
        "Wie wichtig ist kulturelle Vielfalt in einer globalisierten Welt?",
    ],
}

assert all(len(v) == _PROMPTS_PER_LEVEL for v in TEST_PROMPTS.values()), (
    "Each CEFR level must have exactly 20 prompts."
)


# ==============================================================================
# Dictionary loading (exact match, no stemming)
# ==============================================================================

def load_word_map() -> dict[str, str]:
    """Load the CEFR CSV into a word -> level mapping (A1/A2/B1 only, exact
    strings as they appear in the CSV - no stemming/prefix logic applied)."""
    word_map: dict[str, str] = {}
    with _CSV_PATH.open(encoding="utf-8") as f:
        for row in csv_module.DictReader(f):
            lvl  = row.get("level", "").strip().upper()
            stem = row.get("stem", "").strip().lower()
            if lvl in ("A1", "A2", "B1") and stem:
                word_map[stem] = lvl
    return word_map


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"-", " ", text)                     # split compounds on hyphens
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return [t for t in text.split() if t and not t.isdigit()]


# ==============================================================================
# Compliance scoring
# ==============================================================================

def score_compliance(text: str, target_level: str, word_map: dict[str, str]) -> dict:
    """Score a single response's vocabulary against the target CEFR level.

    Stopwords are dropped first. Of the remaining ("main vocabulary") words,
    each is looked up directly in word_map:
      - present & level <= target  -> ok
      - present & level >  target  -> violation
      - absent from the CSV        -> unknown (not penalised)
    """
    tokens      = tokenize(text)
    target_rank = _LEVEL_RANK.get(target_level, 3)

    content_words = [t for t in tokens if t not in _STOPWORDS]

    ok, violation, unknown = 0, 0, 0
    violation_words: list[str] = []

    for word in content_words:
        level = word_map.get(word)
        if level is None:
            unknown += 1
            continue
        if _LEVEL_RANK[level] <= target_rank:
            ok += 1
        else:
            violation += 1
            violation_words.append(f"{word}({level})")

    scored_total  = ok + violation
    compliance_pct = round(ok / scored_total * 100, 1) if scored_total > 0 else None

    return {
        "compliance_pct":  compliance_pct,
        "ok_words":        ok,
        "violation_words": violation,
        "unknown_words":   unknown,
        "total_tokens":    len(tokens),
        "content_tokens":  len(content_words),
        "violations_sample": violation_words[:10],
    }


# ==============================================================================
# Retry helper
# ==============================================================================

def call_with_retry(fn, *args, max_retries: int = _MAX_RETRIES, **kwargs):
    delay = 2.0
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - deliberately broad, this is a CLI eval tool
            last_err = e
            print(f"  [retry {attempt}/{max_retries}] {type(e).__name__}: {e}")
            time.sleep(delay)
            delay *= 2
    raise last_err  # type: ignore[misc]


# ==============================================================================
# Step 1 - Generate conversations
# ==============================================================================

def generate_conversations() -> list[dict]:
    """Send all 80 prompts to chat_german() and collect the responses.
    Also measures real wall-clock latency per call (time.perf_counter),
    excluding the artificial _API_DELAY throttle between calls, so the
    result reflects actual model response time, not our own rate limiting.
    """
    conversations: list[dict] = []
    total = sum(len(v) for v in TEST_PROMPTS.values())
    i = 0

    for level in _LEVELS:
        for prompt in TEST_PROMPTS[level]:
            i += 1
            print(f"[{i}/{total}] ({level}) {prompt!r}")
            t0 = time.perf_counter()
            try:
                result = call_with_retry(chat_german, prompt, [], level)
                german = result.get("german", "")
                english = result.get("english", "")
                error = None
            except Exception as e:  # noqa: BLE001
                german, english = "", ""
                error = str(e)
                print(f"  FAILED after retries: {error}")
            latency_sec = round(time.perf_counter() - t0, 3)
            print(f"  latency: {latency_sec}s")

            conversations.append({
                "level": level,
                "prompt": prompt,
                "response_de": german,
                "response_en": english,
                "error": error,
                "latency_sec": latency_sec,
            })
            time.sleep(_API_DELAY)

    # Real latency summary — printed and available for the results slide
    ok_latencies = [c["latency_sec"] for c in conversations if c["error"] is None]
    if ok_latencies:
        total_time = sum(ok_latencies)
        avg_time = total_time / len(ok_latencies)
        print("\n" + "=" * 60)
        print("LATENCY SUMMARY (measured, excludes the artificial delay)")
        print("=" * 60)
        print(f"Total wall-clock time for {len(ok_latencies)} successful calls: {total_time:.1f}s")
        print(f"Average latency per response: {avg_time:.2f}s")
        print(f"Min: {min(ok_latencies):.2f}s   Max: {max(ok_latencies):.2f}s")
        print("=" * 60)

    return conversations


# ==============================================================================
# Step 2 - Score conversations
# ==============================================================================

def score_conversations(conversations: list[dict]) -> tuple[list[dict], dict]:
    word_map = load_word_map()
    print(f"Loaded {len(word_map)} words from {_CSV_PATH.name}")

    scored: list[dict] = []
    for convo in conversations:
        if convo.get("error") or not convo.get("response_de"):
            scored.append({**convo, "compliance": None})
            continue
        compliance = score_compliance(convo["response_de"], convo["level"], word_map)
        scored.append({**convo, "compliance": compliance})

    # -- Aggregate per level --
    by_level: dict[str, list[float]] = defaultdict(list)
    for row in scored:
        c = row.get("compliance")
        if c and c["compliance_pct"] is not None:
            by_level[row["level"]].append(c["compliance_pct"])

    summary = {}
    for level in _LEVELS:
        scores = by_level[level]
        summary[level] = {
            "responses_scored": len(scores),
            "responses_total":  len(TEST_PROMPTS[level]),
            "avg_compliance_pct": round(sum(scores) / len(scores), 1) if scores else None,
            "min_compliance_pct": round(min(scores), 1) if scores else None,
            "max_compliance_pct": round(max(scores), 1) if scores else None,
        }

    all_scores = [s for scores in by_level.values() for s in scores]
    summary["OVERALL"] = {
        "responses_scored": len(all_scores),
        "responses_total":  sum(len(v) for v in TEST_PROMPTS.values()),
        "avg_compliance_pct": round(sum(all_scores) / len(all_scores), 1) if all_scores else None,
    }

    # -- Latency (measured wall-clock time per call, real data if generate step ran) --
    latencies = [row["latency_sec"] for row in scored
                 if row.get("latency_sec") is not None and row.get("error") is None]
    if latencies:
        summary["OVERALL"]["total_latency_sec"] = round(sum(latencies), 1)
        summary["OVERALL"]["avg_latency_sec"] = round(sum(latencies) / len(latencies), 2)
        summary["OVERALL"]["min_latency_sec"] = round(min(latencies), 2)
        summary["OVERALL"]["max_latency_sec"] = round(max(latencies), 2)

    return scored, summary


# ==============================================================================
# Reporting
# ==============================================================================

def print_report(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("CEFR VOCABULARY COMPLIANCE - SUMMARY")
    print("=" * 60)
    print(f"{'Level':<8}{'Scored':<10}{'Avg %':<10}{'Min %':<10}{'Max %':<10}")
    print("-" * 60)
    for level in _LEVELS:
        s = summary[level]
        avg = f"{s['avg_compliance_pct']}" if s["avg_compliance_pct"] is not None else "N/A"
        mn  = f"{s['min_compliance_pct']}" if s["min_compliance_pct"] is not None else "N/A"
        mx  = f"{s['max_compliance_pct']}" if s["max_compliance_pct"] is not None else "N/A"
        print(f"{level:<8}{s['responses_scored']}/{s['responses_total']:<7}{avg:<10}{mn:<10}{mx:<10}")
    print("-" * 60)
    o = summary["OVERALL"]
    avg = f"{o['avg_compliance_pct']}" if o["avg_compliance_pct"] is not None else "N/A"
    print(f"{'OVERALL':<8}{o['responses_scored']}/{o['responses_total']:<7}{avg:<10}")
    print("=" * 60)
    print("Note: B2 has no vocabulary restriction in the app, so a lower")
    print("compliance % there is expected and not a bug.")
    if "avg_latency_sec" in o:
        print()
        print(f"Avg latency per response: {o['avg_latency_sec']}s "
              f"(min {o['min_latency_sec']}s, max {o['max_latency_sec']}s, "
              f"total {o['total_latency_sec']}s)")


def save_outputs(conversations: list[dict], scored: list[dict], summary: dict) -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    conv_path = _OUTPUT_DIR / "conversations.json"
    conv_path.write_text(json.dumps(conversations, ensure_ascii=False, indent=2), encoding="utf-8")

    scored_path = _OUTPUT_DIR / f"scored_{ts}.json"
    scored_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = _OUTPUT_DIR / f"summary_{ts}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = _OUTPUT_DIR / f"scored_{ts}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv_module.writer(f)
        writer.writerow([
            "level", "prompt", "response_de", "compliance_pct",
            "ok_words", "violation_words", "unknown_words", "violations_sample",
        ])
        for row in scored:
            c = row.get("compliance") or {}
            writer.writerow([
                row["level"],
                row["prompt"],
                row["response_de"],
                c.get("compliance_pct", ""),
                c.get("ok_words", ""),
                c.get("violation_words", ""),
                c.get("unknown_words", ""),
                "; ".join(c.get("violations_sample", [])),
            ])

    print(f"\nSaved:\n  {conv_path}\n  {scored_path}\n  {summary_path}\n  {csv_path}")


# ==============================================================================
# CLI
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="CEFR vocabulary accuracy evaluator")
    parser.add_argument("--generate-only", action="store_true",
                         help="Only generate conversations, skip scoring")
    parser.add_argument("--score-only", action="store_true",
                         help="Only score an existing conversations JSON file")
    parser.add_argument("--input", type=str, default=None,
                         help="Path to conversations JSON (for --score-only)")
    args = parser.parse_args()

    if args.score_only:
        input_path = Path(args.input) if args.input else (_OUTPUT_DIR / "conversations.json")
        if not input_path.exists():
            print(f"ERROR: {input_path} not found. Run without --score-only first, "
                  f"or pass --input <path>.")
            sys.exit(1)
        conversations = json.loads(input_path.read_text(encoding="utf-8"))
        scored, summary = score_conversations(conversations)
        print_report(summary)
        save_outputs(conversations, scored, summary)
        return

    print(f"Generating {sum(len(v) for v in TEST_PROMPTS.values())} conversations "
          f"({_PROMPTS_PER_LEVEL} per level x {len(_LEVELS)} levels)...\n")
    conversations = generate_conversations()

    if args.generate_only:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _OUTPUT_DIR / "conversations.json"
        out_path.write_text(json.dumps(conversations, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved conversations to {out_path}")
        return

    print("\nScoring conversations against CEFR vocabulary...\n")
    scored, summary = score_conversations(conversations)
    print_report(summary)
    save_outputs(conversations, scored, summary)


if __name__ == "__main__":
    main()
