# Loads the CEFR vocabulary CSV and provides helpers used to ground the LLM's
# word choices. Levels are cumulative: A2 includes A1 stems, B1 includes A1+A2,
# B2 includes A1+A2+B1+B2.
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import FrozenSet, Literal

CefrLevel = Literal["A1", "A2", "B1", "B2"]

_CSV_PATH = Path(__file__).resolve().parent / "dictionary_a1a2b1_onlystems.csv"
_LEVEL_STEMS: dict[str, set[str]] = {"A1": set(), "A2": set(), "B1": set(), "B2": set()}


def _load() -> None:
    if not _CSV_PATH.exists():
        return
    with _CSV_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lvl  = row.get("level", "").strip().upper()
            stem = row.get("stem", "").strip().lower()
            if lvl in _LEVEL_STEMS and stem:
                _LEVEL_STEMS[lvl].add(stem)


_load()

# Build cumulative sets once at import time
_CUMULATIVE: dict[str, FrozenSet[str]] = {
    "A1": frozenset(_LEVEL_STEMS["A1"]),
    "A2": frozenset(_LEVEL_STEMS["A1"] | _LEVEL_STEMS["A2"]),
    "B1": frozenset(_LEVEL_STEMS["A1"] | _LEVEL_STEMS["A2"] | _LEVEL_STEMS["B1"]),
    "B2": frozenset(_LEVEL_STEMS["A1"] | _LEVEL_STEMS["A2"] | _LEVEL_STEMS["B1"] | _LEVEL_STEMS["B2"]),
}

_SAMPLE_SIZE = 200  # how many stems to inject into the prompt


def get_sample_for_prompt(level: CefrLevel, seed: int = 42) -> str:
    """Return a comma-separated sample of stems for the given level, drawn
    from the cumulative stem set (all levels at or below `level`)."""
    stems  = sorted(_CUMULATIVE[level])
    rng    = random.Random(seed)
    sample = rng.sample(stems, min(_SAMPLE_SIZE, len(stems)))
    return ", ".join(sorted(sample))


def get_level_description(level: CefrLevel) -> str:
    """Return the CEFR-level instruction that goes into the system prompt."""
    descriptions = {
        "A1": (
            "The user is a BEGINNER (CEFR A1). "
            "Use ONLY the simplest everyday words — greetings, numbers, colours, family, food, "
            "basic verbs (sein, haben, gehen, kommen, machen). "
            "Sentences must be very short (max 8 words). "
            "Never use subordinate clauses, the Konjunktiv, or complex grammar."
        ),
        "A2": (
            "The user is an ELEMENTARY learner (CEFR A2). "
            "Use simple, common vocabulary for everyday topics: shopping, directions, time, weather, hobbies. "
            "Keep sentences short and clear (max 12 words). "
            "Avoid complex subordinate clauses and advanced grammar."
        ),
        "B1": (
            "The user is an INTERMEDIATE learner (CEFR B1). "
            "Use everyday vocabulary and common idiomatic expressions. "
            "You may use subordinate clauses, past tenses, and modal verbs freely. "
            "Avoid rare, specialist, or B2+ vocabulary."
        ),
        "B2": (
            "The user is an UPPER-INTERMEDIATE learner (CEFR B2). "
            "Use varied vocabulary, idiomatic expressions, and complex sentence structures freely. "
            "You may discuss abstract topics, give opinions, and use the Konjunktiv II."
        ),
    }
    return descriptions[level]


def get_vocab_instruction(level: CefrLevel) -> str:
    """Return the permitted-vocabulary block to append to the system prompt,
    or "" if there are no stems to inject.

    A1/A2/B1 get a hard restriction ("use ONLY these stems") since the goal
    at those levels is to keep the learner inside a narrow, known set. B2
    gets the same stem sample but framed as encouragement rather than a
    ceiling, so the model still has room for the varied, idiomatic language
    a B2 learner is expected to handle.
    """
    stem_sample = get_sample_for_prompt(level)
    if not stem_sample:
        return ""
    if level == "B2":
        return (
            f"\n\nB2-WORTSCHATZ (Wortstämme, die dieses Niveau kennzeichnen):\n"
            f"Bevorzuge, wo passend, Wörter aus diesen Stämmen und ihren üblichen Formen: "
            f"{stem_sample}\n"
            f"Du darfst darüber hinaus jedes andere Wort verwenden, das für ein "
            f"vielseitiges, natürliches B2-Gespräch angemessen ist."
        )
    return (
        f"\n\nERLAUBTE WÖRTER (Wortstämme für {level}):\n"
        f"Verwende NUR Wörter aus diesen Stämmen und ihren üblichen Formen: "
        f"{stem_sample}\n"
        f"Vermeide alle Wörter, die nicht zu diesem Niveau passen."
    )

