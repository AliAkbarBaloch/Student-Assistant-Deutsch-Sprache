"""
vocab_service.py
================
Loads the CEFR vocabulary CSV once at startup and provides:
  - Cumulative stem sets per level (A1 ⊆ A1+A2 ⊆ A1+A2+B1)
  - A representative sample of stems to inject into the LLM system prompt
  - A plain-English description of each level for the prompt

Level hierarchy (cumulative):
  A1 → only A1 stems
  A2 → A1 + A2 stems
  B1 → A1 + A2 + B1 stems
  B2 → all stems + no additional restriction (advanced, free vocabulary)
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import FrozenSet, Literal

CefrLevel = Literal["A1", "A2", "B1", "B2"]

# ── CSV loading ────────────────────────────────────────────────────────────────

_CSV_PATH = Path(__file__).resolve().parent / "dictionary_a1a2b1_onlystems.csv"

# Raw sets per individual level
_LEVEL_STEMS: dict[str, set[str]] = {"A1": set(), "A2": set(), "B1": set(), "B2": set()}


def _load() -> None:
    """Parse the CSV and populate _LEVEL_STEMS on first import."""
    if not _CSV_PATH.exists():
        return
    with _CSV_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lvl  = row.get("level", "").strip().upper()
            stem = row.get("stem", "").strip().lower()
            if lvl in _LEVEL_STEMS and stem:
                _LEVEL_STEMS[lvl].add(stem)


_load()

# ── Cumulative sets ────────────────────────────────────────────────────────────

_CUMULATIVE: dict[str, FrozenSet[str]] = {
    "A1": frozenset(_LEVEL_STEMS["A1"]),
    "A2": frozenset(_LEVEL_STEMS["A1"] | _LEVEL_STEMS["A2"]),
    "B1": frozenset(_LEVEL_STEMS["A1"] | _LEVEL_STEMS["A2"] | _LEVEL_STEMS["B1"]),
    # B2 is unrestricted — CSV has no B2 entries; treat as full set
    "B2": frozenset(_LEVEL_STEMS["A1"] | _LEVEL_STEMS["A2"] | _LEVEL_STEMS["B1"]),
}

# ── Public helpers ─────────────────────────────────────────────────────────────

_SAMPLE_SIZE = 200   # number of stem examples injected into the prompt


def get_sample_for_prompt(level: CefrLevel, seed: int = 42) -> str:
    """
    Return a comma-separated string of up to _SAMPLE_SIZE representative stems
    for the given level. Used to ground the LLM's vocabulary choices.
    B2 returns an empty string (no restriction needed).
    """
    if level == "B2":
        return ""
    stems = sorted(_CUMULATIVE[level])
    rng   = random.Random(seed)
    sample = rng.sample(stems, min(_SAMPLE_SIZE, len(stems)))
    return ", ".join(sorted(sample))


def get_level_description(level: CefrLevel) -> str:
    """Return a human-readable CEFR level instruction for the system prompt."""
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


def get_stem_count(level: CefrLevel) -> int:
    """Return the total number of allowed stems for a level."""
    if level == "B2":
        return len(_CUMULATIVE["B1"])   # same pool, unrestricted usage
    return len(_CUMULATIVE[level])
