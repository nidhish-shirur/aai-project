"""Symptom vocabulary lookup (reads knowledge/symptom_vocab.csv).

Purpose: map free text to canonical symptom names. Used by the fallback mode
now and by the LLM extractor in Phase 2 to normalise its output.
"""
from __future__ import annotations

import csv
import re
from functools import lru_cache

import config


@lru_cache(maxsize=1)
def _lookup() -> dict[str, str]:
    path = config.KNOWLEDGE_DIR / "symptom_vocab.csv"
    if not path.exists():
        raise FileNotFoundError("Run `python data/prepare_vocab.py` first to create symptom_vocab.csv")
    table: dict[str, str] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            canonical = row["symptom"]
            table[canonical.replace("_", " ")] = canonical
            for syn in filter(None, row["synonyms"].split("|")):
                table.setdefault(syn.lower().strip(), canonical)
    return table


def normalise(text: str) -> str | None:
    """Exact lookup of one phrase, e.g. 'pet dard' -> 'abdominal_pain'."""
    return _lookup().get(text.lower().strip())


def extract_symptoms(text: str) -> list[str]:
    """All canonical symptoms mentioned in a sentence (longest phrases first,
    each part of the text is used once)."""
    text = " " + re.sub(r"[^a-z0-9' -]+", " ", text.lower()) + " "
    found: list[str] = []
    for phrase in sorted(_lookup(), key=len, reverse=True):
        pattern = " " + re.escape(phrase) + " "
        if re.search(pattern, text):
            canonical = _lookup()[phrase]
            if canonical not in found:
                found.append(canonical)
            text = text.replace(" " + phrase + " ", "  ")
    return found
