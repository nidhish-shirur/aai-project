"""Maps an urgency level to a care pathway and the text shown to the user.

Purpose: keep every user-facing sentence in knowledge/pathways.json so a
clinician can edit the wording without touching code.
"""
from __future__ import annotations

import json
from functools import lru_cache

import config


@lru_cache(maxsize=1)
def _load() -> dict:
    with open(config.KNOWLEDGE_DIR / "pathways.json", encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(text: str) -> str:
    return text.replace("{emergency_number}", config.EMERGENCY_NUMBER)


def describe(urgency: str) -> dict:
    """Label, pathway, timeframe and advice for one urgency level."""
    level = dict(_load()["levels"][urgency])
    level["advice"] = _fmt(level["advice"])
    return level


def emergency_message(category: str = "medical") -> str:
    """Fixed emergency text. Never produced by the LLM."""
    messages = _load()["emergency_messages"]
    return _fmt(messages.get(category, messages["medical"]))


def disclaimer() -> str:
    return _load()["disclaimer"]


def clinician_review_note() -> str:
    return _load()["clinician_review_note"]


def refusal(kinds: list[str]) -> str:
    """Fixed text for 'what disease do I have?' / 'which medicine?'. Never produced by the LLM."""
    texts = _load()["refusals"]
    return " ".join(texts[k] for k in ("diagnosis", "treatment") if k in kinds)
