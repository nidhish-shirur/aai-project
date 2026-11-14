"""LLM-only baseline: the same conversation, but a bare LLM decides the urgency (no rules, no gate).

Fairness: the baseline receives the SAME transcript the rule-based system produced (the patient's first
message plus every question and answer), so both see identical information. Needs a live LLM.
"""
from __future__ import annotations

import json
from typing import Optional

from agent.llm_client import LLM, LLMError
from eval.metrics import LEVELS

BASELINE_SYSTEM = """You are a healthcare triage assistant. Read the conversation and decide how urgently the person needs care.
Levels:
- emergency: call emergency services / go to the emergency department right now
- urgent: be seen at an emergency department or urgent care centre today, within hours
- moderate: see a doctor within 1 to 3 days
- low: self-care and monitoring at home
Reply with ONE JSON object only: {"urgency": "low" | "moderate" | "urgent" | "emergency"}"""


def transcript_text(transcript: list[dict]) -> str:
    """Conversation as plain text (the greeting is dropped: it is boilerplate)."""
    lines = []
    for m in transcript[1:]:
        who = "Patient" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content']}")
    return "\n".join(lines)


def parse_level(reply: object) -> Optional[str]:
    """Pull a valid level out of the model's reply, else None (counted as invalid, never guessed)."""
    if isinstance(reply, dict):
        value = reply.get("urgency")
        if isinstance(value, str) and value.strip().lower() in LEVELS:
            return value.strip().lower()
    return None


def llm_only_urgency(llm: LLM, transcript: list[dict]) -> tuple[Optional[str], str]:
    """(level or None, error text)."""
    try:
        raw = llm.complete_json(BASELINE_SYSTEM, "CONVERSATION:\n" + transcript_text(transcript))
    except LLMError as exc:
        return None, str(exc)
    level = parse_level(raw)
    return level, "" if level else f"invalid reply: {json.dumps(raw)[:80]}"
