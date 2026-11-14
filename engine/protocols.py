"""Loads and validates the triage protocol knowledge base.

Purpose: turn the JSON files in knowledge/ into typed Python objects and
answer the planner's question "what do I still need to ask?".
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

import config
from agent.state import PatientState
from engine.conditions import evaluate

Urgency = Literal["low", "moderate", "urgent", "emergency"]
URGENCY_LEVELS: list[str] = ["low", "moderate", "urgent", "emergency"]


def level_index(level: str) -> int:
    return URGENCY_LEVELS.index(level)


def max_level(*levels: str) -> str:
    return max(levels, key=level_index)


def bump_level(level: str, cap: str) -> str:
    """Raise by one level, but never above `cap`."""
    new = min(level_index(level) + 1, level_index(cap))
    return URGENCY_LEVELS[max(new, level_index(level))]


# ---------------------------------------------------------------- models
class Field_(BaseModel):
    """One piece of information the agent may ask for."""
    key: str
    type: Literal["bool", "float", "int", "enum", "list", "str"]
    question: str
    required: bool = False
    ask_if: Optional[Any] = None
    options: Optional[list[str]] = None
    description: str = ""


class Rule(BaseModel):
    id: str
    urgency: Urgency
    when: Any
    reason: str


class ProtocolRedFlag(BaseModel):
    id: str
    label: str
    when: Any
    category: Literal["medical", "mental_health"] = "medical"


class Protocol(BaseModel):
    id: str
    title: str
    triggers: list[str] = Field(default_factory=list)
    fields: list[Field_] = Field(default_factory=list)
    red_flags: list[ProtocolRedFlag] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)
    default_urgency: Urgency = "low"
    min_urgency_if_incomplete: Urgency = "moderate"
    warning_signs: list[str] = Field(default_factory=list)
    source_notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- loading
# When two protocols match equally, the earlier one in this list wins
# (the more dangerous complaint is preferred).
SAFETY_PRIORITY = [
    "chest_pain", "cough_breathing", "headache", "abdominal_pain",
    "dizziness_weakness", "minor_injury", "gi_upset", "urinary",
    "sore_throat_ear", "fever", "other",
]


def _read_json(path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def get_protocols() -> dict[str, Protocol]:
    protocols: dict[str, Protocol] = {}
    for path in sorted(config.PROTOCOL_DIR.glob("*.json")):
        proto = Protocol(**_read_json(path))
        protocols[proto.id] = proto
    return protocols


def get_protocol(protocol_id: Optional[str]) -> Protocol:
    """Return the protocol, or the generic 'other' protocol if unknown."""
    protocols = get_protocols()
    if protocol_id in protocols:
        return protocols[protocol_id]
    return protocols["other"]


@lru_cache(maxsize=1)
def get_common_fields() -> list[Field_]:
    return [Field_(**f) for f in _read_json(config.KNOWLEDGE_DIR / "common.json")["fields"]]


@lru_cache(maxsize=1)
def get_red_flag_config() -> dict:
    return _read_json(config.KNOWLEDGE_DIR / "red_flags.json")


def get_extra_facts() -> list[Field_]:
    return [Field_(**f) for f in get_red_flag_config()["extra_facts"]]


def known_fact_keys(protocol: Protocol) -> set[str]:
    """Every fact name a rule in this protocol may legally mention."""
    keys = {f.key for f in get_common_fields()}
    keys |= {f.key for f in get_extra_facts()}
    keys |= {f.key for f in protocol.fields}
    keys |= set(get_protocols().keys())  # complaint ids act as True facts
    return keys


def facts_catalogue(protocol: Protocol) -> dict[str, Field_]:
    """key -> field definition. The extractor (Phase 2) uses this as its schema."""
    catalogue: dict[str, Field_] = {}
    for f in [*get_common_fields(), *get_extra_facts(), *protocol.fields]:
        catalogue[f.key] = f
    return catalogue


# ---------------------------------------------------------------- planning helpers
def ordered_fields(protocol: Protocol) -> list[Field_]:
    """Default asking order: age and sex first, then the protocol's own
    questions (red-flag screens come first in each file), then the rest of
    the common fields (pregnancy, long-term conditions, duration, severity)."""
    common = get_common_fields()
    first = [f for f in common if f.key in ("age_years", "sex")]
    rest = [f for f in common if f.key not in ("age_years", "sex")]
    return [*first, *protocol.fields, *rest]


def _applicable(field: Field_, facts: dict[str, Any]) -> bool:
    return field.ask_if is None or evaluate(field.ask_if, facts)


def missing_fields(state: PatientState, protocol: Protocol,
                   include_optional: bool = False) -> list[Field_]:
    """Fields still worth ASKING: applicable, unanswered and not already
    declined with 'I don't know'."""
    facts = state.all_facts()
    out = []
    for f in ordered_fields(protocol):
        if not (f.required or include_optional):
            continue
        if facts.get(f.key) is None and f.key not in state.unknown_fields and _applicable(f, facts):
            out.append(f)
    return out


def required_unanswered(state: PatientState, protocol: Protocol) -> list[str]:
    """Required fields with no value, INCLUDING ones the patient could not
    answer. Used to lower confidence and trigger clinician review."""
    facts = state.all_facts()
    return [f.key for f in ordered_fields(protocol)
            if f.required and facts.get(f.key) is None and _applicable(f, facts)]


def _trigger_regex(trig: str) -> str:
    """Trigger -> regex. Starts at a word boundary; a trailing '$' means the word must also END there
    ('burn$' matches 'burn', not 'burning')."""
    trig = trig.lower()
    whole = trig.endswith("$")
    body = re.escape(trig[:-1] if whole else trig)
    return r"\b" + body + (r"\b" if whole else "")


def _count_trigger(trig: str, text: str) -> int:
    """1 if the trigger appears un-negated ('no chest pain' does not count as chest pain), else 0."""
    from engine.red_flags import _is_negated  # lazy: red_flags imports this module
    return int(any(not _is_negated(text, m.start()) for m in re.finditer(_trigger_regex(trig), text)))


def select_protocol_from_text(text: str) -> str:
    """Keyword fallback (used when the LLM is unavailable). Multi-word triggers count double.
    Negated mentions ('no chest pain') are ignored. Returns 'other' if nothing matches."""
    text = text.lower()
    best_id, best_score = "other", 0
    for pid, proto in get_protocols().items():
        if pid == "other":
            continue
        score = 0
        for trig in proto.triggers:
            if _count_trigger(trig, text):
                score += 2 if " " in trig else 1
        if score > best_score or (score == best_score and score > 0 and
                                  SAFETY_PRIORITY.index(pid) < SAFETY_PRIORITY.index(best_id)):
            best_id, best_score = pid, score
    return best_id
