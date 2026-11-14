"""Deterministic red-flag gate (runs on EVERY turn, before any LLM decision).

Purpose: detect emergency warning signs from (a) the patient's own words and
(b) facts already extracted. Pure Python, no LLM, no network, fully testable.
A single hit means: show emergency guidance immediately.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

from agent.state import PatientState
from engine.conditions import evaluate
from engine.protocols import Protocol, get_red_flag_config

NEGATION_WORDS = {
    "no", "not", "without", "never", "nahi", "nahin", "isn't", "isnt", "aren't", "arent",
    "don't", "dont", "doesn't", "doesnt", "didn't", "didnt", "haven't", "havent",
    "hasn't", "hasnt", "denies", "denied", "deny", "negative",
}
CONTRAST_WORDS = {"but", "however", "although", "though", "except", "yet"}
NEGATION_WINDOW = 3  # how many words before a match may carry the negation


@dataclass
class RedFlagHit:
    id: str
    label: str
    category: str  # "medical" or "mental_health"
    source: str    # "text", "facts" or "protocol"
    evidence: str = ""


@dataclass
class RedFlagResult:
    hits: list[RedFlagHit] = field(default_factory=list)

    @property
    def triggered(self) -> bool:
        return bool(self.hits)

    @property
    def category(self) -> str:
        """mental_health wins so the user gets the right kind of message."""
        return "mental_health" if any(h.category == "mental_health" for h in self.hits) else "medical"

    @property
    def labels(self) -> list[str]:
        return [h.label for h in self.hits]


def _clean(text: str) -> str:
    return text.lower().replace("’", "'").replace("‘", "'")


def _is_negated(text: str, match_start: int) -> bool:
    """True if one of the last few words before the match is a negation and no
    'but/however' comes between it and the match. Conservative on purpose:
    'no fever but chest pain with sweating' is NOT treated as negated."""
    before = re.split(r"[.,;:!?()]", text[:match_start])[-1]
    words = re.findall(r"[a-z']+", before)[-NEGATION_WINDOW:]
    for i in range(len(words) - 1, -1, -1):
        if words[i] in CONTRAST_WORDS:
            return False
        if words[i] in NEGATION_WORDS:
            return True
    return False


def _first_unnegated_match(pattern: str, text: str) -> Optional[str]:
    for m in re.finditer(pattern, text):
        if not _is_negated(text, m.start()):
            return m.group(0)
    return None


@lru_cache(maxsize=1)
def _flags() -> list[dict[str, Any]]:
    return get_red_flag_config()["flags"]


def check_text(text: str) -> list[RedFlagHit]:
    """Scan the patient's words for emergency patterns."""
    text = _clean(text)
    hits: list[RedFlagHit] = []
    for flag in _flags():
        evidence = None
        for pattern in flag.get("patterns", []):
            evidence = _first_unnegated_match(pattern, text)
            if evidence:
                break
        if not evidence:
            for group in flag.get("patterns_all", []):
                found = [_first_unnegated_match(p, text) for p in group]
                if all(found):
                    evidence = " + ".join(found)
                    break
        if evidence:
            hits.append(RedFlagHit(flag["id"], flag["label"], flag.get("category", "medical"),
                                   "text", evidence.strip()))
    return hits


def check_facts(facts: dict[str, Any], protocol: Optional[Protocol] = None) -> list[RedFlagHit]:
    """Evaluate fact-based flags: global ones, then the protocol's own."""
    hits: list[RedFlagHit] = []
    for flag in _flags():
        if "when" in flag and evaluate(flag["when"], facts):
            hits.append(RedFlagHit(flag["id"], flag["label"], flag.get("category", "medical"), "facts"))
    if protocol:
        for flag in protocol.red_flags:
            if evaluate(flag.when, facts):
                hits.append(RedFlagHit(flag.id, flag.label, flag.category, "protocol"))
    return hits


def check_red_flags(state: PatientState, protocol: Optional[Protocol] = None) -> RedFlagResult:
    """Full gate: patient's words + extracted facts + protocol flags (de-duplicated)."""
    hits = check_text(state.text_blob()) + check_facts(state.all_facts(), protocol)
    seen, unique = set(), []
    for h in hits:
        if h.id not in seen:
            seen.add(h.id)
            unique.append(h)
    return RedFlagResult(unique)
