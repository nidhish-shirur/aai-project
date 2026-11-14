"""The rule-based triage engine: decides the urgency level.

Purpose: this is the safety-critical decision. It never calls an LLM.
Order of decisions:
  1. Red flags (text + facts)            -> EMERGENCY, stop.
  2. Protocol rules                      -> highest matching urgency.
  3. Risk modifier (age, conditions, pregnancy) -> raise by ONE level, max 'urgent'.
  4. Missing required information        -> never below the protocol's floor.
  5. Escalation check                    -> clinician-review flag.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Optional

import config
from agent.state import PatientState
from engine import escalation
from engine.conditions import evaluate
from engine.protocols import (
    bump_level, get_protocol, level_index, max_level, required_unanswered,
    select_protocol_from_text,
)
from engine.red_flags import RedFlagHit, check_red_flags


@dataclass
class TriageResult:
    urgency: str
    protocol_id: str
    reasons: list[str] = field(default_factory=list)
    matched_rules: list[str] = field(default_factory=list)
    red_flags: list[RedFlagHit] = field(default_factory=list)
    red_flag_category: str = "medical"
    modifiers_applied: list[str] = field(default_factory=list)
    confidence: str = "high"  # high / medium / low
    missing_required: list[str] = field(default_factory=list)
    needs_clinician_review: bool = False
    clinician_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def _modifiers() -> dict:
    with open(config.KNOWLEDGE_DIR / "modifiers.json", encoding="utf-8") as fh:
        return json.load(fh)


def risk_modifier_reasons(state: PatientState) -> list[str]:
    """Which risk modifiers apply to this patient (each is explained to the user)."""
    m = _modifiers()
    reasons = []
    age = state.age_years
    if age is not None and age < m["young_age_below"]:
        reasons.append(m["reasons"]["young_age"])
    if age is not None and age >= m["old_age_from"]:
        reasons.append(m["reasons"]["old_age"])
    if state.chronic_conditions and set(state.chronic_conditions) & set(m["high_risk_conditions"]):
        reasons.append(m["reasons"]["chronic"])
    if m["pregnancy_applies"] and state.pregnant is True:
        reasons.append(m["reasons"]["pregnancy"])
    return reasons


def compute_urgency(state: PatientState) -> TriageResult:
    complaint = state.complaint or select_protocol_from_text(state.text_blob())
    protocol = get_protocol(complaint)
    facts = state.all_facts()

    # ---- 1. red flags -------------------------------------------------
    gate = check_red_flags(state, protocol)
    if gate.triggered:
        conflicts = escalation.detect_conflicts(facts)
        mh = gate.category == "mental_health"
        reasons = escalation.clinician_review_reasons(state, protocol, [], conflicts, mental_health=mh)
        return TriageResult(
            urgency="emergency", protocol_id=protocol.id,
            reasons=["Red flag: " + label for label in gate.labels],
            red_flags=gate.hits, red_flag_category=gate.category,
            confidence="high", needs_clinician_review=mh, clinician_reasons=[r for r in reasons if mh],
        )

    # ---- 2. protocol rules -------------------------------------------
    matched = [r for r in protocol.rules if evaluate(r.when, facts)]
    if matched:
        urgency = max_level(*[r.urgency for r in matched])
        reasons = [r.reason for r in matched if r.urgency == urgency]
    else:
        urgency = protocol.default_urgency
        reasons = ["No specific warning sign was reported; level set by the default for this type of problem."]

    # ---- 3. risk modifier --------------------------------------------
    modifier_reasons = risk_modifier_reasons(state)
    if modifier_reasons:
        raised = bump_level(urgency, _modifiers()["max_level"])
        if raised != urgency:
            urgency = raised
            reasons = reasons + modifier_reasons

    # ---- 4. incomplete information -----------------------------------
    missing = required_unanswered(state, protocol)
    if missing:
        floor = protocol.min_urgency_if_incomplete
        if level_index(urgency) < level_index(floor):
            urgency = floor
            reasons.append("Some important questions were not answered, so a cautious level was used.")

    # ---- 5. confidence and escalation --------------------------------
    conflicts = escalation.detect_conflicts(facts)
    review = escalation.clinician_review_reasons(state, protocol, missing, conflicts)
    if missing or conflicts or protocol.id == "other":
        confidence = "low"
    elif matched:
        confidence = "high"
    else:
        confidence = "medium"

    return TriageResult(
        urgency=urgency, protocol_id=protocol.id, reasons=reasons,
        matched_rules=[r.id for r in matched], modifiers_applied=modifier_reasons,
        confidence=confidence, missing_required=missing,
        needs_clinician_review=bool(review), clinician_reasons=review,
    )


def escalate_only(rule_urgency: str, llm_urgency: Optional[str]) -> str:
    """Merge the rule result with the LLM safety review. The LLM can only RAISE
    the level, never lower it (this is a core safety property, unit-tested)."""
    if llm_urgency is None or llm_urgency not in ("low", "moderate", "urgent", "emergency"):
        return rule_urgency
    return max_level(rule_urgency, llm_urgency)
