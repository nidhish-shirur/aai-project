"""Escalation logic: when must a human clinician look at this case?

Purpose: LLM self-reported 'confidence' is unreliable, so low confidence is
defined by explicit, testable conditions instead.
"""
from __future__ import annotations

from typing import Any

from agent.state import PatientState
from engine.protocols import Protocol


def detect_conflicts(facts: dict[str, Any]) -> list[str]:
    """Contradictory answers that make the assessment unreliable."""
    conflicts = []
    temp = facts.get("temperature_c")
    if facts.get("fever") is False and temp is not None and temp >= 38.0:
        conflicts.append("Patient denies fever but reports a temperature of 38 C or above")
    age = facts.get("age_years")
    if age is not None and age < 12 and facts.get("pregnant") is True:
        conflicts.append("Pregnancy reported for a child under 12")
    if facts.get("severity") == "mild" and facts.get("unable_to_keep_fluids") is True:
        conflicts.append("Severity 'mild' conflicts with being unable to keep fluids down")
    return conflicts


def clinician_review_reasons(state: PatientState, protocol: Protocol,
                             missing_required: list[str], conflicts: list[str],
                             mental_health: bool = False) -> list[str]:
    """Return the reasons a clinician should review (empty list = no review needed)."""
    reasons = []
    if protocol.id == "other":
        reasons.append("Complaint is not covered by a specific protocol")
    if missing_required:
        reasons.append("Required information is missing: " + ", ".join(missing_required))
    reasons.extend(conflicts)
    if state.unknown_count >= 3:
        reasons.append("Patient could not answer several questions")
    age = state.age_years
    if age is not None and age < 12:
        reasons.append("Child under 12: clinician review recommended")
    if state.pregnant is True:
        reasons.append("Possible pregnancy: clinician review recommended")
    if mental_health:
        reasons.append("Mental health crisis: needs human support")
    return reasons
