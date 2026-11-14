"""Symptom timeline builder (pure functions, no database).

Purpose: turn "fever for 3 days, cough since today" into what a doctor wants:
    Day 1: Fever
    Day 4: Fever + Cough
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from agent.state import PatientState
from engine.protocols import get_protocol

# Facts that are symptoms (and so belong on a timeline) -> display label.
SYMPTOM_FACTS = {
    "fever": "Fever", "headache": "Headache", "chest_pain": "Chest pain",
    "abdominal_pain": "Abdominal pain", "vomiting": "Vomiting",
    "breathing_difficulty": "Breathing difficulty", "wheezing": "Wheezing",
    "confusion": "Confusion", "stiff_neck": "Stiff neck", "palpitations": "Palpitations",
    "sweating": "Sweating", "flank_pain": "Flank pain", "blood_in_urine": "Blood in urine",
    "blood_in_stool": "Blood in stool", "signs_of_dehydration": "Dehydration signs",
    "hearing_loss_sudden": "Sudden hearing loss", "cannot_swallow": "Unable to swallow",
    "loss_of_consciousness": "Fainting / blackout", "neuro_deficit": "Weakness / speech / vision change",
    "seizure": "Seizure", "severe_bleeding": "Heavy bleeding", "rash_nonblanching": "Non-fading rash",
}


def derive_symptom_events(state: PatientState, now: datetime) -> list[tuple[str, datetime, bool]]:
    """(label, onset, onset_comes_from_duration) for every symptom reported so far.

    The main complaint starts `duration_days` ago. Other symptoms are dated when they
    were first reported (the extractor does not capture per-symptom onset yet)."""
    facts = state.all_facts()
    events: list[tuple[str, datetime, bool]] = []
    if state.complaint and state.complaint != "other":
        label = SYMPTOM_FACTS.get(state.complaint) or get_protocol(state.complaint).title
        days = state.facts.get("duration_days")
        events.append((label, now - timedelta(days=days) if days is not None else now, days is not None))
    for key, label in SYMPTOM_FACTS.items():
        if facts.get(key) is True and label not in [e[0] for e in events]:
            events.append((label, now, False))
    return events


def build_timeline(events: list[tuple]) -> list[dict]:
    """events = [(symptom label, onset datetime)] or [(label, onset, resolved_at or None)].
    Returns one entry per day on which something NEW started. `active` lists the symptoms
    present on that day: a symptom is dropped once its resolved_at date has passed."""
    if not events:
        return []
    norm = [(e[0], e[1], e[2] if len(e) > 2 else None) for e in events]
    ordered = sorted(norm, key=lambda e: e[1])
    first_day = ordered[0][1].date()
    by_day: dict[int, list[str]] = {}
    resolved_on: dict[str, object] = {}
    for label, onset, resolved in ordered:
        day = (onset.date() - first_day).days + 1
        by_day.setdefault(day, [])
        if label not in by_day[day]:
            by_day[day].append(label)
        resolved_on[label] = resolved.date() if resolved else None
    timeline, seen = [], []
    for day in sorted(by_day):
        date = first_day + timedelta(days=day - 1)
        for label in by_day[day]:
            if label not in seen:
                seen.append(label)
        # a symptom always shows on the day it first appears, even if it was later marked resolved
        active = [l for l in seen if resolved_on[l] is None or resolved_on[l] > date or l in by_day[day]]
        timeline.append({"day": day, "date": date.isoformat(), "new": list(by_day[day]), "active": active})
    return timeline


def timeline_text(timeline: list[dict]) -> list[str]:
    """['Day 1 (2026-11-06): Fever', 'Day 4 (2026-11-09): Fever + Cough', ...]"""
    return [f"Day {t['day']} ({t['date']}): " + (" + ".join(t["active"]) or "earlier symptoms resolved") for t in timeline]
