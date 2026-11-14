"""Non-interactive demo of the safety core.   Run: python scripts/demo_engine.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.state import PatientState  # noqa: E402
from engine.triage_rules import compute_urgency  # noqa: E402

CASES = [
    ("Fever 3 days (the manual's example)", "fever",
     ["I have had a fever for three days."],
     dict(age_years=42, sex="male", chronic_conditions=[], duration_days=3, severity="moderate",
          temperature_c=38.2, breathing_difficulty=False, chest_pain=False, stiff_neck=False,
          confusion=False, unable_to_drink_fluids=False)),
    ("Chest pain + sweating said in words", "chest_pain",
     ["I have chest pain and I am sweating a lot"], dict(age_years=50, sex="male")),
    ("Negation: 'no chest pain' must NOT trigger", "fever",
     ["I have a mild fever, no chest pain and no difficulty breathing"],
     dict(age_years=24, sex="female", pregnant=False, chronic_conditions=[], duration_days=1, severity="mild",
          breathing_difficulty=False, chest_pain=False, stiff_neck=False, confusion=False,
          unable_to_drink_fluids=False)),
    ("Incomplete answers -> cautious level + clinician review", "fever",
     ["fever since yesterday"], dict(age_years=30, sex="male")),
]

for title, complaint, messages, facts in CASES:
    s = PatientState(complaint=complaint)
    for m in messages:
        s.add_user_message(m)
    s.update(facts)
    r = compute_urgency(s)
    print(f"\n### {title}")
    print(f"  urgency={r.urgency}  confidence={r.confidence}  clinician_review={r.needs_clinician_review}")
    for reason in r.reasons:
        print("   -", reason)
    if r.missing_required:
        print("   missing:", ", ".join(r.missing_required))
