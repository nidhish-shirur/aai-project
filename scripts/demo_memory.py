"""Phase 3 demo: two visits by the same patient, using a throw-away database.

Run: python scripts/demo_memory.py
Shows the continuity note and the symptom timeline. Nothing is saved to your real database.
"""
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory.db import Memory  # noqa: E402

now = datetime.now(timezone.utc)

with tempfile.TemporaryDirectory() as tmp:
    mem = Memory(Path(tmp) / "demo.db")

    # Visit 1: a week ago - fever for 3 days
    visit1 = mem.start_session(now=now - timedelta(days=7))
    visit1.complaint = "fever"
    visit1.update({"age_years": 42, "sex": "male", "duration_days": 3})
    mem.save_state(visit1, now=now - timedelta(days=7))
    mem.record_symptom_events(visit1, now=now - timedelta(days=7))
    mem.save_report(visit1.session_id, {"urgency": "moderate"}, now=now - timedelta(days=7))
    print("Visit 1 saved. Patient code:", visit1.patient_id)

    # Visit 2: today - the same patient now has a cough
    visit2 = mem.start_session(visit1.patient_id, now=now)
    visit2.complaint = "cough_breathing"
    visit2.update({"duration_days": 1})
    mem.save_state(visit2, now=now)
    mem.record_symptom_events(visit2, now=now)

    print("\nWhat the agent remembers at the start of visit 2:")
    print(" ", mem.continuity_note(visit2.patient_id, exclude_session_id=visit2.session_id, now=now))
    print("\nTimeline for the doctor:")
    for line in mem.timeline_text(visit2.patient_id, now=now):
        print("  " + line)
    mem.close()
