"""Memory and timeline tests (use a temporary database file, never your real one)."""
from datetime import datetime, timedelta, timezone

import pytest

from agent.state import PatientState
from memory.db import Memory
from memory.timeline import build_timeline, timeline_text

T0 = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)


def days(n):
    return T0 + timedelta(days=n)


@pytest.fixture
def mem(tmp_path):
    m = Memory(tmp_path / "t.db")
    yield m
    m.close()


def fill_fever_session(mem, now, patient_id=None, duration=3):
    s = mem.start_session(patient_id, now=now)
    s.complaint = "fever"
    s.add_user_message("I have a fever")
    s.update({"age_years": 42, "sex": "male", "duration_days": duration, "chronic_conditions": ["diabetes"],
              "temperature_c": 38.4, "stiff_neck": False})
    s.mark_unknown("severity")
    mem.save_state(s, now=now)
    mem.record_symptom_events(s, now=now)
    return s


# ---------------- plan test: close and reopen the app
def test_state_survives_closing_and_reopening(tmp_path):
    path = tmp_path / "reopen.db"
    m1 = Memory(path)
    s = fill_fever_session(m1, T0)
    m1.log_message(s.session_id, "user", "I have a fever", now=T0)
    m1.close()

    m2 = Memory(path)               # a brand-new connection, as after restarting the app
    loaded = m2.load_state(s.session_id)
    assert loaded.complaint == "fever" and loaded.age_years == 42 and loaded.chronic_conditions == ["diabetes"]
    assert loaded.facts["stiff_neck"] is False             # a recorded "no" is not lost
    assert loaded.unknown_fields == ["severity"]
    assert loaded.user_messages == ["I have a fever"]
    assert m2.get_messages(s.session_id)[0]["content"] == "I have a fever"
    m2.close()


def test_messages_keep_order_and_survive_quotes(mem):
    s = mem.start_session(now=T0)
    nasty = "it's 'bad'; DROP TABLE messages; --"
    mem.log_message(s.session_id, "user", nasty, now=T0)
    mem.log_message(s.session_id, "agent", "How old are you?", now=T0)
    msgs = mem.get_messages(s.session_id)
    assert [m["role"] for m in msgs] == ["user", "agent"] and msgs[0]["content"] == nasty


def test_bad_role_and_unknown_ids_rejected(mem):
    s = mem.start_session(now=T0)
    with pytest.raises(ValueError):
        mem.log_message(s.session_id, "robot", "x")
    with pytest.raises(KeyError):
        mem.load_state("nope")
    with pytest.raises(KeyError):
        mem.start_session("P-DOESNT")
    with pytest.raises(KeyError):
        mem.save_state(PatientState(patient_id="x"))


# ---------------- timeline
def test_single_session_timeline_matches_the_manual_example(mem):
    s = fill_fever_session(mem, T0, duration=3)                  # fever began 3 days ago
    s.update({"wheezing": True, "headache": True})              # reported today
    mem.record_symptom_events(s, now=T0)
    lines = mem.timeline_text(s.patient_id, now=T0)
    assert lines[0].startswith("Day 1") and lines[0].endswith("Fever")
    assert lines[1].startswith("Day 4") and "Fever + " in lines[1] and "Wheezing" in lines[1] and "Headache" in lines[1]


def test_recording_events_twice_adds_no_duplicates(mem):
    s = fill_fever_session(mem, T0)
    mem.record_symptom_events(s, now=T0)
    mem.record_symptom_events(s, now=T0)
    assert len(mem.get_timeline(s.patient_id, now=T0)) == 1


def test_duration_correction_moves_the_onset_but_other_symptoms_keep_first_report(mem):
    s = fill_fever_session(mem, T0, duration=1)
    s.update({"vomiting": True})
    mem.record_symptom_events(s, now=T0)
    s.update({"duration_days": 4})                               # patient corrects: fever is 4 days old
    mem.record_symptom_events(s, now=days(1))                    # recorded a day later
    tl = mem.get_timeline(s.patient_id, now=days(1))
    # fever now began on day 1 (Nov 6); vomiting stays dated at its FIRST report (Nov 9) = day 4.
    # (If it were wrongly re-dated to the second recording, Nov 10, it would be day 5.)
    assert tl[0]["new"] == ["Fever"] and tl[-1]["day"] == 4


def test_timeline_pure_builder():
    ev = [("Cough", days(3)), ("Fever", days(0)), ("Headache", days(1)), ("Fatigue", days(1))]
    tl = build_timeline(ev)
    assert [t["day"] for t in tl] == [1, 2, 4]
    assert tl[1]["new"] == ["Headache", "Fatigue"]
    assert tl[2]["active"] == ["Fever", "Headache", "Fatigue", "Cough"]
    assert timeline_text(tl)[0] == "Day 1 (2026-11-09): Fever"
    assert build_timeline([]) == []


# ---------------- continuity across sessions (plan: recall earlier symptoms)
def test_returning_patient_is_remembered(mem):
    s1 = fill_fever_session(mem, T0)
    mem.save_report(s1.session_id, {"urgency": "moderate"}, now=T0)

    s2 = mem.start_session(s1.patient_id, now=days(7))
    s2.complaint = "cough_breathing"
    s2.update({"duration_days": 1})
    mem.save_state(s2, now=days(7))
    mem.record_symptom_events(s2, now=days(7))

    note = mem.continuity_note(s2.patient_id, exclude_session_id=s2.session_id, now=days(7))
    assert "7 day(s) ago" in note and "fever" in note and "moderate" in note

    lines = mem.timeline_text(s2.patient_id, now=days(7))
    assert any("Fever" in l for l in lines) and any("Cough" in l for l in lines)
    assert lines[0].startswith("Day 1")


def test_first_visit_has_no_continuity_note(mem):
    s = fill_fever_session(mem, T0)
    assert mem.continuity_note(s.patient_id, exclude_session_id=s.session_id, now=T0) == ""


def test_other_patients_never_leak(mem):
    a = fill_fever_session(mem, T0)
    b = mem.start_session(now=T0)
    assert b.patient_id != a.patient_id
    assert mem.previous_sessions(b.patient_id, exclude_session_id=b.session_id, now=T0) == []
    assert a.session_id not in [p["session_id"] for p in mem.previous_sessions(b.patient_id, now=T0)]
    assert mem.get_timeline(b.patient_id, now=T0) == []


def test_old_events_outside_window_are_ignored(mem):
    s = fill_fever_session(mem, T0)
    assert mem.get_timeline(s.patient_id, within_days=30, now=days(60)) == []


@pytest.mark.parametrize("complaint,label", [
    ("chest_pain", "Chest pain"), ("headache", "Headache"), ("abdominal_pain", "Abdominal pain"),
    ("urinary", "Urinary symptoms"), ("gi_upset", "Vomiting or diarrhoea"), ("minor_injury", "Injury, cut, burn or fall"),
])
def test_every_complaint_gets_a_timeline_label(mem, complaint, label):
    s = mem.start_session(now=T0)
    s.complaint = complaint
    s.update({"duration_days": 2})
    mem.record_symptom_events(s, now=T0)
    assert mem.get_timeline(s.patient_id, now=T0)[0]["new"] == [label]


def test_state_without_patient_cannot_record(mem):
    with pytest.raises(ValueError):
        mem.record_symptom_events(PatientState(complaint="fever"))


# ---------------- reports and dashboard ordering
def test_reports_and_sessions_listed_most_urgent_first(mem):
    low = mem.start_session(now=T0)
    emg = mem.start_session(now=T0)
    mod = mem.start_session(now=T0)
    mem.save_report(low.session_id, {"urgency": "low"}, now=T0)
    mem.save_report(emg.session_id, {"urgency": "emergency", "reasons": ["x"]}, now=T0)
    mem.save_report(mod.session_id, {"urgency": "moderate"}, now=T0)
    assert [r["urgency"] for r in mem.list_sessions()] == ["emergency", "moderate", "low"]
    assert mem.get_report(emg.session_id)["reasons"] == ["x"]
    assert mem.get_report(mem.start_session(now=T0).session_id) is None


# ---------------- privacy
def test_delete_patient_erases_everything(mem):
    s = fill_fever_session(mem, T0)
    mem.log_message(s.session_id, "user", "hello", now=T0)
    mem.save_report(s.session_id, {"urgency": "low"}, now=T0)
    mem.delete_patient(s.patient_id)
    assert not mem.patient_exists(s.patient_id)
    for table in ("sessions", "messages", "symptom_events"):
        assert mem._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_purge_old_sessions(mem):
    old = fill_fever_session(mem, T0)
    new = mem.start_session(now=days(40))
    mem.save_state(new, now=days(40))
    removed = mem.purge_older_than(30, now=days(41))
    assert removed == 1
    assert not mem.patient_exists(old.patient_id) and mem.patient_exists(new.patient_id)


def test_no_personal_fields_in_schema(mem):
    cols = {r["name"] for t in ("patients", "sessions") for r in mem._conn.execute(f"PRAGMA table_info({t})")}
    assert not cols & {"name", "phone", "email", "address"}


# ---------------- resolved symptoms (Phase 5)
def test_resolved_symptom_stops_being_active_after_its_resolution_date():
    ev = [("Fever", days(0), days(3)), ("Cough", days(4), None)]
    tl = build_timeline(ev)
    assert tl[0]["active"] == ["Fever"]
    assert tl[1]["active"] == ["Cough"]               # fever was gone before the cough started
    # resolved on the same day it was first reported: it must still appear on that day, then drop off
    tl2 = build_timeline([("Fever", days(0), days(0)), ("Cough", days(2), None)])
    assert tl2[0]["active"] == ["Fever"] and tl2[1]["active"] == ["Cough"]


def test_two_tuple_events_still_work_and_resolution_is_stored(mem):
    s = fill_fever_session(mem, T0)
    s2 = mem.start_session(s.patient_id, now=days(7))
    assert mem.unresolved_earlier_symptoms(s.patient_id, s2.session_id, now=days(7)) == ["Fever"]
    assert mem.mark_earlier_resolved(s.patient_id, s2.session_id, now=days(7)) == 1
    assert mem.unresolved_earlier_symptoms(s.patient_id, s2.session_id, now=days(7)) == []
    assert mem.mark_earlier_resolved(s.patient_id, s2.session_id, now=days(7)) == 0   # idempotent
    # the same patient's current session is never touched
    assert mem.unresolved_earlier_symptoms(s.patient_id, s.session_id, now=days(7)) == []
