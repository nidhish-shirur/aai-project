"""Orchestrator tests. A routing mock plays the LLM: it answers extractor calls and planner
calls from separate scripts, so no API key or internet is needed."""
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from agent.llm_client import LLMError
from agent.orchestrator import PLANNER_SYSTEM, TriageAgent
from agent.safety_review import REVIEW_SYSTEM
from engine.protocols import get_protocol, ordered_fields
from memory.db import Memory

T0 = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)
EMPTY_EXTRACT = {"complaint": None, "facts": {}, "dont_know": []}
FEVER_FIRST = {"complaint": "fever", "facts": {
    "age_years": {"value": 30, "evidence": "30 year old"},
    "sex": {"value": "male", "evidence": "male"},
    "duration_days": {"value": 3, "evidence": "3 days"}}, "dont_know": []}


class RoutingMock:
    def __init__(self, extract=(), plan=(), review=(), lenient_plan=False):
        self.extract, self.plan, self.review, self.calls = list(extract), list(plan), list(review), []
        self.lenient_plan = lenient_plan      # empty planner script returns a valid (overridden) plan instead of an error

    def complete_json(self, system, user, max_retries=1):
        kind = "plan" if system == PLANNER_SYSTEM else "review" if system == REVIEW_SYSTEM else "extract"
        self.calls.append(kind)
        if kind == "review":
            item = self.review.pop(0) if self.review else {"suggested_urgency": None, "concern": ""}
            if isinstance(item, Exception):
                raise item
            return item
        queue = self.plan if kind == "plan" else self.extract
        if not queue:
            if kind == "plan":
                if self.lenient_plan:
                    return {"action": "compute_urgency"}
                raise LLMError("no planner script")
            return dict(EMPTY_EXTRACT)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def complete_text(self, system, user):
        raise LLMError("not used")


class DownLLM:
    calls = []

    def complete_json(self, *a, **k):
        raise LLMError("network down")

    complete_text = complete_json


@pytest.fixture
def mem(tmp_path):
    m = Memory(tmp_path / "o.db")
    yield m
    m.close()


def make(mem, llm=None, **kw):
    agent = TriageAgent(mem, llm, clock=lambda: T0, **kw)
    agent.start()
    return agent


def auto_answer(agent):
    """A patient who answers every question 'no' / 'none' / 'mild' / a number."""
    key = agent._pending["keys"][0]
    f = next(f for f in ordered_fields(get_protocol(agent.state.complaint)) if f.key == key)
    return {"bool": "no", "list": "none", "float": "2", "int": "2"}.get(f.type) or (f.options or ["mild"])[0]


def run_to_end(agent, limit=30):
    asked, turn = [], None
    for _ in range(limit):
        if agent.done:
            break
        turn = agent.handle(auto_answer(agent))
        asked += [t["field"] for t in turn.tool_log if t["tool"] == "ask_question"]
    assert agent.done, "conversation never finished"
    return turn, asked


def tools(turn):
    """Tool names of a turn, without the trailing timing entry ('llm_calls')."""
    return [t["tool"] for t in turn.tool_log if t["tool"] != "llm_calls"]


def action(turn):
    """The last real tool call of a turn (ask_question, compute_urgency, ...)."""
    return [t for t in turn.tool_log if t["tool"] != "llm_calls"][-1]


# ---------------------------------------------------------------- core behaviour
def test_full_fever_conversation_never_repeats_known_facts(mem):
    llm = RoutingMock(extract=[FEVER_FIRST])
    agent = make(mem, llm)
    first = agent.handle("I am a 30 year old male with fever for 3 days")
    assert first.kind == "question"
    turn, asked = run_to_end(agent)
    assert not {"age_years", "sex", "duration_days"} & set(asked)
    assert turn.kind == "result" and turn.urgency in ("low", "moderate", "urgent")
    assert "does not diagnose" in turn.message and agent.state.patient_id in turn.message
    assert agent.state.questions_asked <= 12


def test_tool_log_order_for_a_normal_turn(mem):
    agent = make(mem, RoutingMock(extract=[FEVER_FIRST]))
    turn = agent.handle("30 year old male, fever for 3 days")
    t = tools(turn)
    assert t[:3] == ["red_flag_gate", "extract_facts", "red_flag_gate"]
    assert t[-1] == "ask_question" and "plan_next" in t
    assert turn.tool_log[0]["stage"] == "text_only"


def test_immediate_emergency_uses_zero_llm_calls(mem):
    llm = RoutingMock()
    agent = make(mem, llm)
    turn = agent.handle("my father has crushing chest pain and cannot breathe")
    assert turn.kind == "emergency" and turn.done and turn.urgency == "emergency"
    assert llm.calls == []
    assert "112" in turn.message
    assert mem.get_report(agent.state.session_id)["urgency"] == "emergency"


def test_mental_health_emergency_uses_the_supportive_message(mem):
    llm = RoutingMock()
    turn = make(mem, llm).handle("I want to end my life")
    assert turn.kind == "emergency" and "14416" in turn.message and llm.calls == []


def test_emergency_found_later_in_the_conversation(mem):
    agent = make(mem, RoutingMock(extract=[FEVER_FIRST]))
    agent.handle("30 year old male, fever for 3 days")
    turn = agent.handle("now I also have a seizure")
    assert turn.kind == "emergency" and agent.done


def test_session_is_closed_after_the_result(mem):
    agent = make(mem)
    agent.handle("my father has crushing chest pain and cannot breathe")
    again = agent.handle("hello?")
    assert again.kind == "closed" and again.done


def test_empty_message_is_ignored_without_counting(mem):
    agent = make(mem)
    turn = agent.handle("   ")
    assert turn.kind == "info" and agent.state.user_messages == [] and agent.turn == 0


def test_handle_before_start_is_an_error(mem):
    with pytest.raises(RuntimeError):
        TriageAgent(mem, None).handle("hi")


# ---------------------------------------------------------------- planner is checked, not trusted
def test_early_compute_urgency_is_overridden_while_required_fields_missing(mem):
    llm = RoutingMock(extract=[FEVER_FIRST], plan=[{"action": "compute_urgency"}])
    turn = make(mem, llm).handle("30 year old male, fever for 3 days")
    assert turn.kind == "question"
    plan = [t for t in turn.tool_log if t["tool"] == "plan_next"][0]
    assert plan["source"] == "override" and any("required" in n for n in plan["notes"])


def test_invented_field_and_optional_first_are_overridden(mem):
    llm = RoutingMock(extract=[FEVER_FIRST], plan=[
        {"action": "ask_question", "field": "favourite_colour", "question": "Colour?"},
        {"action": "ask_question", "field": "temperature_c", "question": "What is your temperature?"}])
    agent = make(mem, llm)
    t1 = agent.handle("30 year old male, fever for 3 days")
    assert action(t1)["field"] == "breathing_difficulty" and action(t1)["source"] == "override"
    t2 = agent.handle("no")
    assert action(t2)["source"] == "override" and action(t2)["field"] != "temperature_c"


def test_llm_can_reorder_among_required_fields_and_reword(mem):
    llm = RoutingMock(extract=[FEVER_FIRST], plan=[
        {"action": "ask_question", "field": "stiff_neck", "question": "Is your neck stiff or painful to bend?"}])
    turn = make(mem, llm).handle("30 year old male, fever for 3 days")
    assert turn.message.startswith("Is your neck stiff") and action(turn)["source"] == "llm"


def test_diagnosis_or_medicine_wording_is_rejected(mem):
    protocol_q = {f.key: f.question for f in get_protocol("fever").fields}["stiff_neck"]
    for bad in ("You probably have meningitis, is your neck stiff?", "Did you take paracetamol for the fever?",
                "Is it a diagnosis of flu?"):
        llm = RoutingMock(extract=[FEVER_FIRST], plan=[{"action": "ask_question", "field": "stiff_neck",
                                                        "question": bad}])
        agent = TriageAgent(mem, llm, clock=lambda: T0)
        agent.start()
        turn = agent.handle("30 year old male, fever for 3 days")
        assert turn.message.startswith(protocol_q), bad
        assert action(turn)["source"] == "override"


def test_garbage_plans_fall_back_to_rules(mem):
    for garbage in ("not json at all", {"action": "dance"}, {"action": "ask_question"}, ["x"], {}):
        llm = RoutingMock(extract=[FEVER_FIRST], plan=[garbage])
        turn = TriageAgent(mem, llm, clock=lambda: T0)
        turn.start()
        reply = turn.handle("30 year old male, fever for 3 days")
        assert reply.kind == "question", garbage


def test_planner_failure_does_not_stop_the_conversation(mem):
    llm = RoutingMock(extract=[FEVER_FIRST])  # planner raises LLMError every time
    agent = make(mem, llm)
    agent.handle("30 year old male, fever for 3 days")
    turn, _ = run_to_end(agent)
    assert turn.kind == "result"


def test_combined_question_one_answer_fills_two_fields(mem, monkeypatch):
    monkeypatch.setattr('agent.orchestrator.COMBINE_QUESTIONS', True)
    llm = RoutingMock(
        extract=[FEVER_FIRST, {"complaint": None, "facts": {
            "stiff_neck": {"value": False, "evidence": "neck is fine"},
            "confusion": {"value": False, "evidence": "not confused"}}, "dont_know": []}],
        plan=[{"action": "ask_question", "field": "stiff_neck", "also": ["confusion"],
               "question": "Any stiff neck or confusion?"}])
    agent = make(mem, llm)
    agent.handle("30 year old male, fever for 3 days")
    assert agent._pending["keys"] == ["stiff_neck", "confusion"]
    nxt = agent.handle("neck is fine and not confused")
    assert agent.state.get("stiff_neck") is False and agent.state.get("confusion") is False
    asked_next = [t["field"] for t in nxt.tool_log if t["tool"] == "ask_question"]
    assert asked_next and not {"stiff_neck", "confusion"} & set(asked_next)   # both already answered


def test_escalate_action_adds_a_review_reason_but_never_lowers_urgency(mem):
    llm = RoutingMock(extract=[FEVER_FIRST], plan=[
        {"action": "escalate_to_clinician", "reason": "story is inconsistent"}])
    agent = make(mem, llm)
    turn = agent.handle("30 year old male, fever for 3 days")
    assert turn.done and turn.result["needs_clinician_review"]
    assert any("inconsistent" in r for r in turn.result["clinician_reasons"])
    assert turn.urgency != "low"      # unanswered required safety questions keep the cautious floor
    assert "escalate_to_clinician" in tools(turn)


# ---------------------------------------------------------------- degraded modes
def test_works_with_no_llm_at_all(mem):
    agent = make(mem, None)
    agent.handle("I have a fever")
    turn, _ = run_to_end(agent)
    assert turn.kind == "result" and turn.result["llm_used"] is False


def test_works_when_every_llm_call_fails(mem):
    agent = make(mem, DownLLM())
    first = agent.handle("I have a fever")
    assert first.kind == "question"
    turn, _ = run_to_end(agent)
    assert turn.kind == "result"
    assert any(t.get("error") for t in agent.tool_log if t["tool"] == "extract_facts")


def test_emergency_still_detected_when_llm_is_down(mem):
    turn = make(mem, DownLLM()).handle("I have severe chest pain and sweating")
    assert turn.kind == "emergency"


def test_question_budget_stops_the_questions(mem):
    agent = make(mem, None, max_questions=2)
    agent.handle("I have a fever")
    turn, _ = run_to_end(agent)
    assert agent.state.questions_asked == 2
    assert turn.result["missing_required"] and turn.result["confidence"] != "high"
    assert turn.urgency != "low"


def test_unanswerable_question_is_retried_once_then_marked_unknown(mem):
    agent = make(mem, None)
    agent.handle("I have a fever")
    key = agent._pending["keys"][0]
    again = agent.handle("hmm purple")
    assert again.message.startswith("Sorry, I did not quite catch that")
    agent.handle("hmm purple")
    assert key in agent.state.unknown_fields


def test_unclear_complaint_is_asked_twice_then_defaults_to_other(mem):
    agent = make(mem, RoutingMock())
    first = agent.handle("hello there")
    assert "main problem" in first.message and agent.state.complaint is None
    second = agent.handle("just checking")
    assert agent.state.complaint == "other" and second.kind in ("question", "result")


# ---------------------------------------------------------------- adaptive branching
def test_chest_pain_follow_up_is_asked_only_when_it_applies(mem):
    ex = {"complaint": "chest_pain", "facts": {
        "age_years": {"value": 50, "evidence": "50"}, "sex": {"value": "male", "evidence": "man"},
        "pain_radiates_arm_jaw": {"value": False, "evidence": "does not spread"}}, "dont_know": []}
    agent = make(mem, RoutingMock(extract=[ex]))
    agent.handle("I am a 50 year old man with some chest pain that does not spread")
    _, asked = run_to_end(agent)
    assert "pain_worse_breathing" in asked and "pain_radiates_arm_jaw" not in asked


# ---------------------------------------------------------------- persistence
def test_everything_is_saved(mem):
    agent = make(mem, RoutingMock(extract=[FEVER_FIRST]))
    agent.handle("30 year old male, fever for 3 days")
    turn, _ = run_to_end(agent)
    sid = agent.state.session_id
    loaded = mem.load_state(sid)
    assert loaded.complaint == "fever" and loaded.age_years == 30
    roles = [m["role"] for m in mem.get_messages(sid)]
    assert roles[0] == "agent" and "user" in roles and roles[-1] == "agent"
    report = mem.get_report(sid)
    assert report["urgency"] == turn.urgency and report["tool_log"] and report["questions_asked"] > 0
    assert mem.get_timeline(loaded.patient_id, now=T0)


# ---------------------------------------------------------------- returning patient
def earlier_fever_visit(mem, days_ago=7):
    then = T0 - timedelta(days=days_ago)
    s = mem.start_session(now=then)
    s.complaint = "fever"
    s.update({"duration_days": 3})
    mem.save_state(s, now=then)
    mem.record_symptom_events(s, now=then)
    mem.save_report(s.session_id, {"urgency": "moderate"}, now=then)
    return s.patient_id


def test_returning_patient_is_asked_if_symptoms_are_gone_and_yes_resolves_them(mem):
    pid = earlier_fever_visit(mem)
    agent = TriageAgent(mem, None, clock=lambda: T0)
    greeting = agent.start(pid)
    assert "fever" in greeting.message.lower() and "gone away" in greeting.message
    reply = agent.handle("yes")
    assert "main problem" in reply.message and agent.state.questions_asked == 0
    assert mem.unresolved_earlier_symptoms(pid, agent.state.session_id, now=T0) == []
    agent.handle("I have a cough")                      # new complaint today
    run_to_end(agent)
    assert mem.unresolved_earlier_symptoms(pid, agent.state.session_id, now=T0) == []
    assert any("Cough" in line for line in mem.timeline_text(pid, now=T0))


def test_returning_patient_saying_no_keeps_the_earlier_symptom(mem):
    pid = earlier_fever_visit(mem)
    agent = TriageAgent(mem, None, clock=lambda: T0)
    agent.start(pid)
    agent.handle("no")
    assert mem.unresolved_earlier_symptoms(pid, agent.state.session_id, now=T0) == ["Fever"]


def test_returning_patient_describing_a_problem_instead_of_yes_no(mem):
    pid = earlier_fever_visit(mem)
    agent = TriageAgent(mem, None, clock=lambda: T0)
    agent.start(pid)
    turn = agent.handle("my father has crushing chest pain and cannot breathe")
    assert turn.kind == "emergency"


def test_first_visit_has_no_resolution_question(mem):
    greeting = TriageAgent(mem, None, clock=lambda: T0).start()
    assert "gone away" not in greeting.message and "main problem" in greeting.message


def test_unknown_patient_code_raises(mem):
    with pytest.raises(KeyError):
        TriageAgent(mem, None).start("P-NOPE00")


# ---------------------------------------------------------------- database migration
def test_old_database_without_resolved_at_is_upgraded(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE patients (patient_id TEXT PRIMARY KEY, created_at TEXT NOT NULL);
        CREATE TABLE sessions (session_id TEXT PRIMARY KEY, patient_id TEXT NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active', complaint TEXT, urgency TEXT,
            state_json TEXT NOT NULL, report_json TEXT);
        CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE symptom_events (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
            patient_id TEXT NOT NULL, symptom TEXT NOT NULL, onset_at TEXT NOT NULL, recorded_at TEXT NOT NULL,
            UNIQUE(session_id, symptom));
        INSERT INTO patients VALUES ('P-OLD000', '2026-11-01T00:00:00+00:00');
        INSERT INTO sessions (session_id, patient_id, created_at, updated_at, state_json)
            VALUES ('s1', 'P-OLD000', '2026-11-01T00:00:00+00:00', '2026-11-01T00:00:00+00:00', '{}');
        INSERT INTO symptom_events (session_id, patient_id, symptom, onset_at, recorded_at)
            VALUES ('s1', 'P-OLD000', 'Fever', '2026-11-05T00:00:00+00:00', '2026-11-05T00:00:00+00:00');
    """)
    con.commit()
    con.close()
    m = Memory(path)
    assert m.unresolved_earlier_symptoms("P-OLD000", now=T0) == ["Fever"]
    assert m.mark_earlier_resolved("P-OLD000", now=T0) == 1
    assert m.unresolved_earlier_symptoms("P-OLD000", now=T0) == []
    m.close()


def test_combined_questions_are_ignored_by_default(mem):
    llm = RoutingMock(extract=[FEVER_FIRST], plan=[{"action": "ask_question", "field": "breathing_difficulty",
                                                    "also": ["chest_pain"], "question": "Breathing trouble?"}])
    agent = make(mem, llm)
    agent.handle("30 year old male, fever for 3 days")
    assert agent._pending["keys"] == ["breathing_difficulty"]


def test_other_mentioned_symptom_reaches_the_timeline(mem):
    ex = dict(FEVER_FIRST, facts=dict(FEVER_FIRST["facts"], headache={"value": True, "evidence": "bad headache"}))
    agent = make(mem, RoutingMock(extract=[ex]))
    agent.handle("30 year old male, fever for 3 days and a bad headache")
    run_to_end(agent)
    assert any("Headache" in line for line in mem.timeline_text(agent.state.patient_id, now=T0))


# ---------------------------------------------------------------- latency logging
def test_every_llm_call_is_timed_and_labelled(mem):
    plan = [{"action": "ask_question", "field": "breathing_difficulty", "question": "Any trouble breathing?"}]
    agent = make(mem, RoutingMock(extract=[FEVER_FIRST], plan=plan))
    turn = agent.handle("30 year old male, fever for 3 days")
    entry = [t for t in turn.tool_log if t["tool"] == "llm_calls"][0]
    labels = [c["label"] for c in entry["calls"]]
    assert labels == ["extract", "plan"] and all(c["seconds"] >= 0 and c["ok"] for c in entry["calls"])
    assert entry["total_s"] == round(sum(c["seconds"] for c in entry["calls"]), 2)


def test_report_contains_a_latency_summary_by_purpose(mem):
    agent = make(mem, RoutingMock(extract=[FEVER_FIRST], review=[{"suggested_urgency": "low", "concern": ""}],
                                    lenient_plan=True))
    agent.handle("30 year old male, fever for 3 days")
    turn, _ = run_to_end(agent)
    lat = turn.result["latency"]
    assert lat["llm_calls"] >= 3 and set(lat["by_purpose"]) == {"extract", "plan", "review"}
    assert lat["by_purpose"]["review"]["calls"] == 1
    assert mem.get_report(agent.state.session_id)["latency"]["llm_calls"] == lat["llm_calls"]


def test_failed_calls_are_recorded_with_their_error(mem):
    agent = make(mem, DownLLM())
    agent.handle("I have a fever")
    summary = agent.latency_summary()
    assert summary["by_purpose"]["extract"]["failed"] == 1


def test_no_llm_means_no_latency_entries(mem):
    agent = make(mem, None)
    turn = agent.handle("I have a fever")
    assert not [t for t in turn.tool_log if t["tool"] == "llm_calls"] and agent.latency_summary()["llm_calls"] == 0


def test_timed_wrapper_passes_groq_style_metadata_through():
    from agent.llm_client import TimedLLM

    class Meta:
        last_meta = {"attempts": 2, "rate_limit_hits": 1, "sleep_s": 2, "http_s": 1.5, "completion_tokens": 400}

        def complete_json(self, system, user, max_retries=1):
            return {"ok": True}

    t = TimedLLM(Meta(), lambda s: "x")
    assert t.complete_json("s", "u") == {"ok": True}
    rec = t.drain()[0]
    assert rec["label"] == "x" and rec["attempts"] == 2 and rec["rate_limit_hits"] == 1 and rec["completion_tokens"] == 400
    assert t.drain() == []


# ---------------------------------------------------------------- fast path: bare answers need no LLM
def fever_agent(mem, plan=None):
    plan = plan or [{"action": "ask_question", "field": "breathing_difficulty", "question": "Any trouble breathing?"}]
    llm = RoutingMock(extract=[FEVER_FIRST], plan=plan)
    agent = make(mem, llm)
    agent.handle("30 year old male, fever for 3 days")
    return agent, llm


def test_bare_no_is_read_by_python_without_an_extract_call(mem):
    agent, llm = fever_agent(mem)
    before = llm.calls.count("extract")
    turn = agent.handle("no")
    assert llm.calls.count("extract") == before                      # no new extraction call
    assert agent.state.get("breathing_difficulty") is False
    ex = [t for t in turn.tool_log if t["tool"] == "extract_facts"][0]
    assert ex["fast_path"] is True and ex["applied"] == ["breathing_difficulty"]


def test_bare_yes_still_triggers_the_emergency_gate(mem):
    agent, llm = fever_agent(mem)
    turn = agent.handle("yes")                                       # breathing difficulty
    assert agent.state.get("breathing_difficulty") is True
    assert turn.kind in ("emergency", "question", "result")          # whichever the rules decide, the fact is stored


@pytest.mark.parametrize("answer", ["no, but I also have a rash", "no and my chest hurts", "no I feel very weak"])
def test_longer_answers_still_go_to_the_llm(mem, answer):
    agent, llm = fever_agent(mem)
    before = llm.calls.count("extract")
    agent.handle(answer)
    assert llm.calls.count("extract") == before + 1


def test_bare_unknown_is_recorded_without_the_llm(mem):
    agent, llm = fever_agent(mem)
    before = llm.calls.count("extract")
    agent.handle("idk")
    assert "breathing_difficulty" in agent.state.unknown_fields and llm.calls.count("extract") == before


def test_unparseable_short_answer_falls_back_to_the_llm(mem):
    agent, llm = fever_agent(mem)
    before = llm.calls.count("extract")
    agent.handle("hmm purple")
    assert llm.calls.count("extract") == before + 1


def test_fast_path_unit_rules():
    from agent.extractor import is_bare_answer
    from engine.protocols import Field_
    b = Field_(key="x", type="bool", question="q")
    n = Field_(key="t", type="float", question="q")
    e = Field_(key="s", type="enum", question="q", options=["mild", "moderate", "severe"])
    lst = Field_(key="c", type="list", question="q")
    assert all(is_bare_answer(b, t) for t in ["no", "No.", "yes", "nahi", "haan", "idk"])
    assert not any(is_bare_answer(b, t) for t in ["", "no but rash", "maybe a bit", "mild"])
    assert all(is_bare_answer(n, t) for t in ["101 F", "38.5", "3 days", "2", "39 c"])
    assert not any(is_bare_answer(n, t) for t in ["about three", "101 and shivering", "yes"])
    assert is_bare_answer(e, "Mild") and not is_bare_answer(e, "quite bad")
    assert is_bare_answer(lst, "none") and not is_bare_answer(lst, "sugar and bp")


def test_fast_path_does_not_reset_the_circuit_breaker(mem):
    agent = make(mem, DownLLM())
    agent.handle("I have a fever")
    agent._llm_failures = 3
    agent.handle("no")                                               # bare answer, fast path
    assert agent._llm_failures == 3


def test_a_whole_conversation_of_bare_answers_needs_one_extraction_call(mem):
    llm = RoutingMock(extract=[FEVER_FIRST], lenient_plan=True)
    agent = make(mem, llm)
    agent.handle("30 year old male, fever for 3 days")
    turn, _ = run_to_end(agent)
    assert turn.kind == "result"
    assert llm.calls.count("extract") == 1                           # only the free-text first message
    lat = turn.result["latency"]["by_purpose"]
    assert lat["extract"]["calls"] == 1


def test_a_successful_planner_call_closes_the_breaker(mem):
    agent, llm = fever_agent(mem, plan=[
        {"action": "ask_question", "field": "breathing_difficulty", "question": "Any trouble breathing?"},
        {"action": "ask_question", "field": "stiff_neck", "question": "Is your neck stiff?"}])
    agent._llm_failures = 2
    agent.handle("no")                                               # bare answer -> planner runs and succeeds
    assert agent._llm_failures == 0
