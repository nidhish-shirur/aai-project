"""Headless tests of the Streamlit app (patient chat + clinician dashboard). Skipped if streamlit is missing.
They use a temporary database and force 'no LLM', so they never touch your real data or the network."""
from pathlib import Path

import pytest

st_mod = pytest.importorskip("streamlit")
_ver = tuple(int(x) for x in st_mod.__version__.split(".")[:2])
if _ver < (1, 63):
    pytest.skip(f"Streamlit {st_mod.__version__}: its test tool keeps stale widgets after st.rerun(); "
                "run  pip install -U streamlit  (>= 1.63) to run these tests", allow_module_level=True)
from streamlit.testing.v1 import AppTest  # noqa: E402

import config  # noqa: E402
from memory.db import Memory  # noqa: E402
from tests.test_orchestrator import auto_answer  # noqa: E402


APP = str(Path(__file__).resolve().parents[1] / "app.py")


def new_browser():
    """A fresh app session (like opening the page in a new browser tab)."""
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    return at


@pytest.fixture
def app(tmp_path, monkeypatch):
    import streamlit as st
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "app.db")
    monkeypatch.setattr(config, "DASHBOARD_PIN", "")
    st.cache_resource.clear()
    return new_browser()


def start(at, code=""):
    at.checkbox[0].set_value(False)                      # never call a real LLM in tests
    if code:
        at.text_input[0].set_value(code)
    at.button[0].click().run()
    assert not at.exception
    return at


def say(at, text):
    at.chat_input[0].set_value(text).run()
    assert not at.exception
    return at


def finish_fever(at):
    say(at, "I have a fever")
    for _ in range(25):
        if "agent" in at.session_state and at.session_state["agent"].done:
            break
        say(at, auto_answer(at.session_state["agent"]))
    assert at.session_state["agent"].done
    return at


def test_start_page_shows_disclaimer_and_emergency_number(app):
    assert any("112" in w.value for w in app.warning)
    assert any("does not diagnose" in c.value for c in app.caption)


def test_emergency_message_is_shown_as_an_error_banner_with_zero_llm(app):
    start(app)
    say(app, "my father has crushing chest pain")
    assert any("EMERGENCY" in e.value for e in app.error)
    assert app.session_state["agent"].done and app.session_state["final"].kind == "emergency"


def test_full_fever_assessment_reaches_a_result_with_code_and_trace(app):
    start(app)
    finish_fever(app)
    banners = [e.value for e in [*app.warning, *app.info, *app.success, *app.error]]
    assert any(any(level in b for level in ("LOW", "MODERATE", "URGENT")) for b in banners)
    assert any("anonymous patient code" in b for b in banners)
    assert any("Start a new assessment" == b.label for b in app.button)
    assert len(app.expander) >= 1


def test_user_text_is_shown_as_plain_text_not_markdown(app):
    start(app)
    say(app, "I have a fever **bold** ![x](http://evil.example/p.png)")
    assert any("evil.example" in t.value for t in app.text)
    assert not any("evil.example" in m.value for m in app.markdown)


def test_unknown_patient_code_starts_a_new_record_with_a_notice(app):
    start(app, code="P-NOPE00")
    assert any("not found" in c.value for c in app.caption)
    assert app.session_state["agent"].state.patient_id != "P-NOPE00"


def test_new_assessment_button_resets_the_chat(app):
    start(app)
    finish_fever(app)
    next(b for b in app.button if b.label == "Start a new assessment").click().run()
    assert "agent" not in app.session_state and any(b.label == "Start assessment" for b in app.button)


def test_delete_my_data_erases_the_record(app, tmp_path):
    start(app)
    finish_fever(app)
    pid = app.session_state["agent"].state.patient_id
    next(b for b in app.button if b.label == "Delete my data").click().run()
    mem = Memory(config.DB_PATH)
    assert not mem.patient_exists(pid)
    mem.close()


# ---------------------------------------------------------------- clinician dashboard
def open_dashboard(at):
    at.sidebar.radio[0].set_value("Clinician dashboard").run()
    assert not at.exception
    return at


def test_dashboard_is_empty_before_any_session(app):
    open_dashboard(app)
    assert any("No sessions yet" in i.value for i in app.info)


def test_dashboard_lists_sessions_most_urgent_first_and_shows_the_summary(app):
    start(app)
    finish_fever(app)                                                  # browser 1: a normal fever visit (older)
    other = new_browser()                                              # browser 2: an emergency (newer)
    start(other)
    say(other, "my father has crushing chest pain")
    open_dashboard(other)
    rows = other.dataframe[0].value
    assert list(rows["urgency"])[0] == "emergency" and len(rows) == 2   # urgency order, not time order
    assert [m.label for m in other.metric] == ["Emergency", "Urgent", "Moderate", "Low"]
    md = " ".join(m.value for m in other.markdown)
    assert "Triage summary for clinician" in md and "does not diagnose" in md


def test_dashboard_pin_blocks_until_correct(app, monkeypatch):
    start(app)
    say(app, "my father has crushing chest pain")
    monkeypatch.setattr(config, "DASHBOARD_PIN", "1234")
    open_dashboard(app)
    assert len(app.dataframe) == 0 and len(app.text_input) == 1
    app.text_input[0].set_value("0000").run()
    assert any("Wrong PIN" in e.value for e in app.error) and len(app.dataframe) == 0
    app.text_input[0].set_value("1234").run()
    assert len(app.dataframe) == 1


def test_dashboard_delete_removes_the_patient(app):
    start(app)
    say(app, "my father has crushing chest pain")
    pid = app.session_state["agent"].state.patient_id
    open_dashboard(app)
    next(b for b in app.button if b.label.startswith("Delete this patient")).click().run()
    mem = Memory(config.DB_PATH)
    assert not mem.patient_exists(pid)
    mem.close()


def test_unfinished_sessions_are_hidden_unless_asked_for(app):
    start(app)
    say(app, "I have a fever")                                         # abandoned after one message
    open_dashboard(app)
    assert len(app.dataframe) == 0 and any("No finished assessments" in i.value for i in app.info)
    app.checkbox[1].set_value(True).run()                              # "Also show unfinished sessions"
    assert len(app.dataframe) == 1 and list(app.dataframe[0].value["urgency"]) == ["(unfinished)"]


def test_conditions_question_has_no_raw_option_list(app):
    start(app)
    say(app, "I have a fever")
    agent = app.session_state["agent"]
    for _ in range(20):
        if agent._pending and agent._pending.get("keys") and agent._pending["keys"][0] == "chronic_conditions":
            break
        say(app, auto_answer(agent))
    last = [m.value for m in app.markdown][-1]
    assert "long-term conditions" in last.lower() and "heart_disease" not in last
