"""Clinician summary: structure, honesty (only what is in the record) and Markdown safety."""
import pytest

from agent.state import PatientState
from agent.summary import build_clinician_summary, format_tool_log, label, md_escape, to_markdown
from engine.triage_rules import compute_urgency


def fever_state():
    s = PatientState(patient_id="P-ABC123", complaint="fever")
    s.add_user_message("I have had fever for 3 days")
    s.update({"age_years": 30, "sex": "male", "duration_days": 3, "chronic_conditions": [], "severity": "mild",
              "temperature_c": 38.3, "stiff_neck": False, "breathing_difficulty": False, "confusion": False,
              "chest_pain": False, "unable_to_drink_fluids": False, "headache": True})
    s.mark_unknown("recent_travel")
    return s


def report_for(state, **extra):
    r = compute_urgency(state).to_dict()
    r.update({"rule_urgency": r["urgency"], "llm_review": None, "questions_asked": 7, "llm_used": True})
    r.update(extra)
    return r


def test_summary_lists_present_denied_unknown_and_measurements():
    s = fever_state()
    sm = build_clinician_summary(s, report_for(s), ["Day 1 (2026-11-08): Fever"])
    assert sm["present"] == ["Headache"]                       # the complaint itself is not repeated as a symptom
    assert set(sm["denied"]) == {"Stiff neck", "Breathing difficulty", "Confusion", "Chest pain",
                                 "Unable to drink fluids"}
    assert "Recent travel" in sm["not_answered"]
    assert "Highest temperature: 38.3 °C" in sm["measurements"] and "Duration: 3 day(s)" in sm["measurements"]
    assert sm["main_problem"] and sm["age"] == "30" and sm["sex"] == "male" and sm["chronic_conditions"] == []


def test_summary_only_contains_facts_from_the_record():
    s = fever_state()
    sm = build_clinician_summary(s, report_for(s))
    allowed = {label(k) for k, v in s.all_facts().items() if isinstance(v, bool)}
    assert set(sm["present"]) | set(sm["denied"]) <= allowed
    assert "Rash" not in " ".join(sm["present"] + sm["denied"])       # nothing is invented


def test_markdown_has_urgency_reasons_and_disclaimer():
    s = fever_state()
    md = to_markdown(build_clinician_summary(s, report_for(s), ["Day 1 (2026-11-08): Fever"]))
    assert "## Urgency: Moderate" in md and "P-ABC123" in md and "Fever lasting two days" in md
    assert "Day 1 (2026-11-08): Fever" in md and "does not diagnose" in md and "NOT a diagnosis" in md
    assert "Human clinician review" not in md or "clinician review" in md.lower()


def test_review_section_and_rule_vs_final_urgency():
    s = fever_state()
    r = report_for(s, urgency="urgent", rule_urgency="moderate", needs_clinician_review=True,
                   clinician_reasons=["AI safety review raised the urgency"],
                   llm_review={"raised": True, "concern": "Fever in a vulnerable patient.", "capped": False})
    md = to_markdown(build_clinician_summary(s, r))
    assert "Rule engine said **moderate**" in md and "raised it to **urgent**" in md
    assert "Human clinician review recommended" in md and "AI safety review (second opinion)" in md


def test_unfinished_session_is_marked_not_assessed():
    s = fever_state()
    sm = build_clinician_summary(s, None)
    assert sm["finished"] is False and sm["urgency"] is None
    assert "not assessed" in to_markdown(sm)


def test_emergency_report_with_red_flag_dicts():
    s = PatientState(patient_id="P-E", complaint="chest_pain")
    s.add_user_message("crushing chest pain")
    sm = build_clinician_summary(s, compute_urgency(s).to_dict())
    assert sm["urgency"] == "emergency" and sm["red_flags"]
    assert "Emergency warning signs detected" in to_markdown(sm)


# ---------------------------------------------------------------- markdown safety (patient text is untrusted)
@pytest.mark.parametrize("evil", [
    "![x](http://evil.example/p.png)", "[click](javascript:alert(1))", "<img src=x onerror=alert(1)>",
    "hello\n# Fake heading\n- injected", "**bold** _it_ `code`", "| a | b |",
])
def test_patient_words_cannot_inject_markdown(evil):
    import re
    s = fever_state()
    s.add_user_message(evil)
    md = to_markdown(build_clinician_summary(s, report_for(s)))
    quoted = [l for l in md.splitlines() if l.startswith('- "')]
    assert len(quoted) == 2                                     # one line per message: newlines cannot add lines
    for line in quoted:
        assert not re.search(r"(?<!\\)[<>!\[\]()`*_#|~]", line[2:]), line   # every special character is escaped
    assert not any(l.startswith("# ") and "Fake" in l for l in md.splitlines())


def test_ai_concern_is_escaped_too():
    s = fever_state()
    r = report_for(s, urgency="urgent", llm_review={"raised": True, "concern": "see [this](http://x.example)",
                                                    "capped": False})
    md = to_markdown(build_clinician_summary(s, r))
    assert "](http://x.example)" not in md


def test_md_escape_basics():
    assert md_escape("a\nb\tc") == "a b c"
    assert md_escape("*x*") == "\\*x\\*" and md_escape("") == ""


# ---------------------------------------------------------------- tool-log explanation
def test_format_tool_log_is_one_readable_line_per_turn():
    log = [{"turn": 0, "tool": "start", "patient_id": "P-1"},
           {"turn": 1, "tool": "red_flag_gate", "stage": "text_only", "triggered": False},
           {"turn": 1, "tool": "extract_facts", "applied": ["age_years"], "used_fallback": False, "fast_path": False},
           {"turn": 1, "tool": "plan_next", "source": "llm", "action": "ask_question", "field": "stiff_neck"},
           {"turn": 1, "tool": "ask_question", "field": "stiff_neck", "number": 1},
           {"turn": 1, "tool": "llm_calls", "calls": [], "total_s": 0},
           {"turn": 2, "tool": "extract_facts", "applied": ["stiff_neck"], "fast_path": True},
           {"turn": 2, "tool": "compute_urgency", "urgency": "moderate", "matched_rules": ["fever_two_days"]}]
    lines = format_tool_log(log)
    assert lines[0].startswith("Start: session started") and len(lines) == 3
    assert "emergency check (text only): clear" in lines[1] and "planned next step [llm]" in lines[1]
    assert "read by Python" in lines[2] and "rule engine decided: moderate (rules: fever_two_days)" in lines[2]
    assert "llm calls" not in " ".join(lines)


def test_heading_level_can_be_shifted_for_the_web_page():
    s = fever_state()
    sm = build_clinician_summary(s, report_for(s))
    assert to_markdown(sm).startswith("# Triage summary")
    shifted = to_markdown(sm, base_level=3)
    assert shifted.startswith("### Triage summary") and "\n#### Patient" in shifted
    assert "\n# " not in shifted and "\n## " not in shifted


def test_all_messages_are_listed_in_order_and_truncation_is_stated():
    s = fever_state()
    for i in range(30):
        s.add_user_message(f"answer {i}")
    sm = build_clinician_summary(s, report_for(s))
    assert len(sm["patient_words"]) == 20 and sm["patient_words_hidden"] == 11
    md = to_markdown(sm)
    assert '- "I have had fever for 3 days"' in md and "(11 later message(s) not shown)" in md
    short = fever_state()
    short.add_user_message("no")
    assert "not shown" not in to_markdown(build_clinician_summary(short, report_for(short)))
