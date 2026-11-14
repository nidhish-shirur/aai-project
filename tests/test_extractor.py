"""Extractor tests using the scripted MockLLM (no internet, no key)."""
import pytest

from agent.extractor import (
    build_catalogue, build_user_prompt, coerce_value, extract_into_state, validate_extraction,
)
from agent.llm_client import LLMError, MockLLM, parse_json_loose
from agent.state import PatientState
from engine.protocols import Field_
from engine.red_flags import check_red_flags


def reply(complaint=None, facts=None, dont_know=None):
    return {"complaint": complaint, "facts": facts or {}, "dont_know": dont_know or []}


def fact(value, evidence=""):
    return {"value": value, "evidence": evidence}


# ---------------- parse_json_loose
def test_parse_json_with_fences_and_chatter():
    assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_loose('Sure! Here you go: {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}


@pytest.mark.parametrize("bad", ["", "no json here", "[1, 2]", "{broken"])
def test_parse_json_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_json_loose(bad)


# ---------------- coercion
def F(key, type_, **kw):
    return Field_(key=key, type=type_, question="?", **kw)


def test_coerce_basic_types():
    assert coerce_value(F("x", "bool"), "yes") == (True, True, "")
    assert coerce_value(F("x", "bool"), "maybe")[0] is False
    assert coerce_value(F("duration_days", "float"), "3")[1] == 3.0
    assert coerce_value(F("age_years", "float"), 150)[0] is False          # impossible age
    assert coerce_value(F("duration_days", "float"), True)[0] is False      # bool is not a number
    assert coerce_value(F("severity", "enum", options=["mild", "moderate", "severe"]), "Moderate")[1] == "moderate"
    assert coerce_value(F("severity", "enum", options=["mild", "moderate", "severe"]), "awful")[0] is False


def test_coerce_fahrenheit_safety_net():
    ok, value, _ = coerce_value(F("temperature_c", "float"), 101.3)
    assert ok and value == 38.5
    assert coerce_value(F("temperature_c", "float"), 12)[0] is False        # 12 C is impossible


def test_coerce_list_maps_to_options():
    f = F("chronic_conditions", "list", options=["diabetes", "heart_disease", "other"])
    assert coerce_value(f, ["Diabetes", "gout"])[1] == ["diabetes", "other"]
    assert coerce_value(f, "none")[1] == []


# ---------------- end-to-end with mock
def test_first_message_sets_complaint_and_facts():
    s = PatientState()
    llm = MockLLM([reply("fever", {"duration_days": fact(3), "temperature_c": fact(38.5)})])
    r = extract_into_state(s, "I have had a fever for three days, 101.3 F", llm)
    assert s.complaint == "fever" and s.facts["duration_days"] == 3 and s.facts["temperature_c"] == 38.5
    assert r.complaint_set == "fever" and not r.used_fallback
    assert s.user_messages == ["I have had a fever for three days, 101.3 F"]


def test_denial_needs_a_quote_from_the_patient():
    s = PatientState(complaint="fever")
    llm = MockLLM([reply(facts={
        "chest_pain": fact(False, "no chest pain"),            # quoted -> accepted
        "stiff_neck": fact(False, "she said her neck is fine"),  # invented quote -> dropped
    })])
    r = extract_into_state(s, "No chest pain but I feel hot", llm)
    assert s.facts.get("chest_pain") is False
    assert "stiff_neck" not in s.facts
    assert any(k == "stiff_neck" for k, _ in r.rejected)


def test_unverified_true_is_kept_but_flagged():
    s = PatientState(complaint="fever")
    llm = MockLLM([reply(facts={"stiff_neck": fact(True, "")})])
    r = extract_into_state(s, "my neck hurts to move", llm)
    assert s.facts["stiff_neck"] is True and "stiff_neck" in r.unverified_true


def test_hallucinated_fields_and_values_rejected():
    s = PatientState(complaint="fever")
    llm = MockLLM([reply(facts={"made_up_field": fact(True, "x"), "age_years": fact(500),
                                "severity": fact("terrible")})])
    r = extract_into_state(s, "hello", llm)
    assert s.age_years is None and "severity" not in s.facts
    assert {k for k, _ in r.rejected} == {"made_up_field", "age_years", "severity"}


def test_pending_question_yes_no():
    s = PatientState(complaint="fever")
    llm = MockLLM([reply(facts={"breathing_difficulty": fact(False, "no")})])
    extract_into_state(s, "no", llm, pending_key="breathing_difficulty")
    assert s.facts["breathing_difficulty"] is False


def test_pending_answer_recovered_deterministically_when_llm_misses_it():
    s = PatientState(complaint="fever")
    llm = MockLLM([reply()])  # LLM returned nothing useful
    extract_into_state(s, "yes", llm, pending_key="stiff_neck")
    assert s.facts["stiff_neck"] is True


def test_dont_know_marks_unknown_and_stops_asking():
    s = PatientState(complaint="fever")
    llm = MockLLM([reply(dont_know=["temperature_c"])])
    extract_into_state(s, "I haven't measured it", llm, pending_key="temperature_c")
    assert "temperature_c" in s.unknown_fields


def test_complaint_is_not_switched_mid_session():
    s = PatientState(complaint="fever")
    llm = MockLLM([reply("headache")])
    r = extract_into_state(s, "also a headache", llm)
    assert s.complaint == "fever" and r.complaint_set is None


def test_llm_failure_uses_fallback_and_red_flag_gate_still_works():
    s = PatientState()
    llm = MockLLM([LLMError("network down")])
    r = extract_into_state(s, "I have chest pain and I am sweating", llm)
    assert r.used_fallback and "network down" in r.error
    assert s.complaint == "chest_pain"                     # keyword fallback
    assert check_red_flags(s).triggered                    # raw text was stored first


def test_no_llm_at_all():
    s = PatientState()
    r = extract_into_state(s, "bad headache since morning", None)
    assert s.complaint == "headache" and r.used_fallback


def test_red_flag_text_caught_even_if_llm_says_nothing():
    s = PatientState(complaint="fever")
    extract_into_state(s, "my father suddenly can't breathe", MockLLM([reply()]))
    assert check_red_flags(s).triggered


def test_malformed_payload_shapes_do_not_crash():
    s = PatientState(complaint="fever")
    for bad in ({"facts": "nonsense"}, {"facts": None}, {"dont_know": "x"}, {}):
        extract_into_state(s, "hi", MockLLM([bad]))


def test_prompt_contains_schema_and_pending_but_not_other_patients():
    s = PatientState()
    cat = build_catalogue(s)
    p = build_user_prompt(s, "I have a fever", "age_years", cat)
    assert "COMPLAINTS" in p and "temperature_c" in p and "PENDING_QUESTION: age_years" in p


def test_catalogue_narrows_once_complaint_known():
    before = build_catalogue(PatientState())
    after = build_catalogue(PatientState(complaint="fever"))
    assert "pain_over_15_min" in before and "pain_over_15_min" not in after
    assert "temperature_c" in after


def test_complaint_choice_is_repeatable_safer_protocol_wins():
    from agent.extractor import safer_complaint
    text = "30 year old man, fever for 3 days and a bad headache"
    assert safer_complaint("fever", text) == "headache"       # LLM said fever, words match a safer protocol
    assert safer_complaint("headache", text) == "headache"
    assert safer_complaint("chest_pain", text) == "chest_pain"  # never downgrades the LLM's safer pick
    assert safer_complaint("fever", "I feel hot") == "fever"
    assert safer_complaint("other", "hello") == "other"
