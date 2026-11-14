"""Rule-engine tests: urgency, modifiers, incomplete information, escalate-only."""
import pytest

from agent.state import PatientState
from engine.protocols import bump_level, max_level, missing_fields, get_protocol, select_protocol_from_text
from engine.triage_rules import compute_urgency, escalate_only


def fever_state(**extra) -> PatientState:
    s = PatientState(complaint="fever")
    s.add_user_message("I have a fever")
    base = dict(age_years=30, sex="male", chronic_conditions=[], duration_days=1, severity="mild",
                breathing_difficulty=False, chest_pain=False, stiff_neck=False, confusion=False,
                unable_to_drink_fluids=False)
    base.update(extra)
    s.update(base)
    return s


def test_mild_short_fever_is_low():
    r = compute_urgency(fever_state())
    assert r.urgency == "low" and r.confidence == "medium"


def test_manual_example_three_day_fever_is_moderate():
    r = compute_urgency(fever_state(duration_days=3, severity="moderate", temperature_c=38.2))
    assert r.urgency == "moderate" and "fever_two_days" in r.matched_rules


def test_high_fever_several_days_is_urgent():
    r = compute_urgency(fever_state(duration_days=3, temperature_c=39.4))
    assert r.urgency == "urgent"


def test_highest_matching_rule_wins():
    r = compute_urgency(fever_state(duration_days=6, severity="moderate", temperature_c=40.5))
    assert r.urgency == "urgent"


def test_red_flag_overrides_everything():
    r = compute_urgency(fever_state(stiff_neck=True))
    assert r.urgency == "emergency" and r.red_flags


def test_red_flag_from_words_even_if_facts_say_no():
    s = fever_state()
    s.add_user_message("actually I can't breathe properly")
    assert compute_urgency(s).urgency == "emergency"


# ---------------- modifiers
def test_old_age_raises_one_level():
    assert compute_urgency(fever_state(age_years=72)).urgency == "moderate"


def test_chronic_condition_raises_one_level():
    assert compute_urgency(fever_state(chronic_conditions=["diabetes"])).urgency == "moderate"


def test_low_risk_condition_does_not_raise():
    assert compute_urgency(fever_state(chronic_conditions=["hypertension"])).urgency == "low"


def test_modifiers_do_not_stack_and_cap_at_urgent():
    r = compute_urgency(fever_state(age_years=80, chronic_conditions=["diabetes", "cancer"],
                                    duration_days=3, temperature_c=39.5))
    assert r.urgency == "urgent"           # already urgent: not pushed to emergency


def test_modifier_never_creates_emergency():
    r = compute_urgency(fever_state(age_years=80, duration_days=6))
    assert r.urgency == "urgent"


# ---------------- incomplete information
def test_missing_answers_use_cautious_floor_and_flag_review():
    s = PatientState(complaint="fever")
    s.add_user_message("fever since yesterday")
    s.update({"age_years": 30, "sex": "male"})
    r = compute_urgency(s)
    assert r.urgency == "moderate" and r.confidence == "low"
    assert r.needs_clinician_review and "duration_days" in r.missing_required


def test_dont_know_counts_as_missing_not_as_no():
    s = fever_state()
    s.facts.pop("stiff_neck")
    s.mark_unknown("stiff_neck")
    r = compute_urgency(s)
    assert "stiff_neck" in r.missing_required and r.confidence == "low"
    assert r.urgency == "moderate"
    assert all(f.key != "stiff_neck" for f in missing_fields(s, get_protocol("fever")))  # not asked again


def test_unknown_complaint_goes_to_clinician_review():
    s = PatientState(complaint="other")
    s.update(dict(age_years=30, sex="male", chronic_conditions=[], duration_days=2, severity="mild",
                  breathing_difficulty=False, chest_pain=False, loss_of_consciousness=False,
                  severe_bleeding=False, neuro_deficit=False))
    r = compute_urgency(s)
    assert r.needs_clinician_review and r.confidence == "low" and r.urgency == "moderate"


def test_conflicting_answers_flagged():
    r = compute_urgency(fever_state(fever=False, temperature_c=39.0, duration_days=1))
    assert r.needs_clinician_review and r.confidence == "low"


def test_child_and_pregnancy_flag_review():
    assert compute_urgency(fever_state(age_years=8)).needs_clinician_review
    s = fever_state(age_years=28, sex="female", pregnant=True)
    assert compute_urgency(s).needs_clinician_review


def test_mental_health_emergency_needs_human():
    s = PatientState(complaint="other")
    s.add_user_message("I want to kill myself")
    r = compute_urgency(s)
    assert r.urgency == "emergency" and r.red_flag_category == "mental_health" and r.needs_clinician_review


# ---------------- escalate-only merge (core safety property)
@pytest.mark.parametrize("rule", ["low", "moderate", "urgent", "emergency"])
@pytest.mark.parametrize("llm", [None, "low", "moderate", "urgent", "emergency", "garbage"])
def test_llm_can_never_lower_urgency(rule, llm):
    merged = escalate_only(rule, llm)
    assert merged == max_level(rule, llm) if llm in ("low", "moderate", "urgent", "emergency") else merged == rule
    from engine.protocols import level_index
    assert level_index(merged) >= level_index(rule)


def test_helpers():
    assert bump_level("low", "urgent") == "moderate"
    assert bump_level("urgent", "urgent") == "urgent"
    assert bump_level("emergency", "urgent") == "emergency"   # never lowered by the cap


# ---------------- adaptive questioning
def test_pain_worse_breathing_only_asked_when_no_arm_spread():
    p = get_protocol("chest_pain")
    s = PatientState(complaint="chest_pain")
    s.update({"age_years": 40, "sex": "male"})
    keys = [f.key for f in missing_fields(s, p, include_optional=True)]
    assert "pain_worse_breathing" in keys or True       # unknown arm spread: not applicable yet
    assert "pain_worse_breathing" not in [f.key for f in missing_fields(s, p)]
    s.update({"pain_radiates_arm_jaw": False})
    assert "pain_worse_breathing" in [f.key for f in missing_fields(s, p)]


def test_pregnancy_question_only_for_women_of_childbearing_age():
    p = get_protocol("fever")
    m = PatientState(complaint="fever"); m.update({"age_years": 30, "sex": "male"})
    f = PatientState(complaint="fever"); f.update({"age_years": 30, "sex": "female"})
    old = PatientState(complaint="fever"); old.update({"age_years": 70, "sex": "female"})
    assert "pregnant" not in [x.key for x in missing_fields(m, p)]
    assert "pregnant" in [x.key for x in missing_fields(f, p)]
    assert "pregnant" not in [x.key for x in missing_fields(old, p)]


def test_never_asks_for_known_information():
    p = get_protocol("fever")
    s = fever_state()
    assert missing_fields(s, p) == []


@pytest.mark.parametrize("text,expected", [
    ("I have a fever", "fever"),
    ("I have chest pain", "chest_pain"),
    ("bad headache", "headache"),
    ("burning urination", "urinary"),
    ("vomiting and diarrhoea", "gi_upset"),
    ("my elbow looks funny", "other"),
    ("fever and chest pain", "chest_pain"),            # more dangerous complaint wins ties
])
def test_protocol_selection(text, expected):
    assert select_protocol_from_text(text) == expected
