"""Red-flag gate tests. These are the most important tests in the project:
the gate must catch emergencies and must not be fooled by negation."""
import pytest

from agent.state import PatientState
from engine.red_flags import check_red_flags, check_text
from engine.protocols import get_protocol

MUST_TRIGGER = [
    "I can't breathe",
    "my father is unconscious",
    "he is not responding to me",
    "I have chest pain and I am sweating",
    "chest pain that spreads to my left arm",
    "crushing chest pressure",
    "no fever but chest pain with sweating",          # 'but' must break the negation
    "my face is drooping and my speech is slurred",
    "I am vomiting blood",
    "she is bleeding heavily and it won't stop",
    "I am coughing up blood",
    "I want to kill myself",
    "my throat is closing after eating peanuts",
    "he is having a seizure",
    "my child drank phenyl",
    "he was bitten by a snake",
    "mujhe saans nahi aa rahi",
    "difficulty breathing since last night",
    "I think I'm having a heart attack",
    "the worst headache of my life",
    "I passed out in the kitchen",
    "purple spots on his body and fever",
]

MUST_NOT_TRIGGER = [
    "I have a mild fever and a sore throat",
    "no chest pain, just a headache",
    "I have no difficulty breathing",
    "I do not have any trouble breathing",
    "she did not faint",
    "he never lost consciousness",
    "food poisoning symptoms with some vomiting",
    "no sweating, just chest pain",                    # chest pain alone is handled by the protocol, not a text red flag
    "I have had fever without breathing difficulty",
    "mild cough since yesterday",
    "my stomach hurts after lunch",
]


@pytest.mark.parametrize("text", MUST_TRIGGER)
def test_text_red_flag_fires(text):
    assert check_text(text), f"should have triggered: {text!r}"


@pytest.mark.parametrize("text", MUST_NOT_TRIGGER)
def test_text_red_flag_not_fooled(text):
    assert not check_text(text), f"false alarm: {text!r} -> {[h.id for h in check_text(text)]}"


def test_mental_health_category():
    s = PatientState()
    s.add_user_message("I want to end my life")
    gate = check_red_flags(s)
    assert gate.triggered and gate.category == "mental_health"


def test_medical_category_default():
    s = PatientState()
    s.add_user_message("I can't breathe")
    assert check_red_flags(s).category == "medical"


@pytest.mark.parametrize("facts", [
    {"breathing_difficulty": True},
    {"loss_of_consciousness": True},
    {"face_droop": True},
    {"neuro_deficit": True},
    {"severe_bleeding": True},
    {"vomiting_blood": True},
    {"seizure": True},
    {"suicidal_thoughts": True},
    {"chest_pain": True, "sweating": True},
    {"chest_pain": True, "pain_radiates_arm_jaw": True},
    {"chest_pain": True, "pain_crushing_pressure": True},
])
def test_fact_red_flags_fire(facts):
    s = PatientState()
    s.update(facts)
    assert check_red_flags(s).triggered


@pytest.mark.parametrize("facts", [
    {"chest_pain": True},                               # alone: urgent via protocol, not a global red flag
    {"chest_pain": True, "sweating": False},
    {"breathing_difficulty": False, "loss_of_consciousness": False},
    {"fever": True},
])
def test_fact_red_flags_quiet(facts):
    s = PatientState()
    s.update(facts)
    assert not check_red_flags(s).triggered


def test_unknown_facts_never_trigger():
    s = PatientState()   # nothing known
    assert not check_red_flags(s).triggered


def test_protocol_specific_flags():
    s = PatientState(complaint="fever")
    s.update({"stiff_neck": True})
    assert check_red_flags(s, get_protocol("fever")).triggered
    s2 = PatientState(complaint="fever", age_years=0.1)
    assert check_red_flags(s2, get_protocol("fever")).triggered   # infant under 3 months


def test_red_flag_in_earlier_message_is_remembered():
    s = PatientState()
    s.add_user_message("I have chest pain")
    s.add_user_message("and now I am sweating a lot")
    assert check_red_flags(s).triggered


# ---------------- Phase 6 live finding: common suicidal phrasings must reach the supportive emergency message
@pytest.mark.parametrize("text", [
    "I don't want to live any more", "i do not want to live", "I dont want to live anymore", "no reason to live",
    "everyone is better off dead without me", "life is not worth living", "I can't go on living",
    "I want it all to end", "I'd rather be dead", "I want to end my life", "kill me",
])
def test_suicidal_phrasings_trigger_the_mental_health_flag(text):
    from engine.red_flags import check_text
    hits = check_text(text)
    assert hits and hits[0].category == "mental_health"


@pytest.mark.parametrize("text", [
    "I don't want to live in Mumbai", "I don't want to live with this headache", "no thoughts of suicide",
    "I do not want to die", "I want to live a long life", "I want the pain to stop", "I have a fever and want to rest",
])
def test_ordinary_sentences_do_not_trigger_the_mental_health_flag(text):
    from engine.red_flags import check_text
    assert not check_text(text)
