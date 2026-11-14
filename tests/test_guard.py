"""Output guard, request detector and fixed-wording lint."""
import pytest

from agent.guard import check_output, detect_requests, safe_or
from engine import pathways
from engine.protocols import get_common_fields, get_extra_facts, get_protocols


@pytest.mark.parametrize("text", [
    "Any stiff neck?", "What is the highest temperature you have measured, in Celsius?",
    "Do you have any long-term conditions like diabetes, heart or lung disease?",
    "Is the pain crushing or like pressure?", "Are you unable to drink fluids?",
    "How many days have you had the cough?", "Have you noticed any rash that does not fade when pressed?",
])
def test_normal_questions_pass(text):
    assert check_output(text).ok


@pytest.mark.parametrize("text,cat", [
    ("You probably have the flu. Any cough?", "diagnosis"),
    ("This might be meningitis, is your neck stiff?", "diagnosis"),
    ("It sounds like an infection.", "diagnosis"),
    ("You are suffering from migraine.", "diagnosis"),
    ("Symptoms of dengue include fever. Any rash?", "diagnosis"),
    ("The likely diagnosis is viral.", "diagnosis"),
    ("Did you take paracetamol for it?", "treatment"),
    ("You should take 500 mg twice a day.", "treatment"),
    ("I recommend you take an antibiotic.", "treatment"),
    ("Try a home remedy like ginger tea.", "treatment"),
    ("Use an inhaler if it gets worse.", "treatment"),
])
def test_diagnosis_and_treatment_language_is_caught(text, cat):
    result = check_output(text)
    assert not result.ok and cat in result.categories


def test_empty_and_none_are_safe_to_check():
    assert check_output("").ok and check_output(None).ok


def test_safe_or_substitutes_fixed_wording():
    assert safe_or("Any stiff neck?", "fallback") == ("Any stiff neck?", False)
    assert safe_or("You have the flu", "fallback") == ("fallback", True)


@pytest.mark.parametrize("text,kinds", [
    ("what disease do I have", ["diagnosis"]),
    ("Can you diagnose me?", ["diagnosis"]),
    ("is it cancer?", ["diagnosis"]),
    ("do I have covid", ["diagnosis"]),
    ("my head hurts, what's wrong with me", ["diagnosis"]),
    ("what does my father have", ["diagnosis"]),
    ("which medicine should I take", ["treatment"]),
    ("how much paracetamol can I take", ["treatment"]),
    ("should I take an antibiotic", ["treatment"]),
    ("what is the dose for a child", ["treatment"]),
    ("any home remedy for this?", ["treatment"]),
    ("what is wrong with me and which tablet should I take", ["diagnosis", "treatment"]),
])
def test_requests_for_diagnosis_or_medicine_are_detected(text, kinds):
    assert detect_requests(text) == kinds


@pytest.mark.parametrize("text", [
    "I have had fever for three days", "no", "yes", "none", "I do not have diabetes", "my sugar is high",
    "it started yesterday", "about 101 F", "I take no medicines", "male",
])
def test_ordinary_answers_are_not_treated_as_requests(text):
    assert detect_requests(text) == []


def test_refusal_text_is_fixed_and_passes_its_own_rules():
    both = pathways.refusal(["diagnosis", "treatment"])
    assert "don't diagnose" in both and "medicines" in both
    assert pathways.refusal([]) == ""


def test_all_fixed_wording_is_free_of_diagnosis_language():
    """Lint: protocol questions, warning signs, rule reasons and red-flag labels must never diagnose."""
    allowed = {"Has anything poisonous or too many tablets been swallowed?"}   # overdose screening question
    bad = []

    def chk(where, text):
        if text not in allowed and not check_output(text).ok:
            bad.append((where, text))

    for p in get_protocols().values():
        for f in p.fields:
            chk(f"{p.id} question", f.question)
        for w in p.warning_signs:
            chk(f"{p.id} warning", w)
        for r in p.rules:
            chk(f"{p.id} rule {r.id}", r.reason)
        for rf in p.red_flags:
            chk(f"{p.id} red flag", rf.label)
    for f in [*get_common_fields(), *get_extra_facts()]:
        chk("common", f.question)
    assert not bad, bad
