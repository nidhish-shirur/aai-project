"""Knowledge-base validity, answer parsing, vocabulary and vignette evaluation."""
import pytest

from engine.answer_parser import parse_answer
from engine.protocols import Field_, get_protocol
from eval.vignette_runner import load_vignettes, metrics, run_all
from scripts.check_knowledge import check


def test_knowledge_base_has_no_errors():
    assert check() == []


def test_every_protocol_has_warning_signs_and_required_fields():
    from engine.protocols import get_protocols
    for p in get_protocols().values():
        assert p.warning_signs
        assert any(f.required for f in p.fields)


# ---------------- answer parser
def F(key, type_, **kw):
    return Field_(key=key, type=type_, question="?", **kw)


@pytest.mark.parametrize("text,expected", [("yes", True), ("Haan", True), ("no", False), ("nahi", False), ("nope", False)])
def test_bool(text, expected):
    assert parse_answer(F("x", "bool"), text) == ("ok", expected)


def test_bool_unclear_and_unknown():
    assert parse_answer(F("x", "bool"), "maybe sometimes")[0] == "invalid"
    assert parse_answer(F("x", "bool"), "I don't know")[0] == "unknown"


def test_fahrenheit_converted():
    assert parse_answer(F("temperature_c", "float"), "101.3 F") == ("ok", 38.5)
    assert parse_answer(F("temperature_c", "float"), "38.5") == ("ok", 38.5)


def test_age_and_duration_units():
    assert parse_answer(F("age_years", "float"), "6 months") == ("ok", 0.5)
    assert parse_answer(F("duration_days", "float"), "2 weeks") == ("ok", 14.0)
    assert parse_answer(F("duration_days", "float"), "three days") == ("ok", 3.0)
    assert parse_answer(F("duration_days", "float"), "12 hours") == ("ok", 0.5)


def test_enum_and_list():
    sev = F("severity", "enum", options=["mild", "moderate", "severe"])
    assert parse_answer(sev, "pretty severe") == ("ok", "severe")
    assert parse_answer(sev, "2") == ("ok", "moderate")
    chron = F("chronic_conditions", "list")
    assert parse_answer(chron, "none") == ("ok", [])
    assert parse_answer(chron, "I have sugar and bp") == ("ok", ["diabetes", "hypertension"])


# ---------------- vocabulary
def test_vocab_lookup():
    from engine.vocab import extract_symptoms, normalise
    assert normalise("pet dard") == "abdominal_pain"
    assert normalise("bukhar") == "fever"
    found = extract_symptoms("mujhe bukhar hai aur sar dard, also vomiting")
    assert {"fever", "headache", "vomiting"} <= set(found)


# ---------------- gold vignettes (development set)
def test_vignette_file_is_balanced_and_valid():
    v = load_vignettes()
    assert len(v) >= 30
    assert {x["gold"] for x in v} == {"low", "moderate", "urgent", "emergency"}
    assert len({x["id"] for x in v}) == len(v)
    for x in v:
        get_protocol(x["complaint"])


def test_safety_targets_on_vignettes():
    m = metrics(run_all())
    assert m["emergency_recall"] == 1.0
    assert m["under_triage"] <= 0.05
    assert m["exact"] >= 0.80
    assert m["within_one"] >= 0.95
