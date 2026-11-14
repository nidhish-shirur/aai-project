import pytest

from engine.conditions import evaluate, referenced_facts, validate_condition

FACTS = {"fever": True, "cough": False, "temperature_c": 39.5, "age_years": 4,
         "chronic_conditions": ["diabetes"], "severity": "moderate"}


def test_shorthand_true_fact():
    assert evaluate("fever", FACTS)
    assert not evaluate("cough", FACTS)


def test_comparisons():
    assert evaluate(["temperature_c", ">=", 39], FACTS)
    assert not evaluate(["temperature_c", ">", 40], FACTS)
    assert evaluate(["severity", "in", ["moderate", "severe"]], FACTS)
    assert evaluate(["chronic_conditions", "contains", "diabetes"], FACTS)
    assert evaluate(["severity", "!=", "mild"], FACTS)


def test_all_any_not():
    assert evaluate({"all": ["fever", ["age_years", "<", 5]]}, FACTS)
    assert evaluate({"any": ["cough", "fever"]}, FACTS)
    assert evaluate({"not": "cough"}, FACTS)


def test_unknown_is_never_guessed():
    # an unknown fact makes every comparison False, even '!=' and 'not'-free tests
    assert not evaluate(["duration_days", ">=", 0], FACTS)
    assert not evaluate(["duration_days", "!=", 5], FACTS)
    assert not evaluate("stiff_neck", FACTS)
    assert evaluate(["duration_days", "unknown"], FACTS)
    assert evaluate(["fever", "known"], FACTS)


def test_referenced_facts():
    cond = {"all": ["fever", {"any": [["age_years", "<", 1], "cough"]}]}
    assert referenced_facts(cond) == {"fever", "age_years", "cough"}


@pytest.mark.parametrize("bad", [
    ["x", "~~", 3], {"all": []}, {"foo": ["x"]}, 42, ["x", ">", "high"], {"all": ["a"], "any": ["b"]},
])
def test_bad_conditions_rejected(bad):
    with pytest.raises(ValueError):
        validate_condition(bad)
