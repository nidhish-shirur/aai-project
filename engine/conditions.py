"""Tiny, safe condition language used inside the JSON knowledge files.

Purpose: let clinicians read and edit rules as data (no Python needed, no eval).

Grammar
-------
"fever"                          -> fact `fever` is True
["temperature_c", ">=", 40]      -> comparison  (ops: == != < <= > >= in not_in contains)
["duration_days", "unknown"]     -> fact not known yet   (also: "known")
{"all": [c1, c2]}                -> every condition true
{"any": [c1, c2]}                -> at least one true
{"not": c}                       -> negation
Rule of safety: a comparison against an UNKNOWN fact is False (never guessed).
"""
from __future__ import annotations

from typing import Any

NUMERIC_OPS = {"<", "<=", ">", ">="}
VALUE_OPS = {"==", "!="} | NUMERIC_OPS | {"in", "not_in", "contains"}
NO_VALUE_OPS = {"unknown", "known"}


def _normalise(cond: Any) -> Any:
    """Turn shorthand into the dict/list forms handled below."""
    if isinstance(cond, str):
        return [cond, "==", True]
    return cond


def validate_condition(cond: Any) -> None:
    """Raise ValueError if `cond` is not valid grammar (used by check_knowledge)."""
    cond = _normalise(cond)
    if isinstance(cond, dict):
        if len(cond) != 1:
            raise ValueError(f"condition dict must have exactly one key: {cond}")
        (key, val), = cond.items()
        if key in ("all", "any"):
            if not isinstance(val, list) or not val:
                raise ValueError(f"'{key}' needs a non-empty list: {cond}")
            for sub in val:
                validate_condition(sub)
        elif key == "not":
            validate_condition(val)
        else:
            raise ValueError(f"unknown condition key '{key}'")
    elif isinstance(cond, list):
        if len(cond) == 2 and cond[1] in NO_VALUE_OPS:
            pass
        elif len(cond) == 3 and cond[1] in VALUE_OPS:
            if cond[1] in NUMERIC_OPS and not isinstance(cond[2], (int, float)):
                raise ValueError(f"numeric op needs a number: {cond}")
            if cond[1] in ("in", "not_in") and not isinstance(cond[2], list):
                raise ValueError(f"'in' needs a list: {cond}")
        else:
            raise ValueError(f"bad comparison {cond}")
        if not isinstance(cond[0], str):
            raise ValueError(f"fact name must be text: {cond}")
    else:
        raise ValueError(f"cannot understand condition: {cond!r}")


def referenced_facts(cond: Any) -> set[str]:
    """All fact names a condition looks at."""
    cond = _normalise(cond)
    if isinstance(cond, dict):
        (key, val), = cond.items()
        if key in ("all", "any"):
            return set().union(*(referenced_facts(c) for c in val))
        return referenced_facts(val)
    return {cond[0]}


def _compare(value: Any, op: str, target: Any = None) -> bool:
    if op == "unknown":
        return value is None
    if op == "known":
        return value is not None
    if value is None:
        return False  # never guess
    try:
        if op == "==":
            return value == target
        if op == "!=":
            return value != target
        if op == "<":
            return float(value) < target
        if op == "<=":
            return float(value) <= target
        if op == ">":
            return float(value) > target
        if op == ">=":
            return float(value) >= target
        if op == "in":
            return value in target
        if op == "not_in":
            return value not in target
        if op == "contains":
            return isinstance(value, (list, tuple, set)) and target in value
    except (TypeError, ValueError):
        return False
    raise ValueError(f"unknown operator {op}")


def evaluate(cond: Any, facts: dict[str, Any]) -> bool:
    """Evaluate a condition against a flat facts dictionary."""
    cond = _normalise(cond)
    if isinstance(cond, dict):
        (key, val), = cond.items()
        if key == "all":
            return all(evaluate(c, facts) for c in val)
        if key == "any":
            return any(evaluate(c, facts) for c in val)
        if key == "not":
            return not evaluate(val, facts)
        raise ValueError(f"unknown condition key '{key}'")
    fact, op = cond[0], cond[1]
    target = cond[2] if len(cond) > 2 else None
    return _compare(facts.get(fact), op, target)
