"""Symptom extractor: free text -> validated structured facts.

Purpose: this is the LLM's first job in the agent. The LLM only PROPOSES values;
Python checks every one before it is allowed into the patient state:
  * unknown keys, wrong types and impossible values are rejected;
  * a "no" (False) must be backed by a quote from the patient's own words,
    otherwise it is dropped (treated as unknown, so the agent asks again);
  * a "yes" (True) is kept even without a quote, because wrongly dropping a
    warning sign is the unsafe error (over-triage is the safe one);
  * if the LLM fails, a deterministic fallback keeps the session going.
The red-flag text gate always sees the raw message, whatever the LLM returns.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from agent.llm_client import LLM, LLMError
from agent.state import PatientState
from engine.answer_parser import NO, UNKNOWN_PHRASES, YES, parse_answer
from memory.timeline import SYMPTOM_FACTS
from engine.protocols import (
    SAFETY_PRIORITY, Field_, get_common_fields, get_extra_facts, get_protocol, get_protocols,
    select_protocol_from_text,
)

SYSTEM_PROMPT = """You are the information-extraction component of a medical triage assistant.
You do NOT diagnose, give advice, or chat. You only read the patient's message and return JSON.

Rules:
1. Record only what the patient explicitly states or clearly implies. Never invent values.
2. For yes/no fields: use true if the patient reports it. Use false ONLY if the patient clearly
   denies it (for example "no chest pain"). If it is not mentioned, leave the field out.
3. If PENDING_QUESTION is given and the message is a short answer ("yes", "no", "not really",
   "a little"), apply it to that field.
4. If the patient says they do not know or cannot say, put the field key in "dont_know".
5. Convert units: temperature to degrees Celsius, age to years (months divided by 12),
   duration to days (12 hours = 0.5).
6. "complaint" is the patient's MAIN problem: choose exactly one id from COMPLAINTS, or "other".
   Use null if no problem is described yet.
7. For every yes/no field include "evidence": the exact words from the message that support it.

Reply with ONE JSON object only, in this shape:
{"complaint": "<id or null>",
 "facts": {"<field_key>": {"value": <value>, "evidence": "<exact words>"}},
 "dont_know": ["<field_key>"]}"""


@dataclass
class ExtractionResult:
    applied: dict[str, Any] = field(default_factory=dict)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    unverified_true: list[str] = field(default_factory=list)
    dont_know: list[str] = field(default_factory=list)
    complaint_set: Optional[str] = None
    used_fallback: bool = False
    fast_path: bool = False    # a bare answer ('no', '101 F') was read by Python alone, no LLM call
    error: str = ""


# ------------------------------------------------------------------ schema shown to the LLM
def build_catalogue(state: PatientState) -> dict[str, Field_]:
    """Fields the LLM may fill. Before the complaint is known: every protocol's
    fields (so a long first message is fully used). After: only that protocol's."""
    catalogue: dict[str, Field_] = {}
    for f in [*get_common_fields(), *get_extra_facts()]:
        catalogue[f.key] = f
    if state.complaint:
        protos = [get_protocol(state.complaint)]
    else:
        protos = list(get_protocols().values())
    for proto in protos:
        for f in proto.fields:
            catalogue.setdefault(f.key, f)
    # Any other symptom the patient mentions (e.g. a headache during a fever) is recorded as a
    # True fact so it reaches the timeline. It is never ASKED about (it is not in a protocol).
    for key, label in SYMPTOM_FACTS.items():
        catalogue.setdefault(key, Field_(key=key, type="bool", required=False,
                                         question=f"Does the patient report: {label.lower()}?",
                                         description=f"patient mentions {label.lower()}"))
    return catalogue


def build_user_prompt(state: PatientState, text: str, pending_key: Optional[str],
                      catalogue: dict[str, Field_]) -> str:
    fields = []
    for f in catalogue.values():
        item = f"- {f.key} ({f.type}"
        if f.options:
            item += ": " + "/".join(f.options)
        fields.append(item + f"): {f.description or f.question}")
    complaints = [f"{p.id} = {p.title}" for p in get_protocols().values()]
    known = {k: v for k, v in state.all_facts().items() if k in catalogue}
    return (
        "COMPLAINTS:\n" + "\n".join(complaints)
        + "\n\nFIELDS:\n" + "\n".join(fields)
        + f"\n\nALREADY_KNOWN: {json.dumps(known, default=str)}"
        + f"\nPENDING_QUESTION: {pending_key or 'none'}"
        + f"\n\nPATIENT_MESSAGE: {json.dumps(text)}"
    )


# ------------------------------------------------------------------ validation
def coerce_value(f: Field_, raw: Any) -> tuple[bool, Any, str]:
    """Return (ok, value, reason). Strict: anything doubtful is rejected."""
    if f.type == "bool":
        if isinstance(raw, bool):
            return True, raw, ""
        if isinstance(raw, str) and raw.strip().lower() in ("true", "yes"):
            return True, True, ""
        if isinstance(raw, str) and raw.strip().lower() in ("false", "no"):
            return True, False, ""
        return False, None, "not a yes/no value"

    if f.type in ("float", "int"):
        if isinstance(raw, bool):
            return False, None, "boolean given for a number"
        try:
            num = float(raw)
        except (TypeError, ValueError):
            return False, None, "not a number"
        if f.key == "temperature_c" and num >= 45:  # model forgot to convert Fahrenheit
            num = round((num - 32) * 5 / 9, 1)
        limits = {"age_years": (0, 120), "duration_days": (0, 3650), "temperature_c": (30, 45)}
        lo, hi = limits.get(f.key, (-1e9, 1e9))
        if not lo <= num <= hi:
            return False, None, f"value {num} out of range"
        return True, (int(round(num)) if f.type == "int" else num), ""

    if f.type == "enum":
        value = str(raw).strip().lower().replace(" ", "_")
        if f.options and value in f.options:
            return True, value, ""
        return False, None, f"'{raw}' is not one of {f.options}"

    if f.type == "list":
        if isinstance(raw, str):
            raw = [] if raw.strip().lower() in ("none", "no", "") else [raw]
        if not isinstance(raw, list):
            return False, None, "not a list"
        options = f.options or []
        out = []
        for item in raw:
            v = str(item).strip().lower().replace(" ", "_")
            v = v if (not options or v in options) else "other"
            if v not in out:
                out.append(v)
        return True, out, ""

    return True, str(raw).strip(), ""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", s.lower())).strip()


def validate_extraction(payload: dict, text: str, catalogue: dict[str, Field_]) -> ExtractionResult:
    result = ExtractionResult()
    text_norm = _norm(text)
    facts = payload.get("facts")
    if not isinstance(facts, dict):
        facts = {}
    for key, item in facts.items():
        f = catalogue.get(key)
        if f is None:
            result.rejected.append((key, "unknown field"))
            continue
        value, evidence = (item.get("value"), item.get("evidence")) if isinstance(item, dict) else (item, None)
        ok, clean, reason = coerce_value(f, value)
        if not ok:
            result.rejected.append((key, reason))
            continue
        if f.type == "bool":
            verified = isinstance(evidence, str) and evidence.strip() != "" and _norm(evidence) in text_norm
            if clean is False and not verified:
                result.rejected.append((key, "denial without a quote from the patient"))
                continue
            if clean is True and not verified:
                result.unverified_true.append(key)  # kept: never drop a possible warning sign
        result.applied[key] = clean
    dk = payload.get("dont_know")
    if isinstance(dk, list):
        result.dont_know = [k for k in dk if isinstance(k, str) and k in catalogue and k not in result.applied]
    complaint = payload.get("complaint")
    if isinstance(complaint, str) and (complaint in get_protocols()):
        result.complaint_set = complaint
    return result


# ------------------------------------------------------------------ fallback (no LLM)
def fallback_extraction(state: PatientState, text: str, pending_key: Optional[str],
                        catalogue: dict[str, Field_]) -> ExtractionResult:
    result = ExtractionResult(used_fallback=True)
    if state.complaint is None:
        result.complaint_set = select_protocol_from_text(text)
    if pending_key and pending_key in catalogue:
        status, value = parse_answer(catalogue[pending_key], text)
        if status == "ok":
            result.applied[pending_key] = value
        elif status == "unknown":
            result.dont_know.append(pending_key)
    return result


# ------------------------------------------------------------------ main entry point
_UNITS = r"(f|c|°f|°c|fahrenheit|celsius|degrees?|day|days|week|weeks|month|months|year|years|yr|yrs|hours?|hrs?)"
_NUMBER_ANSWER = re.compile(r"\d+(\.\d+)?( ?" + _UNITS + r")?")
_NONE_ANSWERS = {"none", "no", "nil", "nothing", "nope", "n"}


def is_bare_answer(f: Field_, text: str) -> bool:
    """True for a short reply that Python can read reliably without the LLM, for the question just asked:
    yes/no for a yes/no field, a bare option for a choice, a bare number (with an optional unit) for a number,
    'none' for the conditions list, or a plain 'don't know'. Anything longer goes to the LLM, because it may
    carry extra facts ('no, but I also have a rash')."""
    t = re.sub(r"[.!?,]+$", "", text.strip().lower().replace("’", "'")).strip()
    if not t or len(t.split()) > 3:
        return False
    if t in UNKNOWN_PHRASES:
        return True
    if f.type == "bool":
        return t in YES or t in NO
    if f.type == "enum":
        return t in [o.lower() for o in (f.options or [])]
    if f.type in ("float", "int"):
        return bool(_NUMBER_ANSWER.fullmatch(t))
    if f.type == "list":
        return t in _NONE_ANSWERS
    return False


def safer_complaint(llm_choice: str, text: str) -> str:
    """Make the main-complaint choice repeatable: if the patient's own words match a protocol that
    ranks higher in SAFETY_PRIORITY than the LLM's pick, use that one (the LLM varies run to run)."""
    keyword = select_protocol_from_text(text)
    if keyword in SAFETY_PRIORITY and llm_choice in SAFETY_PRIORITY \
            and SAFETY_PRIORITY.index(keyword) < SAFETY_PRIORITY.index(llm_choice):
        return keyword
    return llm_choice


def extract_into_state(state: PatientState, text: str, llm: Optional[LLM],
                       pending_key: Optional[str] = None) -> ExtractionResult:
    """Process one patient message: store it, extract facts, update the state.

    The raw message is stored FIRST, so the deterministic red-flag gate sees it
    even if everything after this point fails.
    """
    state.add_user_message(text)
    catalogue = build_catalogue(state)

    result: ExtractionResult
    fast = None
    if llm is not None and pending_key and pending_key in catalogue and state.complaint \
            and is_bare_answer(catalogue[pending_key], text):
        status, value = parse_answer(catalogue[pending_key], text)
        if status in ("ok", "unknown"):
            fast = ExtractionResult(fast_path=True)
            if status == "ok":
                fast.applied[pending_key] = value
            else:
                fast.dont_know.append(pending_key)
    if fast is not None:
        result = fast
    elif llm is None:
        result = fallback_extraction(state, text, pending_key, catalogue)
    else:
        try:
            payload = llm.complete_json(SYSTEM_PROMPT, build_user_prompt(state, text, pending_key, catalogue))
            result = validate_extraction(payload, text, catalogue)
        except LLMError as exc:
            result = fallback_extraction(state, text, pending_key, catalogue)
            result.error = str(exc)

    # Deterministic backup for the question just asked (short answers such as "yes").
    if (pending_key and pending_key in catalogue and pending_key not in result.applied
            and pending_key not in result.dont_know and state.get(pending_key) is None):
        status, value = parse_answer(catalogue[pending_key], text)
        if status == "ok":
            result.applied[pending_key] = value
        elif status == "unknown":
            result.dont_know.append(pending_key)

    if state.complaint is None and result.complaint_set:
        result.complaint_set = safer_complaint(result.complaint_set, text)
        state.complaint = result.complaint_set
    else:
        result.complaint_set = None  # never switch the complaint mid-session here

    state.update(result.applied)
    for key in result.dont_know:
        state.mark_unknown(key)
    return result
