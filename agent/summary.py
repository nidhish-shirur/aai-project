"""Clinician summary (Phase 7): a one-page, structured hand-over for a doctor.

Purpose: turn a finished session (state + report + timeline) into the thing a clinician actually
reads. It is built with plain Python from data the system already validated; NO LLM writes any of it,
so it cannot add a symptom, a diagnosis or a recommendation that is not in the record.

Sections: who, main problem, what the patient reports (present / denied / not answered),
measurements, urgency and WHY (rules, red flags, risk modifiers, AI review), why a human should
look, the symptom timeline, and the patient's own words.
"""
from __future__ import annotations

from typing import Any, Optional

from agent.state import TOP_LEVEL_KEYS, PatientState
from engine import pathways
from engine.protocols import get_protocol
from memory.timeline import SYMPTOM_FACTS

# facts shown as measurements / context instead of present-absent symptoms
NON_SYMPTOM = {"age_years", "sex", "pregnant", "chronic_conditions", "duration_days", "severity",
               "temperature_c"}
MAX_WORDS_SHOWN = 20
MAX_WORD_CHARS = 300


def label(key: str) -> str:
    """Human label for a fact key ('stiff_neck' -> 'Stiff neck')."""
    return SYMPTOM_FACTS.get(key) or key.replace("_", " ").capitalize()


_MD_SPECIAL = "\\`*_[]()<>!#|~"


def md_escape(text: str) -> str:
    """Make patient-typed (or LLM-written) text safe to show as Markdown: one line, no formatting, no links or
    images (an image link would make the clinician's browser contact a stranger's server), no raw HTML."""
    flat = " ".join(str(text).split())
    return "".join("\\" + ch if ch in _MD_SPECIAL else ch for ch in flat)


def _num(value: Any) -> str:
    try:
        f = float(value)
        return str(int(f)) if f == int(f) else f"{f:.1f}"
    except (TypeError, ValueError):
        return str(value)


def build_clinician_summary(state: PatientState, report: Optional[dict], timeline: Optional[list[str]] = None,
                            continuity: str = "") -> dict:
    """Structured summary. `report` is the saved triage report (None for an unfinished session)."""
    report = report or {}
    facts = state.all_facts()
    protocol_id = report.get("protocol_id") or state.complaint or "other"
    protocol = get_protocol(protocol_id)

    present, denied = [], []
    for key, value in facts.items():
        if key in NON_SYMPTOM or key in TOP_LEVEL_KEYS or key == state.complaint or not isinstance(value, bool):
            continue
        (present if value else denied).append(label(key))
    present = sorted(present)
    denied = sorted(denied)

    measurements = []
    if facts.get("temperature_c") is not None:
        measurements.append(f"Highest temperature: {_num(facts['temperature_c'])} °C")
    if facts.get("duration_days") is not None:
        measurements.append(f"Duration: {_num(facts['duration_days'])} day(s)")
    if facts.get("severity"):
        measurements.append(f"Patient-rated severity: {facts['severity']}")

    not_answered = sorted({label(k) for k in state.unknown_fields} |
                          {label(k) for k in report.get("missing_required", [])})

    urgency = report.get("urgency")
    info = pathways.describe(urgency) if urgency in ("low", "moderate", "urgent", "emergency") else None
    review = report.get("llm_review") or {}
    words = [w[:MAX_WORD_CHARS] for w in state.user_messages[:MAX_WORDS_SHOWN]]
    hidden_words = max(0, len(state.user_messages) - MAX_WORDS_SHOWN)

    return {
        "patient_code": state.patient_id,
        "session_id": state.session_id,
        "finished": bool(report),
        "age": None if state.age_years is None else _num(state.age_years),
        "sex": state.sex,
        "pregnant": state.pregnant,
        "chronic_conditions": state.chronic_conditions,
        "main_problem": protocol.title,
        "measurements": measurements,
        "present": present,
        "denied": denied,
        "not_answered": not_answered,
        "urgency": urgency,
        "urgency_label": info["label"] if info else None,
        "suggested_care": info["pathway"] if info else None,
        "timeframe": info["timeframe"] if info else None,
        "rule_urgency": report.get("rule_urgency"),
        "confidence": report.get("confidence"),
        "reasons": list(report.get("reasons", [])),
        "red_flags": [h.get("label") if isinstance(h, dict) else str(h) for h in report.get("red_flags", [])],
        "modifiers": list(report.get("modifiers_applied", [])),
        "matched_rules": list(report.get("matched_rules", [])),
        "needs_review": bool(report.get("needs_clinician_review")),
        "review_reasons": list(report.get("clinician_reasons", [])),
        "ai_review": {"raised": bool(review.get("raised")), "concern": review.get("concern") or "",
                      "capped": bool(review.get("capped"))} if report.get("llm_review") else None,
        "timeline": list(timeline or []),
        "continuity": continuity,
        "questions_asked": report.get("questions_asked", state.questions_asked),
        "llm_used": report.get("llm_used"),
        "patient_words": words,
        "patient_words_hidden": hidden_words,
    }


def to_markdown(s: dict, base_level: int = 1) -> str:
    """Printable / downloadable text version of the summary. `base_level` shifts the heading sizes
    (the web page uses 3 so the summary does not shout)."""
    def bullets(items: list[str], empty: str = "none recorded") -> list[str]:
        return [f"- {i}" for i in items] if items else [f"- ({empty})"]

    who = [f"Patient code: {s['patient_code']}", f"Session: {s['session_id']}"]
    if s["age"] is not None:
        who.append(f"Age: {s['age']}")
    if s["sex"]:
        who.append(f"Sex: {s['sex']}")
    if s["pregnant"]:
        who.append("Possible pregnancy: yes")
    if s["chronic_conditions"]:
        who.append("Long-term conditions: " + ", ".join(c.replace("_", " ") for c in s["chronic_conditions"]))
    elif s["chronic_conditions"] == []:
        who.append("Long-term conditions: none reported")

    lines = ["# Triage summary for clinician", "",
             "_Decision-support output of an educational prototype. It is NOT a diagnosis._", ""]
    if s["urgency"]:
        lines += [f"## Urgency: {s['urgency_label']}", f"- Suggested care: {s['suggested_care']}",
                  f"- Timeframe: {s['timeframe']}", f"- Confidence: {s['confidence']}"]
        if s["rule_urgency"] and s["rule_urgency"] != s["urgency"]:
            lines.append(f"- Rule engine said **{s['rule_urgency']}**; the AI safety review raised it to **{s['urgency']}**")
    else:
        lines += ["## Urgency: not assessed (session unfinished)"]
    if s["needs_review"]:
        lines += ["", "**Human clinician review recommended:**"] + bullets(s["review_reasons"])
    lines += ["", "## Patient", *[f"- {w}" for w in who], "", f"## Main problem: {s['main_problem']}",
              *bullets(s["measurements"], "no measurements")]
    lines += ["", "### Reported (yes)", *bullets(s["present"]), "", "### Denied (no)", *bullets(s["denied"]),
              "", "### Not answered / unknown", *bullets(s["not_answered"], "all asked questions answered")]
    lines += ["", "## Why this urgency", *bullets(s["reasons"], "no reason recorded")]
    if s["red_flags"]:
        lines += ["", "### Emergency warning signs detected", *bullets(s["red_flags"])]
    if s["modifiers"]:
        lines += ["", "### Risk modifiers applied", *bullets(s["modifiers"])]
    if s["ai_review"] and s["ai_review"]["raised"]:
        lines += ["", "### AI safety review (second opinion)", f"- {md_escape(s['ai_review']['concern'])}"]
    lines += ["", "## Symptom timeline", *bullets(s["timeline"], "no earlier data")]
    if s["continuity"]:
        lines += ["", md_escape(s["continuity"])]
    lines += ["", "## Patient's own words (as typed, in order)", *[f'- "{md_escape(w)}"' for w in s["patient_words"]]]
    if s.get("patient_words_hidden"):
        lines.append(f"- ({s['patient_words_hidden']} later message(s) not shown)")
    lines += ["", "---", "", pathways.disclaimer(), "",
              f"Questions asked: {s['questions_asked']}. AI model used: {'yes' if s['llm_used'] else 'no'}."]
    text = "\n".join(lines)
    if base_level > 1:
        shift = "#" * (base_level - 1)
        text = "\n".join(shift + l if l.startswith("#") else l for l in text.splitlines())
    return text


def format_tool_log(tool_log: list[dict]) -> list[str]:
    """One readable line per turn for the 'how did the agent decide?' panel."""
    turns: dict[int, list[str]] = {}
    for t in tool_log:
        name, n = t["tool"], t.get("turn", 0)
        if name == "llm_calls":
            continue
        if name == "red_flag_gate":
            text = f"emergency check ({t.get('stage', '').replace('_', ' ')}): " + \
                   ("TRIGGERED" if t.get("triggered") else "clear")
        elif name == "extract_facts":
            how = "read by Python" if t.get("fast_path") else ("fallback" if t.get("used_fallback") else "LLM")
            text = f"extracted facts ({how}): " + (", ".join(t.get("applied") or []) or "none")
        elif name == "plan_next":
            text = f"planned next step [{t.get('source')}]: {t.get('action')}" + \
                   (f" -> {t['field']}" if t.get("field") else "")
            if t.get("notes"):
                text += " (" + "; ".join(t["notes"]) + ")"
        elif name == "ask_question":
            text = f"asked question #{t.get('number')}: {t.get('field')}"
        elif name == "refusal":
            text = "refused a diagnosis/medicine request: " + ", ".join(t.get("kinds", []))
        elif name == "safety_review":
            text = "AI safety review: " + ("raised urgency" if t.get("raised") else "no change")
        elif name == "compute_urgency":
            text = f"rule engine decided: {t.get('urgency')}" + \
                   (f" (rules: {', '.join(t['matched_rules'])})" if t.get("matched_rules") else "")
        elif name == "start":
            text = "session started"
        else:
            text = name.replace("_", " ")
        turns.setdefault(n, []).append(text)
    return [f"Turn {n}: " + " | ".join(parts) if n else "Start: " + " | ".join(parts)
            for n, parts in sorted(turns.items())]
