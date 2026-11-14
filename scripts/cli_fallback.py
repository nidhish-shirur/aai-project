"""Deterministic triage chat in the terminal - works with NO LLM and NO internet.

Purpose: (1) this is the project's fallback mode (feature E14): if the LLM API is
down, the same safety core still runs; (2) it lets you try the engine by hand;
(3) it saves each session to SQLite, so a returning patient code is remembered.

Run from the project root:   python scripts/cli_fallback.py
Answer yes / no / a number / "dont know". Type "quit" to stop.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from agent.state import PatientState  # noqa: E402
from engine import pathways  # noqa: E402
from engine.answer_parser import parse_answer  # noqa: E402
from engine.protocols import get_protocol, missing_fields, select_protocol_from_text  # noqa: E402
from engine.red_flags import check_red_flags  # noqa: E402
from engine.triage_rules import compute_urgency  # noqa: E402
from memory.db import Memory  # noqa: E402

BAR = "=" * 64


def persist(mem: Memory, state: PatientState) -> dict:
    """Save state, symptoms and the final report. Returns the report dict."""
    result = compute_urgency(state).to_dict()
    mem.save_state(state)
    mem.record_symptom_events(state)
    mem.save_report(state.session_id, result)
    return result


def show_emergency(state: PatientState, protocol) -> None:
    gate = check_red_flags(state, protocol)
    print("\n" + BAR)
    print(pathways.emergency_message(gate.category))
    print("\nWhy: " + "; ".join(gate.labels))
    print(BAR)
    print(pathways.disclaimer())


def show_result(state: PatientState) -> None:
    result = compute_urgency(state)
    protocol = get_protocol(result.protocol_id)
    if result.urgency == "emergency":
        show_emergency(state, protocol)
        return
    info = pathways.describe(result.urgency)
    print("\n" + BAR)
    print(f"URGENCY:        {info['label']}")
    print(f"SUGGESTED CARE: {info['pathway']}")
    print(f"WHEN:           {info['timeframe']}")
    print(f"CONFIDENCE:     {result.confidence}")
    print(BAR)
    print(info["advice"])
    print("\nWhy this level:")
    for reason in result.reasons:
        print(" -", reason)
    print("\nSeek help sooner if you notice:")
    for sign in protocol.warning_signs:
        print(" -", sign)
    if result.needs_clinician_review:
        print("\n" + pathways.clinician_review_note())
        for reason in result.clinician_reasons:
            print(" -", reason)
    print("\n" + pathways.disclaimer())


def finish(mem: Memory, state: PatientState, protocol, emergency: bool) -> None:
    if emergency:
        show_emergency(state, protocol)
    else:
        show_result(state)
    persist(mem, state)
    timeline = mem.timeline_text(state.patient_id)
    if timeline:
        print("\nSymptom timeline (for your doctor):")
        for line in timeline:
            print("  " + line)
    print(f"\nYour anonymous patient code: {state.patient_id}")
    print("Enter it next time to let the system remember this visit. No name or phone is stored.")


def main() -> None:
    print(BAR)
    print("TriageGuide - fallback mode (no LLM). Educational prototype.")
    print(pathways.disclaimer())
    print(f"In an emergency call {config.EMERGENCY_NUMBER} now.")
    print(BAR)

    mem = Memory()
    code = input("\nPatient code from an earlier visit (press Enter if this is your first time) > ").strip().upper()
    try:
        state = mem.start_session(code or None)
    except KeyError:
        print("That code was not found, so I will start a new record.")
        state = mem.start_session()
    note = mem.continuity_note(state.patient_id, exclude_session_id=state.session_id)
    if note:
        print("\n" + note)

    text = input("\nWhat is the problem? > ").strip()
    if text.lower() in ("quit", "exit", ""):
        return
    state.add_user_message(text)
    mem.log_message(state.session_id, "user", text)
    state.complaint = select_protocol_from_text(text)
    protocol = get_protocol(state.complaint)
    print(f"(Protocol selected: {protocol.title})")

    if check_red_flags(state, protocol).triggered:
        finish(mem, state, protocol, emergency=True)
        return

    invalid_in_a_row = 0
    while state.questions_asked < config.MAX_QUESTIONS:
        todo = missing_fields(state, protocol) or missing_fields(state, protocol, include_optional=True)
        if not todo:
            break
        field = todo[0]
        hint = f" ({'/'.join(field.options)})" if field.options else ""
        mem.log_message(state.session_id, "agent", field.question)
        answer = input(f"\n{field.question}{hint}\n> ").strip()
        if answer.lower() in ("quit", "exit"):
            mem.save_state(state)
            return
        status, value = parse_answer(field, answer)
        if status == "invalid":
            invalid_in_a_row += 1
            if invalid_in_a_row < 2:
                print("Sorry, I did not understand. Please try again.")
                continue
            status = "unknown"  # give up on this question, never loop forever
        invalid_in_a_row = 0
        state.questions_asked += 1
        state.add_user_message(answer)
        mem.log_message(state.session_id, "user", answer)
        if status == "unknown":
            state.mark_unknown(field.key)
        else:
            state.update({field.key: value})
        if check_red_flags(state, protocol).triggered:
            finish(mem, state, protocol, emergency=True)
            return

    finish(mem, state, protocol, emergency=False)


if __name__ == "__main__":
    main()
