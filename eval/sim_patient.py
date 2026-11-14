"""Simulated patient and case runner for the held-out evaluation.

The patient answers ONLY from the case's `facts` (the ground truth written by the case author):
* a yes/no question -> 'yes' if the fact is True, otherwise 'no' (nothing else to report),
* a number / choice question -> the stated value, or "I don't know" if the case does not say,
* the conditions list -> the stated conditions or 'none'.
The first message is the case's free text. The agent under test sees exactly what a real user would type.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from agent.llm_client import LLM
from agent.orchestrator import TriageAgent
from engine.protocols import get_protocol, ordered_fields, get_extra_facts
from memory.db import Memory

MAX_TURNS = 40
CLOCK = lambda: datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)  # noqa: E731  (fixed time: reproducible runs)


def _field(agent: TriageAgent, key: str):
    complaint = agent.state.complaint
    pools = list(ordered_fields(get_protocol(complaint))) if complaint else []
    pools += get_extra_facts()
    return next((f for f in pools if f.key == key), None)


def answer_text(field_type: str, key: str, facts: dict[str, Any]) -> str:
    """What the simulated patient types for one question."""
    value = facts.get(key)
    if field_type == "bool":
        return "yes" if value is True else "no"
    if field_type == "list":
        return ", ".join(v.replace("_", " ") for v in value) if value else "none"
    if value is None:
        return "I don't know"
    if field_type in ("float", "int"):
        return f"{value:g}"
    return str(value)


def run_case(case: dict, llm: Optional[LLM] = None, safety_review: bool = False) -> dict:
    """Run one held-out case through the real TriageAgent. Returns the final result and the transcript."""
    mem = Memory(":memory:")
    agent = TriageAgent(mem, llm, clock=CLOCK, safety_review=safety_review)
    agent.start()
    turn = agent.handle(case["text"])
    turns = 1
    while not agent.done and turns < MAX_TURNS:
        kind, key = agent.pending_kind, agent.pending_key
        if kind == "complaint":
            reply = case["text"]
        elif kind == "field" and key:
            f = _field(agent, key)
            reply = answer_text(f.type if f else "bool", key, case["facts"])
        else:
            reply = "I don't know"
        turn = agent.handle(reply)
        turns += 1
    out = {"id": case["id"], "done": agent.done, "turns": turns, "state": agent.state,
           "transcript": mem.get_messages(agent.state.session_id), "tool_log": agent.tool_log,
           "latency": agent.latency_summary()}
    if agent.done:
        out.update(pred=turn.urgency, kind=turn.kind, result=turn.result, message=turn.message)
    else:
        out.update(pred=None, kind="unfinished", result=None, message="")
    mem.close()
    return out
