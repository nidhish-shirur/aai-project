from __future__ import annotations

from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.extractor import extract_into_state
from agent.safety_review import review_and_apply
from agent.state import PatientState
from engine.protocols import get_protocol, missing_fields
from engine.red_flags import check_red_flags, check_text
from engine.triage_rules import compute_urgency


class TriageGraphState(TypedDict, total=False):
    patient: PatientState
    message: str
    pending_field: Optional[str]
    next_question: str
    urgency: Optional[str]
    emergency: bool
    done: bool
    clinician_review: bool
    report: dict[str, Any]
    events: list[str]


def red_flag_agent(
    state: TriageGraphState,
) -> dict[str, Any]:
    patient = state["patient"]
    message = state.get("message", "")

    text_hits = check_text(
        patient.text_blob() + " . " + message
    )

    if text_hits:
        return {
            "emergency": True,
            "done": True,
            "urgency": "emergency",
            "events": [
                *state.get("events", []),
                "red_flag_agent: emergency detected",
            ],
        }

    protocol = (
        get_protocol(patient.complaint)
        if patient.complaint
        else None
    )

    fact_result = check_red_flags(patient, protocol)

    if fact_result.triggered:
        return {
            "emergency": True,
            "done": True,
            "urgency": "emergency",
            "events": [
                *state.get("events", []),
                "red_flag_agent: fact-based emergency detected",
            ],
        }

    return {
        "emergency": False,
        "events": [
            *state.get("events", []),
            "red_flag_agent: no emergency detected",
        ],
    }


def route_after_red_flags(
    state: TriageGraphState,
) -> str:
    if state.get("emergency"):
        return "finish"

    return "extract"


def extraction_agent(
    state: TriageGraphState,
) -> dict[str, Any]:
    patient = state["patient"]
    message = state.get("message", "")

    result = extract_into_state(
        patient,
        message,
        llm=None,
        pending_key=state.get("pending_field"),
    )

    return {
        "events": [
            *state.get("events", []),
            "extraction_agent: facts extracted",
        ],
    }


def questioning_agent(
    state: TriageGraphState,
) -> dict[str, Any]:
    patient = state["patient"]

    if not patient.complaint:
        return {
            "next_question": (
                "What is the main problem today?"
            ),
            "events": [
                *state.get("events", []),
                "questioning_agent: complaint required",
            ],
        }

    protocol = get_protocol(patient.complaint)
    fields = missing_fields(patient, protocol)

    if not fields:
        return {
            "next_question": "",
            "events": [
                *state.get("events", []),
                "questioning_agent: required information complete",
            ],
        }

    field = fields[0]

    return {
        "pending_field": field.key,
        "next_question": field.question,
        "events": [
            *state.get("events", []),
            f"questioning_agent: asking about {field.key}",
        ],
    }


def route_after_question(
    state: TriageGraphState,
) -> str:
    if state.get("next_question"):
        return "ask"

    return "triage"


def triage_agent(
    state: TriageGraphState,
) -> dict[str, Any]:
    patient = state["patient"]
    result = compute_urgency(patient)

    return {
        "urgency": result.urgency,
        "report": result.to_dict(),
        "events": [
            *state.get("events", []),
            f"triage_agent: urgency={result.urgency}",
        ],
    }


def safety_agent(
    state: TriageGraphState,
) -> dict[str, Any]:
    patient = state["patient"]
    result = compute_urgency(patient)

    review = review_and_apply(
        llm=None,
        state=patient,
        result=result,
    )

    report = result.to_dict()
    report["safety_review"] = review.to_dict()

    return {
        "urgency": result.urgency,
        "report": report,
        "clinician_review": result.needs_clinician_review,
        "events": [
            *state.get("events", []),
            "safety_agent: safety review completed",
        ],
    }


def finish_agent(
    state: TriageGraphState,
) -> dict[str, Any]:
    return {
        "done": True,
        "events": [
            *state.get("events", []),
            "workflow: finished",
        ],
    }


def build_triage_graph():
    graph = StateGraph(TriageGraphState)

    graph.add_node("red_flags", red_flag_agent)
    graph.add_node("extract", extraction_agent)
    graph.add_node("question", questioning_agent)
    graph.add_node("triage", triage_agent)
    graph.add_node("safety", safety_agent)
    graph.add_node("finish", finish_agent)

    graph.add_edge(START, "red_flags")

    graph.add_conditional_edges(
        "red_flags",
        route_after_red_flags,
        {
            "finish": "finish",
            "extract": "extract",
        },
    )

    graph.add_edge("extract", "question")

    graph.add_conditional_edges(
        "question",
        route_after_question,
        {
            "ask": END,
            "triage": "triage",
        },
    )

    graph.add_edge("triage", "safety")
    graph.add_edge("safety", "finish")
    graph.add_edge("finish", END)

    return graph.compile()