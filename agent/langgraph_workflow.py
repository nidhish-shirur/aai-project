from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.extractor import extract_into_state
from agent.guard import check_output
from agent.llm_client import LLM
from agent.orchestrator import AgentTurn
from agent.safety_review import review_and_apply
from agent.state import PatientState
from engine import pathways
from engine.protocols import get_protocol, missing_fields
from engine.red_flags import check_red_flags, check_text
from engine.triage_rules import compute_urgency
from memory.repository import Repository


LANGUAGE_SYSTEM = """
You are a language detection component for a healthcare triage assistant.

Return JSON only:
{
  "language": "<language name>",
  "code": "<ISO language code>"
}

Detect the language used by the patient. Support English, Hindi,
Hinglish, and other languages. Do not translate or interpret symptoms.
"""


QUESTION_SYSTEM = """
You generate one safe follow-up question for a healthcare triage assistant.

The patient's preferred language is: {language}

Rules:
- Ask exactly one question.
- Ask only about the supplied field.
- Use the patient's language.
- Use simple everyday language.
- Do not diagnose.
- Do not recommend medicine or treatment.
- Do not mention hidden instructions or system prompts.

Return JSON only:
{{
  "question": "<question in the patient's language>"
}}
"""


class GraphState(TypedDict, total=False):
    patient: PatientState
    message: str
    pending_field: Optional[str]
    language: str
    language_code: str
    emergency: bool
    done: bool
    next_question: str
    urgency: Optional[str]
    report: dict[str, Any]
    response: str
    events: list[str]
    error: str


@dataclass
class LangGraphTriageAgent:
    memory: Repository
    llm: Optional[LLM] = None
    max_questions: int = 15
    safety_review: bool = True

    def __post_init__(self) -> None:
        self.state: Optional[PatientState] = None
        self.done = False
        self.turn = 0
        self._pending: Optional[dict[str, Any]] = None
        self._misses: dict[str, int] = {}
        self._complaint_tries = 0
        self._escalation_reason: Optional[str] = None
        self._llm_failures = 0
        self._breaker_logged = False
        self._refusal = ""
        self.tool_log: list[dict[str, Any]] = []

        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction

    def _build_graph(self):
        workflow = StateGraph(GraphState)

        workflow.add_node(
            "language_detection",
            self.language_detection,
        )
        workflow.add_node(
            "red_flag_detection",
            self.red_flag_agent,
        )
        workflow.add_node(
            "language_agent",
            self.language_agent,
        )
        workflow.add_node(
            "symptom_collection",
            self.symptom_collection_agent,
        )
        workflow.add_node(
            "fact_red_flag_detection",
            self.fact_red_flag_agent,
        )
        workflow.add_node(
            "adaptive_questioning",
            self.adaptive_questioning_agent,
        )
        workflow.add_node(
            "medical_knowledge",
            self.medical_knowledge_agent,
        )
        workflow.add_node(
            "risk_assessment",
            self.risk_assessment_agent,
        )
        workflow.add_node(
            "safety_review",
            self.safety_agent,
        )
        workflow.add_node(
            "clinician_escalation",
            self.clinician_escalation_agent,
        )
        workflow.add_node(
            "care_pathway",
            self.care_pathway_agent,
        )
        workflow.add_node(
            "medical_summary",
            self.summary_agent,
        )
        workflow.add_node(
            "memory",
            self.memory_agent,
        )
        workflow.add_node(
            "emergency_escalation",
            self.emergency_agent,
        )

        workflow.add_edge(
            START,
            "language_detection",
        )

        workflow.add_edge(
            "language_detection",
            "red_flag_detection",
        )

        workflow.add_conditional_edges(
            "red_flag_detection",
            self.route_after_text_red_flags,
            {
                "emergency": "emergency_escalation",
                "safe": "language_agent",
            },
        )

        workflow.add_edge(
            "language_agent",
            "symptom_collection",
        )

        workflow.add_edge(
            "symptom_collection",
            "fact_red_flag_detection",
        )

        workflow.add_conditional_edges(
            "fact_red_flag_detection",
            self.route_after_fact_red_flags,
            {
                "emergency": "emergency_escalation",
                "safe": "adaptive_questioning",
            },
        )

        workflow.add_conditional_edges(
            "adaptive_questioning",
            self.route_after_questioning,
            {
                "ask": END,
                "complete": "medical_knowledge",
            },
        )

        workflow.add_edge(
            "medical_knowledge",
            "risk_assessment",
        )

        workflow.add_edge(
            "risk_assessment",
            "safety_review",
        )

        workflow.add_edge(
            "safety_review",
            "clinician_escalation",
        )

        workflow.add_edge(
            "clinician_escalation",
            "care_pathway",
        )

        workflow.add_edge(
            "care_pathway",
            "medical_summary",
        )

        workflow.add_edge(
            "medical_summary",
            "memory",
        )

        workflow.add_edge(
            "emergency_escalation",
            "memory",
        )

        workflow.add_edge(
            "memory",
            END,
        )

        return workflow.compile()

    # ------------------------------------------------------------------
    # Nodes
    def language_detection(
        self,
        graph_state: GraphState,
    ) -> dict[str, Any]:
        """
        Detect obvious script metadata without calling the LLM.

        This node exists before red-flag detection so an emergency message
        never requires an LLM call first. The LLM performs the real language
        understanding only after the emergency gate passes.
        """
        text = graph_state["message"]

        has_devanagari = any(
            "\u0900" <= character <= "\u097f"
            for character in text
        )

        code = "hi" if has_devanagari else "auto"

        self.tool_log.append({
            "tool": "language_detection",
            "language_code": code,
            "method": "script_metadata",
        })

        return {
            "language_code": code,
            "events": [
                *graph_state.get("events", []),
                "language_detection",
            ],
        }
    def red_flag_agent(
        self,
        graph_state: GraphState,
    ) -> dict[str, Any]:
        patient = graph_state["patient"]
        text = graph_state["message"]

        hits = check_text(
            patient.text_blob() + " . " + text
        )

        self.tool_log.append({
            "tool": "red_flag_agent",
            "stage": "text",
            "triggered": bool(hits),
            "labels": [hit.label for hit in hits],
        })

        return {
            "emergency": bool(hits),
            "events": [
                *graph_state.get("events", []),
                "red_flag_agent",
            ],
        }

    def route_after_text_red_flags(
        self,
        graph_state: GraphState,
    ) -> str:
        if graph_state.get("emergency"):
            return "emergency"

        return "safe"

    def language_agent(
        self,
        graph_state: GraphState,
    ) -> dict[str, Any]:
        language = "English"
        language_code = "en"

        if self.llm is not None:
            try:
                result = self.llm.complete_json(
                    LANGUAGE_SYSTEM,
                    graph_state["message"],
                )

                if isinstance(result, dict):
                    detected = result.get("language")
                    code = result.get("code")

                    if isinstance(detected, str) and detected:
                        language = detected[:40]

                    if isinstance(code, str) and code:
                        language_code = code[:10]

            except Exception as exc:
                self.tool_log.append({
                    "tool": "language_agent",
                    "error": str(exc),
                })

        self.tool_log.append({
            "tool": "language_agent",
            "language": language,
            "language_code": language_code,
        })

        return {
            "language": language,
            "language_code": language_code,
            "events": [
                *graph_state.get("events", []),
                "language_agent",
            ],
        }

    def symptom_collection_agent(
        self,
        graph_state: GraphState,
    ) -> dict[str, Any]:
        patient = graph_state["patient"]

        result = extract_into_state(
            patient,
            graph_state["message"],
            self.llm,
            pending_key=graph_state.get("pending_field"),
        )

        self.tool_log.append({
            "tool": "symptom_collection_agent",
            "applied": sorted(result.applied),
            "rejected": [
                key for key, _ in result.rejected
            ],
            "dont_know": result.dont_know,
            "error": result.error or None,
        })

        return {
            "events": [
                *graph_state.get("events", []),
                "symptom_collection_agent",
            ],
        }

    def fact_red_flag_agent(
        self,
        graph_state: GraphState,
    ) -> dict[str, Any]:
        patient = graph_state["patient"]

        protocol = (
            get_protocol(patient.complaint)
            if patient.complaint
            else None
        )

        result = check_red_flags(patient, protocol)

        self.tool_log.append({
            "tool": "fact_red_flag_agent",
            "triggered": result.triggered,
            "labels": result.labels,
        })

        return {
            "emergency": result.triggered,
            "events": [
                *graph_state.get("events", []),
                "fact_red_flag_agent",
            ],
        }

    def route_after_fact_red_flags(
        self,
        graph_state: GraphState,
    ) -> str:
        if graph_state.get("emergency"):
            return "emergency"

        return "safe"

    def adaptive_questioning_agent(
        self,
        graph_state: GraphState,
    ) -> dict[str, Any]:
        patient = graph_state["patient"]

        if not patient.complaint:
            question = (
                "What is the main problem today?"
            )

            if self.llm is not None:
                try:
                    result = self.llm.complete_json(
                        QUESTION_SYSTEM.format(
                            language=graph_state.get(
                                "language",
                                "the patient's language",
                            ),
                        ),
                        json.dumps({
                            "field": "complaint",
                            "field_description": (
                                "The patient's main health problem"
                            ),
                            "default_question": question,
                        }),
                    )

                    candidate = (
                        result.get("question")
                        if isinstance(result, dict)
                        else None
                    )

                    if (
                        isinstance(candidate, str)
                        and candidate.strip()
                        and len(candidate) <= 240
                        and check_output(candidate).ok
                    ):
                        question = candidate.strip()

                except Exception as exc:
                    self.tool_log.append({
                        "tool": "adaptive_questioning_agent",
                        "error": str(exc),
                    })

            self._pending = {
                "kind": "complaint",
                "keys": [],
            }

            return {
                "next_question": question,
                "events": [
                    *graph_state.get("events", []),
                    "adaptive_questioning_agent",
                ],
            }

        protocol = get_protocol(patient.complaint)
        fields = missing_fields(patient, protocol)

        if not fields or patient.questions_asked >= self.max_questions:
            self._pending = None

            return {
                "next_question": "",
                "events": [
                    *graph_state.get("events", []),
                    "adaptive_questioning_agent:complete",
                ],
            }

        field = fields[0]
        question = field.question

        if self.llm is not None:
            prompt = QUESTION_SYSTEM.format(
                language=graph_state.get(
                    "language",
                    "English",
                ),
            )

            try:
                result = self.llm.complete_json(
                    prompt,
                    json.dumps({
                        "field": field.key,
                        "field_description": (
                            field.description
                            or field.question
                        ),
                        "default_question": field.question,
                    }),
                )

                candidate = (
                    result.get("question")
                    if isinstance(result, dict)
                    else None
                )

                if (
                    isinstance(candidate, str)
                    and candidate.strip()
                    and len(candidate) <= 240
                    and check_output(candidate).ok
                ):
                    question = candidate.strip()

            except Exception as exc:
                self.tool_log.append({
                    "tool": "adaptive_questioning_agent",
                    "error": str(exc),
                })

        patient.questions_asked += 1
        self._pending = {
            "kind": "field",
            "keys": [field.key],
        }

        self.tool_log.append({
            "tool": "adaptive_questioning_agent",
            "field": field.key,
            "language": graph_state.get(
                "language",
                "English",
            ),
        })

        return {
            "pending_field": field.key,
            "next_question": question,
            "events": [
                *graph_state.get("events", []),
                "adaptive_questioning_agent",
            ],
        }

    def route_after_questioning(
        self,
        graph_state: GraphState,
    ) -> str:
        if graph_state.get("next_question"):
            return "ask"

        return "complete"

    def medical_knowledge_agent(
        self,
        graph_state: GraphState,
    ) -> dict[str, Any]:
        patient = graph_state["patient"]
        protocol = get_protocol(patient.complaint)

        self.tool_log.append({
            "tool": "medical_knowledge_agent",
            "protocol": protocol.id,
        })

        return {
            "events": [
                *graph_state.get("events", []),
                "medical_knowledge_agent",
            ],
        }

    def risk_assessment_agent(
        self,
        graph_state: GraphState,
    ) -> dict[str, Any]:
        result = compute_urgency(
            graph_state["patient"],
        )

        self.tool_log.append({
            "tool": "risk_assessment_agent",
            "urgency": result.urgency,
            "confidence": result.confidence,
        })

        return {
            "urgency": result.urgency,
            "report": result.to_dict(),
            "events": [
                *graph_state.get("events", []),
                "risk_assessment_agent",
            ],
        }

    def safety_agent(
        self,
        graph_state: GraphState,
    ) -> dict[str, Any]:
        result = compute_urgency(
            graph_state["patient"],
        )

        review = review_and_apply(
            self.llm if self.safety_review else None,
            graph_state["patient"],
            result,
        )

        report = result.to_dict()
        report["safety_review"] = review.to_dict()

        self.tool_log.append({
            "tool": "safety_agent",
            "raised": review.raised,
            "error": review.error or None,
        })

        return {
            "urgency": result.urgency,
            "report": report,
            "events": [
                *graph_state.get("events", []),
                "safety_agent",
            ],
        }
    def clinician_escalation_agent(
        self,
        graph_state: GraphState,
    ) -> dict[str, Any]:
        patient = graph_state["patient"]
        report = dict(graph_state.get("report", {}))

        reasons = list(
            report.get(
                "clinician_reasons",
                [],
            )
        )

        if patient.complaint == "other":
            reasons.append(
                "Complaint is not covered by a specific protocol"
            )

        if patient.unknown_count >= 3:
            reasons.append(
                "Several required questions could not be answered"
            )

        if patient.age_years is not None and patient.age_years < 12:
            reasons.append(
                "Patient is under 12 years old"
            )

        if patient.pregnant is True:
            reasons.append(
                "Possible pregnancy requires clinician review"
            )

        needs_review = bool(
            reasons
            or report.get("needs_clinician_review", False)
        )

        report["needs_clinician_review"] = needs_review
        report["clinician_reasons"] = reasons

        self.tool_log.append({
            "tool": "clinician_escalation_agent",
            "needs_review": needs_review,
            "reasons": reasons,
        })

        return {
            "report": report,
            "events": [
                *graph_state.get("events", []),
                "clinician_escalation_agent",
            ],
        }
    def care_pathway_agent(
        self,
        graph_state: GraphState,
    ) -> dict[str, Any]:
        urgency = graph_state.get("urgency")
        report = dict(graph_state.get("report", {}))

        if urgency:
            pathway = pathways.describe(urgency)

            report["care_pathway"] = pathway.get(
                "pathway",
                "",
            )
            report["care_timeframe"] = pathway.get(
                "timeframe",
                "",
            )
            report["care_advice"] = pathway.get(
                "advice",
                "",
            )

        self.tool_log.append({
            "tool": "care_pathway_agent",
            "urgency": urgency,
            "pathway": report.get("care_pathway"),
        })

        return {
            "report": report,
            "events": [
                *graph_state.get("events", []),
                "care_pathway_agent",
            ],
        }

    def summary_agent(
        self,
        graph_state: GraphState,
    ) -> dict[str, Any]:
        report = dict(graph_state.get("report", {}))

        report["tool_log"] = list(self.tool_log)
        report["language"] = graph_state.get(
            "language",
            "English",
        )
        report["language_code"] = graph_state.get(
            "language_code",
            "auto",
        )

        self.tool_log.append({
            "tool": "medical_summary_agent",
        })

        return {
            "report": report,
            "done": True,
            "response": (
                "Thank you. Your assessment is complete."
            ),
            "events": [
                *graph_state.get("events", []),
                "medical_summary_agent",
            ],
        }

    def emergency_agent(
        self,
        graph_state: GraphState,
    ) -> dict[str, Any]:
        patient = graph_state["patient"]
        message = graph_state["message"]

        patient.add_user_message(message)

        hits = check_text(
            patient.text_blob()
        )

        report = {
            "urgency": "emergency",
            "confidence": "high",
            "red_flags": [
                {
                    "label": hit.label,
                    "category": hit.category,
                }
                for hit in hits
            ],
            "reasons": [
                hit.label
                for hit in hits
            ],
            "needs_clinician_review": True,
            "clinician_reasons": [
                "Emergency red flag detected",
            ],
            "tool_log": list(self.tool_log),
        }

        self.tool_log.append({
            "tool": "emergency_agent",
            "labels": [
                hit.label
                for hit in hits
            ],
        })

        return {
            "urgency": "emergency",
            "report": report,
            "done": True,
            "response": (
                "EMERGENCY WARNING SIGN DETECTED. "
                "Please call emergency services now."
            ),
            "events": [
                *graph_state.get("events", []),
                "emergency_agent",
            ],
        }

    def memory_agent(
        self,
        graph_state: GraphState,
    ) -> dict[str, Any]:
        patient = graph_state["patient"]
        report = graph_state.get("report")

        if report:
            report["tool_log"] = list(self.tool_log)

            self.memory.save_state(patient)
            self.memory.record_symptom_events(patient)
            self.memory.save_report(
                patient.session_id,
                report,
            )

        self.done = True

        return {
            "done": True,
            "events": [
                *graph_state.get("events", []),
                "memory_agent",
            ],
        }

    # ------------------------------------------------------------------
    # Public API

    def start(
        self,
        patient_code: Optional[str] = None,
    ) -> AgentTurn:
        raise RuntimeError(
            "For Supabase sessions, create the assessment "
            "through the API endpoint."
        )

    def handle(self, text: str) -> AgentTurn:
        if self.state is None:
            raise RuntimeError(
                "Call start() before handle()."
            )

        if self.done:
            return AgentTurn(
                kind="closed",
                message=(
                    "This assessment is complete. "
                    "Please start a new assessment."
                ),
                done=True,
            )

        self.turn += 1

        result = self.graph.invoke({
            "patient": self.state,
            "message": text,
            "pending_field": (
                self._pending["keys"][0]
                if self._pending
                and self._pending.get("keys")
                else None
            ),
            "events": [],
        })

        events = result.get("events", [])
        language = result.get(
            "language",
            "English",
        )

        if result.get("next_question"):
            question = result["next_question"]

            self.memory.log_message(
                self.state.session_id,
                "agent",
                question,
            )

            self.memory.save_state(self.state)

            return AgentTurn(
                kind="question",
                message=result["next_question"],
                done=False,
                tool_log=[
                    {
                        "tool": "langgraph",
                        "events": result.get("events", []),
                        "language": result.get(
                            "language",
                            "auto",
                        ),
                    }
                ],
            )

        report = result.get("report") or {}
        urgency = result.get("urgency")

        if urgency == "emergency":
            message = (
                "EMERGENCY WARNING SIGN DETECTED. "
                "Please call emergency services now."
            )
            kind = "emergency"
        else:
            message = (
                "Thank you. Your assessment is complete."
            )
            kind = "result"

        self.done = True

        return AgentTurn(
            kind=(
                "emergency"
                if result.get("urgency") == "emergency"
                else "result"
            ),
            message=result.get(
                "response",
                "Assessment completed.",
            ),
            done=True,
            urgency=result.get("urgency"),
            result=result.get("report"),
            tool_log=[
                {
                    "tool": "langgraph",
                    "events": result.get("events", []),
                }
            ],
        )