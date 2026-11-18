from __future__ import annotations
from fastapi.responses import (PlainTextResponse, Response) 
import os
from typing import Any
from api.pdf_report import build_clinician_pdf
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from agent.workflow_visualization import workflow_diagram
from api.rate_limit import enforce_rate_limit
import config
from agent.llm_client import get_llm
from agent.langgraph_workflow import LangGraphTriageAgent
from agent.summary import build_clinician_summary
from api.auth import supabase_admin
from api.dependencies import (
    current_user,
    require_clinician,
    require_patient,
)
from api.schemas import (
    MessageRequest,
    ClinicianFeedbackRequest,
)
from memory.supabase_repository import SupabaseRepository
from agent.workflow_visualization import (
    WORKFLOW_MERMAID,
    executed_nodes,
)


app = FastAPI(title="TriageGuide API")
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8501",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
repository = SupabaseRepository(supabase_admin)


def create_agent() -> LangGraphTriageAgent:
    use_llm = (
        os.getenv("API_USE_LLM", "false").lower()
        == "true"
    )

    llm = get_llm() if use_llm else None

    return LangGraphTriageAgent(
        repository,
        llm=llm,
        safety_review=True,
    )


def runtime_from_agent(
    agent: LangGraphTriageAgent,
) -> dict[str, Any]:
    return {
        "done": agent.done,
        "turn": agent.turn,
        "pending": agent._pending,
        "misses": agent._misses,
        "complaint_tries": agent._complaint_tries,
        "escalation_reason": agent._escalation_reason,
        "llm_failures": agent._llm_failures,
        "breaker_logged": agent._breaker_logged,
        "refusal": agent._refusal,
        "tool_log": agent.tool_log,
    }


def agent_from_session(
    session_id: str,
) -> LangGraphTriageAgent:
    agent = create_agent()

    state = repository.load_state(session_id)
    runtime = repository.get_runtime(session_id)

    agent.state = state
    agent.done = bool(runtime.get("done", False))
    agent.turn = int(runtime.get("turn", 0))
    agent._pending = runtime.get("pending")
    agent._misses = runtime.get("misses", {})
    agent._complaint_tries = int(
        runtime.get("complaint_tries", 0)
    )
    agent._escalation_reason = runtime.get(
        "escalation_reason"
    )
    agent._llm_failures = int(
        runtime.get("llm_failures", 0)
    )
    agent._breaker_logged = bool(
        runtime.get("breaker_logged", False)
    )
    agent._refusal = runtime.get("refusal", "")
    agent.tool_log = runtime.get("tool_log", [])

    return agent


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/auth/me")
def me(
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    return user


@app.post("/patient/assessments")
def create_assessment(
    user: dict[str, Any] = Depends(require_patient),
) -> dict[str, Any]:
    state = repository.start_user_session(
        user["user_id"],
    )

    agent = create_agent()
    agent.state = state
    agent._pending = {
        "kind": "complaint",
        "keys": [],
    }

    message = (
        "What is the main problem today?"
    )

    repository.log_message(
        state.session_id,
        "agent",
        message,
    )

    repository.save_runtime(
        state.session_id,
        runtime_from_agent(agent),
    )

    repository.audit(
        action="assessment_created",
        entity_type="session",
        entity_id=state.session_id,
        actor_user_id=user["user_id"],
    )

    return {
        "session_id": state.session_id,
        "patient_code": state.patient_id,
        "message": message,
        "kind": "greeting",
        "done": False,
    }


@app.post("/patient/assessments/{session_id}/messages")
def send_message(
    session_id: str,
    request: MessageRequest,
    raw_request: Request,
    user: dict[str, Any] = Depends(require_patient),
) -> dict[str, Any]:
    enforce_rate_limit(raw_request)

    if not repository.session_belongs_to_user(
        session_id,
        user["user_id"],
    ):
        raise HTTPException(
            status_code=404,
            detail="Assessment not found",
        )

    agent = agent_from_session(session_id)

    if agent.done:
        raise HTTPException(
            status_code=409,
            detail="Assessment is already complete",
        )

    turn = agent.handle(request.text)

    repository.save_runtime(
        session_id,
        runtime_from_agent(agent),
    )

    repository.audit(
        action="message_submitted",
        entity_type="session",
        entity_id=session_id,
        actor_user_id=user["user_id"],
        metadata={
            "completed": turn.done,
            "urgency": turn.urgency,
        },
    )

    if turn.done:
        repository.audit(
            action="assessment_completed",
            entity_type="session",
            entity_id=session_id,
            actor_user_id=user["user_id"],
            metadata={"urgency": turn.urgency},
        )

    return {
        "session_id": session_id,
        "message": turn.message,
        "kind": turn.kind,
        "done": turn.done,
        "urgency": turn.urgency,
        "result": turn.result,
        "refusal": turn.refusal,
    }


@app.get("/patient/assessments/{session_id}")
def get_patient_assessment(
    session_id: str,
    user: dict[str, Any] = Depends(require_patient),
) -> dict[str, Any]:
    if not repository.session_belongs_to_user(
        session_id,
        user["user_id"],
    ):
        raise HTTPException(
            status_code=404,
            detail="Assessment not found",
        )

    state = repository.load_state(session_id)

    return {
        "session_id": session_id,
        "state": state.model_dump(mode="json"),
        "report": repository.get_report(session_id),
    }


@app.delete("/patient/me")
def delete_patient_data(
    user: dict[str, Any] = Depends(require_patient),
) -> dict[str, str]:
    repository.delete_user_patient(
        user["user_id"],
    )

    repository.audit(
        action="patient_deleted",
        entity_type="patient",
        actor_user_id=user["user_id"],
    )

    return {"status": "deleted"}


@app.get("/clinician/ping")
def clinician_ping(
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, str]:
    return {
        "status": "authenticated",
        "role": user["role"],
    }


@app.get("/clinician/sessions")
def list_sessions(
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    sessions = repository.list_sessions(limit=200)

    repository.audit(
        action="sessions_listed",
        entity_type="session",
        actor_user_id=user["user_id"],
    )

    return {
        "sessions": sessions,
        "requested_by": user["user_id"],
    }


@app.get("/clinician/sessions/{session_id}")
def get_session(
    session_id: str,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    try:
        state = repository.load_state(session_id)
        report = repository.get_report(session_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        ) from exc

    timeline = repository.timeline_text(
        state.patient_id,
    )

    continuity = repository.continuity_note(
        state.patient_id,
        exclude_session_id=session_id,
    )

    summary = build_clinician_summary(
        state,
        report,
        timeline,
        continuity,
    )

    repository.audit(
        action="session_viewed",
        entity_type="session",
        entity_id=session_id,
        actor_user_id=user["user_id"],
    )

    return {
        "session_id": session_id,
        "state": state.model_dump(mode="json"),
        "report": report,
        "summary": summary,
        "timeline": timeline,
        "workflow": {
            "diagram": workflow_diagram(report),
            "executed_nodes": executed_nodes(report),
        },
    }


@app.delete("/clinician/patients/{patient_code}")
def clinician_delete_patient(
    patient_code: str,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, str]:
    try:
        repository.delete_patient(patient_code)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Patient not found",
        ) from exc

    repository.audit(
        action="patient_deleted_by_clinician",
        entity_type="patient",
        entity_id=patient_code,
        actor_user_id=user["user_id"],
    )

    return {
        "status": "deleted",
        "patient_code": patient_code,
    }

@app.get("/patient/assessments/{session_id}/messages")
def get_patient_messages(
    session_id: str,
    user: dict[str, Any] = Depends(require_patient),
) -> dict[str, Any]:
    if not repository.session_belongs_to_user(
        session_id,
        user["user_id"],
    ):
        raise HTTPException(
            status_code=404,
            detail="Assessment not found",
        )

    return {
        "session_id": session_id,
        "messages": repository.get_messages(session_id),
    }

@app.get(
    "/patient/assessments/{session_id}/download",
    response_class=PlainTextResponse,
)
def download_patient_report(
    session_id: str,
    user: dict[str, Any] = Depends(require_patient),
) -> Response:
    if not repository.session_belongs_to_user(
        session_id,
        user["user_id"],
    ):
        raise HTTPException(
            status_code=404,
            detail="Assessment not found",
        )

    state = repository.load_state(session_id)
    report = repository.get_report(session_id)

    if report is None:
        raise HTTPException(
            status_code=409,
            detail="Assessment is not complete",
        )

    timeline = repository.timeline_text(state.patient_id)
    continuity = repository.continuity_note(
        state.patient_id,
        exclude_session_id=session_id,
    )

    summary = build_clinician_summary(
        state,
        report,
        timeline,
        continuity,
    )

    repository.audit(
        action="report_downloaded",
        entity_type="session",
        entity_id=session_id,
        actor_user_id=user["user_id"],
    )

    lines = [
        "TriageGuide assessment summary",
        "",
        f"Patient code: {summary.get('patient_code', '')}",
        f"Urgency: {summary.get('urgency_label', '')}",
        f"Suggested care: {summary.get('suggested_care', '')}",
        f"Timeframe: {summary.get('timeframe', '')}",
        "",
        "Summary:",
        summary.get("summary", ""),
        "",
        "Reasons:",
    ]

    lines.extend(
        f"- {reason}"
        for reason in summary.get("reasons", [])
    )

    lines.extend([
        "",
        "Timeline:",
    ])

    lines.extend(
        f"- {item}"
        for item in timeline
    )

    return PlainTextResponse(
        "\n".join(lines),
        headers={
            "Content-Disposition": (
                f'attachment; filename="triage-{session_id}.txt"'
            )
        },
    )

@app.get("/clinician/sessions/{session_id}/summary")
def get_session_summary(
    session_id: str,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    try:
        state = repository.load_state(session_id)
        report = repository.get_report(session_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        ) from exc

    timeline = repository.timeline_text(state.patient_id)
    continuity = repository.continuity_note(
        state.patient_id,
        exclude_session_id=session_id,
    )

    summary = build_clinician_summary(
        state,
        report,
        timeline,
        continuity,
    )

    repository.audit(
        action="summary_viewed",
        entity_type="session",
        entity_id=session_id,
        actor_user_id=user["user_id"],
    )

    return {
        "session_id": session_id,
        "summary": summary,
        "timeline": timeline,
        "continuity": continuity,
        "tool_log": (
            report.get("tool_log", [])
            if report
            else []
        ),
        "latency": (
            report.get("latency", {})
            if report
            else {}
        ),
    }

@app.get(
    "/clinician/sessions/{session_id}/pdf",
)
def download_clinician_pdf(
    session_id: str,
    user: dict[str, Any] = Depends(require_clinician),
) -> PlainTextResponse:
    try:
        state = repository.load_state(session_id)
        report = repository.get_report(session_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        ) from exc

    if report is None:
        raise HTTPException(
            status_code=409,
            detail="Assessment is not complete",
        )

    timeline = repository.timeline_text(
        state.patient_id,
    )

    continuity = repository.continuity_note(
        state.patient_id,
        exclude_session_id=session_id,
    )

    summary = build_clinician_summary(
        state,
        report,
        timeline,
        continuity,
    )

    pdf_bytes = build_clinician_pdf(
        summary,
        report,
        timeline,
    )

    repository.audit(
        action="clinician_pdf_downloaded",
        entity_type="session",
        entity_id=session_id,
        actor_user_id=user["user_id"],
    )

    from fastapi.responses import Response

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="triage-{session_id}.pdf"'
            ),
        },
    )

@app.post("/clinician/sessions/{session_id}/feedback")
def submit_clinician_feedback(
    session_id: str,
    request: ClinicianFeedbackRequest,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    if request.outcome == "incorrect_urgency":
        if request.corrected_urgency is None:
            raise HTTPException(
                status_code=422,
                detail="Corrected urgency is required",
            )

    try:
        repository._session(session_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        ) from exc

    feedback = repository.save_clinician_feedback(
        session_id=session_id,
        clinician_user_id=user["user_id"],
        outcome=request.outcome,
        corrected_urgency=request.corrected_urgency,
        comment=request.comment,
    )

    repository.audit(
        action="clinician_feedback_submitted",
        entity_type="session",
        entity_id=session_id,
        actor_user_id=user["user_id"],
        metadata={
            "outcome": request.outcome,
            "corrected_urgency": request.corrected_urgency,
        },
    )

    return {
        "status": "saved",
        "feedback": feedback,
    }


@app.get("/clinician/sessions/{session_id}/feedback")
def get_clinician_feedback(
    session_id: str,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    return {
        "feedback": repository.get_clinician_feedback(
            session_id
        )
    }