from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, FastAPI, HTTPException

import config
from agent.llm_client import get_llm
from agent.orchestrator import TriageAgent
from api.auth import supabase_admin
from api.dependencies import current_user, require_clinician
from api.schemas import CreateAssessmentRequest, MessageRequest
from memory.supabase_repository import SupabaseRepository


app = FastAPI(title="TriageGuide API")

repository = SupabaseRepository(supabase_admin)

ACTIVE_AGENTS: dict[str, TriageAgent] = {}


def create_agent() -> TriageAgent:
    use_llm = os.getenv("API_USE_LLM", "false").lower() == "true"

    llm = get_llm() if use_llm else None

    return TriageAgent(
        repository,
        llm=llm,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/auth/me")
def me(
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    return user


@app.get("/clinician/ping")
def clinician_ping(
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, str]:
    return {
        "status": "authenticated",
        "role": user["role"],
    }


@app.post("/patient/assessments")
def create_assessment(
    request: CreateAssessmentRequest,
) -> dict[str, Any]:
    agent = create_agent()

    try:
        turn = agent.start(request.patient_code)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail="Patient code was not found",
        )

    session_id = agent.state.session_id
    ACTIVE_AGENTS[session_id] = agent

    return {
        "session_id": session_id,
        "patient_code": agent.state.patient_id,
        "message": turn.message,
        "kind": turn.kind,
        "done": turn.done,
    }


@app.post("/patient/assessments/{session_id}/messages")
def send_message(
    session_id: str,
    request: MessageRequest,
) -> dict[str, Any]:
    agent = ACTIVE_AGENTS.get(session_id)

    if agent is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Assessment is not active in this API process. "
                "Start a new assessment."
            ),
        )

    turn = agent.handle(request.text)

    if turn.done:
        ACTIVE_AGENTS.pop(session_id, None)

    return {
        "session_id": session_id,
        "message": turn.message,
        "kind": turn.kind,
        "done": turn.done,
        "urgency": turn.urgency,
        "result": turn.result,
        "refusal": turn.refusal,
    }


@app.get("/clinician/sessions")
def list_sessions(
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    return {
        "sessions": repository.list_sessions(limit=200),
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

    return {
        "session_id": session_id,
        "state": state.model_dump(mode="json"),
        "report": report,
        "timeline": repository.timeline_text(
            state.patient_id,
        ),
        "requested_by": user["user_id"],
    }


@app.delete("/clinician/patients/{patient_code}")
def delete_patient(
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

    return {
        "status": "deleted",
        "patient_code": patient_code,
    }