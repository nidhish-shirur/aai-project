from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from agent.state import PatientState


class Repository(Protocol):
    def start_session(
        self,
        patient_id: str | None = None,
        now: datetime | None = None,
    ) -> PatientState:
        ...

    def start_user_session(
        self,
        user_id: str,
        now: datetime | None = None,
    ) -> PatientState:
        ...

    def save_state(
        self,
        state: PatientState,
        now: datetime | None = None,
    ) -> None:
        ...

    def load_state(self, session_id: str) -> PatientState:
        ...

    def log_message(
        self,
        session_id: str,
        role: str,
        content: str,
        now: datetime | None = None,
    ) -> None:
        ...

    def get_messages(
        self,
        session_id: str,
    ) -> list[dict[str, Any]]:
        ...

    def save_report(
        self,
        session_id: str,
        report: dict[str, Any],
        now: datetime | None = None,
    ) -> None:
        ...

    def get_report(
        self,
        session_id: str,
    ) -> dict[str, Any] | None:
        ...

    def list_sessions(
        self,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        ...

    def record_symptom_events(
        self,
        state: PatientState,
        now: datetime | None = None,
    ) -> None:
        ...

    def timeline_text(
        self,
        patient_id: str,
        within_days: int = 30,
        now: datetime | None = None,
    ) -> list[str]:
        ...

    def continuity_note(
        self,
        patient_id: str,
        exclude_session_id: str | None = None,
        now: datetime | None = None,
    ) -> str:
        ...

    def unresolved_earlier_symptoms(
        self,
        patient_id: str,
        exclude_session_id: str | None = None,
        within_days: int = 30,
        now: datetime | None = None,
    ) -> list[str]:
        ...

    def mark_earlier_resolved(
        self,
        patient_id: str,
        exclude_session_id: str | None = None,
        now: datetime | None = None,
    ) -> int:
        ...

    def previous_sessions(
        self,
        patient_id: str,
        exclude_session_id: str | None = None,
        limit: int = 3,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def delete_patient(self, patient_id: str) -> None:
        ...

    def delete_user_patient(self, user_id: str) -> None:
        ...

    def save_runtime(
        self,
        session_id: str,
        runtime: dict[str, Any],
    ) -> None:
        ...

    def get_runtime(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        ...

    def session_belongs_to_user(
        self,
        session_id: str,
        user_id: str,
    ) -> bool:
        ...

    def audit(
        self,
        action: str,
        entity_type: str,
        entity_id: str | None = None,
        actor_user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        ...