from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from supabase import Client

from agent.state import PatientState
from memory.timeline import (
    build_timeline,
    derive_symptom_events,
    timeline_text,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.isoformat()


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


class SupabaseRepository:
    """Supabase-backed implementation compatible with TriageAgent."""

    def __init__(self, client: Client):
        self.client = client

    # ------------------------------------------------------------ helpers

    def _patient(self, public_code: str) -> dict[str, Any]:
        response = (
            self.client
            .table("patients")
            .select("id, public_code, created_at")
            .eq("public_code", public_code)
            .limit(1)
            .execute()
        )

        if not response.data:
            raise KeyError(f"Unknown patient code: {public_code}")

        return response.data[0]

    def _session(self, session_id: str) -> dict[str, Any]:
        response = (
            self.client
            .table("sessions")
            .select("*")
            .eq("id", session_id)
            .limit(1)
            .execute()
        )

        if not response.data:
            raise KeyError(f"Unknown session: {session_id}")

        return response.data[0]

    def _public_code(self) -> str:
        while True:
            code = "P-" + secrets.token_hex(3).upper()
            response = (
                self.client
                .table("patients")
                .select("id")
                .eq("public_code", code)
                .limit(1)
                .execute()
            )
            if not response.data:
                return code

    # ------------------------------------------------------------ patients

    def patient_exists(self, patient_id: str) -> bool:
        response = (
            self.client
            .table("patients")
            .select("id")
            .eq("public_code", patient_id)
            .limit(1)
            .execute()
        )
        return bool(response.data)

    def create_patient(self, now: Optional[datetime] = None) -> str:
        response = (
            self.client
            .table("patients")
            .insert({
                "public_code": self._public_code(),
                "created_at": iso(now or utcnow()),
            })
            .execute()
        )
        return response.data[0]["public_code"]

    def start_session(
        self,
        patient_id: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> PatientState:
        now = now or utcnow()

        if patient_id is None:
            patient_id = self.create_patient(now)

        patient = self._patient(patient_id)

        session_id = str(uuid.uuid4())
        state = PatientState(
            session_id=session_id,
            patient_id=patient["public_code"],
        )

        self.client.table("sessions").insert({
            "id": session_id,
            "patient_id": patient["id"],
            "created_at": iso(now),
            "updated_at": iso(now),
            "status": "active",
            "state": state.model_dump(mode="json"),
        }).execute()

        return state

    # ------------------------------------------------------------ state

    def save_state(
        self,
        state: PatientState,
        now: Optional[datetime] = None,
    ) -> None:
        self._session(state.session_id)

        self.client.table("sessions").update({
            "state": state.model_dump(mode="json"),
            "complaint": state.complaint,
            "updated_at": iso(now or utcnow()),
        }).eq("id", state.session_id).execute()

    def load_state(self, session_id: str) -> PatientState:
        row = self._session(session_id)
        return PatientState.model_validate(row["state"])

    # ------------------------------------------------------------ messages

    def log_message(
        self,
        session_id: str,
        role: str,
        content: str,
        now: Optional[datetime] = None,
    ) -> None:
        if role not in {"user", "agent"}:
            raise ValueError("role must be 'user' or 'agent'")

        self._session(session_id)

        self.client.table("messages").insert({
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": iso(now or utcnow()),
        }).execute()

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        response = (
            self.client
            .table("messages")
            .select("role, content, created_at")
            .eq("session_id", session_id)
            .order("id")
            .execute()
        )
        return response.data

    # ------------------------------------------------------------ reports

    def save_report(
        self,
        session_id: str,
        report: dict[str, Any],
        now: Optional[datetime] = None,
    ) -> None:
        self._session(session_id)

        self.client.table("sessions").update({
            "report": report,
            "urgency": report.get("urgency"),
            "status": "completed",
            "updated_at": iso(now or utcnow()),
        }).eq("id", session_id).execute()

    def get_report(self, session_id: str) -> Optional[dict[str, Any]]:
        row = self._session(session_id)
        return row.get("report")

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        response = (
            self.client
            .table("sessions")
            .select(
                "id, patient_id, created_at, updated_at, "
                "status, complaint, urgency"
            )
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )

        patients = self.client.table("patients").select(
            "id, public_code"
        ).execute()

        patient_codes = {
            row["id"]: row["public_code"]
            for row in patients.data
        }

        rank = {
            "emergency": 4,
            "urgent": 3,
            "moderate": 2,
            "low": 1,
            None: 0,
        }

        rows = []
        for row in response.data:
            row["session_id"] = row.pop("id")
            row["patient_id"] = patient_codes.get(row["patient_id"])
            rows.append(row)

        rows.sort(
            key=lambda row: (
                rank.get(row.get("urgency"), 0),
                row.get("updated_at") or "",
            ),
            reverse=True,
        )

        return rows

    # ------------------------------------------------------------ symptoms

    def record_symptom_events(
        self,
        state: PatientState,
        now: Optional[datetime] = None,
    ) -> None:
        if not state.patient_id:
            raise ValueError("state has no patient_id")

        now = now or utcnow()
        patient = self._patient(state.patient_id)

        for symptom, onset, from_duration in derive_symptom_events(state, now):
            existing = (
                self.client
                .table("symptom_events")
                .select("id")
                .eq("session_id", state.session_id)
                .eq("symptom", symptom)
                .limit(1)
                .execute()
            )

            payload = {
                "session_id": state.session_id,
                "patient_id": patient["id"],
                "symptom": symptom,
                "onset_at": iso(onset),
                "recorded_at": iso(now),
            }

            if not existing.data:
                self.client.table("symptom_events").insert(payload).execute()
            elif from_duration:
                self.client.table("symptom_events").update({
                    "onset_at": iso(onset),
                }).eq("id", existing.data[0]["id"]).execute()

    def get_timeline(
        self,
        patient_id: str,
        within_days: int = 30,
        now: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        now = now or utcnow()
        patient = self._patient(patient_id)
        cutoff = now - timedelta(days=within_days)

        response = (
            self.client
            .table("symptom_events")
            .select("symptom, onset_at, resolved_at")
            .eq("patient_id", patient["id"])
            .execute()
        )

        grouped: dict[str, list[dict[str, Any]]] = {}

        for row in response.data:
            onset = parse_datetime(row["onset_at"])
            if onset < cutoff:
                continue
            grouped.setdefault(row["symptom"], []).append(row)

        events = []

        for symptom, rows in grouped.items():
            earliest = min(
                rows,
                key=lambda row: row["onset_at"],
            )

            resolved_dates = [
                parse_datetime(row["resolved_at"])
                for row in rows
                if row.get("resolved_at")
            ]

            unresolved = any(
                not row.get("resolved_at")
                for row in rows
            )

            resolved = None
            if not unresolved and resolved_dates:
                resolved = max(resolved_dates)

            events.append((
                symptom,
                parse_datetime(earliest["onset_at"]),
                resolved,
            ))

        return build_timeline(events)

    def timeline_text(
        self,
        patient_id: str,
        within_days: int = 30,
        now: Optional[datetime] = None,
    ) -> list[str]:
        return timeline_text(
            self.get_timeline(patient_id, within_days, now)
        )

    # ------------------------------------------------------------ continuity

    def unresolved_earlier_symptoms(
        self,
        patient_id: str,
        exclude_session_id: Optional[str] = None,
        within_days: int = 30,
        now: Optional[datetime] = None,
    ) -> list[str]:
        now = now or utcnow()
        patient = self._patient(patient_id)
        cutoff = now - timedelta(days=within_days)

        response = (
            self.client
            .table("symptom_events")
            .select("symptom, session_id, onset_at, resolved_at")
            .eq("patient_id", patient["id"])
            .execute()
        )

        rows = [
            row for row in response.data
            if row["session_id"] != exclude_session_id
            and not row.get("resolved_at")
            and parse_datetime(row["onset_at"]) >= cutoff
        ]

        rows.sort(key=lambda row: row["onset_at"])

        output = []
        for row in rows:
            if row["symptom"] not in output:
                output.append(row["symptom"])

        return output

    def mark_earlier_resolved(
        self,
        patient_id: str,
        exclude_session_id: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> int:
        now = now or utcnow()
        patient = self._patient(patient_id)

        response = (
            self.client
            .table("symptom_events")
            .select("id, session_id, resolved_at")
            .eq("patient_id", patient["id"])
            .execute()
        )

        count = 0

        for row in response.data:
            if (
                row["session_id"] != exclude_session_id
                and not row.get("resolved_at")
            ):
                self.client.table("symptom_events").update({
                    "resolved_at": iso(now),
                }).eq("id", row["id"]).execute()
                count += 1

        return count

    def previous_sessions(
        self,
        patient_id: str,
        exclude_session_id: Optional[str] = None,
        limit: int = 3,
        now: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        now = now or utcnow()
        patient = self._patient(patient_id)

        response = (
            self.client
            .table("sessions")
            .select("id, created_at, complaint, urgency")
            .eq("patient_id", patient["id"])
            .order("created_at", desc=True)
            .limit(limit + 1)
            .execute()
        )

        output = []

        for row in response.data:
            if row["id"] == exclude_session_id:
                continue

            events = (
                self.client
                .table("symptom_events")
                .select("symptom")
                .eq("session_id", row["id"])
                .order("onset_at")
                .execute()
            )

            output.append({
                "session_id": row["id"],
                "complaint": row.get("complaint"),
                "urgency": row.get("urgency"),
                "days_ago": (
                    now - parse_datetime(row["created_at"])
                ).days,
                "symptoms": [
                    event["symptom"] for event in events.data
                ],
            })

            if len(output) >= limit:
                break

        return output

    def continuity_note(
        self,
        patient_id: str,
        exclude_session_id: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> str:
        previous = [
            session
            for session in self.previous_sessions(
                patient_id,
                exclude_session_id,
                now=now,
            )
            if session["symptoms"]
        ]

        if not previous:
            return ""

        parts = []

        for session in previous:
            days = session["days_ago"]
            when = "today" if days == 0 else f"{days} day(s) ago"
            urgency = (
                f", assessed as {session['urgency']}"
                if session["urgency"]
                else ""
            )
            symptoms = ", ".join(session["symptoms"]).lower()
            parts.append(f"{when}: {symptoms}{urgency}")

        return "Earlier sessions for this patient: " + "; ".join(parts) + "."

    # ------------------------------------------------------------ privacy

    def delete_patient(self, patient_id: str) -> None:
        patient = self._patient(patient_id)

        self.client.table("patients").delete().eq(
            "id", patient["id"]
        ).execute()

    def purge_older_than(
        self,
        days: int,
        now: Optional[datetime] = None,
    ) -> int:
        cutoff = now or utcnow()
        cutoff = cutoff - timedelta(days=days)

        response = (
            self.client
            .table("sessions")
            .select("id")
            .lt("updated_at", iso(cutoff))
            .execute()
        )

        for row in response.data:
            self.client.table("sessions").delete().eq(
                "id", row["id"]
            ).execute()

        return len(response.data)