from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from supabase import create_client

import config
import uuid

def rows(
    connection: sqlite3.Connection,
    table: str,
) -> list[dict[str, Any]]:
    cursor = connection.execute(
        f"SELECT * FROM {table}"
    )
    columns = [
        description[0]
        for description in cursor.description
    ]
    return [
        dict(zip(columns, row))
        for row in cursor.fetchall()
    ]


def main() -> None:
    if not config.SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL is missing")

    if not config.SUPABASE_SECRET_KEY:
        raise RuntimeError(
            "SUPABASE_SECRET_KEY is missing"
        )

    database = Path(config.DB_PATH)

    if not database.exists():
        raise FileNotFoundError(database)

    client = create_client(
        config.SUPABASE_URL,
        config.SUPABASE_SECRET_KEY,
    )

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row

    patient_rows = rows(connection, "patients")
    session_rows = rows(connection, "sessions")
    message_rows = rows(connection, "messages")
    event_rows = rows(connection, "symptom_events")

    patient_map: dict[str, str] = {}
    session_map: dict[str, str] = {}

    for patient in patient_rows:
        result = (
            client
            .table("patients")
            .insert({
                "public_code": patient["patient_id"],
                "created_at": patient["created_at"],
            })
            .execute()
        )

        if not result.data:
            raise RuntimeError(
                f"Patient import failed: {patient['patient_id']}"
            )

        patient_map[patient["patient_id"]] = result.data[0]["id"]

    for session in session_rows:
        patient_uuid = patient_map[session["patient_id"]]

        state = json.loads(session["state_json"])

        report = (
            json.loads(session["report_json"])
            if session["report_json"]
            else None
        )

        new_session_id = str(uuid.uuid4())

        result = (
            client
            .table("sessions")
            .insert({
                "id": new_session_id,
                "patient_id": patient_uuid,
                "created_at": session["created_at"],
                "updated_at": session["updated_at"],
                "status": session["status"],
                "complaint": session["complaint"],
                "urgency": session["urgency"],
                "state": state,
                "report": report,
                "agent_runtime": {},
            })
            .execute()
        )

        if not result.data:
            raise RuntimeError(
                f"Session import failed: "
                f"{session['session_id']}"
            )

        session_map[session["session_id"]] = new_session_id

    for message in message_rows:
        client.table("messages").insert({
            "session_id": session_map[message["session_id"]],
            "role": message["role"],
            "content": message["content"],
            "created_at": message["created_at"],
        }).execute()

    for event in event_rows:
        client.table("symptom_events").insert({
            "session_id": session_map[event["session_id"]],
            "patient_id": patient_map[event["patient_id"]],
            "symptom": event["symptom"],
            "onset_at": event["onset_at"],
            "recorded_at": event["recorded_at"],
            "resolved_at": event.get("resolved_at"),
        }).execute()

    print("Migration completed.")
    print(f"Patients: {len(patient_rows)}")
    print(f"Sessions: {len(session_rows)}")
    print(f"Messages: {len(message_rows)}")
    print(f"Symptom events: {len(event_rows)}")

    connection.close()


if __name__ == "__main__":
    main()