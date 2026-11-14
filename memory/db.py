"""SQLite session memory.

Purpose: (1) keep the whole conversation and patient state so a session can be
reloaded, (2) link sessions of the same person through an ANONYMOUS patient code
(no names or phone numbers are ever stored), (3) provide the continuity note and
symptom timeline that make the agent remember "you reported fever last week".
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import config
from agent.state import PatientState
from memory.timeline import build_timeline, derive_symptom_events, timeline_text

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'active',
    complaint  TEXT,
    urgency    TEXT,
    state_json TEXT NOT NULL,
    report_json TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS symptom_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    patient_id TEXT NOT NULL,
    symptom TEXT NOT NULL,
    onset_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    UNIQUE(session_id, symptom)
);
CREATE INDEX IF NOT EXISTS idx_sessions_patient ON sessions(patient_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_events_patient ON symptom_events(patient_id, onset_at);
"""

URGENCY_RANK_SQL = ("CASE urgency WHEN 'emergency' THEN 4 WHEN 'urgent' THEN 3 "
                    "WHEN 'moderate' THEN 2 WHEN 'low' THEN 1 ELSE 0 END")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse(text: str) -> datetime:
    return datetime.fromisoformat(text)


class Memory:
    """Thread-safe wrapper around one SQLite file (use ':memory:' in tests)."""

    def __init__(self, path: Any = None):
        self.path = str(path if path is not None else config.DB_PATH)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Upgrade databases created by earlier phases (adds symptom_events.resolved_at)."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(symptom_events)")}
        if "resolved_at" not in cols:
            self._conn.execute("ALTER TABLE symptom_events ADD COLUMN resolved_at TEXT")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------ patients and sessions
    def create_patient(self, now: Optional[datetime] = None) -> str:
        now = now or _now()
        with self._lock:
            while True:
                code = "P-" + secrets.token_hex(3).upper()
                if not self.patient_exists(code):
                    break
            self._conn.execute("INSERT INTO patients VALUES (?, ?)", (code, _iso(now)))
            self._conn.commit()
        return code

    def patient_exists(self, patient_id: str) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
        return row is not None

    def start_session(self, patient_id: Optional[str] = None, now: Optional[datetime] = None) -> PatientState:
        """Begin a new session. Pass an existing patient code for a returning patient."""
        now = now or _now()
        if patient_id is None:
            patient_id = self.create_patient(now)
        elif not self.patient_exists(patient_id):
            raise KeyError(f"Unknown patient code: {patient_id}")
        state = PatientState(patient_id=patient_id)
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (session_id, patient_id, created_at, updated_at, state_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (state.session_id, patient_id, _iso(now), _iso(now), state.model_dump_json()))
            self._conn.commit()
        return state

    def save_state(self, state: PatientState, now: Optional[datetime] = None) -> None:
        now = now or _now()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sessions SET state_json = ?, complaint = ?, updated_at = ? WHERE session_id = ?",
                (state.model_dump_json(), state.complaint, _iso(now), state.session_id))
            self._conn.commit()
        if cur.rowcount == 0:
            raise KeyError(f"Unknown session: {state.session_id}")

    def load_state(self, session_id: str) -> PatientState:
        with self._lock:
            row = self._conn.execute("SELECT state_json FROM sessions WHERE session_id = ?",
                                     (session_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown session: {session_id}")
        return PatientState.model_validate_json(row["state_json"])

    # ------------------------------------------------------------ messages
    def log_message(self, session_id: str, role: str, content: str, now: Optional[datetime] = None) -> None:
        if role not in ("user", "agent"):
            raise ValueError("role must be 'user' or 'agent'")
        with self._lock:
            self._conn.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
                               (session_id, role, content, _iso(now or _now())))
            self._conn.commit()

    def get_messages(self, session_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT role, content, created_at FROM messages "
                                      "WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ reports
    def save_report(self, session_id: str, report: dict, now: Optional[datetime] = None) -> None:
        """Store the final triage result and mark the session completed."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sessions SET report_json = ?, urgency = ?, status = 'completed', updated_at = ? "
                "WHERE session_id = ?",
                (json.dumps(report, default=str), report.get("urgency"), _iso(now or _now()), session_id))
            self._conn.commit()
        if cur.rowcount == 0:
            raise KeyError(f"Unknown session: {session_id}")

    def get_report(self, session_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute("SELECT report_json FROM sessions WHERE session_id = ?",
                                     (session_id,)).fetchone()
        return json.loads(row["report_json"]) if row and row["report_json"] else None

    def list_sessions(self, limit: int = 50) -> list[dict]:
        """Newest-urgency-first list (used by the doctor dashboard later)."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT session_id, patient_id, created_at, status, complaint, urgency FROM sessions "
                f"ORDER BY {URGENCY_RANK_SQL} DESC, updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ timeline and continuity
    def record_symptom_events(self, state: PatientState, now: Optional[datetime] = None) -> None:
        """Save symptoms reported so far. Re-running is safe (no duplicates)."""
        now = now or _now()
        if not state.patient_id:
            raise ValueError("state has no patient_id: create it with Memory.start_session()")
        with self._lock:
            for label, onset, from_duration in derive_symptom_events(state, now):
                action = "DO UPDATE SET onset_at = excluded.onset_at" if from_duration else "DO NOTHING"
                self._conn.execute(
                    "INSERT INTO symptom_events (session_id, patient_id, symptom, onset_at, recorded_at) "
                    f"VALUES (?,?,?,?,?) ON CONFLICT(session_id, symptom) {action}",
                    (state.session_id, state.patient_id, label, _iso(onset), _iso(now)))
            self._conn.commit()

    def get_timeline(self, patient_id: str, within_days: int = 30,
                     now: Optional[datetime] = None) -> list[dict]:
        """Timeline across all of this patient's recent sessions (earliest onset per symptom).
        Symptoms the patient said have gone away stop appearing as 'active' after that day."""
        now = now or _now()
        cutoff = _iso(now - timedelta(days=within_days))
        with self._lock:
            rows = self._conn.execute(
                "SELECT symptom, MIN(onset_at) AS onset_at, MAX(resolved_at) AS resolved_at, "
                "SUM(resolved_at IS NULL) AS open_count FROM symptom_events "
                "WHERE patient_id = ? AND onset_at >= ? GROUP BY symptom", (patient_id, cutoff)).fetchall()
        events = []
        for r in rows:
            # resolved only if EVERY record of that symptom was marked resolved
            resolved = _parse(r["resolved_at"]) if r["resolved_at"] and not r["open_count"] else None
            events.append((r["symptom"], _parse(r["onset_at"]), resolved))
        return build_timeline(events)

    def unresolved_earlier_symptoms(self, patient_id: str, exclude_session_id: Optional[str] = None,
                                    within_days: int = 30, now: Optional[datetime] = None) -> list[str]:
        """Labels reported in earlier sessions and not yet marked as gone."""
        now = now or _now()
        cutoff = _iso(now - timedelta(days=within_days))
        with self._lock:
            rows = self._conn.execute(
                "SELECT symptom, MIN(onset_at) AS first FROM symptom_events WHERE patient_id = ? "
                "AND session_id != ? AND resolved_at IS NULL AND onset_at >= ? GROUP BY symptom ORDER BY first",
                (patient_id, exclude_session_id or "", cutoff)).fetchall()
        return [r["symptom"] for r in rows]

    def mark_earlier_resolved(self, patient_id: str, exclude_session_id: Optional[str] = None,
                              now: Optional[datetime] = None) -> int:
        """The patient said earlier symptoms are gone. Returns how many records were updated."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE symptom_events SET resolved_at = ? WHERE patient_id = ? AND session_id != ? "
                "AND resolved_at IS NULL", (_iso(now or _now()), patient_id, exclude_session_id or ""))
            self._conn.commit()
        return cur.rowcount

    def timeline_text(self, patient_id: str, within_days: int = 30, now: Optional[datetime] = None) -> list[str]:
        return timeline_text(self.get_timeline(patient_id, within_days, now))

    def previous_sessions(self, patient_id: str, exclude_session_id: Optional[str] = None,
                          limit: int = 3, now: Optional[datetime] = None) -> list[dict]:
        now = now or _now()
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id, created_at, complaint, urgency FROM sessions "
                "WHERE patient_id = ? AND session_id != ? ORDER BY created_at DESC LIMIT ?",
                (patient_id, exclude_session_id or "", limit)).fetchall()
            out = []
            for r in rows:
                syms = [x["symptom"] for x in self._conn.execute(
                    "SELECT symptom FROM symptom_events WHERE session_id = ? ORDER BY onset_at",
                    (r["session_id"],)).fetchall()]
                out.append({"session_id": r["session_id"], "complaint": r["complaint"], "urgency": r["urgency"],
                            "days_ago": (now - _parse(r["created_at"])).days, "symptoms": syms})
        return out

    def continuity_note(self, patient_id: str, exclude_session_id: Optional[str] = None,
                        now: Optional[datetime] = None) -> str:
        """One short paragraph for the LLM prompt and the doctor summary ('' if first visit)."""
        prev = [p for p in self.previous_sessions(patient_id, exclude_session_id, now=now) if p["symptoms"]]
        if not prev:
            return ""
        parts = []
        for p in prev:
            when = "today" if p["days_ago"] == 0 else f"{p['days_ago']} day(s) ago"
            urg = f", assessed as {p['urgency']}" if p["urgency"] else ""
            parts.append(f"{when}: {', '.join(p['symptoms']).lower()}{urg}")
        return "Earlier sessions for this patient: " + "; ".join(parts) + "."

    # ------------------------------------------------------------ privacy
    def delete_patient(self, patient_id: str) -> None:
        """Erase a patient and everything linked to them (right to delete)."""
        with self._lock:
            self._conn.execute("DELETE FROM patients WHERE patient_id = ?", (patient_id,))
            self._conn.commit()

    def purge_older_than(self, days: int, now: Optional[datetime] = None) -> int:
        """Delete sessions older than `days` and patients left without sessions. Returns sessions removed."""
        cutoff = _iso((now or _now()) - timedelta(days=days))
        with self._lock:
            cur = self._conn.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,))
            self._conn.execute("DELETE FROM patients WHERE patient_id NOT IN (SELECT patient_id FROM sessions)")
            self._conn.commit()
        return cur.rowcount
