"""Session state: everything the agent knows about the current patient.

Purpose: one validated object that the extractor fills, the red-flag gate and
rule engine read, and the memory module saves. Facts use three values:
True / False (answered) and None (not known yet).
"""
from __future__ import annotations

import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# These facts live as real attributes; every other fact lives in `facts`.
TOP_LEVEL_KEYS = {"age_years", "sex", "pregnant", "chronic_conditions"}


class PatientState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    patient_id: Optional[str] = None  # anonymous code linking sessions of the same person
    age_years: Optional[float] = Field(default=None, ge=0, le=120)
    sex: Optional[Literal["male", "female", "other"]] = None
    pregnant: Optional[bool] = None
    chronic_conditions: Optional[list[str]] = None  # None = not asked, [] = "none"
    complaint: Optional[str] = None  # protocol id, e.g. "fever"
    facts: dict[str, Any] = Field(default_factory=dict)
    unknown_fields: list[str] = Field(default_factory=list)  # patient said "don't know"
    user_messages: list[str] = Field(default_factory=list)
    questions_asked: int = 0

    # ---- updating -------------------------------------------------------
    def update(self, values: dict[str, Any]) -> None:
        """Merge extracted facts. None values never overwrite known facts."""
        for key, value in values.items():
            if value is None:
                continue
            if key in TOP_LEVEL_KEYS:
                setattr(self, key, value)
            else:
                self.facts[key] = value
            if key in self.unknown_fields:
                self.unknown_fields.remove(key)

    def mark_unknown(self, key: str) -> None:
        if key not in self.unknown_fields and self.get(key) is None:
            self.unknown_fields.append(key)

    def add_user_message(self, text: str) -> None:
        self.user_messages.append(text)

    # ---- reading --------------------------------------------------------
    def get(self, key: str) -> Any:
        return self.all_facts().get(key)

    def all_facts(self) -> dict[str, Any]:
        """Flat dictionary used by rules. The complaint id itself counts as a
        True fact (so a 'fever' complaint means fever == True)."""
        out: dict[str, Any] = dict(self.facts)
        for key in TOP_LEVEL_KEYS:
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.complaint and self.complaint != "other":
            out.setdefault(self.complaint, True)
        return out

    def text_blob(self) -> str:
        """All patient messages joined with sentence breaks (used by the red-flag text gate)."""
        return " . ".join(self.user_messages)

    @property
    def unknown_count(self) -> int:
        return len(self.unknown_fields)
