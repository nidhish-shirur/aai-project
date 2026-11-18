from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


class CreateAssessmentRequest(BaseModel):
    patient_code: str | None = Field(default=None, max_length=32)
    patient_token: str | None = Field(default=None, max_length=200)


class MessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class ClinicianFeedbackRequest(BaseModel):
    outcome: Literal[
        "correct",
        "needs_review",
        "incorrect_urgency",
    ]
    corrected_urgency: Literal[
        "low",
        "moderate",
        "urgent",
        "emergency",
    ] | None = None
    comment: str | None = Field(
        default=None,
        max_length=2000,
    )