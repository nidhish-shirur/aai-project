from __future__ import annotations

from pydantic import BaseModel, Field


class CreateAssessmentRequest(BaseModel):
    patient_code: str | None = Field(
        default=None,
        max_length=32,
    )


class MessageRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=2000,
    )