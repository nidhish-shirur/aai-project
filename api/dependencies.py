from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.auth import authenticate


bearer = HTTPBearer(auto_error=True)


def current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
) -> dict[str, Any]:
    return authenticate(credentials)


def require_patient(
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    if user["role"] != "patient":
        raise HTTPException(
            status_code=403,
            detail="Patient account required",
        )

    return user


def require_clinician(
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    if user["role"] not in {
        "clinician",
        "reviewer",
        "admin",
    }:
        raise HTTPException(
            status_code=403,
            detail="Clinician role required",
        )

    return user


def require_admin(
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    if user["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="Administrator role required",
        )

    return user