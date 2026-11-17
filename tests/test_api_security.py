from __future__ import annotations

from unittest.mock import Mock

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from api import auth
from api.dependencies import require_admin, require_clinician
from api.dependencies import require_patient


def credentials(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials=token,
    )


def test_patient_cannot_access_clinician_route() -> None:
    with pytest.raises(HTTPException) as error:
        require_clinician({
            "user_id": "patient-id",
            "role": "patient",
        })

    assert error.value.status_code == 403


def test_reviewer_cannot_access_admin_route() -> None:
    with pytest.raises(HTTPException) as error:
        require_admin({
            "user_id": "reviewer-id",
            "role": "reviewer",
        })

    assert error.value.status_code == 403


def test_clinician_cannot_access_patient_route() -> None:
    with pytest.raises(HTTPException) as error:
        require_patient({
            "user_id": "clinician-id",
            "role": "clinician",
        })

    assert error.value.status_code == 403


def test_invalid_token_is_rejected(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise jwt.InvalidTokenError("invalid")

    monkeypatch.setattr(
        auth.jwks_client,
        "get_signing_key_from_jwt",
        fail,
    )

    with pytest.raises(HTTPException) as error:
        auth.verify_access_token("bad-token")

    assert error.value.status_code == 401


def test_authenticated_claim_is_required(monkeypatch) -> None:
    key = Mock()
    key.key = "secret"

    monkeypatch.setattr(
        auth.jwks_client,
        "get_signing_key_from_jwt",
        lambda token: key,
    )

    monkeypatch.setattr(
        jwt,
        "decode",
        lambda *args, **kwargs: {
            "sub": "user-id",
            "role": "anon",
        },
    )

    with pytest.raises(HTTPException) as error:
        auth.verify_access_token("token")

    assert error.value.status_code == 401