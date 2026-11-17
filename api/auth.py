from __future__ import annotations

from typing import Any

import jwt
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jwt import PyJWKClient
from supabase import Client, create_client

import config


if not config.SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL is missing")

if not config.SUPABASE_SECRET_KEY:
    raise RuntimeError("SUPABASE_SECRET_KEY is missing")

if not config.SUPABASE_JWKS_URL:
    raise RuntimeError("SUPABASE_JWKS_URL is missing")

if not config.SUPABASE_JWT_ISSUER:
    raise RuntimeError("SUPABASE_JWT_ISSUER is missing")


supabase_admin: Client = create_client(
    config.SUPABASE_URL,
    config.SUPABASE_SECRET_KEY,
)

jwks_client = PyJWKClient(config.SUPABASE_JWKS_URL)

ALGORITHMS = ["RS256", "ES256"]


def verify_access_token(token: str) -> dict[str, Any]:
    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)

        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=ALGORITHMS,
            audience="authenticated",
            issuer=config.SUPABASE_JWT_ISSUER,
            options={
                "require": [
                    "exp",
                    "iat",
                    "sub",
                    "aud",
                    "iss",
                ],
            },
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired access token",
        ) from exc

    if payload.get("role") != "authenticated":
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication role",
        )

    user_id = payload.get("sub")

    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(
            status_code=401,
            detail="Token has no user ID",
        )

    return payload


def get_profile(user_id: str) -> dict[str, Any]:
    response = (
        supabase_admin
        .table("profiles")
        .select("user_id, display_name, role")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )

    if not response.data:
        raise HTTPException(
            status_code=403,
            detail="User profile does not exist",
        )

    return response.data[0]


def authenticate(
    credentials: HTTPAuthorizationCredentials,
) -> dict[str, Any]:
    payload = verify_access_token(credentials.credentials)
    profile = get_profile(payload["sub"])

    return {
        "user_id": payload["sub"],
        "email": payload.get("email"),
        "role": profile["role"],
        "display_name": profile.get("display_name"),
    }