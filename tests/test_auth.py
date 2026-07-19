"""Unit tests for app/core/auth.py — JWT access-token validation and RBAC.

Calls the dependency functions directly with a manually-built
HTTPAuthorizationCredentials, bypassing FastAPI's Depends() wiring — no
live server, DB, or Redis needed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt

from app.core.auth import get_current_user, require_role
from app.core.config import settings

ALGORITHM = "HS256"


def _make_token(claims: dict, expires_in: timedelta = timedelta(minutes=20)) -> str:
    payload = {**claims, "exp": datetime.now(timezone.utc) + expires_in}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


def _bearer(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


# ── get_current_user ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_token_rejected():
    with pytest.raises(HTTPException) as exc:
        await get_current_user(token=None)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_valid_access_token_returns_payload():
    token = _make_token({"sub": "user-1", "role": "clinician", "type": "access"})
    payload = await get_current_user(token=_bearer(token))
    assert payload["sub"] == "user-1"
    assert payload["role"] == "clinician"


@pytest.mark.asyncio
async def test_refresh_token_rejected_as_access_token():
    token = _make_token({"sub": "user-1", "jti": "abc", "type": "refresh"}, expires_in=timedelta(days=7))
    with pytest.raises(HTTPException) as exc:
        await get_current_user(token=_bearer(token))
    assert exc.value.status_code == 403
    assert "Invalid token type" in exc.value.detail


@pytest.mark.asyncio
async def test_expired_token_rejected():
    token = _make_token({"sub": "user-1", "role": "clinician", "type": "access"}, expires_in=timedelta(minutes=-5))
    with pytest.raises(HTTPException) as exc:
        await get_current_user(token=_bearer(token))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_token_signed_with_wrong_secret_rejected():
    bad_token = jwt.encode(
        {"sub": "user-1", "role": "clinician", "type": "access",
         "exp": datetime.now(timezone.utc) + timedelta(minutes=20)},
        "not-the-real-secret",
        algorithm=ALGORITHM,
    )
    with pytest.raises(HTTPException) as exc:
        await get_current_user(token=_bearer(bad_token))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_malformed_token_rejected():
    with pytest.raises(HTTPException) as exc:
        await get_current_user(token=_bearer("not-a-jwt"))
    assert exc.value.status_code == 403


# ── require_role ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_require_role_allows_matching_role():
    token = _make_token({"sub": "u", "role": "admin", "type": "access"})
    dependency = require_role("admin")
    payload = await dependency(user=await get_current_user(token=_bearer(token)))
    assert payload["role"] == "admin"


@pytest.mark.asyncio
async def test_require_role_denies_non_matching_role():
    token = _make_token({"sub": "u", "role": "clinician", "type": "access"})
    dependency = require_role("admin")
    with pytest.raises(HTTPException) as exc:
        await dependency(user=await get_current_user(token=_bearer(token)))
    assert exc.value.status_code == 403
    assert "Insufficient permissions" in exc.value.detail


@pytest.mark.asyncio
async def test_require_role_accepts_any_of_multiple_roles():
    token = _make_token({"sub": "u", "role": "reviewer", "type": "access"})
    dependency = require_role("admin", "reviewer")
    payload = await dependency(user=await get_current_user(token=_bearer(token)))
    assert payload["role"] == "reviewer"
