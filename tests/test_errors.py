"""Unit tests for app/core/errors.py — exception handler response shaping."""
from __future__ import annotations

import json

import pytest
from fastapi.exceptions import RequestValidationError

from app.core.errors import (
    ERROR_CODES,
    format_error,
    general_exception_handler,
    validation_exception_handler,
)


def _body(response) -> dict:
    return json.loads(response.body)


def test_format_error_shape():
    response = format_error("PATIENT_NOT_FOUND", "no such patient", 404)
    assert response.status_code == 404
    body = _body(response)
    assert body["error"]["code"] == "PATIENT_NOT_FOUND"
    assert body["error"]["message"] == "no such patient"
    assert body["error"]["details"] == {}
    assert "timestamp" in body["error"]


def test_format_error_with_details():
    response = format_error("VALIDATION_ERROR", "bad input", 422, details={"field": "age"})
    body = _body(response)
    assert body["error"]["details"] == {"field": "age"}


def test_error_codes_map_expected_status():
    assert ERROR_CODES["RATE_LIMIT_EXCEEDED"] == 429
    assert ERROR_CODES["REQUEST_TIMEOUT"] == 504
    assert ERROR_CODES["JOB_NOT_FOUND"] == 404


@pytest.mark.asyncio
async def test_validation_exception_handler_returns_422():
    exc = RequestValidationError(
        [{"loc": ("body", "claim_text"), "msg": "field required", "type": "value_error.missing"}]
    )
    response = await validation_exception_handler(request=None, exc=exc)
    assert response.status_code == 422
    body = _body(response)
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert body["error"]["message"] == "field required"
    assert body["error"]["retryable"] is False
    assert len(body["error"]["details"]) == 1


@pytest.mark.asyncio
async def test_general_exception_handler_returns_500_and_hides_detail():
    response = await general_exception_handler(request=None, exc=RuntimeError("db password leaked"))
    assert response.status_code == 500
    body = _body(response)
    assert body["error"]["code"] == "INTERNAL_ERROR"
    # The raw exception message must never leak to the client.
    assert "db password leaked" not in json.dumps(body)
    assert body["error"]["retryable"] is True
