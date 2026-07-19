"""
VaidyaAI Middleware
- MaxBodySizeMiddleware: Enforces upload size limits at the ASGI layer.
- APIContractMiddleware: Injects API contract headers and body fields.
"""

import asyncio
import time
import uuid
import json
import logging
from datetime import datetime, timezone
UTC = timezone.utc
from starlette.types import ASGIApp, Scope, Receive, Send
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from app.core.config import settings
from app.core.errors import format_error
from app.core.logging import request_id_var

MAX_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
logger = logging.getLogger("vaidya")


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.scope.get("request_id", str(uuid.uuid4()))
        request.scope["request_id"] = request_id
        request.state.request_id = request_id
        request_id_var.set(request_id)
        response = await call_next(request)
        if "X-Request-ID" not in response.headers:
            response.headers["X-Request-ID"] = request_id
        return response


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Append-only HIPAA-technical-safeguard access log.

    Writes one audit_logs row for every request under a tracked prefix
    (claims, image/report analysis, patient history). Runs after
    CorrelationIDMiddleware (added later in main.py, so it wraps this one)
    so request.state.request_id is already set. Logs resource references
    only — never claim text, report contents, or image bytes.
    """

    _TRACKED_PREFIXES = (
        ("/api/v1/verify", "claim"),
        ("/api/v1/analyze/image", "image_analysis"),
        ("/api/v1/analyze/report", "report"),
        ("/patients", "patient_history"),
    )

    async def dispatch(self, request: Request, call_next):
        resource_type = next(
            (rtype for prefix, rtype in self._TRACKED_PREFIXES if request.url.path.startswith(prefix)),
            None,
        )
        if resource_type is None:
            return await call_next(request)

        response = await call_next(request)

        try:
            await self._write_log(request, response, resource_type)
        except Exception as exc:
            logger.warning("Audit log write failed (non-fatal): %s", exc)

        return response

    @staticmethod
    def _decode_user_id(request: Request):
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return None
        try:
            from jose import jwt

            payload = jwt.decode(auth.split(" ", 1)[1], settings.SECRET_KEY, algorithms=["HS256"])
            return payload.get("sub")
        except Exception:
            return None

    @staticmethod
    def _resource_id(request: Request):
        for key, value in request.path_params.items():
            if key != "patient_id" and value is not None:
                return str(value)
        return None

    async def _write_log(self, request: Request, response, resource_type: str):
        import uuid as uuid_module

        from app.db.session import AsyncSessionLocal
        from app.models.audit_log import AuditLog

        def _as_uuid(value):
            try:
                return uuid_module.UUID(str(value))
            except (TypeError, ValueError):
                return None

        request_id = getattr(request.state, "request_id", None)
        async with AsyncSessionLocal() as session:
            session.add(
                AuditLog(
                    user_id=_as_uuid(self._decode_user_id(request)),
                    action=f"{request.method.lower()}_{resource_type}",
                    resource_type=resource_type,
                    resource_id=self._resource_id(request),
                    patient_id=_as_uuid(request.path_params.get("patient_id")),
                    ip_address=request.client.host if request.client else None,
                    user_agent=request.headers.get("user-agent"),
                    status="success" if response.status_code < 400 else "failure",
                    details={
                        "request_id": request_id,
                        "path": request.url.path,
                        "method": request.method,
                        "status_code": response.status_code,
                    },
                )
            )
            await session.commit()


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Safety net only — long-running work belongs in Celery, not a request handler."""

    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(
                call_next(request), timeout=settings.REQUEST_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            request_id = getattr(request.state, "request_id", "unknown")
            logger.error(
                "Request timed out request_id=%s path=%s timeout=%ss",
                request_id, request.url.path, settings.REQUEST_TIMEOUT_SECONDS,
            )
            return format_error(
                "REQUEST_TIMEOUT",
                f"Request exceeded {settings.REQUEST_TIMEOUT_SECONDS}s time limit.",
                504,
            )


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            request_id = getattr(request.state, "request_id", "unknown")
            logger.error("Unhandled error request_id=%s error=%s", request_id, exc)
            return format_error("INTERNAL_ERROR", str(exc), 500)


class MaxBodySizeMiddleware:
    """Pure ASGI Middleware to reject oversized requests."""
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        content_length = 0
        for header, value in scope.get("headers", []):
            if header.lower() == b"content-length":
                content_length = int(value)
                break
        
        if content_length > MAX_BYTES:
            response = JSONResponse(
                status_code=413,
                content={
                    "detail": f"Request body exceeds {settings.MAX_UPLOAD_SIZE_MB} MB limit."
                },
            )
            return await response(scope, receive, send)

        return await self.app(scope, receive, send)


class APIContractMiddleware:
    """
    Pure ASGI Middleware for Day 3 API Contract Compliance.
    Handles Headers and Body injection safely.
    """
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        start_time = time.time()
        request_id = scope.get("request_id", str(uuid.uuid4()))
        scope["request_id"] = request_id

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = []
                # Filter out content-length because we might change body size
                for header, value in message.get("headers", []):
                    if header.lower() != b"content-length":
                        headers.append((header, value))
                
                headers.append((b"X-Medical-Disclaimer", b"AI-assisted analysis. NOT diagnostic."))
                
                process_time = (time.time() - start_time) * 1000
                headers.append((b"X-Process-Time-Ms", f"{process_time:.2f}".encode()))
                
                message["headers"] = headers
                await send(message)
            
            elif message["type"] == "http.response.body":
                try:
                    # Only attempt to modify if it looks like JSON
                    body = message.get("body", b"")
                    if body and body.startswith(b"{"):
                        content = json.loads(body.decode())
                        if isinstance(content, dict):
                            content["request_id"] = request_id
                            content["medical_disclaimer"] = settings.MEDICAL_DISCLAIMER
                            content["timestamp"] = datetime.now(UTC).isoformat()
                            
                            message["body"] = json.dumps(content).encode()
                except Exception:
                    pass
                await send(message)
            else:
                await send(message)

        await self.app(scope, receive, send_wrapper)
