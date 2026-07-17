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
