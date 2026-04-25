"""
VaidyaAI Middleware
- MaxBodySizeMiddleware: Enforces upload size limits at the ASGI layer.
- APIContractMiddleware: Injects API contract headers and body fields.
"""

import time
import uuid
import json
from datetime import datetime, UTC
from starlette.types import ASGIApp, Scope, Receive, Send
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from app.core.config import settings

MAX_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Rejects oversized requests before they hit any route handler."""
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            if int(content_length) > MAX_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": (
                            f"Request body exceeds {settings.MAX_UPLOAD_SIZE_MB} MB limit. "
                            f"Declared Content-Length: {int(content_length) / 1024 / 1024:.1f} MB."
                        )
                    },
                )
        return await call_next(request)


class APIContractMiddleware:
    """
    Pure ASGI Middleware for Day 3 API Contract Compliance.
    Handles Headers and Body injection without BaseHTTPMiddleware loop issues.
    """
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Scope-level request_id for logging
        scope["request_id"] = request_id

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"X-Medical-Disclaimer", b"AI-assisted analysis. NOT diagnostic."))
                headers.append((b"X-Request-ID", request_id.encode()))
                
                process_time = (time.time() - start_time) * 1000
                headers.append((b"X-Process-Time-Ms", f"{process_time:.2f}".encode()))
                
                message["headers"] = headers
                await send(message)
            
            elif message["type"] == "http.response.body":
                try:
                    content = json.loads(message["body"].decode())
                    if isinstance(content, dict):
                        content["request_id"] = request_id
                        content["medical_disclaimer"] = settings.MEDICAL_DISCLAIMER
                        content["timestamp"] = datetime.now(UTC).isoformat()
                        
                        new_body = json.dumps(content).encode()
                        message["body"] = new_body
                except:
                    pass
                await send(message)
            else:
                await send(message)

        await self.app(scope, receive, send_wrapper)
