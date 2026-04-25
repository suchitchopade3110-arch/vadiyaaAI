"""
MaxBodySizeMiddleware
Enforces MAX_UPLOAD_SIZE_MB at the ASGI layer.
Rejects oversized requests before they hit any route handler.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from app.core.config import settings

MAX_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
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
