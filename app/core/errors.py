from datetime import timezone, datetime
UTC = timezone.utc

from fastapi.responses import JSONResponse


ERROR_CODES = {
    "VALIDATION_ERROR": 422,
    "FILE_FORMAT_UNSUPPORTED": 400,
    "PATIENT_NOT_FOUND": 404,
    "JOB_NOT_FOUND": 404,
    "NER_EXTRACTION_FAILED": 500,
    "SEGMENTATION_FAILED": 500,
    "LLM_TIMEOUT": 500,
    "RAG_RETRIEVAL_EMPTY": 500,
    "RATE_LIMIT_EXCEEDED": 429,
    "REQUEST_TIMEOUT": 504,
    "INTERNAL_ERROR": 500,
}


def format_error(code: str, message: str, status: int, details: dict | None = None):
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
                "timestamp": datetime.now(UTC).isoformat(),
            }
        },
    )

from fastapi import Request
from fastapi.exceptions import RequestValidationError

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={
        "error": {
            "code": "VALIDATION_ERROR",
            "message": str(exc.errors()[0]["msg"]),
            "details": exc.errors(),
            "retryable": False
        }
    })

async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "retryable": True
        }
    })
