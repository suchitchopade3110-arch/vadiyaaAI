import logging
import time
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger("vaidya")


def with_retry(
    max_retries: int = 2,
    backoff_seconds: float = 5.0,
    exceptions: tuple = (Exception,),
    fallback: Callable | None = None,
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        wait = backoff_seconds * (2 ** attempt)
                        logger.warning(
                            "Retry %d/%d for %s after %.1fs - %s",
                            attempt + 1,
                            max_retries,
                            func.__name__,
                            wait,
                            exc,
                        )
                        time.sleep(wait)
                    else:
                        logger.error(
                            "All retries exhausted for %s - %s",
                            func.__name__,
                            exc,
                        )
            if fallback:
                logger.warning("Using fallback for %s", func.__name__)
                return fallback(*args, **kwargs)
            raise last_exc

        return wrapper

    return decorator

