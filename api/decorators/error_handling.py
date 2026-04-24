import asyncio
import logging
from functools import wraps
from fastapi import HTTPException

logger = logging.getLogger("api.errors")


def _translate(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=f"Resource missing: {exc}")
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(status_code=422, detail=f"Missing key: {exc}")
    logger.exception("unhandled error")
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


def handle_errors(func):
    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                raise _translate(exc) from exc
        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            raise _translate(exc) from exc
    return sync_wrapper
