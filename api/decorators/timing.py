import time
import logging
from functools import wraps

logger = logging.getLogger("api.timing")


def timed(label: str = ""):
    def decorator(func):
        name = label or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            logger.info("%s took %.2f ms", name, (time.perf_counter() - t0) * 1000)
            return result
        return wrapper
    return decorator
