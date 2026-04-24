from functools import wraps
from api.server import server

def require_model(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        server.require_ready()
        return func(*args, **kwargs)
    return wrapper