import threading
from functools import wraps

locks = {}


def wait_lock(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        locks[func.__name__] = locks.get(func.__name__, threading.Lock())
        with locks[func.__name__]:
            result = func(*args, **kwargs)
        return result

    return wrapper
