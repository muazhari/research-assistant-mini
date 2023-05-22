import threading
from functools import wraps
from time import sleep

lock = threading.Lock()


def wait_lock(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        while lock.locked():
            sleep(0.001)
        with lock:
            result = func(*args, **kwargs)
        return result

    return wrapper
