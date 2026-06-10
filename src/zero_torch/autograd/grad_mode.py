"Autograd context managers."

import threading


class GradMode(threading.local):
    def __init__(self):
        pass


_grad_mode = GradMode()


def is_grad_enabled() -> bool:
    pass


class set_grad_enabled:
    def __init__(self, mode: bool):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class no_grad(set_grad_enabled):
    def __init__(self):
        pass
