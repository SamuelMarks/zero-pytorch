"Autograd context managers."

import threading


class GradMode(threading.local):
    def __init__(self):
        self.enabled = True


_grad_mode = GradMode()


def is_grad_enabled() -> bool:
    return getattr(_grad_mode, "enabled", True)


class set_grad_enabled:
    def __init__(self, mode: bool):
        self.prev = is_grad_enabled()
        self.mode = mode

    def __enter__(self):
        _grad_mode.enabled = self.mode

    def __exit__(self, exc_type, exc_val, exc_tb):
        _grad_mode.enabled = self.prev


class no_grad(set_grad_enabled):
    def __init__(self):
        super().__init__(False)
