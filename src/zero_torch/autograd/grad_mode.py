"Autograd context managers."

import threading


class GradMode(threading.local):
    """Class."""

    def __init__(self):
        """Function."""
        self.enabled = True


_grad_mode = GradMode()


def is_grad_enabled() -> bool:
    """Function."""
    return getattr(_grad_mode, "enabled", True)


class set_grad_enabled:
    """Class."""

    def __init__(self, mode: bool):
        """Function."""
        self.prev = is_grad_enabled()
        self.mode = mode

    def __enter__(self):
        """Function."""
        _grad_mode.enabled = self.mode

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Function."""
        _grad_mode.enabled = self.prev


class no_grad(set_grad_enabled):
    """Class."""

    def __init__(self):
        """Function."""
        super().__init__(False)
