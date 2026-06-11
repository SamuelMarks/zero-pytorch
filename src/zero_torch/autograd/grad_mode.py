"Autograd context managers."

import threading


class GradMode(threading.local):
    """Thread-local storage for gradient computation mode."""

    def __init__(self):
        """Initializes the GradMode with gradients enabled."""
        self.enabled = True


_grad_mode = GradMode()


def is_grad_enabled() -> bool:
    """Returns True if grad mode is currently enabled.

    Returns:
        bool: True if gradients are enabled, False otherwise.
    """
    return getattr(_grad_mode, "enabled", True)


class set_grad_enabled:
    """Context-manager that sets gradient calculation on or off."""

    def __init__(self, mode: bool):
        """Initializes the context manager.

        Args:
            mode (bool): Flag whether to enable grad (True), or disable (False).
        """
        self.prev = is_grad_enabled()
        self.mode = mode

    def __enter__(self) -> None:
        """Enters the context manager and sets the specified mode."""
        _grad_mode.enabled = self.mode

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exits the context manager and restores the previous mode.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Exception traceback.
        """
        _grad_mode.enabled = self.prev


class no_grad(set_grad_enabled):
    """Context-manager that disabled gradient calculation."""

    def __init__(self):
        """Initializes the no_grad context manager."""
        super().__init__(False)
