"""Autograd context managers."""

from __future__ import annotations

import threading
from types import TracebackType


class GradMode(threading.local):
    """Thread-local storage for gradient computation mode."""

    enabled: bool = True

    def __init__(self) -> None:
        """Initializes the GradMode with gradients enabled."""
        self.enabled: bool = True


_grad_mode = GradMode()


def is_grad_enabled() -> bool:
    """Returns True if grad mode is currently enabled.

    Returns:
        True if gradients are enabled, False otherwise.
    """
    return _grad_mode.enabled


class set_grad_enabled:
    """Context-manager that sets gradient calculation on or off."""

    def __init__(self, mode: bool) -> None:
        """Initializes the context manager.

        Args:
            mode: Flag whether to enable grad (True) or disable (False).
        """
        self.prev: bool = is_grad_enabled()
        self.mode: bool = mode

    def __enter__(self) -> None:
        """Enters the context manager and sets the specified mode."""
        _grad_mode.enabled = self.mode

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exits the context manager and restores the previous mode.

        Args:
            exc_type: Exception type if raised inside the block.
            exc_val: Exception instance if raised inside the block.
            exc_tb: Exception traceback if raised inside the block.
        """
        _ = (exc_type, exc_val, exc_tb)
        _grad_mode.enabled = self.prev


class no_grad(set_grad_enabled):
    """Context-manager that disables gradient calculation."""

    def __init__(self) -> None:
        """Initializes the no_grad context manager."""
        super().__init__(False)
