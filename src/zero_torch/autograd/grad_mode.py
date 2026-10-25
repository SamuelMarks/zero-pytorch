"""Autograd context managers."""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from types import TracebackType
from typing import TypeVar

from zero_torch._types import OpArg

_A = TypeVar("_A")
_R = TypeVar("_R")
_ = OpArg


class GradMode(threading.local):
    """Thread-local storage for gradient computation mode."""

    def __init__(self) -> None:
        """Initializes thread-local gradient mode with `enabled = True`."""
        self.enabled: bool = True


_grad_mode = GradMode()


def is_grad_enabled() -> bool:
    """Returns True if grad mode is currently enabled.

    Returns:
        True if gradient computation is enabled, False otherwise.
    """
    return _grad_mode.enabled


class set_grad_enabled:
    """Context-manager and decorator that sets gradient calculation on or off."""

    def __init__(self, mode: bool) -> None:
        """Initializes the gradient mode context manager.

        Args:
            mode: Whether to enable (`True`) or disable (`False`) gradient tracking.
        """
        self.prev: bool = _grad_mode.enabled
        self.new_mode: bool = mode
        _grad_mode.enabled = mode

    def __enter__(self) -> None:
        """Enters the gradient-mode context."""
        _grad_mode.enabled = self.new_mode

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Restores the previous gradient mode on exit.

        Args:
            exc_type: Exception type if raised, else None.
            exc_val: Exception instance if raised, else None.
            exc_tb: Traceback if raised, else None.
        """
        _ = (exc_type, exc_val, exc_tb)
        _grad_mode.enabled = self.prev

    def __call__(self, func: Callable[[_A], _R]) -> Callable[[_A], _R]:
        """Decorates a callable so it executes under `self.new_mode` gradient mode.

        Args:
            func: Callable to wrap.

        Returns:
            Wrapped callable that runs with gradient mode set to `self.new_mode`.
        """
        _grad_mode.enabled = self.prev
        target_mode = self.new_mode

        @functools.wraps(func)
        def wrapper(arg: _A) -> _R:
            """Executes `func` inside `set_grad_enabled(target_mode)`.

            Args:
                arg: Argument forwarded to `func`.

            Returns:
                Return value of `func`.
            """
            with set_grad_enabled(target_mode):
                return func(arg)

        return wrapper


class no_grad(set_grad_enabled):
    """Context-manager and decorator that disables gradient calculation."""

    def __init__(self) -> None:
        """Initializes the no_grad context manager."""
        super().__init__(False)
