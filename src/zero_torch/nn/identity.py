"Identity module."

from typing import Any
from .module import Module


class Identity(Module):
    """A placeholder identity operator that is argument-insensitive."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the Identity module."""
        pass

    def forward(self, input: Any) -> Any:
        """Forward pass."""
        pass
