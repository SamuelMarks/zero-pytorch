"""Identity module."""

from __future__ import annotations

from zero_torch._types import OpArg
from zero_torch.tensor import Tensor

from .module import Module


class Identity(Module):
    """A placeholder identity operator that is argument-insensitive."""

    def __init__(self, *args: OpArg, **kwargs: OpArg) -> None:
        """Initializes the Identity module.

        Args:
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.
        """
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        """Returns the `input` tensor unmodified.

        Args:
            input: Input tensor.

        Returns:
            The input tensor unmodified.
        """
        return input
