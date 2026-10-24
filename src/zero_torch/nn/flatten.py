"""Flatten and Unflatten modules."""

from __future__ import annotations

from collections.abc import Sequence

import zero_torch
from zero_torch.tensor import Tensor

from .module import Module


class Flatten(Module):
    """Flattens a contiguous range of dims into a tensor."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        """Initializes Flatten.

        Args:
            start_dim: First dimension to flatten.
            end_dim: Last dimension to flatten.
        """
        super().__init__()
        self.start_dim: int = start_dim
        self.end_dim: int = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """Flattens `input` from `start_dim` to `end_dim`.

        Args:
            input: Input tensor.

        Returns:
            Flattened tensor.
        """
        return zero_torch.flatten(input, self.start_dim, self.end_dim)


class Unflatten(Module):
    """Unflattens a tensor dimension into multiple dimensions."""

    def __init__(self, dim: int, unflattened_size: int | Sequence[int]) -> None:
        """Initializes Unflatten.

        Args:
            dim: Dimension of the input tensor to be unflattened.
            unflattened_size: New shape of the unflattened dimension.
        """
        super().__init__()
        self.dim: int = dim
        self.unflattened_size: tuple[int, ...] = (
            (unflattened_size,)
            if isinstance(unflattened_size, int)
            else tuple(unflattened_size)
        )

    def forward(self, input: Tensor) -> Tensor:
        """Unflattens dimension `dim` of `input` into `unflattened_size`.

        Args:
            input: Input tensor.

        Returns:
            Unflattened tensor.
        """
        shape = list(input.shape)
        idx = self.dim if self.dim >= 0 else len(shape) + self.dim
        new_shape = tuple(shape[:idx]) + self.unflattened_size + tuple(shape[idx + 1 :])
        return zero_torch.reshape(input, new_shape)
