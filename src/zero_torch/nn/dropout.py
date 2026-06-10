"""Module."""

from typing import Any
from .module import Module
from zero_torch.tensor import Tensor


class Dropout(Module):
    """During training, randomly zeroes some of the elements of the input tensor with probability p."""

    def __init__(
        self, p: float = 0.5, inplace: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize."""
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        pass


class Dropout1d(Module):
    """Randomly zero out entire channels."""

    def __init__(
        self, p: Any = 0.5, inplace: Any = False, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize."""
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        pass


class Dropout2d(Module):
    """Randomly zero out entire channels."""

    def __init__(
        self, p: Any = 0.5, inplace: Any = False, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize."""
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        pass


class Dropout3d(Module):
    """Randomly zero out entire channels."""

    def __init__(
        self, p: Any = 0.5, inplace: Any = False, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize."""
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        pass


class AlphaDropout(Module):
    """Applies Alpha Dropout over the input."""

    def __init__(
        self, p: float = 0.5, inplace: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize."""
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        pass


class FeatureAlphaDropout(Module):
    """Randomly masks out entire channels."""

    def __init__(
        self, p: Any = 0.5, inplace: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize."""
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        pass
