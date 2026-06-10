"ReLU and Sequential modules."

from typing import Any
from .module import Module
from zero_torch.tensor import Tensor


class ReLU(Module):
    """Applies the rectified linear unit function element-wise."""

    def __init__(
        self,
        inplace: bool = False,
        __constants__=["inplace"],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize ReLU module."""
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        pass


class Sequential(Module):
    """A sequential container."""

    def __init__(self, *args: Any) -> None:
        """Initialize Sequential module."""
        pass

    def forward(self, input: Any) -> Any:
        """Forward pass through all modules."""
        pass
