"Linear module."

from typing import Any
from zero_torch.tensor import Tensor
from .module import Module


class Linear(Module):
    """Applies an affine linear transformation to the incoming data."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize Linear module."""
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        pass
