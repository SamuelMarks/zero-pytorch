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
        super().__init__()
        from .module import Parameter
        import zero_torch

        self.weight = Parameter(zero_torch.ones((out_features, in_features)))
        self.bias = Parameter(zero_torch.ones((out_features,)))

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        import zero_torch.nn.functional as F

        return F.linear(input, self.weight, self.bias)
