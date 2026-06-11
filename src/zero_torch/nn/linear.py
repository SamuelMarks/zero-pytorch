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
        """Initialize Linear module.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
            device (Any, optional): The device on which to create the parameters. Defaults to None.
            dtype (Any, optional): The data type to use for parameters. Defaults to None.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        from .module import Parameter
        import zero_torch

        self.weight = Parameter(zero_torch.ones((out_features, in_features)))
        self.bias = Parameter(zero_torch.ones((out_features,)))

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the linear module.

        Args:
            input (Tensor): The input tensor to the linear layer.

        Returns:
            Tensor: The output tensor resulting from the linear transformation.
        """
        import zero_torch.nn.functional as F

        return F.linear(input, self.weight, self.bias)
