"""Linear module."""

from __future__ import annotations

import math

import zero_torch
import zero_torch.nn.functional as F
from zero_torch._types import DeviceLike, DTypeLike, OpArg
from zero_torch.nn.init import _calculate_fan_in_and_fan_out, kaiming_uniform_, uniform_
from zero_torch.tensor import Tensor

from .module import Module, Parameter


class Linear(Module):
    """Applies an affine linear transformation to the incoming data: `y = xA^T + b`."""

    __constants__: list[str] = ["in_features", "out_features"]
    weight: Parameter | None = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
        *args: OpArg,
        **kwargs: OpArg,
    ) -> None:
        """Initializes the Linear module.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If set to False, the layer will not learn an additive bias.
            device: Desired device of the parameters.
            dtype: Desired data type of the parameters.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        _ = (device, dtype, args, kwargs)
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.weight: Parameter = Parameter(
            zero_torch.empty((out_features, in_features))
        )
        self.bias: Parameter | None = (
            Parameter(zero_torch.empty((out_features,))) if bias else None
        )
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets weight and bias using Kaiming uniform initialization."""
        assert self.weight is not None
        kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        """Applies the linear transformation to `input`.

        Args:
            input: Input tensor of shape `(*, in_features)`.

        Returns:
            Output tensor of shape `(*, out_features)`.
        """
        assert self.weight is not None
        return F.linear(input, self.weight, self.bias)


class Bilinear(Module):
    """Applies a bilinear transformation to the incoming data: `y = x_1^T A x_2 + b`."""

    __constants__: list[str] = ["in1_features", "in2_features", "out_features"]
    weight: Parameter | None = None

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes the Bilinear module.

        Args:
            in1_features: Size of each first input sample.
            in2_features: Size of each second input sample.
            out_features: Size of each output sample.
            bias: If set to False, the layer will not learn an additive bias.
            device: Desired device of the parameters.
            dtype: Desired data type of the parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.in1_features: int = in1_features
        self.in2_features: int = in2_features
        self.out_features: int = out_features
        self.weight: Parameter = Parameter(
            zero_torch.empty((out_features, in1_features, in2_features))
        )
        self.bias: Parameter | None = (
            Parameter(zero_torch.empty((out_features,))) if bias else None
        )
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets weight and bias using uniform bounds `[-1/sqrt(in1), 1/sqrt(in1)]`."""
        assert self.weight is not None
        bound = 1.0 / math.sqrt(self.in1_features) if self.in1_features > 0 else 0.0
        uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            uniform_(self.bias, -bound, bound)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        """Applies the bilinear transformation to `input1` and `input2`.

        Args:
            input1: First input tensor of shape `(*, in1_features)`.
            input2: Second input tensor of shape `(*, in2_features)`.

        Returns:
            Output tensor of shape `(*, out_features)`.
        """
        assert self.weight is not None
        return F.bilinear(input1, input2, self.weight, self.bias)
