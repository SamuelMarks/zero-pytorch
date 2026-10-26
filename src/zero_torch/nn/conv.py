"""Convolution modules."""

from __future__ import annotations

from collections.abc import Sequence

import zero_torch
import zero_torch.nn.functional as F
from zero_torch._types import (
    DeviceLike,
    DTypeLike,
    PaddingArg1d,
    PaddingArg2d,
    PaddingArg3d,
    Size1d,
    Size2d,
    Size3d,
)
from zero_torch.tensor import Tensor

from .functional_pooling import fold as _fp_fold
from .functional_pooling import unfold as _fp_unfold
from .module import Module, Parameter


class Conv1d(Module):
    """Applies a 1D convolution over an input signal composed of several input planes."""

    transposed: bool = False
    output_padding: tuple[int, ...] = (0,)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size1d,
        stride: Size1d = 1,
        padding: PaddingArg1d = 0,
        dilation: Size1d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes Conv1d.

        Args:
            in_channels: Number of channels in the input signal.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Padding added to both sides of the input.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input channels to output channels.
            bias: If True, adds a learnable bias to the output.
            padding_mode: Padding mode (`'zeros'`, `'reflect'`, `'replicate'`, `'circular'`).
            device: Desired device of the parameters.
            dtype: Desired data type of the parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: Size1d = kernel_size
        self.stride: Size1d = stride
        self.padding: PaddingArg1d = padding
        self.dilation: Size1d = dilation
        self.groups: int = groups
        self.padding_mode: str = padding_mode

        k = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
        self.weight: Parameter = Parameter(
            zero_torch.zeros((out_channels, in_channels // groups, k[0]))
        )
        self.bias: Parameter | None = (
            Parameter(zero_torch.zeros((out_channels,))) if bias else None
        )
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 1D convolution to `input`.

        Args:
            input: Input tensor of shape `(N, C_in, L_in)`.

        Returns:
            Convolved tensor of shape `(N, C_out, L_out)`.
        """
        return F.conv1d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Conv2d(Module):
    """Applies a 2D convolution over an input signal composed of several input planes."""

    transposed: bool = False
    output_padding: tuple[int, ...] = (0, 0)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2d,
        stride: Size2d = 1,
        padding: PaddingArg2d = 0,
        dilation: Size2d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes Conv2d.

        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Padding added to all four sides of the input.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input channels to output channels.
            bias: If True, adds a learnable bias to the output.
            padding_mode: Padding mode (`'zeros'`, `'reflect'`, `'replicate'`, `'circular'`).
            device: Desired device of the parameters.
            dtype: Desired data type of the parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: Size2d = kernel_size
        self.stride: Size2d = stride
        self.padding: PaddingArg2d = padding
        self.dilation: Size2d = dilation
        self.groups: int = groups
        self.padding_mode: str = padding_mode

        k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.weight: Parameter = Parameter(
            zero_torch.zeros((out_channels, in_channels // groups, k[0], k[1]))
        )
        self.bias: Parameter | None = (
            Parameter(zero_torch.zeros((out_channels,))) if bias else None
        )
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 2D convolution to `input`.

        Args:
            input: Input tensor of shape `(N, C_in, H_in, W_in)`.

        Returns:
            Convolved tensor of shape `(N, C_out, H_out, W_out)`.
        """
        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Conv3d(Module):
    """Applies a 3D convolution over an input signal composed of several input planes."""

    transposed: bool = False
    output_padding: tuple[int, ...] = (0, 0, 0)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size3d,
        stride: Size3d = 1,
        padding: PaddingArg3d = 0,
        dilation: Size3d = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes Conv3d.

        Args:
            in_channels: Number of channels in the input volume.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Padding added to all six sides of the input.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input channels to output channels.
            bias: If True, adds a learnable bias to the output.
            padding_mode: Padding mode (`'zeros'`, `'reflect'`, `'replicate'`, `'circular'`).
            device: Desired device of the parameters.
            dtype: Desired data type of the parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: Size3d = kernel_size
        self.stride: Size3d = stride
        self.padding: PaddingArg3d = padding
        self.dilation: Size3d = dilation
        self.groups: int = groups
        self.padding_mode: str = padding_mode

        k = (
            (kernel_size, kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else kernel_size
        )
        self.weight: Parameter = Parameter(
            zero_torch.zeros((out_channels, in_channels // groups, k[0], k[1], k[2]))
        )
        self.bias: Parameter | None = (
            Parameter(zero_torch.zeros((out_channels,))) if bias else None
        )
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 3D convolution to `input`.

        Args:
            input: Input tensor of shape `(N, C_in, D_in, H_in, W_in)`.

        Returns:
            Convolved tensor of shape `(N, C_out, D_out, H_out, W_out)`.
        """
        return F.conv3d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ConvTranspose1d(Module):
    """Applies a 1D transposed convolution operator over an input signal."""

    transposed: bool = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size1d,
        stride: Size1d = 1,
        padding: Size1d = 0,
        output_padding: Size1d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Size1d = 1,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes ConvTranspose1d.

        Args:
            in_channels: Number of channels in the input signal.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Implicit zero-padding awareness.
            output_padding: Additional size added to one side of the output shape.
            groups: Number of blocked connections.
            bias: If True, adds a learnable bias to the output.
            dilation: Spacing between kernel elements.
            padding_mode: Padding mode (`'zeros'`).
            device: Desired device of the parameters.
            dtype: Desired data type of the parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: Size1d = kernel_size
        self.stride: Size1d = stride
        self.padding: Size1d = padding
        self.output_padding: Size1d = output_padding
        self.groups: int = groups
        self.dilation: Size1d = dilation
        self.padding_mode: str = padding_mode

        k = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
        self.weight: Parameter = Parameter(
            zero_torch.zeros((in_channels, out_channels // groups, k[0]))
        )
        self.bias: Parameter | None = (
            Parameter(zero_torch.zeros((out_channels,))) if bias else None
        )
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)

    def forward(
        self, input: Tensor, output_size: Sequence[int] | None = None
    ) -> Tensor:
        """Applies 1D transposed convolution to `input`.

        Args:
            input: Input tensor.
            output_size: Optional target output spatial size.

        Returns:
            Transposed convolved tensor.
        """
        _ = output_size
        return F.conv_transpose1d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )


class ConvTranspose2d(Module):
    """Applies a 2D transposed convolution operator over an input image."""

    transposed: bool = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2d,
        stride: Size2d = 1,
        padding: Size2d = 0,
        output_padding: Size2d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Size2d = 1,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes ConvTranspose2d.

        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Implicit zero-padding awareness.
            output_padding: Additional size added to one side of each dimension in the output shape.
            groups: Number of blocked connections.
            bias: If True, adds a learnable bias to the output.
            dilation: Spacing between kernel elements.
            padding_mode: Padding mode (`'zeros'`).
            device: Desired device of the parameters.
            dtype: Desired data type of the parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: Size2d = kernel_size
        self.stride: Size2d = stride
        self.padding: Size2d = padding
        self.output_padding: Size2d = output_padding
        self.groups: int = groups
        self.dilation: Size2d = dilation
        self.padding_mode: str = padding_mode

        k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.weight: Parameter = Parameter(
            zero_torch.zeros((in_channels, out_channels // groups, k[0], k[1]))
        )
        self.bias: Parameter | None = (
            Parameter(zero_torch.zeros((out_channels,))) if bias else None
        )
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)

    def forward(
        self, input: Tensor, output_size: Sequence[int] | None = None
    ) -> Tensor:
        """Applies 2D transposed convolution to `input`.

        Args:
            input: Input tensor.
            output_size: Optional target output spatial size.

        Returns:
            Transposed convolved tensor.
        """
        _ = output_size
        return F.conv_transpose2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )


class ConvTranspose3d(Module):
    """Applies a 3D transposed convolution operator over an input volume."""

    transposed: bool = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size3d,
        stride: Size3d = 1,
        padding: Size3d = 0,
        output_padding: Size3d = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Size3d = 1,
        padding_mode: str = "zeros",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes ConvTranspose3d.

        Args:
            in_channels: Number of channels in the input volume.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Implicit zero-padding awareness.
            output_padding: Additional size added to one side of each dimension in the output shape.
            groups: Number of blocked connections.
            bias: If True, adds a learnable bias to the output.
            dilation: Spacing between kernel elements.
            padding_mode: Padding mode (`'zeros'`).
            device: Desired device of the parameters.
            dtype: Desired data type of the parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: Size3d = kernel_size
        self.stride: Size3d = stride
        self.padding: Size3d = padding
        self.output_padding: Size3d = output_padding
        self.groups: int = groups
        self.dilation: Size3d = dilation
        self.padding_mode: str = padding_mode

        k = (
            (kernel_size, kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else kernel_size
        )
        self.weight: Parameter = Parameter(
            zero_torch.zeros((in_channels, out_channels // groups, k[0], k[1], k[2]))
        )
        self.bias: Parameter | None = (
            Parameter(zero_torch.zeros((out_channels,))) if bias else None
        )
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)

    def forward(
        self, input: Tensor, output_size: Sequence[int] | None = None
    ) -> Tensor:
        """Applies 3D transposed convolution to `input`.

        Args:
            input: Input tensor.
            output_size: Optional target output spatial size.

        Returns:
            Transposed convolved tensor.
        """
        _ = output_size
        return F.conv_transpose3d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )


class Unfold(Module):
    """Extracts sliding local blocks from a batched input tensor."""

    __constants__: list[str] = ["kernel_size", "dilation", "padding", "stride"]

    def __init__(
        self,
        kernel_size: Size2d,
        dilation: Size2d = 1,
        padding: Size2d = 0,
        stride: Size2d = 1,
    ) -> None:
        """Initializes Unfold.

        Args:
            kernel_size: Size of the sliding blocks.
            dilation: Parameter that controls the stride of elements within the neighborhood.
            padding: Implicit zero padding to be added on both sides of input.
            stride: Stride of the sliding blocks in the input spatial dimensions.
        """
        super().__init__()
        self.kernel_size: Size2d = kernel_size
        self.dilation: Size2d = dilation
        self.padding: Size2d = padding
        self.stride: Size2d = stride

    def forward(self, input: Tensor) -> Tensor:
        """Extracts sliding blocks from `input`.

        Args:
            input: 4D input tensor `(N, C, H, W)` or 3D `(C, H, W)`.

        Returns:
            Unfolded tensor `(N, C * prod(kernel_size), L)`.
        """
        return _fp_unfold(
            input,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )


class Fold(Module):
    """Combines an array of sliding local blocks into a large containing tensor."""

    __constants__: list[str] = [
        "output_size",
        "kernel_size",
        "dilation",
        "padding",
        "stride",
    ]

    def __init__(
        self,
        output_size: Size2d,
        kernel_size: Size2d,
        dilation: Size2d = 1,
        padding: Size2d = 0,
        stride: Size2d = 1,
    ) -> None:
        """Initializes Fold.

        Args:
            output_size: Shape of the spatial dimensions of the output.
            kernel_size: Size of the sliding blocks.
            dilation: Parameter that controls the stride of elements within the neighborhood.
            padding: Implicit zero padding added on both sides of input.
            stride: Stride of the sliding blocks in the input spatial dimensions.
        """
        super().__init__()
        self.output_size: Size2d = output_size
        self.kernel_size: Size2d = kernel_size
        self.dilation: Size2d = dilation
        self.padding: Size2d = padding
        self.stride: Size2d = stride

    def forward(self, input: Tensor) -> Tensor:
        """Combines sliding blocks in `input` into a spatial tensor.

        Args:
            input: 3D input tensor `(N, C * prod(kernel_size), L)` or 2D `(C * prod(kernel_size), L)`.

        Returns:
            Folded tensor `(N, C, H_out, W_out)`.
        """
        return _fp_fold(
            input,
            output_size=self.output_size,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
