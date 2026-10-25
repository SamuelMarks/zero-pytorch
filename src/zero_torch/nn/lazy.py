"""Lazy neural network modules with deferred channel/feature initialization."""

from __future__ import annotations

from collections.abc import Sequence

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

from .conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from .linear import Linear
from .module import Module, Parameter
from .norm import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
)


class _LazyModuleBase(Module):
    """Typed base class for modules with lazy parameter/buffer materialization."""

    weight: Parameter | None = None
    bias: Parameter | None = None
    running_mean: Tensor | None = None
    running_var: Tensor | None = None
    num_batches_tracked: Tensor | None = None


def _materialize_onto_self(lazy_mod: _LazyModuleBase, impl_mod: Module) -> None:
    """Materializes parameters and buffers from `impl_mod` directly onto `lazy_mod`.

    Args:
        lazy_mod: Lazy module instance receiving parameters and buffers.
        impl_mod: Materialized underlying module instance.
    """
    for name, param in impl_mod._parameters.items():
        lazy_mod.register_parameter(name, param)
    for name, buf in impl_mod._buffers.items():
        lazy_mod.register_buffer(name, buf)
    lazy_mod.weight = impl_mod._parameters.get("weight")
    lazy_mod.bias = impl_mod._parameters.get("bias")
    lazy_mod.running_mean = impl_mod._buffers.get("running_mean")
    lazy_mod.running_var = impl_mod._buffers.get("running_var")
    lazy_mod.num_batches_tracked = impl_mod._buffers.get("num_batches_tracked")


class LazyBatchNorm1d(_LazyModuleBase):
    """A `BatchNorm1d` module with lazy initialization of `num_features`."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes LazyBatchNorm1d.

        Args:
            eps: Value added to the denominator for numerical stability.
            momentum: Value used for the running_mean and running_var computation.
            affine: If True, this module has learnable affine parameters.
            track_running_stats: If True, tracks running mean and variance.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.eps: float = eps
        self.momentum: float = momentum
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats
        self._module: BatchNorm1d | None = None

    def forward(self, input: Tensor) -> Tensor:
        """Initializes the underlying `BatchNorm1d` and applies it to `input`.

        Args:
            input: Input tensor of shape `(N, C)` or `(N, C, L)`.

        Returns:
            Normalized tensor.
        """
        if self._module is None:
            self._module = BatchNorm1d(
                input.shape[1],
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
            )
            _materialize_onto_self(self, self._module)
        self._module.training = self.training
        return self._module(input)


class LazyBatchNorm2d(_LazyModuleBase):
    """A `BatchNorm2d` module with lazy initialization of `num_features`."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes LazyBatchNorm2d.

        Args:
            eps: Value added to the denominator for numerical stability.
            momentum: Value used for the running_mean and running_var computation.
            affine: If True, this module has learnable affine parameters.
            track_running_stats: If True, tracks running mean and variance.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.eps: float = eps
        self.momentum: float = momentum
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats
        self._module: BatchNorm2d | None = None

    def forward(self, input: Tensor) -> Tensor:
        """Initializes the underlying `BatchNorm2d` and applies it to `input`.

        Args:
            input: Input tensor of shape `(N, C, H, W)`.

        Returns:
            Normalized tensor.
        """
        if self._module is None:
            self._module = BatchNorm2d(
                input.shape[1],
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
            )
            _materialize_onto_self(self, self._module)
        self._module.training = self.training
        return self._module(input)


class LazyBatchNorm3d(_LazyModuleBase):
    """A `BatchNorm3d` module with lazy initialization of `num_features`."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes LazyBatchNorm3d.

        Args:
            eps: Value added to the denominator for numerical stability.
            momentum: Value used for the running_mean and running_var computation.
            affine: If True, this module has learnable affine parameters.
            track_running_stats: If True, tracks running mean and variance.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.eps: float = eps
        self.momentum: float = momentum
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats
        self._module: BatchNorm3d | None = None

    def forward(self, input: Tensor) -> Tensor:
        """Initializes the underlying `BatchNorm3d` and applies it to `input`.

        Args:
            input: Input tensor of shape `(N, C, D, H, W)`.

        Returns:
            Normalized tensor.
        """
        if self._module is None:
            self._module = BatchNorm3d(
                input.shape[1],
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
            )
            _materialize_onto_self(self, self._module)
        self._module.training = self.training
        return self._module(input)


class LazyConv1d(_LazyModuleBase):
    """A `Conv1d` module with lazy initialization of `in_channels`."""

    def __init__(
        self,
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
        """Initializes LazyConv1d.

        Args:
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Padding added to both sides of the input.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections.
            bias: If True, adds a learnable bias to the output.
            padding_mode: Padding mode (`'zeros'`, `'reflect'`, `'replicate'`, `'circular'`).
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.out_channels: int = out_channels
        self.kernel_size: Size1d = kernel_size
        self.stride: Size1d = stride
        self.padding: PaddingArg1d = padding
        self.dilation: Size1d = dilation
        self.groups: int = groups
        self.bias_flag: bool = bias
        self.padding_mode: str = padding_mode
        self._module: Conv1d | None = None

    def forward(self, input: Tensor) -> Tensor:
        """Initializes the underlying `Conv1d` and applies it to `input`.

        Args:
            input: Input tensor of shape `(N, C_in, L_in)`.

        Returns:
            Convolved tensor.
        """
        if self._module is None:
            self._module = Conv1d(
                input.shape[1],
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias_flag,
                padding_mode=self.padding_mode,
            )
            _materialize_onto_self(self, self._module)
        return self._module(input)


class LazyConv2d(_LazyModuleBase):
    """A `Conv2d` module with lazy initialization of `in_channels`."""

    def __init__(
        self,
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
        """Initializes LazyConv2d.

        Args:
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Padding added to all four sides of the input.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections.
            bias: If True, adds a learnable bias to the output.
            padding_mode: Padding mode (`'zeros'`, `'reflect'`, `'replicate'`, `'circular'`).
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.out_channels: int = out_channels
        self.kernel_size: Size2d = kernel_size
        self.stride: Size2d = stride
        self.padding: PaddingArg2d = padding
        self.dilation: Size2d = dilation
        self.groups: int = groups
        self.bias_flag: bool = bias
        self.padding_mode: str = padding_mode
        self._module: Conv2d | None = None

    def forward(self, input: Tensor) -> Tensor:
        """Initializes the underlying `Conv2d` and applies it to `input`.

        Args:
            input: Input tensor of shape `(N, C_in, H_in, W_in)`.

        Returns:
            Convolved tensor.
        """
        if self._module is None:
            self._module = Conv2d(
                input.shape[1],
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias_flag,
                padding_mode=self.padding_mode,
            )
            _materialize_onto_self(self, self._module)
        return self._module(input)


class LazyConv3d(_LazyModuleBase):
    """A `Conv3d` module with lazy initialization of `in_channels`."""

    def __init__(
        self,
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
        """Initializes LazyConv3d.

        Args:
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Padding added to all six sides of the input.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections.
            bias: If True, adds a learnable bias to the output.
            padding_mode: Padding mode (`'zeros'`, `'reflect'`, `'replicate'`, `'circular'`).
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.out_channels: int = out_channels
        self.kernel_size: Size3d = kernel_size
        self.stride: Size3d = stride
        self.padding: PaddingArg3d = padding
        self.dilation: Size3d = dilation
        self.groups: int = groups
        self.bias_flag: bool = bias
        self.padding_mode: str = padding_mode
        self._module: Conv3d | None = None

    def forward(self, input: Tensor) -> Tensor:
        """Initializes the underlying `Conv3d` and applies it to `input`.

        Args:
            input: Input tensor of shape `(N, C_in, D_in, H_in, W_in)`.

        Returns:
            Convolved tensor.
        """
        if self._module is None:
            self._module = Conv3d(
                input.shape[1],
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias_flag,
                padding_mode=self.padding_mode,
            )
            _materialize_onto_self(self, self._module)
        return self._module(input)


class LazyConvTranspose1d(_LazyModuleBase):
    """A `ConvTranspose1d` module with lazy initialization of `in_channels`."""

    def __init__(
        self,
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
        """Initializes LazyConvTranspose1d.

        Args:
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Implicit zero-padding awareness.
            output_padding: Additional size added to one side of the output shape.
            groups: Number of blocked connections.
            bias: If True, adds a learnable bias to the output.
            dilation: Spacing between kernel elements.
            padding_mode: Padding mode (`'zeros'`).
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.out_channels: int = out_channels
        self.kernel_size: Size1d = kernel_size
        self.stride: Size1d = stride
        self.padding: Size1d = padding
        self.output_padding: Size1d = output_padding
        self.groups: int = groups
        self.bias_flag: bool = bias
        self.dilation: Size1d = dilation
        self.padding_mode: str = padding_mode
        self._module: ConvTranspose1d | None = None

    def forward(
        self, input: Tensor, output_size: Sequence[int] | None = None
    ) -> Tensor:
        """Initializes the underlying `ConvTranspose1d` on first call and applies it to `input`.

        Args:
            input: Input tensor.
            output_size: Optional target output spatial size.

        Returns:
            Transposed convolved tensor.
        """
        if self._module is None:
            self._module = ConvTranspose1d(
                input.shape[1],
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                groups=self.groups,
                bias=self.bias_flag,
                dilation=self.dilation,
                padding_mode=self.padding_mode,
            )
            _materialize_onto_self(self, self._module)
        return self._module.forward(input, output_size=output_size)


class LazyConvTranspose2d(_LazyModuleBase):
    """A `ConvTranspose2d` module with lazy initialization of `in_channels`."""

    def __init__(
        self,
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
        """Initializes LazyConvTranspose2d.

        Args:
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Implicit zero-padding awareness.
            output_padding: Additional size added to one side of each dimension in the output shape.
            groups: Number of blocked connections.
            bias: If True, adds a learnable bias to the output.
            dilation: Spacing between kernel elements.
            padding_mode: Padding mode (`'zeros'`).
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.out_channels: int = out_channels
        self.kernel_size: Size2d = kernel_size
        self.stride: Size2d = stride
        self.padding: Size2d = padding
        self.output_padding: Size2d = output_padding
        self.groups: int = groups
        self.bias_flag: bool = bias
        self.dilation: Size2d = dilation
        self.padding_mode: str = padding_mode
        self._module: ConvTranspose2d | None = None

    def forward(
        self, input: Tensor, output_size: Sequence[int] | None = None
    ) -> Tensor:
        """Initializes the underlying `ConvTranspose2d` on first call and applies it to `input`.

        Args:
            input: Input tensor.
            output_size: Optional target output spatial size.

        Returns:
            Transposed convolved tensor.
        """
        if self._module is None:
            self._module = ConvTranspose2d(
                input.shape[1],
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                groups=self.groups,
                bias=self.bias_flag,
                dilation=self.dilation,
                padding_mode=self.padding_mode,
            )
            _materialize_onto_self(self, self._module)
        return self._module.forward(input, output_size=output_size)


class LazyConvTranspose3d(_LazyModuleBase):
    """A `ConvTranspose3d` module with lazy initialization of `in_channels`."""

    def __init__(
        self,
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
        """Initializes LazyConvTranspose3d.

        Args:
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Implicit zero-padding awareness.
            output_padding: Additional size added to one side of each dimension in the output shape.
            groups: Number of blocked connections.
            bias: If True, adds a learnable bias to the output.
            dilation: Spacing between kernel elements.
            padding_mode: Padding mode (`'zeros'`).
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.out_channels: int = out_channels
        self.kernel_size: Size3d = kernel_size
        self.stride: Size3d = stride
        self.padding: Size3d = padding
        self.output_padding: Size3d = output_padding
        self.groups: int = groups
        self.bias_flag: bool = bias
        self.dilation: Size3d = dilation
        self.padding_mode: str = padding_mode
        self._module: ConvTranspose3d | None = None

    def forward(
        self, input: Tensor, output_size: Sequence[int] | None = None
    ) -> Tensor:
        """Initializes the underlying `ConvTranspose3d` on first call and applies it to `input`.

        Args:
            input: Input tensor.
            output_size: Optional target output spatial size.

        Returns:
            Transposed convolved tensor.
        """
        if self._module is None:
            self._module = ConvTranspose3d(
                input.shape[1],
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                groups=self.groups,
                bias=self.bias_flag,
                dilation=self.dilation,
                padding_mode=self.padding_mode,
            )
            _materialize_onto_self(self, self._module)
        return self._module.forward(input, output_size=output_size)


class LazyInstanceNorm1d(_LazyModuleBase):
    """An `InstanceNorm1d` module with lazy initialization of `num_features`."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes LazyInstanceNorm1d.

        Args:
            eps: Value added to the denominator for numerical stability.
            momentum: Value used for the running_mean and running_var computation.
            affine: If True, this module has learnable affine parameters.
            track_running_stats: If True, tracks running mean and variance.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.eps: float = eps
        self.momentum: float = momentum
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats
        self._module: InstanceNorm1d | None = None

    def forward(self, input: Tensor) -> Tensor:
        """Initializes the underlying `InstanceNorm1d` and applies it to `input`.

        Args:
            input: Input tensor of shape `(N, C, L)`.

        Returns:
            Normalized tensor.
        """
        if self._module is None:
            self._module = InstanceNorm1d(
                input.shape[1],
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
            )
            _materialize_onto_self(self, self._module)
        self._module.training = self.training
        return self._module(input)


class LazyInstanceNorm2d(_LazyModuleBase):
    """An `InstanceNorm2d` module with lazy initialization of `num_features`."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes LazyInstanceNorm2d.

        Args:
            eps: Value added to the denominator for numerical stability.
            momentum: Value used for the running_mean and running_var computation.
            affine: If True, this module has learnable affine parameters.
            track_running_stats: If True, tracks running mean and variance.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.eps: float = eps
        self.momentum: float = momentum
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats
        self._module: InstanceNorm2d | None = None

    def forward(self, input: Tensor) -> Tensor:
        """Initializes the underlying `InstanceNorm2d` and applies it to `input`.

        Args:
            input: Input tensor of shape `(N, C, H, W)`.

        Returns:
            Normalized tensor.
        """
        if self._module is None:
            self._module = InstanceNorm2d(
                input.shape[1],
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
            )
            _materialize_onto_self(self, self._module)
        self._module.training = self.training
        return self._module(input)


class LazyInstanceNorm3d(_LazyModuleBase):
    """An `InstanceNorm3d` module with lazy initialization of `num_features`."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes LazyInstanceNorm3d.

        Args:
            eps: Value added to the denominator for numerical stability.
            momentum: Value used for the running_mean and running_var computation.
            affine: If True, this module has learnable affine parameters.
            track_running_stats: If True, tracks running mean and variance.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.eps: float = eps
        self.momentum: float = momentum
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats
        self._module: InstanceNorm3d | None = None

    def forward(self, input: Tensor) -> Tensor:
        """Initializes the underlying `InstanceNorm3d` and applies it to `input`.

        Args:
            input: Input tensor of shape `(N, C, D, H, W)`.

        Returns:
            Normalized tensor.
        """
        if self._module is None:
            self._module = InstanceNorm3d(
                input.shape[1],
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
            )
            _materialize_onto_self(self, self._module)
        self._module.training = self.training
        return self._module(input)


class LazyLinear(_LazyModuleBase):
    """A `Linear` module where `in_features` is inferred on the first `forward` call."""

    def __init__(
        self,
        out_features: int,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes LazyLinear.

        Args:
            out_features: Size of each output sample.
            bias: If set to False, the layer will not learn an additive bias.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.out_features: int = out_features
        self.bias_flag: bool = bias
        self._module: Linear | None = None

    def forward(self, input: Tensor) -> Tensor:
        """Initializes the underlying `Linear` and applies it to `input`.

        Args:
            input: Input tensor of shape `(*, in_features)`.

        Returns:
            Output tensor of shape `(*, out_features)`.
        """
        if self._module is None:
            self._module = Linear(
                input.shape[-1], self.out_features, bias=self.bias_flag
            )
            _materialize_onto_self(self, self._module)
        return self._module(input)
