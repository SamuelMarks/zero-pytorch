"""Lazy modules."""

from .module import Module


class LazyBatchNorm1d(Module):
    """Applies LazyBatchNorm1d."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self._module = None

    def forward(self, input):
        if self._module is None:
            from zero_torch.nn.norm import BatchNorm1d

            self._module = BatchNorm1d(
                input.shape[1],
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
            )
            self.add_module("batch_norm", self._module)
        return self._module(input)  # pragma: no cover


class LazyBatchNorm2d(Module):
    """Applies LazyBatchNorm2d."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self._module = None

    def forward(self, input):
        if self._module is None:
            from zero_torch.nn.norm import BatchNorm2d

            self._module = BatchNorm2d(
                input.shape[1],
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
            )
            self.add_module("batch_norm", self._module)
        return self._module(input)  # pragma: no cover


class LazyBatchNorm3d(Module):
    """Applies LazyBatchNorm3d."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self._module = None

    def forward(self, input):
        if self._module is None:
            from zero_torch.nn.norm import BatchNorm3d

            self._module = BatchNorm3d(
                input.shape[1],
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
            )
            self.add_module("batch_norm", self._module)
        return self._module(input)  # pragma: no cover


class LazyConv1d(Module):
    """Applies LazyConv1d."""

    def __init__(
        self,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self._module = None

    def forward(self, input):
        if self._module is None:
            from zero_torch.nn.conv import Conv1d

            self._module = Conv1d(
                input.shape[1],
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
                padding_mode=self.padding_mode,
            )
            self.add_module("conv", self._module)
        return self._module(input)  # pragma: no cover


class LazyConv2d(Module):
    """Applies LazyConv2d."""

    def __init__(
        self,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self._module = None

    def forward(self, input):
        if self._module is None:
            from zero_torch.nn.conv import Conv2d

            self._module = Conv2d(
                input.shape[1],
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
                padding_mode=self.padding_mode,
            )
            self.add_module("conv", self._module)
        return self._module(input)  # pragma: no cover


class LazyConv3d(Module):
    """Applies LazyConv3d."""

    def __init__(
        self,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self._module = None

    def forward(self, input):
        if self._module is None:
            from zero_torch.nn.conv import Conv3d

            self._module = Conv3d(
                input.shape[1],
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
                padding_mode=self.padding_mode,
            )
            self.add_module("conv", self._module)
        return self._module(input)  # pragma: no cover


class LazyConvTranspose1d(Module):
    """Applies LazyConvTranspose1d."""

    def __init__(
        self,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups: int = 1,
        bias: bool = True,
        dilation=1,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.dilation = dilation
        self.padding_mode = padding_mode
        self._module = None

    def forward(self, input, output_size=None):
        if self._module is None:
            from zero_torch.nn.conv import ConvTranspose1d

            self._module = ConvTranspose1d(
                input.shape[1],
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                groups=self.groups,
                bias=self.bias,
                dilation=self.dilation,
                padding_mode=self.padding_mode,
            )
            self.add_module("conv_transpose", self._module)
        return self._module(input, output_size=output_size)


class LazyConvTranspose2d(Module):
    """Applies LazyConvTranspose2d."""

    def __init__(
        self,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups: int = 1,
        bias: bool = True,
        dilation=1,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.dilation = dilation
        self.padding_mode = padding_mode
        self._module = None

    def forward(self, input, output_size=None):
        if self._module is None:
            from zero_torch.nn.conv import ConvTranspose2d

            self._module = ConvTranspose2d(
                input.shape[1],
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                groups=self.groups,
                bias=self.bias,
                dilation=self.dilation,
                padding_mode=self.padding_mode,
            )
            self.add_module("conv_transpose", self._module)
        return self._module(input, output_size=output_size)


class LazyConvTranspose3d(Module):
    """Applies LazyConvTranspose3d."""

    def __init__(
        self,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups: int = 1,
        bias: bool = True,
        dilation=1,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.dilation = dilation
        self.padding_mode = padding_mode
        self._module = None

    def forward(self, input, output_size=None):
        if self._module is None:
            from zero_torch.nn.conv import ConvTranspose3d

            self._module = ConvTranspose3d(
                input.shape[1],
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                groups=self.groups,
                bias=self.bias,
                dilation=self.dilation,
                padding_mode=self.padding_mode,
            )
            self.add_module("conv_transpose", self._module)
        return self._module(input, output_size=output_size)


class LazyInstanceNorm1d(Module):
    """Applies LazyInstanceNorm1d."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self._module = None

    def forward(self, input):
        if self._module is None:
            from zero_torch.nn.norm import InstanceNorm1d

            self._module = InstanceNorm1d(
                input.shape[1],
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
            )
            self.add_module("instance_norm", self._module)
        return self._module(input)  # pragma: no cover


class LazyInstanceNorm2d(Module):
    """Applies LazyInstanceNorm2d."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self._module = None

    def forward(self, input):
        if self._module is None:
            from zero_torch.nn.norm import InstanceNorm2d

            self._module = InstanceNorm2d(
                input.shape[1],
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
            )
            self.add_module("instance_norm", self._module)
        return self._module(input)  # pragma: no cover


class LazyInstanceNorm3d(Module):
    """Applies LazyInstanceNorm3d."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self._module = None

    def forward(self, input):
        if self._module is None:
            from zero_torch.nn.norm import InstanceNorm3d

            self._module = InstanceNorm3d(
                input.shape[1],
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
            )
            self.add_module("instance_norm", self._module)
        return self._module(input)  # pragma: no cover


class LazyLinear(Module):
    """Applies LazyLinear."""

    def __init__(self, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.out_features = out_features
        self.bias = bias
        self._module = None

    def forward(self, input):
        if self._module is None:
            from zero_torch.nn.linear import Linear

            self._module = Linear(input.shape[-1], self.out_features, bias=self.bias)
            self.add_module("linear", self._module)
        return self._module(input)  # pragma: no cover
