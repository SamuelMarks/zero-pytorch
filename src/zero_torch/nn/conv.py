"""Convolution modules."""

from .module import Module


class Conv1d(Module):
    """Applies a 1D convolution over an input signal."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        """Initializes Conv1d."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        from .module import Parameter
        import zero_torch

        if isinstance(kernel_size, int):
            k = (kernel_size,)
        else:
            k = kernel_size
        self.weight = Parameter(
            zero_torch.zeros((out_channels, in_channels // groups, k[0]))
        )
        if bias:
            self.bias = Parameter(zero_torch.zeros((out_channels,)))
        else:
            self.bias = None

    def forward(self, input):
        """Forward pass."""
        import zero_torch.nn.functional as F

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
    """Applies a 2D convolution over an input signal."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        """Initializes Conv2d."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        from .module import Parameter
        import zero_torch

        if isinstance(kernel_size, int):
            k = (kernel_size, kernel_size)
        else:
            k = kernel_size
        self.weight = Parameter(
            zero_torch.zeros((out_channels, in_channels // groups, k[0], k[1]))
        )
        if bias:
            self.bias = Parameter(zero_torch.zeros((out_channels,)))
        else:
            self.bias = None

    def forward(self, input):
        """Forward pass."""
        import zero_torch.nn.functional as F

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
    """Applies a 3D convolution over an input signal."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        """Initializes Conv3d."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        from .module import Parameter
        import zero_torch

        if isinstance(kernel_size, int):
            k = (kernel_size, kernel_size, kernel_size)
        else:
            k = kernel_size
        self.weight = Parameter(
            zero_torch.zeros((out_channels, in_channels // groups, k[0], k[1], k[2]))
        )
        if bias:
            self.bias = Parameter(zero_torch.zeros((out_channels,)))
        else:
            self.bias = None

    def forward(self, input):
        """Forward pass."""
        import zero_torch.nn.functional as F

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
    """Applies a 1D transposed convolution operator."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups: int = 1,
        bias: bool = True,
        dilation=1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        """Initializes ConvTranspose1d."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.dilation = dilation
        self.padding_mode = padding_mode

    def forward(self, input, output_size=None):
        """Forward pass."""
        import zero_torch.nn.functional as F

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
    """Applies a 2D transposed convolution operator."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups: int = 1,
        bias: bool = True,
        dilation=1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        """Initializes ConvTranspose2d."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.dilation = dilation
        self.padding_mode = padding_mode

    def forward(self, input, output_size=None):
        """Forward pass."""
        import zero_torch.nn.functional as F

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
    """Applies a 3D transposed convolution operator."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups: int = 1,
        bias: bool = True,
        dilation=1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        """Initializes ConvTranspose3d."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.dilation = dilation
        self.padding_mode = padding_mode

    def forward(self, input, output_size=None):
        """Forward pass."""
        import zero_torch.nn.functional as F

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
    """Implementation of the Unfold module."""

    def __init__(self, kernel_size, dilation=1, padding=0, stride=1) -> None:
        """Initializes Unfold."""
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input):
        """Forward pass."""
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class Fold(Module):
    """Implementation of the Fold module."""

    def __init__(
        self, output_size, kernel_size, dilation=1, padding=0, stride=1
    ) -> None:
        """Initializes Fold."""
        super().__init__()
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input):
        """Forward pass."""
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError
