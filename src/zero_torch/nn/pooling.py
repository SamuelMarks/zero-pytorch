"Pooling layers."

from typing import Any
from .module import Module
from zero_torch.tensor import Tensor

from .functional_pooling import (
    adaptive_avg_pool1d,
    adaptive_avg_pool2d,
    adaptive_avg_pool3d,
)


class AdaptiveAvgPool1d(Module):
    """Applies a 1D adaptive average pooling over an input signal composed of several input planes."""

    def __init__(self, output_size: Any, *args: Any, **kwargs: Any) -> None:
        """Initialize the module.

        Args:
            output_size (Any): The target output size.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The pooled tensor.
        """
        return adaptive_avg_pool1d(input, output_size=self.output_size)


class AdaptiveAvgPool2d(Module):
    """Applies a 2D adaptive average pooling over an input signal composed of several input planes."""

    def __init__(self, output_size: Any, *args: Any, **kwargs: Any) -> None:
        """Initialize the module.

        Args:
            output_size (Any): The target output size.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The pooled tensor.
        """
        return adaptive_avg_pool2d(input, output_size=self.output_size)


class AdaptiveAvgPool3d(Module):
    """Applies a 3D adaptive average pooling over an input signal composed of several input planes."""

    def __init__(self, output_size: Any, *args: Any, **kwargs: Any) -> None:
        """Initialize the module.

        Args:
            output_size (Any): The target output size.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The pooled tensor.
        """
        return adaptive_avg_pool3d(input, output_size=self.output_size)


class AdaptiveMaxPool1d(Module):
    """Applies a adaptive max pooling over an input signal."""

    def __init__(
        self, output_size: Any, return_indices: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the module.

        Args:
            output_size (Any): The target output size.
            return_indices (bool, optional): Whether to return indices.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.output_size = output_size
        self.return_indices = return_indices

    def forward(self, input: Tensor) -> Any:
        """Forward pass.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Any: The pooled tensor or a tuple of (tensor, indices).
        """
        from .functional_pooling import adaptive_max_pool1d

        return adaptive_max_pool1d(
            input, output_size=self.output_size, return_indices=self.return_indices
        )


class AdaptiveMaxPool2d(Module):
    """Applies a adaptive max pooling over an input signal."""

    def __init__(
        self, output_size: Any, return_indices: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the module.

        Args:
            output_size (Any): The target output size.
            return_indices (bool, optional): Whether to return indices.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.output_size = output_size
        self.return_indices = return_indices

    def forward(self, input: Tensor) -> Any:
        """Forward pass.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Any: The pooled tensor or a tuple of (tensor, indices).
        """
        from .functional_pooling import adaptive_max_pool2d

        return adaptive_max_pool2d(
            input, output_size=self.output_size, return_indices=self.return_indices
        )


class AdaptiveMaxPool3d(Module):
    """Applies a adaptive max pooling over an input signal."""

    def __init__(
        self, output_size: Any, return_indices: bool = False, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the module.

        Args:
            output_size (Any): The target output size.
            return_indices (bool, optional): Whether to return indices.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.output_size = output_size
        self.return_indices = return_indices

    def forward(self, input: Tensor) -> Any:
        """Forward pass.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Any: The pooled tensor or a tuple of (tensor, indices).
        """
        from .functional_pooling import adaptive_max_pool3d

        return adaptive_max_pool3d(
            input, output_size=self.output_size, return_indices=self.return_indices
        )


class AvgPool1d(Module):
    """Applies a 1D average pooling over an input signal composed of several input planes."""

    def __init__(
        self,
        kernel_size: Any,
        stride: Any = None,
        padding: Any = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the module.

        Args:
            kernel_size: The size of the window.
            stride: The stride of the window.
            padding: Implicit zero padding to be added on both sides.
            ceil_mode: When True, will use ceil instead of floor to compute the output shape.
            count_include_pad: When True, will include the zero-padding in the averaging calculation.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The pooled tensor.
        """
        from .functional_pooling import avg_pool1d

        return avg_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )


class AvgPool2d(Module):
    """Applies a 2D average pooling over an input signal composed of several input planes."""

    def __init__(
        self,
        kernel_size: Any,
        stride: Any = None,
        padding: Any = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the module.

        Args:
            kernel_size: The size of the window.
            stride: The stride of the window.
            padding: Implicit zero padding to be added on both sides.
            ceil_mode: When True, will use ceil instead of floor to compute the output shape.
            count_include_pad: When True, will include the zero-padding in the averaging calculation.
            divisor_override: If specified, it will be used as divisor.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The pooled tensor.
        """
        from .functional_pooling import avg_pool2d

        return avg_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )


class AvgPool3d(Module):
    """Applies a 3D average pooling over an input signal composed of several input planes."""

    def __init__(
        self,
        kernel_size: Any,
        stride: Any = None,
        padding: Any = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the module.

        Args:
            kernel_size: The size of the window.
            stride: The stride of the window.
            padding: Implicit zero padding to be added on both sides.
            ceil_mode: When True, will use ceil instead of floor to compute the output shape.
            count_include_pad: When True, will include the zero-padding in the averaging calculation.
            divisor_override: If specified, it will be used as divisor.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The pooled tensor.
        """
        from .functional_pooling import avg_pool3d

        return avg_pool3d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )


class MaxPool1d(Module):
    """Applies a 1D max pooling over an input signal composed of several input planes."""

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Pooled tensor.
        """
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class FractionalMaxPool2d(Module):
    """Applies a 2D fractional max pooling."""

    def __init__(self, kernel_size: int, output_size: int = None) -> None:
        """Initialize FractionalMaxPool2d.

        Args:
            kernel_size (int): Size of the pooling window.
            output_size (int, optional): Target output size. Defaults to None.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Pooled tensor.
        """
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class MaxPool2d(Module):
    """Applies a 2D max pooling."""

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Pooled tensor.
        """
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class FractionalMaxPool3d(Module):
    """Applies a 3D fractional max pooling."""

    def __init__(self, kernel_size: int, output_size: int = None) -> None:
        """Initialize FractionalMaxPool3d.

        Args:
            kernel_size (int): Size of the pooling window.
            output_size (int, optional): Target output size. Defaults to None.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Pooled tensor.
        """
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class MaxPool3d(Module):
    """Applies a 3D max pooling."""

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Pooled tensor.
        """
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class LPPool1d(Module):
    """Applies a 1D power-average pooling."""

    def __init__(self, norm_type: float, kernel_size: int, stride: int = None) -> None:
        """Initialize LPPool1d.

        Args:
            norm_type (float): Power for the pooling.
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling window. Defaults to None.
        """
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Pooled tensor.
        """
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class LPPool2d(Module):
    """Applies a 2D power-average pooling."""

    def __init__(self, norm_type: float, kernel_size: int, stride: int = None) -> None:
        """Initialize LPPool2d.

        Args:
            norm_type (float): Power for the pooling.
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling window. Defaults to None.
        """
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Pooled tensor.
        """
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class LPPool3d(Module):
    """Applies a 3D power-average pooling."""

    def __init__(self, norm_type: float, kernel_size: int, stride: int = None) -> None:
        """Initialize LPPool3d.

        Args:
            norm_type (float): Power for the pooling.
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling window. Defaults to None.
        """
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Pooled tensor.
        """
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class MaxUnpool1d(Module):
    """Computes a partial inverse of MaxPool1d."""

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0) -> None:
        """Initialize MaxUnpool1d.

        Args:
            kernel_size (int): Size of the unpooling window.
            stride (int, optional): Stride of the window. Defaults to None.
            padding (int, optional): Implicit zero padding. Defaults to 0.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input: Tensor, indices: Tensor, output_size=None) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Unpooled tensor.
        """
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class MaxUnpool2d(Module):
    """Computes a partial inverse of MaxPool2d."""

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0) -> None:
        """Initialize MaxUnpool2d.

        Args:
            kernel_size (int): Size of the unpooling window.
            stride (int, optional): Stride of the window. Defaults to None.
            padding (int, optional): Implicit zero padding. Defaults to 0.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input: Tensor, indices: Tensor, output_size=None) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Unpooled tensor.
        """
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class MaxUnpool3d(Module):
    """Computes a partial inverse of MaxPool3d."""

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0) -> None:
        """Initialize MaxUnpool3d.

        Args:
            kernel_size (int): Size of the unpooling window.
            stride (int, optional): Stride of the window. Defaults to None.
            padding (int, optional): Implicit zero padding. Defaults to 0.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input: Tensor, indices: Tensor, output_size=None) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Unpooled tensor.
        """
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError
