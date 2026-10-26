"""Pooling layers."""

from __future__ import annotations

from collections.abc import Sequence

from zero_torch._types import (
    OpArg,
    OptSize1d,
    OptSize2d,
    OptSize3d,
    Size1d,
    Size2d,
    Size3d,
)
from zero_torch.tensor import Tensor

from .functional_pooling import (
    adaptive_avg_pool1d,
    adaptive_avg_pool2d,
    adaptive_avg_pool3d,
    adaptive_max_pool1d,
    adaptive_max_pool2d,
    adaptive_max_pool3d,
    avg_pool1d,
    avg_pool2d,
    avg_pool3d,
    fractional_max_pool2d,
    fractional_max_pool3d,
    lp_pool1d,
    lp_pool2d,
    lp_pool3d,
    max_pool1d,
    max_pool2d,
    max_pool3d,
    max_unpool1d,
    max_unpool2d,
    max_unpool3d,
)
from .module import Module


class AdaptiveAvgPool1d(Module):
    """Applies a 1D adaptive average pooling over an input signal."""

    def __init__(self, output_size: OptSize1d, *args: OpArg, **kwargs: OpArg) -> None:
        """Initializes AdaptiveAvgPool1d.

        Args:
            output_size: Target output size `L_out`.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__()
        self.output_size: OptSize1d = output_size

    def forward(self, input: Tensor) -> Tensor:
        """Applies 1D adaptive average pooling to `input`.

        Args:
            input: Input tensor.

        Returns:
            Pooled tensor.
        """
        return adaptive_avg_pool1d(input, output_size=self.output_size)


class AdaptiveAvgPool2d(Module):
    """Applies a 2D adaptive average pooling over an input signal."""

    def __init__(self, output_size: OptSize2d, *args: OpArg, **kwargs: OpArg) -> None:
        """Initializes AdaptiveAvgPool2d.

        Args:
            output_size: Target output spatial size `(H_out, W_out)` or `H_out`.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__()
        self.output_size: OptSize2d = output_size

    def forward(self, input: Tensor) -> Tensor:
        """Applies 2D adaptive average pooling to `input`.

        Args:
            input: Input tensor.

        Returns:
            Pooled tensor.
        """
        return adaptive_avg_pool2d(input, output_size=self.output_size)


class AdaptiveAvgPool3d(Module):
    """Applies a 3D adaptive average pooling over an input signal."""

    def __init__(self, output_size: OptSize3d, *args: OpArg, **kwargs: OpArg) -> None:
        """Initializes AdaptiveAvgPool3d.

        Args:
            output_size: Target output volumetric size `(D_out, H_out, W_out)` or `D_out`.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__()
        self.output_size: OptSize3d = output_size

    def forward(self, input: Tensor) -> Tensor:
        """Applies 3D adaptive average pooling to `input`.

        Args:
            input: Input tensor.

        Returns:
            Pooled tensor.
        """
        return adaptive_avg_pool3d(input, output_size=self.output_size)


class AdaptiveMaxPool1d(Module):
    """Applies a 1D adaptive max pooling over an input signal."""

    def __init__(
        self,
        output_size: OptSize1d,
        return_indices: bool = False,
        *args: OpArg,
        **kwargs: OpArg,
    ) -> None:
        """Initializes AdaptiveMaxPool1d.

        Args:
            output_size: Target output size `L_out`.
            return_indices: Whether to return argmax indices alongside pooled outputs.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__()
        self.output_size: OptSize1d = output_size
        self.return_indices: bool = return_indices

    def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Applies 1D adaptive max pooling to `input`.

        Args:
            input: Input tensor.

        Returns:
            Pooled tensor, or `(output, indices)` if `return_indices=True`.
        """
        return adaptive_max_pool1d(
            input, output_size=self.output_size, return_indices=self.return_indices
        )


class AdaptiveMaxPool2d(Module):
    """Applies a 2D adaptive max pooling over an input signal."""

    def __init__(
        self,
        output_size: OptSize2d,
        return_indices: bool = False,
        *args: OpArg,
        **kwargs: OpArg,
    ) -> None:
        """Initializes AdaptiveMaxPool2d.

        Args:
            output_size: Target output spatial size `(H_out, W_out)`.
            return_indices: Whether to return argmax indices alongside pooled outputs.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__()
        self.output_size: OptSize2d = output_size
        self.return_indices: bool = return_indices

    def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Applies 2D adaptive max pooling to `input`.

        Args:
            input: Input tensor.

        Returns:
            Pooled tensor, or `(output, indices)` if `return_indices=True`.
        """
        return adaptive_max_pool2d(
            input, output_size=self.output_size, return_indices=self.return_indices
        )


class AdaptiveMaxPool3d(Module):
    """Applies a 3D adaptive max pooling over an input signal."""

    def __init__(
        self,
        output_size: OptSize3d,
        return_indices: bool = False,
        *args: OpArg,
        **kwargs: OpArg,
    ) -> None:
        """Initializes AdaptiveMaxPool3d.

        Args:
            output_size: Target output volumetric size `(D_out, H_out, W_out)`.
            return_indices: Whether to return argmax indices alongside pooled outputs.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__()
        self.output_size: OptSize3d = output_size
        self.return_indices: bool = return_indices

    def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Applies 3D adaptive max pooling to `input`.

        Args:
            input: Input tensor.

        Returns:
            Pooled tensor, or `(output, indices)` if `return_indices=True`.
        """
        return adaptive_max_pool3d(
            input, output_size=self.output_size, return_indices=self.return_indices
        )


class AvgPool1d(Module):
    """Applies a 1D average pooling over an input signal."""

    def __init__(
        self,
        kernel_size: Size1d,
        stride: Size1d | None = None,
        padding: Size1d = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        *args: OpArg,
        **kwargs: OpArg,
    ) -> None:
        """Initializes AvgPool1d.

        Args:
            kernel_size: Size of the pooling window.
            stride: Stride of the pooling window.
            padding: Implicit zero padding added on both sides.
            ceil_mode: When True, uses ceil instead of floor to compute output shape.
            count_include_pad: When True, includes zero-padding in the averaging calculation.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__()
        self.kernel_size: Size1d = kernel_size
        self.stride: Size1d = stride if stride is not None else kernel_size
        self.padding: Size1d = padding
        self.ceil_mode: bool = ceil_mode
        self.count_include_pad: bool = count_include_pad

    def forward(self, input: Tensor) -> Tensor:
        """Applies 1D average pooling to `input`.

        Args:
            input: Input tensor.

        Returns:
            Pooled tensor.
        """
        return avg_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )


class AvgPool2d(Module):
    """Applies a 2D average pooling over an input signal."""

    __constants__: list[str] = [
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
        "divisor_override",
    ]

    def __init__(
        self,
        kernel_size: Size2d,
        stride: Size2d | None = None,
        padding: Size2d = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
        *args: OpArg,
        **kwargs: OpArg,
    ) -> None:
        """Initializes AvgPool2d.

        Args:
            kernel_size: Size of the pooling window.
            stride: Stride of the pooling window.
            padding: Implicit zero padding added on both sides.
            ceil_mode: When True, uses ceil instead of floor to compute output shape.
            count_include_pad: When True, includes zero-padding in the averaging calculation.
            divisor_override: If specified, used as divisor instead of `kernel_size`.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__()
        self.kernel_size: Size2d = kernel_size
        self.stride: Size2d = stride if stride is not None else kernel_size
        self.padding: Size2d = padding
        self.ceil_mode: bool = ceil_mode
        self.count_include_pad: bool = count_include_pad
        self.divisor_override: int | None = divisor_override

    def forward(self, input: Tensor) -> Tensor:
        """Applies 2D average pooling to `input`.

        Args:
            input: Input tensor.

        Returns:
            Pooled tensor.
        """
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
    """Applies a 3D average pooling over an input signal."""

    __constants__: list[str] = [
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
        "divisor_override",
    ]

    def __init__(
        self,
        kernel_size: Size3d,
        stride: Size3d | None = None,
        padding: Size3d = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
        *args: OpArg,
        **kwargs: OpArg,
    ) -> None:
        """Initializes AvgPool3d.

        Args:
            kernel_size: Size of the pooling window.
            stride: Stride of the pooling window.
            padding: Implicit zero padding added on both sides.
            ceil_mode: When True, uses ceil instead of floor to compute output shape.
            count_include_pad: When True, includes zero-padding in the averaging calculation.
            divisor_override: If specified, used as divisor instead of `kernel_size`.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        _ = (args, kwargs)
        super().__init__()
        self.kernel_size: Size3d = kernel_size
        self.stride: Size3d = stride if stride is not None else kernel_size
        self.padding: Size3d = padding
        self.ceil_mode: bool = ceil_mode
        self.count_include_pad: bool = count_include_pad
        self.divisor_override: int | None = divisor_override

    def forward(self, input: Tensor) -> Tensor:
        """Applies 3D average pooling to `input`.

        Args:
            input: Input tensor.

        Returns:
            Pooled tensor.
        """
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
    """Applies a 1D max pooling over an input signal."""

    def __init__(
        self,
        kernel_size: Size1d,
        stride: Size1d | None = None,
        padding: Size1d = 0,
        dilation: Size1d = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        """Initializes MaxPool1d.

        Args:
            kernel_size: Size of the pooling window.
            stride: Stride of the pooling window.
            padding: Implicit negative infinity padding added on both sides.
            dilation: Parameter that controls the stride of elements in the window.
            return_indices: If True, returns the max indices along with the outputs.
            ceil_mode: When True, uses ceil instead of floor to compute output shape.
        """
        super().__init__()
        self.kernel_size: Size1d = kernel_size
        self.stride: Size1d | None = stride
        self.padding: Size1d = padding
        self.dilation: Size1d = dilation
        self.return_indices: bool = return_indices
        self.ceil_mode: bool = ceil_mode

    def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Applies 1D max pooling to `input`.

        Args:
            input: Input tensor.

        Returns:
            Pooled tensor, or `(output, indices)` if `return_indices=True`.
        """
        return max_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )


class FractionalMaxPool2d(Module):
    """Applies a 2D fractional max pooling over an input signal."""

    __constants__: list[str] = [
        "kernel_size",
        "return_indices",
        "output_size",
        "output_ratio",
    ]

    def __init__(
        self,
        kernel_size: Size2d,
        output_size: Size2d | None = None,
        output_ratio: float | tuple[float, float] | None = None,
        return_indices: bool = False,
        _random_samples: Tensor | None = None,
    ) -> None:
        """Initializes FractionalMaxPool2d.

        Args:
            kernel_size: Size of the pooling window.
            output_size: Target output spatial size `(H_out, W_out)`.
            output_ratio: Target output spatial ratio in `(0, 1)`.
            return_indices: If True, returns the indices along with the outputs.
            _random_samples: Optional pre-generated random step samples.
        """
        super().__init__()
        self.kernel_size: Size2d = kernel_size
        self.output_size: Size2d | None = output_size
        self.output_ratio: float | tuple[float, float] | None = output_ratio
        self.return_indices: bool = return_indices
        self._random_samples: Tensor | None = _random_samples

    def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Applies 2D fractional max pooling to `input`.

        Args:
            input: Input tensor.

        Returns:
            Pooled tensor, or `(output, indices)` if `return_indices=True`.
        """
        return fractional_max_pool2d(
            input,
            kernel_size=self.kernel_size,
            output_size=self.output_size,
            output_ratio=self.output_ratio,
            return_indices=self.return_indices,
            _random_samples=self._random_samples,
        )


class MaxPool2d(Module):
    """Applies a 2D max pooling over an input signal."""

    def __init__(
        self,
        kernel_size: Size2d,
        stride: Size2d | None = None,
        padding: Size2d = 0,
        dilation: Size2d = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        """Initializes MaxPool2d.

        Args:
            kernel_size: Size of the pooling window.
            stride: Stride of the pooling window.
            padding: Implicit negative infinity padding added on both sides.
            dilation: Parameter that controls the stride of elements in the window.
            return_indices: If True, returns the max indices along with the outputs.
            ceil_mode: When True, uses ceil instead of floor to compute output shape.
        """
        super().__init__()
        self.kernel_size: Size2d = kernel_size
        self.stride: Size2d | None = stride
        self.padding: Size2d = padding
        self.dilation: Size2d = dilation
        self.return_indices: bool = return_indices
        self.ceil_mode: bool = ceil_mode

    def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Applies 2D max pooling to `input`.

        Args:
            input: Input tensor.

        Returns:
            Pooled tensor, or `(output, indices)` if `return_indices=True`.
        """
        return max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )


class FractionalMaxPool3d(Module):
    """Applies a 3D fractional max pooling over an input signal."""

    __constants__: list[str] = [
        "kernel_size",
        "return_indices",
        "output_size",
        "output_ratio",
    ]

    def __init__(
        self,
        kernel_size: Size3d,
        output_size: Size3d | None = None,
        output_ratio: float | tuple[float, float, float] | None = None,
        return_indices: bool = False,
        _random_samples: Tensor | None = None,
    ) -> None:
        """Initializes FractionalMaxPool3d.

        Args:
            kernel_size: Size of the pooling window.
            output_size: Target output volumetric size `(D_out, H_out, W_out)`.
            output_ratio: Target output volumetric ratio in `(0, 1)`.
            return_indices: If True, returns the indices along with the outputs.
            _random_samples: Optional pre-generated random step samples.
        """
        super().__init__()
        self.kernel_size: Size3d = kernel_size
        self.output_size: Size3d | None = output_size
        self.output_ratio: float | tuple[float, float, float] | None = output_ratio
        self.return_indices: bool = return_indices
        self._random_samples: Tensor | None = _random_samples

    def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Applies 3D fractional max pooling to `input`.

        Args:
            input: Input tensor.

        Returns:
            Pooled tensor, or `(output, indices)` if `return_indices=True`.
        """
        return fractional_max_pool3d(
            input,
            kernel_size=self.kernel_size,
            output_size=self.output_size,
            output_ratio=self.output_ratio,
            return_indices=self.return_indices,
            _random_samples=self._random_samples,
        )


class MaxPool3d(Module):
    """Applies a 3D max pooling over an input signal."""

    def __init__(
        self,
        kernel_size: Size3d,
        stride: Size3d | None = None,
        padding: Size3d = 0,
        dilation: Size3d = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        """Initializes MaxPool3d.

        Args:
            kernel_size: Size of the pooling window.
            stride: Stride of the pooling window.
            padding: Implicit negative infinity padding added on both sides.
            dilation: Parameter that controls the stride of elements in the window.
            return_indices: If True, returns the max indices along with the outputs.
            ceil_mode: When True, uses ceil instead of floor to compute output shape.
        """
        super().__init__()
        self.kernel_size: Size3d = kernel_size
        self.stride: Size3d | None = stride
        self.padding: Size3d = padding
        self.dilation: Size3d = dilation
        self.return_indices: bool = return_indices
        self.ceil_mode: bool = ceil_mode

    def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Applies 3D max pooling to `input`.

        Args:
            input: Input tensor.

        Returns:
            Pooled tensor, or `(output, indices)` if `return_indices=True`.
        """
        return max_pool3d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )


class LPPool1d(Module):
    """Applies a 1D power-average pooling over an input signal."""

    def __init__(
        self,
        norm_type: float,
        kernel_size: Size1d,
        stride: Size1d | None = None,
        ceil_mode: bool = False,
    ) -> None:
        """Initializes LPPool1d.

        Args:
            norm_type: Exponent `p` for the LP-pooling.
            kernel_size: Size of the pooling window.
            stride: Stride of the pooling window.
            ceil_mode: When True, uses ceil instead of floor to compute output shape.
        """
        super().__init__()
        self.norm_type: float = norm_type
        self.kernel_size: Size1d = kernel_size
        self.stride: Size1d | None = stride
        self.ceil_mode: bool = ceil_mode

    def forward(self, input: Tensor) -> Tensor:
        """Applies 1D power-average pooling to `input`.

        Args:
            input: Input tensor.

        Returns:
            Pooled tensor.
        """
        return lp_pool1d(
            input, self.norm_type, self.kernel_size, self.stride, self.ceil_mode
        )


class LPPool2d(Module):
    """Applies a 2D power-average pooling over an input signal."""

    def __init__(
        self,
        norm_type: float,
        kernel_size: Size2d,
        stride: Size2d | None = None,
        ceil_mode: bool = False,
    ) -> None:
        """Initializes LPPool2d.

        Args:
            norm_type: Exponent `p` for the LP-pooling.
            kernel_size: Size of the pooling window.
            stride: Stride of the pooling window.
            ceil_mode: When True, uses ceil instead of floor to compute output shape.
        """
        super().__init__()
        self.norm_type: float = norm_type
        self.kernel_size: Size2d = kernel_size
        self.stride: Size2d | None = stride
        self.ceil_mode: bool = ceil_mode

    def forward(self, input: Tensor) -> Tensor:
        """Applies 2D power-average pooling to `input`.

        Args:
            input: Input tensor.

        Returns:
            Pooled tensor.
        """
        return lp_pool2d(
            input, self.norm_type, self.kernel_size, self.stride, self.ceil_mode
        )


class LPPool3d(Module):
    """Applies a 3D power-average pooling over an input signal."""

    def __init__(
        self,
        norm_type: float,
        kernel_size: Size3d,
        stride: Size3d | None = None,
        ceil_mode: bool = False,
    ) -> None:
        """Initializes LPPool3d.

        Args:
            norm_type: Exponent `p` for the LP-pooling.
            kernel_size: Size of the pooling window.
            stride: Stride of the pooling window.
            ceil_mode: When True, uses ceil instead of floor to compute output shape.
        """
        super().__init__()
        self.norm_type: float = norm_type
        self.kernel_size: Size3d = kernel_size
        self.stride: Size3d | None = stride
        self.ceil_mode: bool = ceil_mode

    def forward(self, input: Tensor) -> Tensor:
        """Applies 3D power-average pooling to `input`.

        Args:
            input: Input tensor.

        Returns:
            Pooled tensor.
        """
        return lp_pool3d(
            input, self.norm_type, self.kernel_size, self.stride, self.ceil_mode
        )


class MaxUnpool1d(Module):
    """Computes a partial inverse of `MaxPool1d`."""

    def __init__(
        self,
        kernel_size: Size1d,
        stride: Size1d | None = None,
        padding: Size1d = 0,
    ) -> None:
        """Initializes MaxUnpool1d.

        Args:
            kernel_size: Size of the max pooling window.
            stride: Stride of the max pooling window.
            padding: Padding that was added to the input.
        """
        super().__init__()
        self.kernel_size: Size1d = kernel_size
        self.stride: Size1d | None = stride
        self.padding: Size1d = padding

    def forward(
        self,
        input: Tensor,
        indices: Tensor,
        output_size: Sequence[int] | None = None,
    ) -> Tensor:
        """Applies 1D max unpooling to `input` using `indices`.

        Args:
            input: Pooled input tensor to invert.
            indices: Indices given out by `MaxPool1d`.
            output_size: Optional targeted output size.

        Returns:
            Unpooled tensor.
        """
        return max_unpool1d(
            input,
            indices,
            self.kernel_size,
            self.stride,
            self.padding,
            output_size=output_size,
        )


class MaxUnpool2d(Module):
    """Computes a partial inverse of `MaxPool2d`."""

    def __init__(
        self,
        kernel_size: Size2d,
        stride: Size2d | None = None,
        padding: Size2d = 0,
    ) -> None:
        """Initializes MaxUnpool2d.

        Args:
            kernel_size: Size of the max pooling window.
            stride: Stride of the max pooling window.
            padding: Padding that was added to the input.
        """
        super().__init__()
        self.kernel_size: Size2d = kernel_size
        self.stride: Size2d | None = stride
        self.padding: Size2d = padding

    def forward(
        self,
        input: Tensor,
        indices: Tensor,
        output_size: Sequence[int] | None = None,
    ) -> Tensor:
        """Applies 2D max unpooling to `input` using `indices`.

        Args:
            input: Pooled input tensor to invert.
            indices: Indices given out by `MaxPool2d`.
            output_size: Optional targeted output size.

        Returns:
            Unpooled tensor.
        """
        return max_unpool2d(
            input,
            indices,
            self.kernel_size,
            self.stride,
            self.padding,
            output_size=output_size,
        )


class MaxUnpool3d(Module):
    """Computes a partial inverse of `MaxPool3d`."""

    def __init__(
        self,
        kernel_size: Size3d,
        stride: Size3d | None = None,
        padding: Size3d = 0,
    ) -> None:
        """Initializes MaxUnpool3d.

        Args:
            kernel_size: Size of the max pooling window.
            stride: Stride of the max pooling window.
            padding: Padding that was added to the input.
        """
        super().__init__()
        self.kernel_size: Size3d = kernel_size
        self.stride: Size3d | None = stride
        self.padding: Size3d = padding

    def forward(
        self,
        input: Tensor,
        indices: Tensor,
        output_size: Sequence[int] | None = None,
    ) -> Tensor:
        """Applies 3D max unpooling to `input` using `indices`.

        Args:
            input: Pooled input tensor to invert.
            indices: Indices given out by `MaxPool3d`.
            output_size: Optional targeted output size.

        Returns:
            Unpooled tensor.
        """
        return max_unpool3d(
            input,
            indices,
            self.kernel_size,
            self.stride,
            self.padding,
            output_size=output_size,
        )
