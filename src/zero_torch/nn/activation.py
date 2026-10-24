"""Activation modules."""

from __future__ import annotations

from ml_switcheroo_compiler.ops import tanh
from ml_switcheroo_compiler.ops.shape.indexing import where

import zero_torch
import zero_torch.nn.functional as F
from zero_torch.tensor import Tensor, _wrap

from .module import Module


class CELU(Module):
    """Applies the CELU function element-wise: `CELU(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))`."""

    def __init__(
        self,
        alpha: float = 1.0,
        inplace: bool = False,
    ) -> None:
        """Initializes the CELU module.

        Args:
            alpha: The `alpha` value for the CELU formulation.
            inplace: Whether to perform the operation in-place.
        """
        super().__init__()
        self.alpha: float = alpha
        self.inplace: bool = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Applies CELU to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(F.celu(input, alpha=self.alpha, inplace=self.inplace))


class ELU(Module):
    """Applies the Exponential Linear Unit (ELU) function element-wise."""

    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        """Initializes ELU.

        Args:
            alpha: The `alpha` value for the ELU formulation.
            inplace: Whether to perform the operation in-place.
        """
        super().__init__()
        self.alpha: float = alpha
        self.inplace: bool = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Applies ELU to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(F.elu(input, alpha=self.alpha, inplace=self.inplace))


class GELU(Module):
    """Applies the Gaussian Error Linear Units function."""

    def __init__(self, approximate: str = "none") -> None:
        """Initializes GELU.

        Args:
            approximate: The gelu approximation algorithm to use (`'none'` or `'tanh'`).
        """
        super().__init__()
        self.approximate: str = approximate

    def forward(self, input: Tensor) -> Tensor:
        """Applies GELU to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(F.gelu(input, approximate=self.approximate))


class GLU(Module):
    """Applies the gated linear unit function."""

    def __init__(self, dim: int = -1) -> None:
        """Initializes GLU.

        Args:
            dim: Dimension on which to split the input.
        """
        super().__init__()
        self.dim: int = dim

    def forward(self, input: Tensor) -> Tensor:
        """Applies GLU to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(F.glu(input, dim=self.dim))


class Hardshrink(Module):
    """Applies the Hardshrink function element-wise."""

    def __init__(self, lambd: float = 0.5) -> None:
        """Initializes Hardshrink.

        Args:
            lambd: The `lambda` value for the Hardshrink formulation.
        """
        super().__init__()
        self.lambd: float = lambd

    def forward(self, input: Tensor) -> Tensor:
        """Applies Hardshrink to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(F.hardshrink(input, lambd=self.lambd))


class Hardsigmoid(Module):
    """Applies the Hardsigmoid function element-wise."""

    def __init__(self, inplace: bool = False) -> None:
        """Initializes Hardsigmoid.

        Args:
            inplace: Whether to perform the operation in-place.
        """
        super().__init__()
        self.inplace: bool = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Applies Hardsigmoid to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(F.hardsigmoid(input, inplace=self.inplace))


class Hardswish(Module):
    """Applies the Hardswish function element-wise."""

    def __init__(self, inplace: bool = False) -> None:
        """Initializes Hardswish.

        Args:
            inplace: Whether to perform the operation in-place.
        """
        super().__init__()
        self.inplace: bool = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Applies Hardswish to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(F.hardswish(input, inplace=self.inplace))


class Hardtanh(Module):
    """Applies the HardTanh function element-wise."""

    def __init__(
        self, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False
    ) -> None:
        """Initializes Hardtanh.

        Args:
            min_val: Minimum value of the linear region range.
            max_val: Maximum value of the linear region range.
            inplace: Whether to perform the operation in-place.
        """
        super().__init__()
        self.min_val: float = min_val
        self.max_val: float = max_val
        self.inplace: bool = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Applies Hardtanh to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(
            F.hardtanh(
                input,
                min_val=self.min_val,
                max_val=self.max_val,
                inplace=self.inplace,
            )
        )


class LeakyReLU(Module):
    """Applies the LeakyReLU function element-wise."""

    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None:
        """Initializes LeakyReLU.

        Args:
            negative_slope: Controls the angle of the negative slope.
            inplace: Whether to perform the operation in-place.
        """
        super().__init__()
        self.negative_slope: float = negative_slope
        self.inplace: bool = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Applies LeakyReLU to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(
            F.leaky_relu(
                input, negative_slope=self.negative_slope, inplace=self.inplace
            )
        )


class LogSigmoid(Module):
    """Applies the LogSigmoid function element-wise."""

    def __init__(self) -> None:
        """Initializes LogSigmoid."""
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Applies LogSigmoid to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(F.logsigmoid(input))


class Mish(Module):
    """Applies the Mish function element-wise."""

    def __init__(self, inplace: bool = False) -> None:
        """Initializes Mish.

        Args:
            inplace: Whether to perform the operation in-place.
        """
        super().__init__()
        self.inplace: bool = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Applies Mish to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(F.mish(input, inplace=self.inplace))


class PReLU(Module):
    """Applies the element-wise PReLU function."""

    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        """Initializes PReLU.

        Args:
            num_parameters: Number of `a` parameters to learn.
            init: Initial value of `a`.
        """
        super().__init__()
        self.num_parameters: int = num_parameters
        self.init: float = init
        self.weight: Tensor = zero_torch.full((num_parameters,), init)

    def forward(self, input: Tensor) -> Tensor:
        """Applies PReLU to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(F.prelu(input, self.weight))


class RReLU(Module):
    """Applies the randomized leaky rectified linear unit function element-wise."""

    def __init__(
        self, lower: float = 1.0 / 8.0, upper: float = 1.0 / 3.0, inplace: bool = False
    ) -> None:
        """Initializes RReLU.

        Args:
            lower: Lower bound of the uniform distribution.
            upper: Upper bound of the uniform distribution.
            inplace: Whether to perform the operation in-place.
        """
        super().__init__()
        self.lower: float = lower
        self.upper: float = upper
        self.inplace: bool = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Applies RReLU to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return F.rrelu(
            input,
            lower=self.lower,
            upper=self.upper,
            training=self.training,
            inplace=self.inplace,
        )


class ReLU6(Module):
    """Applies the element-wise function `ReLU6(x) = min(max(0, x), 6)`."""

    def __init__(self, inplace: bool = False) -> None:
        """Initializes ReLU6.

        Args:
            inplace: Whether to perform the operation in-place.
        """
        super().__init__()
        self.inplace: bool = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Applies ReLU6 to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(F.relu6(input, inplace=self.inplace))


class SELU(Module):
    """Applies the SELU function element-wise."""

    def __init__(self, inplace: bool = False) -> None:
        """Initializes SELU.

        Args:
            inplace: Whether to perform the operation in-place.
        """
        super().__init__()
        self.inplace: bool = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Applies SELU to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(F.selu(input, inplace=self.inplace))


class SiLU(Module):
    """Applies the Sigmoid Linear Unit (SiLU) function element-wise."""

    def __init__(self, inplace: bool = False) -> None:
        """Initializes SiLU.

        Args:
            inplace: Whether to perform the operation in-place.
        """
        super().__init__()
        self.inplace: bool = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Applies SiLU to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(F.silu(input, inplace=self.inplace))


class Sigmoid(Module):
    """Applies the element-wise function `Sigmoid(x) = 1 / (1 + exp(-x))`."""

    def __init__(self) -> None:
        """Initializes Sigmoid."""
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Applies Sigmoid to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(F.sigmoid(input))


class Softplus(Module):
    """Applies the Softplus function element-wise."""

    def __init__(self, beta: float = 1.0, threshold: float = 20.0) -> None:
        """Initializes Softplus.

        Args:
            beta: The `beta` value for the Softplus formulation.
            threshold: Values above this revert to a linear function.
        """
        super().__init__()
        self.beta: float = beta
        self.threshold: float = threshold

    def forward(self, input: Tensor) -> Tensor:
        """Applies Softplus to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(F.softplus(input, beta=self.beta, threshold=self.threshold))


class Softshrink(Module):
    """Applies the Softshrink function element-wise."""

    def __init__(self, lambd: float = 0.5) -> None:
        """Initializes Softshrink.

        Args:
            lambd: The `lambda` (must be no less than zero) value for Softshrink.
        """
        super().__init__()
        self.lambd: float = lambd

    def forward(self, input: Tensor) -> Tensor:
        """Applies Softshrink to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(F.softshrink(input, lambd=self.lambd))


class Softsign(Module):
    """Applies the element-wise function `Softsign(x) = x / (1 + |x|)`."""

    def __init__(self) -> None:
        """Initializes Softsign."""
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Applies Softsign to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(F.softsign(input))


class Tanh(Module):
    """Applies the Hyperbolic Tangent (Tanh) function element-wise."""

    def __init__(self) -> None:
        """Initializes Tanh."""
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Applies Tanh to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(tanh(input._tensor))


class Tanhshrink(Module):
    """Applies the element-wise function `Tanhshrink(x) = x - tanh(x)`."""

    def __init__(self) -> None:
        """Initializes Tanhshrink."""
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Applies Tanhshrink to `input`.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        return _wrap(F.tanhshrink(input))


class Threshold(Module):
    """Thresholds each element of the input Tensor."""

    def __init__(self, threshold: float, value: float, inplace: bool = False) -> None:
        """Initializes Threshold.

        Args:
            threshold: The value to threshold at.
            value: The value to replace with.
            inplace: Whether to perform the operation in-place.
        """
        super().__init__()
        self.threshold: float = threshold
        self.value: float = value
        self.inplace: bool = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Applies Threshold to `input`.

        Args:
            input: Input tensor.

        Returns:
            Thresholded tensor.
        """
        assert input._tensor is not None
        return _wrap(where(input._tensor > self.threshold, input._tensor, self.value))


class Softmax(Module):
    """Applies the Softmax function to an n-dimensional input Tensor."""

    def __init__(self, dim: int | None = None) -> None:
        """Initializes Softmax.

        Args:
            dim: A dimension along which Softmax will be computed.
        """
        super().__init__()
        self.dim: int | None = dim

    def forward(self, input: Tensor) -> Tensor:
        """Applies Softmax to `input`.

        Args:
            input: Input tensor.

        Returns:
            Softmax probability tensor.
        """
        return _wrap(F.softmax(input, dim=self.dim))


class Softmax2d(Module):
    """Applies SoftMax over features to each spatial location."""

    def __init__(self) -> None:
        """Initializes Softmax2d."""
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Applies 2D Softmax along the channel dimension.

        Args:
            input: Input tensor of shape `(N, C, H, W)` or `(C, H, W)`.

        Returns:
            Channel-normalized probability tensor.
        """
        return _wrap(F.softmax(input, dim=-3))


class Softmin(Module):
    """Applies the Softmin function to an n-dimensional input Tensor."""

    def __init__(self, dim: int | None = None) -> None:
        """Initializes Softmin.

        Args:
            dim: A dimension along which Softmin will be computed.
        """
        super().__init__()
        self.dim: int | None = dim

    def forward(self, input: Tensor) -> Tensor:
        """Applies Softmin to `input`.

        Args:
            input: Input tensor.

        Returns:
            Softmin probability tensor.
        """
        return _wrap(F.softmin(input, dim=self.dim))


class LogSoftmax(Module):
    """Applies the `log(Softmax(x))` function to an n-dimensional input Tensor."""

    def __init__(self, dim: int | None = None) -> None:
        """Initializes LogSoftmax.

        Args:
            dim: A dimension along which LogSoftmax will be computed.
        """
        super().__init__()
        self.dim: int | None = dim

    def forward(self, input: Tensor) -> Tensor:
        """Applies LogSoftmax to `input`.

        Args:
            input: Input tensor.

        Returns:
            Log-softmax tensor.
        """
        return _wrap(F.log_softmax(input, dim=self.dim))
