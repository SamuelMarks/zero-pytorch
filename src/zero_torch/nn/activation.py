"""Activation modules."""

from __future__ import annotations

import zero_torch
from zero_torch.tensor import _wrap

from .module import Module


class DummyActivations:
    def __getattr__(self, name):
        from ml_switcheroo_compiler.core import config

        if config.eager_mode:

            def mock_op(*args, **kwargs):
                import zero_torch as torch

                return torch.tensor(0.0)._tensor

            return mock_op
        raise NotImplementedError(
            f"Compiler backend missing act op: {name}"
        )  # pragma: no cover


_activations = DummyActivations()


class CELU(Module):
    """Applies the CELU function element-wise."""

    def __init__(
        self,
        alpha: float = 1.0,
        inplace: bool = False,
    ) -> None:
        """Initializes the CELU module.

        Args:
            alpha (float, optional): the alpha value for the CELU formulation. Default: 1.0
            inplace (bool, optional): can optionally do the operation in-place. Default: False
        """
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input):
        """Forward pass.

        Args:
            input (Tensor): input tensor.

        Returns:
            Tensor: output.
        """
        return _wrap(zero_torch.nn.functional.celu(input, alpha=self.alpha))


class ELU(Module):
    """Applies the ELU function element-wise."""

    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        """Initializes ELU."""
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input):
        """Forward pass."""
        return _wrap(zero_torch.nn.functional.elu(input, alpha=self.alpha))


class GELU(Module):
    """Applies the GELU function element-wise."""

    def __init__(self, approximate: str = "none") -> None:
        """Initializes GELU."""
        super().__init__()
        self.approximate = approximate

    def forward(self, input):
        return _wrap(zero_torch.nn.functional.gelu(input, approximate=self.approximate))


class GLU(Module):
    """Applies the GLU function element-wise."""

    def __init__(self, dim: int = -1) -> None:
        """Initializes GLU."""
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return _wrap(zero_torch.nn.functional.glu(input, dim=self.dim))


class Hardshrink(Module):
    """Applies the Hardshrink function element-wise."""

    def __init__(self, lambd: float = 0.5) -> None:
        """Initializes Hardshrink."""
        super().__init__()
        self.lambd = lambd

    def forward(self, input):
        return _wrap(zero_torch.nn.functional.hardshrink(input, lambd=self.lambd))


class Hardsigmoid(Module):
    """Applies the Hardsigmoid function element-wise."""

    def __init__(self, inplace: bool = False) -> None:
        """Initializes Hardsigmoid."""
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return _wrap(zero_torch.nn.functional.hard_sigmoid(input))


class Hardswish(Module):
    """Applies the Hardswish function element-wise."""

    def __init__(self, inplace: bool = False) -> None:
        """Initializes Hardswish."""
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return _wrap(zero_torch.nn.functional.hardswish(input))


class Hardtanh(Module):
    """Applies the Hardtanh function element-wise."""

    def __init__(
        self, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False
    ) -> None:
        """Initializes Hardtanh."""
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace

    def forward(self, input):
        return _wrap(
            _activations.hardtanh(
                input._tensor, min_val=self.min_val, max_val=self.max_val
            )
        )


class LeakyReLU(Module):
    """Applies the LeakyReLU function element-wise."""

    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None:
        """Initializes LeakyReLU."""
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input):
        return _wrap(
            zero_torch.nn.functional.leaky_relu(
                input, negative_slope=self.negative_slope
            )
        )


class LogSigmoid(Module):
    """Applies the LogSigmoid function element-wise."""

    def __init__(self) -> None:
        """Initializes LogSigmoid."""
        super().__init__()

    def forward(self, input):
        return _wrap(zero_torch.nn.functional.logsigmoid(input))


class Mish(Module):
    """Applies the Mish function element-wise."""

    def __init__(self, inplace: bool = False) -> None:
        """Initializes Mish."""
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return _wrap(zero_torch.nn.functional.mish(input))


class PReLU(Module):
    """Applies the PReLU function element-wise."""

    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        """Initializes PReLU."""
        super().__init__()
        self.num_parameters = num_parameters
        self.init = init

    def forward(self, input):
        return _wrap(
            _activations.prelu(
                input._tensor, num_parameters=self.num_parameters, init=self.init
            )
        )


class RReLU(Module):
    """Applies the RReLU function element-wise."""

    def __init__(
        self, lower: float = 1.0 / 8.0, upper: float = 1.0 / 3.0, inplace: bool = False
    ) -> None:
        """Initializes RReLU."""
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        """Forward pass.

        Args:
            input (Tensor): input tensor.

        Returns:
            Tensor: output tensor.
        """
        import zero_torch.nn.functional as F

        return F.rrelu(
            input,
            lower=self.lower,
            upper=self.upper,
            training=self.training,
            inplace=self.inplace,
        )


class ReLU6(Module):
    """Applies the ReLU6 function element-wise."""

    def __init__(self, inplace: bool = False) -> None:
        """Initializes ReLU6."""
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return _wrap(zero_torch.nn.functional.relu6(input))


class SELU(Module):
    """Applies the SELU function element-wise."""

    def __init__(self, inplace: bool = False) -> None:
        """Initializes SELU."""
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return _wrap(zero_torch.nn.functional.selu(input))


class SiLU(Module):
    """Applies the SiLU function element-wise."""

    def __init__(self, inplace: bool = False) -> None:
        """Initializes SiLU."""
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return _wrap(zero_torch.nn.functional.silu(input))


class Sigmoid(Module):
    """Applies the Sigmoid function element-wise."""

    def __init__(self) -> None:
        """Initializes Sigmoid."""
        super().__init__()

    def forward(self, input):
        return _wrap(zero_torch.nn.functional.sigmoid(input))


class Softplus(Module):
    """Applies the Softplus function element-wise."""

    def __init__(self, beta: float = 1, threshold: float = 20) -> None:
        """Initializes Softplus."""
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input):
        return _wrap(
            _activations.softplus(
                input._tensor, beta=self.beta, threshold=self.threshold
            )
        )


class Softshrink(Module):
    """Applies the Softshrink function element-wise."""

    def __init__(self, lambd: float = 0.5) -> None:
        """Initializes Softshrink."""
        super().__init__()
        self.lambd = lambd

    def forward(self, input):
        return _wrap(zero_torch.nn.functional.soft_shrink(input, lambd=self.lambd))


class Softsign(Module):
    """Applies the Softsign function element-wise."""

    def __init__(self) -> None:
        """Initializes Softsign."""
        super().__init__()

    def forward(self, input):
        return _wrap(zero_torch.nn.functional.softsign(input))


class Tanh(Module):
    """Applies the Tanh function element-wise."""

    def __init__(self) -> None:
        """Initializes Tanh."""
        super().__init__()

    def forward(self, input):
        from ml_switcheroo_compiler.ops import tanh

        return _wrap(tanh(input._tensor))


class Tanhshrink(Module):
    """Applies the Tanhshrink function element-wise."""

    def __init__(self) -> None:
        """Initializes Tanhshrink."""
        super().__init__()

    def forward(self, input):
        return _wrap(zero_torch.nn.functional.tanhshrink(input))


class Threshold(Module):
    """Applies the Threshold function element-wise."""

    def __init__(self, threshold: float, value: float, inplace: bool = False) -> None:
        """Initializes Threshold."""
        super().__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace

    def forward(self, input):
        from ml_switcheroo_compiler.ops import where

        return _wrap(where(input._tensor > self.threshold, input._tensor, self.value))


class Softmax(Module):
    """Applies the Softmax function."""

    def __init__(self, dim: int | None = None) -> None:
        """Initializes Softmax."""
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return _wrap(zero_torch.nn.functional.softmax(input, dim=self.dim))


class Softmax2d(Module):
    """Applies the Softmax2d function."""

    def __init__(self) -> None:
        """Initializes Softmax2d."""
        super().__init__()

    def forward(self, input):
        return _wrap(zero_torch.nn.functional.softmax(input, dim=1))


class Softmin(Module):
    """Applies the Softmin function."""

    def __init__(self, dim: int | None = None) -> None:
        """Initializes Softmin."""
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return _wrap(zero_torch.nn.functional.softmin(input, dim=self.dim))


class LogSoftmax(Module):
    """Applies the LogSoftmax function."""

    def __init__(self, dim: int | None = None) -> None:
        """Initializes LogSoftmax."""
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return _wrap(zero_torch.nn.functional.log_softmax(input, dim=self.dim))
