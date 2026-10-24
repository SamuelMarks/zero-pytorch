"""Autograd math functions."""

from __future__ import annotations

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.ops.base import get_op

import zero_torch
from zero_torch._types import OpArg, TensorLike
from zero_torch.tensor import Tensor, _to_tensor, _wrap

from .function import Context, Function


def _unbroadcast(grad: Tensor, target_shape: tuple[int, ...]) -> Tensor:
    """Reduces `grad` along broadcasted dimensions so its shape matches `target_shape`.

    Args:
        grad: Upstream gradient tensor.
        target_shape: Desired shape of the input tensor.

    Returns:
        Unbroadcasted gradient tensor with shape `target_shape`.
    """
    if grad.shape == target_shape:
        return grad
    res = grad
    while len(res.shape) > len(target_shape):
        res = zero_torch.sum(res, axis=0)
    for i, (g_dim, t_dim) in enumerate(zip(res.shape, target_shape)):
        if t_dim == 1 and g_dim != 1:
            res = zero_torch.sum(res, axis=i, keepdims=True)
    return res


class AddBackward(Function):
    """Addition with autograd."""

    @staticmethod
    def forward(ctx: Context, a: TensorLike, b: TensorLike) -> Tensor:
        """Performs element-wise addition forward pass.

        Args:
            ctx: Autograd context object.
            a: First input tensor or scalar.
            b: Second input tensor or scalar.

        Returns:
            Element-wise sum tensor.
        """
        a_tensor = _to_tensor(a)
        b_tensor = _to_tensor(b)
        ctx.save_for_backward(a, b)
        return _wrap(ops.add(a_tensor, b_tensor))

    @staticmethod
    def backward(
        ctx: Context, *grad_outputs: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        """Computes gradients for element-wise addition.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Tuple of gradients `(grad_a, grad_b)`.
        """
        grad_output = grad_outputs[0]
        a, b = ctx.saved_tensors
        grad_a = (
            _unbroadcast(grad_output, a.shape)
            if isinstance(a, Tensor) and a.requires_grad
            else None
        )
        grad_b = (
            _unbroadcast(grad_output, b.shape)
            if isinstance(b, Tensor) and b.requires_grad
            else None
        )
        return grad_a, grad_b


class SubBackward(Function):
    """Subtraction with autograd."""

    @staticmethod
    def forward(ctx: Context, a: TensorLike, b: TensorLike) -> Tensor:
        """Performs element-wise subtraction forward pass.

        Args:
            ctx: Autograd context object.
            a: Minuend tensor or scalar.
            b: Subtrahend tensor or scalar.

        Returns:
            Element-wise difference tensor.
        """
        a_tensor = _to_tensor(a)
        b_tensor = _to_tensor(b)
        ctx.save_for_backward(a, b)
        return _wrap(ops.subtract(a_tensor, b_tensor))

    @staticmethod
    def backward(
        ctx: Context, *grad_outputs: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        """Computes gradients for element-wise subtraction.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Tuple of gradients `(grad_a, grad_b)`.
        """
        grad_output = grad_outputs[0]
        a, b = ctx.saved_tensors
        grad_a = (
            _unbroadcast(grad_output, a.shape)
            if isinstance(a, Tensor) and a.requires_grad
            else None
        )
        grad_b = (
            _unbroadcast(-grad_output, b.shape)
            if isinstance(b, Tensor) and b.requires_grad
            else None
        )
        return grad_a, grad_b


class MulBackward(Function):
    """Multiplication with autograd."""

    @staticmethod
    def forward(ctx: Context, a: TensorLike, b: TensorLike) -> Tensor:
        """Performs element-wise multiplication forward pass.

        Args:
            ctx: Autograd context object.
            a: First factor tensor or scalar.
            b: Second factor tensor or scalar.

        Returns:
            Element-wise product tensor.
        """
        a_tensor = _to_tensor(a)
        b_tensor = _to_tensor(b)
        ctx.save_for_backward(a, b)
        return _wrap(ops.multiply(a_tensor, b_tensor))

    @staticmethod
    def backward(
        ctx: Context, *grad_outputs: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        """Computes gradients for element-wise multiplication.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Tuple of gradients `(grad_a, grad_b)`.
        """
        grad_output = grad_outputs[0]
        a, b = ctx.saved_tensors
        grad_a = (
            _unbroadcast(grad_output * b, a.shape)
            if isinstance(a, Tensor) and a.requires_grad
            else None
        )
        grad_b = (
            _unbroadcast(grad_output * a, b.shape)
            if isinstance(b, Tensor) and b.requires_grad
            else None
        )
        return grad_a, grad_b


class DivBackward(Function):
    """Division with autograd."""

    @staticmethod
    def forward(ctx: Context, a: TensorLike, b: TensorLike) -> Tensor:
        """Performs element-wise division forward pass.

        Args:
            ctx: Autograd context object.
            a: Dividend tensor or scalar.
            b: Divisor tensor or scalar.

        Returns:
            Element-wise quotient tensor.
        """
        a_tensor = _to_tensor(a)
        b_tensor = _to_tensor(b)
        ctx.save_for_backward(a, b)
        return _wrap(ops.divide(a_tensor, b_tensor))

    @staticmethod
    def backward(
        ctx: Context, *grad_outputs: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        """Computes gradients for element-wise division.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Tuple of gradients `(grad_a, grad_b)`.
        """
        grad_output = grad_outputs[0]
        a, b = ctx.saved_tensors
        grad_a = (
            _unbroadcast(grad_output / b, a.shape)
            if isinstance(a, Tensor) and a.requires_grad
            else None
        )
        grad_b = (
            _unbroadcast(-grad_output * a / (b * b), b.shape)
            if isinstance(b, Tensor) and b.requires_grad
            else None
        )
        return grad_a, grad_b


class MatmulBackward(Function):
    """Matrix multiplication with autograd."""

    @staticmethod
    def forward(ctx: Context, a: TensorLike, b: TensorLike) -> Tensor:
        """Performs matrix multiplication forward pass.

        Args:
            ctx: Autograd context object.
            a: Left-hand matrix tensor.
            b: Right-hand matrix tensor.

        Returns:
            Matrix product tensor.
        """
        a_tensor = _to_tensor(a)
        b_tensor = _to_tensor(b)
        ctx.save_for_backward(a, b)
        return _wrap(ops.matmul(a_tensor, b_tensor))

    @staticmethod
    def backward(
        ctx: Context, *grad_outputs: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        """Computes gradients for matrix multiplication.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Tuple of gradients `(grad_a, grad_b)`.
        """
        grad_output = grad_outputs[0]
        a, b = ctx.saved_tensors
        grad_a: Tensor | None = None
        if isinstance(a, Tensor) and a.requires_grad:
            b_t = _wrap(get_op("Transpose")()(_to_tensor(b)))
            grad_a = _wrap(ops.matmul(_to_tensor(grad_output), _to_tensor(b_t)))

        grad_b: Tensor | None = None
        if isinstance(b, Tensor) and b.requires_grad:
            a_t = _wrap(get_op("Transpose")()(_to_tensor(a)))
            grad_b = _wrap(ops.matmul(_to_tensor(a_t), _to_tensor(grad_output)))

        return grad_a, grad_b


class PowBackward(Function):
    """Power with autograd."""

    @staticmethod
    def forward(ctx: Context, a: TensorLike, b: TensorLike) -> Tensor:
        """Performs element-wise power forward pass.

        Args:
            ctx: Autograd context object.
            a: Base tensor or scalar.
            b: Exponent tensor or scalar.

        Returns:
            Exponentiated tensor.
        """
        a_tensor = _to_tensor(a)
        b_tensor = _to_tensor(b)
        ctx.save_for_backward(a, b)
        return _wrap(ops.power(a_tensor, b_tensor))

    @staticmethod
    def backward(
        ctx: Context, *grad_outputs: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        """Computes gradients for element-wise power.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Tuple of gradients `(grad_a, grad_b)`.
        """
        grad_output = grad_outputs[0]
        a, b = ctx.saved_tensors
        grad_a: Tensor | None = None
        if isinstance(a, Tensor) and a.requires_grad:
            b_minus_1 = _wrap(ops.subtract(_to_tensor(b), _to_tensor(1)))
            a_pow = _wrap(ops.power(_to_tensor(a), _to_tensor(b_minus_1)))
            grad_a = _unbroadcast(grad_output * b * a_pow, a.shape)

        grad_b: Tensor | None = None
        if isinstance(b, Tensor) and b.requires_grad:
            ln_a = _wrap(ops.log(_to_tensor(a)))
            a_pow_b = _wrap(ops.power(_to_tensor(a), _to_tensor(b)))
            grad_b = _unbroadcast(grad_output * a_pow_b * ln_a, b.shape)

        return grad_a, grad_b


class NegBackward(Function):
    """Negation with autograd."""

    @staticmethod
    def forward(ctx: Context, a: TensorLike) -> Tensor:
        """Performs element-wise negation forward pass.

        Args:
            ctx: Autograd context object.
            a: Input tensor or scalar.

        Returns:
            Negated tensor.
        """
        a_tensor = _to_tensor(a)
        ctx.save_for_backward(a)
        return _wrap(ops.negative(a_tensor))

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor) -> Tensor | None:
        """Computes gradient for element-wise negation.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Negated upstream gradient if input requires grad, else None.
        """
        grad_output = grad_outputs[0]
        (a,) = ctx.saved_tensors
        return -grad_output if isinstance(a, Tensor) and a.requires_grad else None


class SumBackward(Function):
    """Sum reduction with autograd."""

    @staticmethod
    def forward(
        ctx: Context,
        a: TensorLike,
        axis: OpArg = None,
        keepdims: OpArg = False,
    ) -> Tensor:
        """Performs sum reduction forward pass.

        Args:
            ctx: Autograd context object.
            a: Input tensor.
            axis: Dimension(s) to reduce.
            keepdims: Whether reduced dimensions are retained with size 1.

        Returns:
            Reduced sum tensor.
        """
        a_t = a if isinstance(a, Tensor) else Tensor(a)
        ctx.save_for_backward(a_t)
        if axis is None:
            return _wrap(ops.reductions.sum(_to_tensor(a_t)))
        return _wrap(
            ops.reductions.sum(_to_tensor(a_t), axis=axis, keepdims=bool(keepdims))
        )

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor) -> Tensor | None:
        """Computes gradient for sum reduction by broadcasting upstream gradient to input shape.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Broadcasted gradient tensor matching input shape.
        """
        grad_output = grad_outputs[0]
        (a,) = ctx.saved_tensors
        if isinstance(a, Tensor) and a.requires_grad:
            return zero_torch.ones_like(a) * grad_output
        return None
