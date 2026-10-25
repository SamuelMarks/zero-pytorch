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


def _expand_reduced_grad(
    grad: Tensor,
    input_shape: tuple[int, ...],
    axis: OpArg,
    keepdims: bool,
) -> tuple[Tensor, int]:
    """Expands a reduced gradient tensor to broadcast with `input_shape` and returns the reduction count.

    Args:
        grad: Upstream gradient tensor.
        input_shape: Shape of the forward input tensor.
        axis: Reduced axis or sequence of axes, or None.
        keepdims: Whether reduced axes were kept as size-1 dimensions.

    Returns:
        Tuple `(expanded_grad, reduced_count)`.
    """
    ndim = len(input_shape)
    if axis is None:
        count = 1
        for d in input_shape:
            count *= d
        return grad, max(1, count)

    raw_axes = (
        (axis,)
        if isinstance(axis, int)
        else tuple(axis)
        if isinstance(axis, (tuple, list))
        else ()
    )
    norm_axes = sorted(
        {
            int(ax) if int(ax) >= 0 else ndim + int(ax)
            for ax in raw_axes
            if isinstance(ax, int)
        }
    )
    count = 1
    for ax in norm_axes:
        count *= input_shape[ax]
    if keepdims:
        return grad, max(1, count)
    bcast_shape = tuple(1 if i in norm_axes else input_shape[i] for i in range(ndim))
    return zero_torch.reshape(grad, bcast_shape), max(1, count)


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


class ExpBackward(Function):
    """Exponential function with autograd."""

    @staticmethod
    def forward(ctx: Context, a: TensorLike) -> Tensor:
        """Computes `exp(a)` and saves output for backward.

        Args:
            ctx: Autograd context object.
            a: Input tensor or scalar.

        Returns:
            Exponentiated tensor.
        """
        res = _wrap(ops.exp(_to_tensor(a)))
        ctx.save_for_backward(a, res)
        return res

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor) -> Tensor | None:
        """Computes `grad_output * exp(a)`.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Input gradient tensor or None.
        """
        a, res = ctx.saved_tensors
        if isinstance(a, Tensor) and a.requires_grad and isinstance(res, Tensor):
            return grad_outputs[0] * res
        return None


class LogBackward(Function):
    """Natural logarithm with autograd."""

    @staticmethod
    def forward(ctx: Context, a: TensorLike) -> Tensor:
        """Computes `log(a)`.

        Args:
            ctx: Autograd context object.
            a: Positive input tensor or scalar.

        Returns:
            Logarithm tensor.
        """
        ctx.save_for_backward(a)
        return _wrap(ops.log(_to_tensor(a)))

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor) -> Tensor | None:
        """Computes `grad_output / a`.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Input gradient tensor or None.
        """
        (a,) = ctx.saved_tensors
        if isinstance(a, Tensor) and a.requires_grad:
            return grad_outputs[0] / a
        return None


class SinBackward(Function):
    """Sine function with autograd."""

    @staticmethod
    def forward(ctx: Context, a: TensorLike) -> Tensor:
        """Computes `sin(a)`.

        Args:
            ctx: Autograd context object.
            a: Input tensor or scalar.

        Returns:
            Sine tensor.
        """
        ctx.save_for_backward(a)
        return _wrap(ops.sin(_to_tensor(a)))

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor) -> Tensor | None:
        """Computes `grad_output * cos(a)`.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Input gradient tensor or None.
        """
        (a,) = ctx.saved_tensors
        if isinstance(a, Tensor) and a.requires_grad:
            return grad_outputs[0] * _wrap(ops.cos(_to_tensor(a)))
        return None


class CosBackward(Function):
    """Cosine function with autograd."""

    @staticmethod
    def forward(ctx: Context, a: TensorLike) -> Tensor:
        """Computes `cos(a)`.

        Args:
            ctx: Autograd context object.
            a: Input tensor or scalar.

        Returns:
            Cosine tensor.
        """
        ctx.save_for_backward(a)
        return _wrap(ops.cos(_to_tensor(a)))

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor) -> Tensor | None:
        """Computes `-grad_output * sin(a)`.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Input gradient tensor or None.
        """
        (a,) = ctx.saved_tensors
        if isinstance(a, Tensor) and a.requires_grad:
            return -grad_outputs[0] * _wrap(ops.sin(_to_tensor(a)))
        return None


class TanhBackward(Function):
    """Hyperbolic tangent with autograd."""

    @staticmethod
    def forward(ctx: Context, a: TensorLike) -> Tensor:
        """Computes `tanh(a)`.

        Args:
            ctx: Autograd context object.
            a: Input tensor or scalar.

        Returns:
            Hyperbolic tangent tensor.
        """
        res = _wrap(ops.tanh(_to_tensor(a)))
        ctx.save_for_backward(a, res)
        return res

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor) -> Tensor | None:
        """Computes `grad_output * (1 - tanh(a)^2)`.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Input gradient tensor or None.
        """
        a, res = ctx.saved_tensors
        if isinstance(a, Tensor) and a.requires_grad and isinstance(res, Tensor):
            return grad_outputs[0] * (1.0 - res * res)
        return None


class SigmoidBackward(Function):
    """Logistic sigmoid with autograd."""

    @staticmethod
    def forward(ctx: Context, a: TensorLike) -> Tensor:
        """Computes `1 / (1 + exp(-a))`.

        Args:
            ctx: Autograd context object.
            a: Input tensor or scalar.

        Returns:
            Sigmoid tensor.
        """
        neg_a = _wrap(ops.negative(_to_tensor(a)))
        exp_neg = _wrap(ops.exp(_to_tensor(neg_a)))
        denom = _wrap(ops.add(_to_tensor(1.0), _to_tensor(exp_neg)))
        res = _wrap(ops.divide(_to_tensor(1.0), _to_tensor(denom)))
        ctx.save_for_backward(a, res)
        return res

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor) -> Tensor | None:
        """Computes `grad_output * sigmoid(a) * (1 - sigmoid(a))`.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Input gradient tensor or None.
        """
        a, res = ctx.saved_tensors
        if isinstance(a, Tensor) and a.requires_grad and isinstance(res, Tensor):
            return grad_outputs[0] * res * (1.0 - res)
        return None


class ReluBackward(Function):
    """Rectified linear unit (`max(0, a)`) with autograd."""

    @staticmethod
    def forward(ctx: Context, a: TensorLike) -> Tensor:
        """Computes `max(a, 0)`.

        Args:
            ctx: Autograd context object.
            a: Input tensor or scalar.

        Returns:
            ReLU-activated tensor.
        """
        a_t = a if isinstance(a, Tensor) else Tensor(a)
        ctx.save_for_backward(a_t)
        return _wrap(ops.maximum(_to_tensor(a_t), _to_tensor(0.0)))

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor) -> Tensor | None:
        """Computes `grad_output * (a > 0)`.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Input gradient tensor or None.
        """
        (a,) = ctx.saved_tensors
        if isinstance(a, Tensor) and a.requires_grad:
            mask = zero_torch.where(
                a > 0.0, zero_torch.ones_like(a), zero_torch.zeros_like(a)
            )
            return grad_outputs[0] * mask
        return None


class SqrtBackward(Function):
    """Square root with autograd."""

    @staticmethod
    def forward(ctx: Context, a: TensorLike) -> Tensor:
        """Computes `sqrt(a)`.

        Args:
            ctx: Autograd context object.
            a: Non-negative input tensor or scalar.

        Returns:
            Square-root tensor.
        """
        res = _wrap(ops.sqrt(_to_tensor(a)))
        ctx.save_for_backward(a, res)
        return res

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor) -> Tensor | None:
        """Computes `grad_output / (2 * sqrt(a))`.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Input gradient tensor or None.
        """
        a, res = ctx.saved_tensors
        if isinstance(a, Tensor) and a.requires_grad and isinstance(res, Tensor):
            return grad_outputs[0] / (2.0 * res)
        return None


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
        ctx.save_for_backward(a_t, axis, bool(keepdims))
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
            Broadcasted gradient tensor matching input shape, or None.
        """
        grad_output = grad_outputs[0]
        saved = ctx.saved_tensors
        a = saved[0]
        axis = saved[1] if len(saved) > 1 else None
        keepdims = bool(saved[2]) if len(saved) > 2 else False
        if isinstance(a, Tensor) and a.requires_grad:
            expanded_grad, _ = _expand_reduced_grad(
                grad_output, a.shape, axis, keepdims
            )
            return zero_torch.ones_like(a) * expanded_grad
        return None


class MeanBackward(Function):
    """Mean reduction with autograd."""

    @staticmethod
    def forward(
        ctx: Context,
        a: TensorLike,
        axis: OpArg = None,
        keepdims: OpArg = False,
    ) -> Tensor:
        """Performs mean reduction forward pass.

        Args:
            ctx: Autograd context object.
            a: Input tensor.
            axis: Dimension(s) to reduce.
            keepdims: Whether reduced dimensions are retained with size 1.

        Returns:
            Reduced mean tensor.
        """
        a_t = a if isinstance(a, Tensor) else Tensor(a)
        ctx.save_for_backward(a_t, axis, bool(keepdims))
        if axis is None:
            return _wrap(ops.reductions.mean(_to_tensor(a_t)))
        return _wrap(
            ops.reductions.mean(_to_tensor(a_t), axis=axis, keepdims=bool(keepdims))
        )

    @staticmethod
    def backward(
        ctx: Context, *grad_outputs: Tensor
    ) -> tuple[Tensor | None, None, None]:
        """Computes gradient for mean reduction by scaling and broadcasting upstream gradient.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Tuple `(grad_a, None, None)`.
        """
        grad_output = grad_outputs[0]
        a, axis, keepdims = ctx.saved_tensors
        if isinstance(a, Tensor) and a.requires_grad:
            expanded_grad, count = _expand_reduced_grad(
                grad_output, a.shape, axis, bool(keepdims)
            )
            return (zero_torch.ones_like(a) * expanded_grad) / float(count), None, None
        return None, None, None


class ReshapeBackward(Function):
    """Reshape / view operation with autograd."""

    @staticmethod
    def forward(ctx: Context, a: TensorLike, shape: OpArg) -> Tensor:
        """Reshapes `a` to `shape` and saves input shape for backward.

        Args:
            ctx: Autograd context object.
            a: Input tensor.
            shape: Target shape tuple or sequence.

        Returns:
            Reshaped tensor.
        """
        a_t = a if isinstance(a, Tensor) else Tensor(a)
        ctx.save_for_backward(a_t)
        target_shape = (
            (shape,)
            if isinstance(shape, int)
            else tuple(int(s) for s in shape if isinstance(s, int))
            if isinstance(shape, (tuple, list))
            else a_t.shape
        )
        return _wrap(get_op("Reshape")()(_to_tensor(a_t), newshape=target_shape))

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor) -> tuple[Tensor | None, None]:
        """Reshapes upstream gradient back to original input shape.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Tuple `(grad_a, None)`.
        """
        (a,) = ctx.saved_tensors
        if isinstance(a, Tensor) and a.requires_grad:
            return (
                _wrap(
                    get_op("Reshape")()(_to_tensor(grad_outputs[0]), newshape=a.shape)
                ),
                None,
            )
        return None, None


class TransposeBackward(Function):
    """2D dimension swap (`transpose(dim0, dim1)`) with autograd."""

    @staticmethod
    def forward(
        ctx: Context,
        a: TensorLike,
        dim0: OpArg = 0,
        dim1: OpArg = 1,
    ) -> Tensor:
        """Swaps dimensions `dim0` and `dim1` of `a`.

        Args:
            ctx: Autograd context object.
            a: Input tensor.
            dim0: First dimension index.
            dim1: Second dimension index.

        Returns:
            Transposed tensor.
        """
        a_t = a if isinstance(a, Tensor) else Tensor(a)
        ndim = len(a_t.shape)
        d0 = int(dim0) if isinstance(dim0, int) else 0
        d1 = int(dim1) if isinstance(dim1, int) else 1
        d0 = d0 if d0 >= 0 else ndim + d0
        d1 = d1 if d1 >= 0 else ndim + d1
        perm = list(range(ndim))
        perm[d0], perm[d1] = perm[d1], perm[d0]
        ctx.save_for_backward(a_t, d0, d1)
        return _wrap(get_op("Transpose")()(_to_tensor(a_t), axes=tuple(perm)))

    @staticmethod
    def backward(
        ctx: Context, *grad_outputs: Tensor
    ) -> tuple[Tensor | None, None, None]:
        """Swaps dimensions `dim0` and `dim1` of upstream gradient.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Tuple `(grad_a, None, None)`.
        """
        a, d0, d1 = ctx.saved_tensors
        if (
            isinstance(a, Tensor)
            and a.requires_grad
            and isinstance(d0, int)
            and isinstance(d1, int)
        ):
            ndim = len(grad_outputs[0].shape)
            perm = list(range(ndim))
            perm[d0], perm[d1] = perm[d1], perm[d0]
            return (
                _wrap(
                    get_op("Transpose")()(_to_tensor(grad_outputs[0]), axes=tuple(perm))
                ),
                None,
                None,
            )
        return None, None, None


class PermuteBackward(Function):
    """Arbitrary axis permutation (`permute`) with autograd."""

    @staticmethod
    def forward(ctx: Context, a: TensorLike, dims: OpArg) -> Tensor:
        """Permutes dimensions of `a` according to `dims`.

        Args:
            ctx: Autograd context object.
            a: Input tensor.
            dims: Sequence of dimension indices.

        Returns:
            Permuted tensor.
        """
        a_t = a if isinstance(a, Tensor) else Tensor(a)
        ndim = len(a_t.shape)
        perm = (
            tuple(
                int(d) if int(d) >= 0 else ndim + int(d)
                for d in dims
                if isinstance(d, int)
            )
            if isinstance(dims, (tuple, list))
            else tuple(range(ndim))
        )
        ctx.save_for_backward(a_t, perm)
        return _wrap(get_op("Transpose")()(_to_tensor(a_t), axes=perm))

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor) -> tuple[Tensor | None, None]:
        """Applies inverse permutation to upstream gradient.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Tuple `(grad_a, None)`.
        """
        a, perm = ctx.saved_tensors
        if isinstance(a, Tensor) and a.requires_grad and isinstance(perm, tuple):
            inv_perm = [0] * len(perm)
            for i, p in enumerate(perm):
                inv_perm[int(p) if isinstance(p, int) else 0] = i
            return (
                _wrap(
                    get_op("Transpose")()(
                        _to_tensor(grad_outputs[0]), axes=tuple(inv_perm)
                    )
                ),
                None,
            )
        return None, None


class UnsqueezeBackward(Function):
    """Dimension insertion (`unsqueeze`) with autograd."""

    @staticmethod
    def forward(ctx: Context, a: TensorLike, dim: OpArg = 0) -> Tensor:
        """Inserts a singleton dimension at `dim`.

        Args:
            ctx: Autograd context object.
            a: Input tensor.
            dim: Dimension index to insert.

        Returns:
            Unsqueezed tensor.
        """
        a_t = a if isinstance(a, Tensor) else Tensor(a)
        d = int(dim) if isinstance(dim, int) else 0
        shape_list = list(a_t.shape)
        norm_d = d + len(shape_list) + 1 if d < 0 else d
        shape_list.insert(norm_d, 1)
        ctx.save_for_backward(a_t)
        return _wrap(get_op("Reshape")()(_to_tensor(a_t), newshape=tuple(shape_list)))

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor) -> tuple[Tensor | None, None]:
        """Reshapes upstream gradient back to original input shape.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Tuple `(grad_a, None)`.
        """
        (a,) = ctx.saved_tensors
        if isinstance(a, Tensor) and a.requires_grad:
            return (
                _wrap(
                    get_op("Reshape")()(_to_tensor(grad_outputs[0]), newshape=a.shape)
                ),
                None,
            )
        return None, None


class SqueezeBackward(Function):
    """Singleton dimension removal (`squeeze`) with autograd."""

    @staticmethod
    def forward(ctx: Context, a: TensorLike, axis: OpArg = None) -> Tensor:
        """Removes singleton dimension(s) from `a`.

        Args:
            ctx: Autograd context object.
            a: Input tensor.
            axis: Optional dimension index or tuple to squeeze.

        Returns:
            Squeezed tensor.
        """
        a_t = a if isinstance(a, Tensor) else Tensor(a)
        ctx.save_for_backward(a_t)
        return _wrap(get_op("Squeeze")()(_to_tensor(a_t), axis=axis))

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor) -> tuple[Tensor | None, None]:
        """Reshapes upstream gradient back to original input shape.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Tuple `(grad_a, None)`.
        """
        (a,) = ctx.saved_tensors
        if isinstance(a, Tensor) and a.requires_grad:
            return (
                _wrap(
                    get_op("Reshape")()(_to_tensor(grad_outputs[0]), newshape=a.shape)
                ),
                None,
            )
        return None, None


class ConcatenateBackward(Function):
    """Tensor concatenation (`concatenate` / `cat`) with autograd."""

    @staticmethod
    def forward(ctx: Context, axis: OpArg, *tensors: TensorLike) -> Tensor:
        """Concatenates `tensors` along `axis`.

        Args:
            ctx: Autograd context object.
            axis: Dimension along which to concatenate.
            *tensors: Input tensors to concatenate.

        Returns:
            Concatenated tensor.
        """
        ax = int(axis) if isinstance(axis, int) else 0
        t_list = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
        ctx.save_for_backward(ax, *t_list)
        return _wrap(get_op("Concatenate")()([_to_tensor(t) for t in t_list], axis=ax))

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor) -> tuple[Tensor | None, ...]:
        """Splits upstream gradient along `axis` back to each concatenated input.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Tuple `(None, grad_t0, grad_t1, ...)`.
        """
        grad_output = grad_outputs[0]
        saved = ctx.saved_tensors
        ax_raw = saved[0]
        tensors = saved[1:]
        ndim = len(grad_output.shape)
        ax = int(ax_raw) if isinstance(ax_raw, int) else 0
        norm_ax = ax if ax >= 0 else ndim + ax
        split_points: list[int] = []
        running = 0
        for t in tensors[:-1]:
            running += t.shape[norm_ax] if isinstance(t, Tensor) else 0
            split_points.append(running)
        parts = (
            list(zero_torch.split(grad_output, split_points, axis=norm_ax))
            if split_points
            else [grad_output]
        )
        grads: list[Tensor | None] = [None]
        for t, part in zip(tensors, parts):
            grads.append(part if isinstance(t, Tensor) and t.requires_grad else None)
        return tuple(grads)


class StackBackward(Function):
    """Tensor stacking (`stack`) with autograd."""

    @staticmethod
    def forward(ctx: Context, axis: OpArg, *tensors: TensorLike) -> Tensor:
        """Stacks `tensors` along a new dimension `axis`.

        Args:
            ctx: Autograd context object.
            axis: Dimension index to insert.
            *tensors: Input tensors to stack.

        Returns:
            Stacked tensor.
        """
        ax = int(axis) if isinstance(axis, int) else 0
        t_list = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
        ctx.save_for_backward(ax, *t_list)
        return _wrap(get_op("Stack")()([_to_tensor(t) for t in t_list], axis=ax))

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor) -> tuple[Tensor | None, ...]:
        """Splits upstream gradient along stacked dimension and reshapes to each input.

        Args:
            ctx: Autograd context object.
            *grad_outputs: Upstream gradient tensor(s).

        Returns:
            Tuple `(None, grad_t0, grad_t1, ...)`.
        """
        grad_output = grad_outputs[0]
        saved = ctx.saved_tensors
        ax_raw = saved[0]
        tensors = saved[1:]
        ax = int(ax_raw) if isinstance(ax_raw, int) else 0
        parts = list(zero_torch.split(grad_output, len(tensors), axis=ax))
        grads: list[Tensor | None] = [None]
        for t, part in zip(tensors, parts):
            if isinstance(t, Tensor) and t.requires_grad:
                grads.append(
                    _wrap(get_op("Reshape")()(_to_tensor(part), newshape=t.shape))
                )
            else:
                grads.append(None)
        return tuple(grads)
