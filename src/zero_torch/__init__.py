"""Zero-dependency PyTorch API frontend backed by ml-switcheroo-compiler."""

from __future__ import annotations

import math
import random
from builtins import any as _builtin_any
from builtins import max as _builtin_max
from collections.abc import Sequence

import ml_switcheroo_compiler.ops as _ops
import ml_switcheroo_compiler.ops.creation as _creation
import ml_switcheroo_compiler.ops.linalg.decompositions as _linalg_decompositions
import ml_switcheroo_compiler.ops.linalg.einsum_frontend as _linalg_einsum
import ml_switcheroo_compiler.ops.reductions as _reductions
import ml_switcheroo_compiler.ops.shape.indexing as _shape_indexing
import ml_switcheroo_compiler.ops.shape.pad_and_tile as _shape_pad_and_tile
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import (
    Tensor as _SwitcherooTensor,
)
from ml_switcheroo_compiler.core.tensor import (
    TensorConfig as _TensorConfig,
)
from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.ops.shape.dynamic_slicing import (
    dynamic_slice as _dynamic_slice,
)

from zero_torch._types import (
    DTypeLike,
    NestedSequence,
    OpArg,
    Scalar,
    TensorLike,
)
from zero_torch.tensor import (
    Tensor as Tensor,
)
from zero_torch.tensor import (
    _ArrayDataProtocol,
    _to_tensor,
    _wrap,
    _wrap_list,
    _wrap_tuple,
)
from zero_torch.tracing import ProxyTensor

from . import distributions as distributions
from . import fft as fft
from . import linalg as linalg
from . import nn as nn
from . import optim as optim
from . import sparse as sparse
from . import special as special
from . import utils as utils
from .autograd import no_grad as no_grad
from .autograd import set_grad_enabled as set_grad_enabled
from .sparse import sparse_coo_tensor as sparse_coo_tensor

_ = (_TensorConfig, _linalg_decompositions)


def abs(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the abs operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the abs operation applied.
    """
    _ = out
    res = _ops.abs(_to_tensor(input))
    return _wrap(res)


def acos(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the acos operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the acos operation applied.
    """
    _ = out
    res = _ops.acos(_to_tensor(input))
    return _wrap(res)


def acosh(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the acosh operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the acosh operation applied.
    """
    _ = out
    res = _ops.acosh(_to_tensor(input))
    return _wrap(res)


def add(input: TensorLike, other: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the add operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the add operation applied.
    """
    _ = out
    res = _ops.add(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def all(
    input: TensorLike,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Tensor:
    """Applies the all reduction operation.

    Args:
        input: Input tensor or scalar.
        dim: Optional dimension or sequence of dimensions to reduce.
        keepdim: Whether the output tensor has `dim` retained or not.
        axis: Optional alias for `dim`.
        keepdims: Optional alias for `keepdim`.
        dtype: Optional desired data type of returned tensor.

    Returns:
        Tensor: A new tensor with the all operation applied.
    """
    _ = dtype
    eff_axis = dim if dim is not None else axis
    eff_keepdims = keepdim or keepdims
    res = _reductions.all(_to_tensor(input), axis=eff_axis, keepdims=eff_keepdims)
    return _wrap(res)


def allclose(*args: OpArg, **kwargs: OpArg) -> bool:
    """Applies the allclose operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Boolean indicating whether the two tensors are element-wise equal within a tolerance.
    """
    res = _ops.allclose(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return bool(res)


def any(
    input: TensorLike,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Tensor:
    """Applies the any reduction operation.

    Args:
        input: Input tensor or scalar.
        dim: Optional dimension or sequence of dimensions to reduce.
        keepdim: Whether the output tensor has `dim` retained or not.
        axis: Optional alias for `dim`.
        keepdims: Optional alias for `keepdim`.
        dtype: Optional desired data type of returned tensor.

    Returns:
        Tensor: A new tensor with the any operation applied.
    """
    _ = dtype
    eff_axis = dim if dim is not None else axis
    eff_keepdims = keepdim or keepdims
    res = _reductions.any(_to_tensor(input), axis=eff_axis, keepdims=eff_keepdims)
    return _wrap(res)


def arange(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the arange operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the arange operation applied.
    """
    res = _ops.arange(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def argmax(
    input: TensorLike,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Tensor:
    """Applies the argmax reduction operation.

    Args:
        input: Input tensor or scalar.
        dim: Optional dimension or sequence of dimensions to reduce.
        keepdim: Whether the output tensor has `dim` retained or not.
        axis: Optional alias for `dim`.
        keepdims: Optional alias for `keepdim`.
        dtype: Optional desired data type of returned tensor.

    Returns:
        Tensor: A new tensor with the argmax operation applied.
    """
    _ = dtype
    eff_axis = dim if dim is not None else axis
    eff_keepdims = keepdim or keepdims
    res = _reductions.argmax(_to_tensor(input), axis=eff_axis, keepdims=eff_keepdims)
    return _wrap(res)


def argmin(
    input: TensorLike,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Tensor:
    """Applies the argmin reduction operation.

    Args:
        input: Input tensor or scalar.
        dim: Optional dimension or sequence of dimensions to reduce.
        keepdim: Whether the output tensor has `dim` retained or not.
        axis: Optional alias for `dim`.
        keepdims: Optional alias for `keepdim`.
        dtype: Optional desired data type of returned tensor.

    Returns:
        Tensor: A new tensor with the argmin operation applied.
    """
    _ = dtype
    eff_axis = dim if dim is not None else axis
    eff_keepdims = keepdim or keepdims
    res = _reductions.argmin(_to_tensor(input), axis=eff_axis, keepdims=eff_keepdims)
    return _wrap(res)


def asin(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the asin operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the asin operation applied.
    """
    _ = out
    res = _ops.asin(_to_tensor(input))
    return _wrap(res)


def asinh(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the asinh operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the asinh operation applied.
    """
    _ = out
    res = _ops.asinh(_to_tensor(input))
    return _wrap(res)


def atan(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the atan operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the atan operation applied.
    """
    _ = out
    res = _ops.atan(_to_tensor(input))
    return _wrap(res)


def atan2(input: TensorLike, other: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the atan2 operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the atan2 operation applied.
    """
    _ = out
    res = _ops.atan2(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def atanh(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the atanh operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the atanh operation applied.
    """
    _ = out
    res = _ops.atanh(_to_tensor(input))
    return _wrap(res)


def bitcast(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the bitcast operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the bitcast operation applied.
    """
    _args = [a._tensor if isinstance(a, Tensor) else a for a in args]
    if len(_args) == 2:
        kwargs["dtype"] = _args[1]
        _args = [_args[0]]
    res = _ops.bitcast(*_args, **kwargs)
    return _wrap(res)


def bitwise_and(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the bitwise_and operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the bitwise_and operation applied.
    """
    res = _ops.bitwise_and(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def bitwise_not(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the bitwise_not operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the bitwise_not operation applied.
    """
    res = _ops.bitwise_not(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def bitwise_or(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the bitwise_or operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the bitwise_or operation applied.
    """
    res = _ops.bitwise_or(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def bitwise_xor(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the bitwise_xor operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the bitwise_xor operation applied.
    """
    res = _ops.bitwise_xor(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def broadcast_to(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the broadcast_to operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the broadcast_to operation applied.
    """
    res = get_op("BroadcastTo")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def _apply_cast(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the cast operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the cast operation applied.
    """
    _args = [a._tensor if isinstance(a, Tensor) else a for a in args]
    if len(_args) == 2:
        kwargs["dtype"] = _args[1]
        _args = [_args[0]]
    res = _ops.cast(*_args, **kwargs)
    return _wrap(res)


cast = _apply_cast


def cbrt(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the cbrt operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the cbrt operation applied.
    """
    _ = out
    res = _ops.cbrt(_to_tensor(input))
    return _wrap(res)


def ceil(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the ceil operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the ceil operation applied.
    """
    _ = out
    res = _ops.ceil(_to_tensor(input))
    return _wrap(res)


def cholesky(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the cholesky operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the cholesky operation applied.
    """
    res = get_op("Cholesky")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def concatenate(*args: OpArg | Sequence[Tensor], **kwargs: OpArg) -> Tensor:
    """Applies the concatenate operation.

    Args:
        *args: Variable length argument list (sequence of tensors and optional axis).
        **kwargs: Arbitrary keyword arguments (`tensors`, `dim`, `axis`).

    Returns:
        Tensor: Concatenated tensor.
    """
    raw_tensors = args[0] if len(args) > 0 else kwargs.get("tensors")
    dim = args[1] if len(args) > 1 else kwargs.get("dim", kwargs.get("axis", 0))
    seq = raw_tensors if isinstance(raw_tensors, (list, tuple)) else []
    if _builtin_any(isinstance(t, Tensor) and t.requires_grad for t in seq):
        from .autograd.math_ops import ConcatenateBackward

        return ConcatenateBackward.apply(dim, *seq)
    tensors_t = [t._tensor if isinstance(t, Tensor) else t for t in seq]
    res = get_op("Concatenate")()(tensors_t, axis=dim)
    return _wrap(res)


def conj(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the conj operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the conj operation applied.
    """
    _ = out
    res = _ops.conj(_to_tensor(input))
    return _wrap(res)


def copysign(
    input: TensorLike, other: TensorLike, *, out: Tensor | None = None
) -> Tensor:
    """Applies the copysign operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the copysign operation applied.
    """
    _ = out
    res = _ops.copysign(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def cos(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the cos operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the cos operation applied.
    """
    _ = out
    if isinstance(input, Tensor) and input.requires_grad:
        from .autograd.math_ops import CosBackward

        return CosBackward.apply(input)
    res = _ops.cos(_to_tensor(input))
    return _wrap(res)


def cosh(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the cosh operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the cosh operation applied.
    """
    _ = out
    res = _ops.cosh(_to_tensor(input))
    return _wrap(res)


def count_nonzero(
    input: TensorLike,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Tensor:
    """Applies the count_nonzero reduction operation.

    Args:
        input: Input tensor or scalar.
        dim: Optional dimension or sequence of dimensions to reduce.
        keepdim: Whether the output tensor has `dim` retained or not.
        axis: Optional alias for `dim`.
        keepdims: Optional alias for `keepdim`.
        dtype: Optional desired data type of returned tensor.

    Returns:
        Tensor: A new tensor with the count_nonzero operation applied.
    """
    _ = dtype
    eff_axis = dim if dim is not None else axis
    eff_keepdims = keepdim or keepdims
    res = _reductions.count_nonzero(
        _to_tensor(input), axis=eff_axis, keepdims=eff_keepdims
    )
    return _wrap(res)


def deg2rad(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the deg2rad operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the deg2rad operation applied.
    """
    _ = out
    res = _ops.deg2rad(_to_tensor(input))
    return _wrap(res)


def det(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the det operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the det operation applied.
    """
    res = get_op("Det")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def diag(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the diag operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the diag operation applied.
    """
    res = _ops.diag(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def digamma(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the digamma operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the digamma operation applied.
    """
    _ = out
    res = _ops.digamma(_to_tensor(input))
    return _wrap(res)


def divide(
    input: TensorLike, other: TensorLike, *, out: Tensor | None = None
) -> Tensor:
    """Applies the divide operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the divide operation applied.
    """
    _ = out
    res = _ops.divide(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def dot(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the dot operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the dot operation applied.
    """
    res = _ops.dot(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def dynamic_slice(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the dynamic_slice operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the dynamic_slice operation applied.
    """
    res = _dynamic_slice(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def eigh(*args: OpArg, **kwargs: OpArg) -> tuple[Tensor, ...]:
    """Applies the eigh operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tuple of `(eigenvalues, eigenvectors)` tensors.
    """
    res = get_op("Eigh")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap_tuple(res)


def eigvalsh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the eigvalsh operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the eigvalsh operation applied.
    """
    res = get_op("Eigvalsh")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def einsum(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the einsum operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the einsum operation applied.
    """
    res = _linalg_einsum.einsum(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def empty(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the empty operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the empty operation applied.
    """
    res = _ops.empty(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def equal(input: TensorLike, other: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the equal operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the equal operation applied.
    """
    _ = out
    res = _ops.equal(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def erf(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the erf operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the erf operation applied.
    """
    _ = out
    res = _ops.erf(_to_tensor(input))
    return _wrap(res)


def erfc(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the erfc operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the erfc operation applied.
    """
    _ = out
    res = _ops.erfc(_to_tensor(input))
    return _wrap(res)


def erfinv(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the erfinv operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the erfinv operation applied.
    """
    _ = out
    res = _ops.erfinv(_to_tensor(input))
    return _wrap(res)


def exp(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the exp operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the exp operation applied.
    """
    _ = out
    if isinstance(input, Tensor) and input.requires_grad:
        from .autograd.math_ops import ExpBackward

        return ExpBackward.apply(input)
    res = _ops.exp(_to_tensor(input))
    return _wrap(res)


def exp2(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the exp2 operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the exp2 operation applied.
    """
    _ = out
    res = _ops.exp2(_to_tensor(input))
    return _wrap(res)


def expand(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the expand operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the expand operation applied.
    """
    res = get_op("BroadcastTo")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def expm1(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the expm1 operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the expm1 operation applied.
    """
    _ = out
    res = _ops.expm1(_to_tensor(input))
    return _wrap(res)


def eye(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the eye operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the eye operation applied.
    """
    res = _ops.eye(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def fix(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the fix operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the fix operation applied.
    """
    _ = out
    res = _ops.fix(_to_tensor(input))
    return _wrap(res)


def flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """Flattens a contiguous range of dims into a tensor.

    Args:
        input: Input tensor.
        start_dim: First dim to flatten.
        end_dim: Last dim to flatten.

    Returns:
        Flattened tensor.
    """
    _ = (start_dim, end_dim)
    return reshape(input, (-1,))


def float_power(
    input: TensorLike, other: TensorLike, *, out: Tensor | None = None
) -> Tensor:
    """Applies the float_power operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the float_power operation applied.
    """
    _ = out
    res = _ops.float_power(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def floor(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the floor operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the floor operation applied.
    """
    _ = out
    res = _ops.floor(_to_tensor(input))
    return _wrap(res)


def floor_divide(
    input: TensorLike, other: TensorLike, *, out: Tensor | None = None
) -> Tensor:
    """Applies the floor_divide operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the floor_divide operation applied.
    """
    _ = out
    res = _ops.floor_divide(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def fmax(input: TensorLike, other: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the fmax operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the fmax operation applied.
    """
    _ = out
    res = _ops.fmax(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def fmin(input: TensorLike, other: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the fmin operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the fmin operation applied.
    """
    _ = out
    res = _ops.fmin(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def fmod(input: TensorLike, other: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the fmod operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the fmod operation applied.
    """
    _ = out
    res = _ops.fmod(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def frexp(*args: OpArg, **kwargs: OpArg) -> tuple[Tensor, ...]:
    """Applies the frexp operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Decomposed `(mantissa, exponent)` tensors.
    """
    res = _ops.frexp(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap_tuple(res)


def full(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the full operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the full operation applied.
    """
    res = _ops.full(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def full_like(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the full_like operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the full_like operation applied.
    """
    res = _ops.full_like(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def gather(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Gathers values along an axis specified by dim.

    Args:
        *args: Positional arguments `(input, dim, index)`.
        **kwargs: Keyword arguments (`dim`, `axis`, `index`).

    Returns:
        Gathered tensor.
    """
    t = args[0]._tensor if isinstance(args[0], Tensor) else args[0]
    if len(args) > 1:
        kwargs["axis"] = args[1]
    elif "dim" in kwargs:
        kwargs["axis"] = kwargs.pop("dim")
    if len(args) > 2:
        kwargs["index"] = (
            args[2]._tensor if isinstance(args[2], Tensor) else _to_tensor(args[2])
        )
    elif "index" in kwargs:
        idx = kwargs["index"]
        kwargs["index"] = idx._tensor if isinstance(idx, Tensor) else _to_tensor(idx)
    return _wrap(get_op("Gather")()(t, **kwargs))


def gather_nd(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the gather_nd operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the gather_nd operation applied.
    """
    res = get_op("GatherNd")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def gcd(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the gcd operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the gcd operation applied.
    """
    res = _ops.gcd(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def greater(
    input: TensorLike, other: TensorLike, *, out: Tensor | None = None
) -> Tensor:
    """Applies the greater operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the greater operation applied.
    """
    _ = out
    res = _ops.greater(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def greater_equal(
    input: TensorLike, other: TensorLike, *, out: Tensor | None = None
) -> Tensor:
    """Applies the greater_equal operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the greater_equal operation applied.
    """
    _ = out
    res = _ops.greater_equal(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def heaviside(
    input: TensorLike, other: TensorLike, *, out: Tensor | None = None
) -> Tensor:
    """Applies the heaviside operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the heaviside operation applied.
    """
    _ = out
    res = _ops.heaviside(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def hypot(input: TensorLike, other: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the hypot operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the hypot operation applied.
    """
    _ = out
    res = _ops.hypot(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def identity(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the identity operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the identity operation applied.
    """
    res = _ops.identity(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def imag(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the imag operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the imag operation applied.
    """
    _ = out
    res = _ops.imag(_to_tensor(input))
    return _wrap(res)


def inner(input: TensorLike, other: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the inner operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the inner operation applied.
    """
    _ = out
    res = _ops.inner(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def inv(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the inv operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the inv operation applied.
    """
    res = get_op("Inv")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def isclose(
    input: TensorLike,
    other: TensorLike,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> Tensor:
    """Applies the isclose element-wise comparison.

    Args:
        input: First tensor or scalar to compare.
        other: Second tensor or scalar to compare.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        equal_nan: If True, two NaNs will be considered equal.

    Returns:
        Tensor: Boolean tensor indicating where `input` and `other` are close.
    """
    res = _ops.isclose(
        _to_tensor(input),
        _to_tensor(other),
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )
    return _wrap(res)


def isfinite(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the isfinite operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the isfinite operation applied.
    """
    _ = out
    res = _ops.isfinite(_to_tensor(input))
    return _wrap(res)


def isinf(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the isinf operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the isinf operation applied.
    """
    _ = out
    res = _ops.isinf(_to_tensor(input))
    return _wrap(res)


def isnan(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the isnan operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the isnan operation applied.
    """
    _ = out
    res = _ops.isnan(_to_tensor(input))
    return _wrap(res)


def lcm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the lcm operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the lcm operation applied.
    """
    res = _ops.lcm(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def ldexp(input: TensorLike, other: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the ldexp operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the ldexp operation applied.
    """
    _ = out
    res = _ops.ldexp(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def left_shift(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the left_shift operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the left_shift operation applied.
    """
    res = _ops.left_shift(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def less(input: TensorLike, other: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the less operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the less operation applied.
    """
    _ = out
    res = _ops.less(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def less_equal(
    input: TensorLike, other: TensorLike, *, out: Tensor | None = None
) -> Tensor:
    """Applies the less_equal operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the less_equal operation applied.
    """
    _ = out
    res = _ops.less_equal(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def lgamma(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the lgamma operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the lgamma operation applied.
    """
    _ = out
    t = input if isinstance(input, Tensor) else Tensor(input)
    return special.gammaln(t)


def linspace(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the linspace operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the linspace operation applied.
    """
    res = _ops.linspace(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def log(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the log operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the log operation applied.
    """
    _ = out
    if isinstance(input, Tensor) and input.requires_grad:
        from .autograd.math_ops import LogBackward

        return LogBackward.apply(input)
    res = _ops.log(_to_tensor(input))
    return _wrap(res)


def log10(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the log10 operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the log10 operation applied.
    """
    _ = out
    res = _ops.log10(_to_tensor(input))
    return _wrap(res)


def log1p(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the log1p operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the log1p operation applied.
    """
    _ = out
    res = _ops.log1p(_to_tensor(input))
    return _wrap(res)


def log2(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the log2 operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the log2 operation applied.
    """
    _ = out
    res = _ops.log2(_to_tensor(input))
    return _wrap(res)


def logaddexp(
    input: TensorLike, other: TensorLike, *, out: Tensor | None = None
) -> Tensor:
    """Applies the logaddexp operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the logaddexp operation applied.
    """
    _ = out
    res = _ops.logaddexp(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def logaddexp2(
    input: TensorLike, other: TensorLike, *, out: Tensor | None = None
) -> Tensor:
    """Applies the logaddexp2 operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the logaddexp2 operation applied.
    """
    _ = out
    res = _ops.logaddexp2(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def logical_and(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the logical_and operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the logical_and operation applied.
    """
    res = _ops.logical_and(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def logical_not(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the logical_not operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the logical_not operation applied.
    """
    res = _ops.logical_not(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def logical_or(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the logical_or operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the logical_or operation applied.
    """
    res = _ops.logical_or(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def logical_xor(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the logical_xor operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the logical_xor operation applied.
    """
    res = _ops.logical_xor(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def logsumexp(
    input: TensorLike,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Tensor:
    """Applies the logsumexp reduction operation.

    Args:
        input: Input tensor or scalar.
        dim: Optional dimension or sequence of dimensions to reduce.
        keepdim: Whether the output tensor has `dim` retained or not.
        axis: Optional alias for `dim`.
        keepdims: Optional alias for `keepdim`.
        dtype: Optional desired data type of returned tensor.

    Returns:
        Tensor: A new tensor with the logsumexp operation applied.
    """
    _ = dtype
    eff_axis = dim if dim is not None else axis
    eff_keepdims = keepdim or keepdims
    res = _reductions.logsumexp(_to_tensor(input), axis=eff_axis, keepdims=eff_keepdims)
    return _wrap(res)


def matmul(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the matmul operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the matmul operation applied.
    """
    if (
        len(args) == 2
        and not kwargs
        and (
            (isinstance(args[0], Tensor) and args[0].requires_grad)
            or (isinstance(args[1], Tensor) and args[1].requires_grad)
        )
    ):
        from .autograd.math_ops import MatmulBackward

        return MatmulBackward.apply(args[0], args[1])
    res = _ops.matmul(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def matrix_power(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the matrix_power operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the matrix_power operation applied.
    """
    res = get_op("MatrixPower")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def max(
    input: TensorLike,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Tensor:
    """Applies the max reduction operation.

    Args:
        input: Input tensor or scalar.
        dim: Optional dimension or sequence of dimensions to reduce.
        keepdim: Whether the output tensor has `dim` retained or not.
        axis: Optional alias for `dim`.
        keepdims: Optional alias for `keepdim`.
        dtype: Optional desired data type of returned tensor.

    Returns:
        Tensor: A new tensor with the max operation applied.
    """
    _ = dtype
    eff_axis = dim if dim is not None else axis
    eff_keepdims = keepdim or keepdims
    res = _reductions.max(_to_tensor(input), axis=eff_axis, keepdims=eff_keepdims)
    return _wrap(res)


def maximum(
    input: TensorLike, other: TensorLike, *, out: Tensor | None = None
) -> Tensor:
    """Applies the maximum operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the maximum operation applied.
    """
    _ = out
    res = _ops.maximum(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def mean(
    input: TensorLike,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Tensor:
    """Applies the mean reduction operation.

    Args:
        input: Input tensor or scalar.
        dim: Optional dimension or sequence of dimensions to reduce.
        keepdim: Whether the output tensor has `dim` retained or not.
        axis: Optional alias for `dim`.
        keepdims: Optional alias for `keepdim`.
        dtype: Optional desired data type of returned tensor.

    Returns:
        Tensor: A new tensor with the mean operation applied.
    """
    _ = dtype
    eff_axis = dim if dim is not None else axis
    eff_keepdims = keepdim or keepdims
    if isinstance(input, Tensor) and input.requires_grad:
        from .autograd.math_ops import MeanBackward

        return MeanBackward.apply(input, eff_axis, eff_keepdims)
    res = _reductions.mean(_to_tensor(input), axis=eff_axis, keepdims=eff_keepdims)
    return _wrap(res)


def meshgrid(*args: OpArg, **kwargs: OpArg) -> tuple[Tensor, ...]:
    """Applies the meshgrid operation.

    Args:
        *args: 1D input tensors defining grid coordinates.
        **kwargs: Optional `indexing` mode (`ij` or `xy`).

    Returns:
        Grid coordinate tensors.
    """
    if "indexing" not in kwargs:
        kwargs["indexing"] = "ij"
    res = _shape_pad_and_tile.meshgrid(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap_tuple(res)


def min(
    input: TensorLike,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Tensor:
    """Applies the min reduction operation.

    Args:
        input: Input tensor or scalar.
        dim: Optional dimension or sequence of dimensions to reduce.
        keepdim: Whether the output tensor has `dim` retained or not.
        axis: Optional alias for `dim`.
        keepdims: Optional alias for `keepdim`.
        dtype: Optional desired data type of returned tensor.

    Returns:
        Tensor: A new tensor with the min operation applied.
    """
    _ = dtype
    eff_axis = dim if dim is not None else axis
    eff_keepdims = keepdim or keepdims
    res = _reductions.min(_to_tensor(input), axis=eff_axis, keepdims=eff_keepdims)
    return _wrap(res)


def minimum(
    input: TensorLike, other: TensorLike, *, out: Tensor | None = None
) -> Tensor:
    """Applies the minimum operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the minimum operation applied.
    """
    _ = out
    res = _ops.minimum(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def mod(input: TensorLike, other: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the mod operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the mod operation applied.
    """
    _ = out
    res = _ops.mod(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def moveaxis(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the moveaxis operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the moveaxis operation applied.
    """
    res = _ops.moveaxis(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def multiply(
    input: TensorLike, other: TensorLike, *, out: Tensor | None = None
) -> Tensor:
    """Applies the multiply operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the multiply operation applied.
    """
    _ = out
    res = _ops.multiply(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def negative(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the negative operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the negative operation applied.
    """
    _ = out
    res = _ops.negative(_to_tensor(input))
    return _wrap(res)


def nextafter(
    input: TensorLike, other: TensorLike, *, out: Tensor | None = None
) -> Tensor:
    """Applies the nextafter operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the nextafter operation applied.
    """
    _ = out
    res = _ops.nextafter(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes the matrix or vector norm of a given tensor.

    Args:
        *args: Input tensor and optional positional arguments.
        **kwargs: Optional `p`, `dim`, and `keepdim` keyword arguments.

    Returns:
        Computed norm tensor.
    """
    tensor_in = args[0]._tensor if isinstance(args[0], Tensor) else args[0]
    p_val = kwargs.get("p", 2)
    p = float(p_val) if isinstance(p_val, (int, float)) else 2.0
    dim = kwargs.get("dim")
    keepdim = bool(kwargs.get("keepdim", False))

    sum_kwargs: dict[str, OpArg] = {}
    if dim is not None:
        sum_kwargs["axis"] = dim
    if keepdim:
        sum_kwargs["keepdims"] = keepdim

    p_tensor = Tensor(p)._tensor
    pow_tensor = _ops.power(tensor_in, p_tensor)
    summed = _reductions.sum(pow_tensor, **sum_kwargs)

    if p == 2.0:
        res = _ops.sqrt(summed)
    else:
        inv_p_tensor = Tensor(1.0 / p)._tensor
        res = _ops.power(summed, inv_p_tensor)

    return _wrap(res)


def not_equal(
    input: TensorLike, other: TensorLike, *, out: Tensor | None = None
) -> Tensor:
    """Applies the not_equal operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the not_equal operation applied.
    """
    _ = out
    res = _ops.not_equal(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def ones(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the ones operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the ones operation applied.
    """
    res = _ops.ones(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def ones_like(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the ones_like operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the ones_like operation applied.
    """
    res = _ops.ones_like(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def outer(input: TensorLike, other: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the outer operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the outer operation applied.
    """
    _ = out
    res = _ops.outer(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def permute(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the permute operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the permute operation applied.
    """
    if (
        len(args) == 2
        and not kwargs
        and isinstance(args[0], Tensor)
        and args[0].requires_grad
    ):
        from .autograd.math_ops import PermuteBackward

        return PermuteBackward.apply(args[0], args[1])
    res = get_op("Transpose")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def pinv(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the pinv operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the pinv operation applied.
    """
    res = get_op("Pinv")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def positive(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the positive operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the positive operation applied.
    """
    _ = out
    res = _ops.positive(_to_tensor(input))
    return _wrap(res)


def power(input: TensorLike, other: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the power operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the power operation applied.
    """
    _ = out
    res = _ops.power(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def prod(
    input: TensorLike,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Tensor:
    """Applies the prod reduction operation.

    Args:
        input: Input tensor or scalar.
        dim: Optional dimension or sequence of dimensions to reduce.
        keepdim: Whether the output tensor has `dim` retained or not.
        axis: Optional alias for `dim`.
        keepdims: Optional alias for `keepdim`.
        dtype: Optional desired data type of returned tensor.

    Returns:
        Tensor: A new tensor with the prod operation applied.
    """
    _ = dtype
    eff_axis = dim if dim is not None else axis
    eff_keepdims = keepdim or keepdims
    res = _reductions.prod(_to_tensor(input), axis=eff_axis, keepdims=eff_keepdims)
    return _wrap(res)


def qr(*args: OpArg, **kwargs: OpArg) -> tuple[Tensor, ...]:
    """Applies the qr operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tuple of `(Q, R)` tensors.
    """
    res = get_op("Qr")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap_tuple(res)


def rad2deg(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the rad2deg operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the rad2deg operation applied.
    """
    _ = out
    res = _ops.rad2deg(_to_tensor(input))
    return _wrap(res)


def real(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the real operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the real operation applied.
    """
    _ = out
    res = _ops.real(_to_tensor(input))
    return _wrap(res)


def reciprocal(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the reciprocal operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the reciprocal operation applied.
    """
    _ = out
    res = _ops.reciprocal(_to_tensor(input))
    return _wrap(res)


def remainder(
    input: TensorLike, other: TensorLike, *, out: Tensor | None = None
) -> Tensor:
    """Applies the remainder operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the remainder operation applied.
    """
    _ = out
    res = _ops.remainder(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def repeat(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the repeat operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the repeat operation applied.
    """
    res = get_op("Repeat")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def reshape(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Returns a tensor with the same data and number of elements as input, with the specified shape.

    Args:
        *args: Input tensor and target shape tuple.
        **kwargs: Keyword arguments (`newshape`).

    Returns:
        Reshaped tensor.
    """
    if len(args) >= 1 and isinstance(args[0], Tensor) and args[0].requires_grad:
        from .autograd.math_ops import ReshapeBackward

        target_shape = (
            args[1] if len(args) > 1 else kwargs.get("newshape", args[0].shape)
        )
        return ReshapeBackward.apply(args[0], target_shape)
    t = args[0]._tensor if isinstance(args[0], Tensor) else args[0]
    if len(args) > 1:
        kwargs["newshape"] = args[1]
    res = get_op("Reshape")()(t, **kwargs)
    return _wrap(res)


def right_shift(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the right_shift operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the right_shift operation applied.
    """
    res = _ops.right_shift(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def roll(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the roll operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the roll operation applied.
    """
    res = get_op("Roll")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def round(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the round operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the round operation applied.
    """
    _ = out
    res = _ops.round(_to_tensor(input))
    return _wrap(res)


def rsqrt(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the rsqrt operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the rsqrt operation applied.
    """
    _ = out
    res = _ops.rsqrt(_to_tensor(input))
    return _wrap(res)


def scatter_add(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the scatter_add operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the scatter_add operation applied.
    """
    res = get_op("ScatterAdd")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def scatter_nd(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the scatter_nd operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the scatter_nd operation applied.
    """
    res = get_op("ScatterNd")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def sign(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the sign operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the sign operation applied.
    """
    _ = out
    res = _ops.sign(_to_tensor(input))
    return _wrap(res)


def sin(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the sin operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the sin operation applied.
    """
    _ = out
    if isinstance(input, Tensor) and input.requires_grad:
        from .autograd.math_ops import SinBackward

        return SinBackward.apply(input)
    res = _ops.sin(_to_tensor(input))
    return _wrap(res)


def sinc(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the sinc operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the sinc operation applied.
    """
    _ = out
    res = _ops.sinc(_to_tensor(input))
    return _wrap(res)


def sinh(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the sinh operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the sinh operation applied.
    """
    _ = out
    res = _ops.sinh(_to_tensor(input))
    return _wrap(res)


def slice(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the slice operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the slice operation applied.
    """
    res = get_op("Slice")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def slogdet(*args: OpArg, **kwargs: OpArg) -> tuple[Tensor, ...]:
    """Computes the sign and natural logarithm of the absolute value of the determinant of a square matrix.

    Args:
        *args: Input square matrix tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Tuple of `(sign, logabsdet)` tensors.
    """
    if len(args) > 0 and isinstance(args[0], Tensor):
        return linalg.slogdet(args[0], **kwargs)
    return tensor(1.0), tensor(0.0)


def split(*args: OpArg, **kwargs: OpArg) -> tuple[Tensor, ...]:
    """Applies the split operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tuple of split tensors.
    """
    res = get_op("Split")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap_tuple(res)


def sqrt(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the sqrt operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the sqrt operation applied.
    """
    _ = out
    if isinstance(input, Tensor) and input.requires_grad:
        from .autograd.math_ops import SqrtBackward

        return SqrtBackward.apply(input)
    res = _ops.sqrt(_to_tensor(input))
    return _wrap(res)


def square(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the square operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the square operation applied.
    """
    _ = out
    res = _ops.square(_to_tensor(input))
    return _wrap(res)


def squeeze(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Returns a tensor with all specified dimensions of input of size 1 removed.

    Args:
        *args: Input tensor and optional dimension index.
        **kwargs: Optional `dim` or `axis` keyword arguments.

    Returns:
        Squeezed tensor.
    """
    dim = kwargs.get("dim", kwargs.get("axis"))
    if len(args) > 1:
        dim = args[1]
    if len(args) >= 1 and isinstance(args[0], Tensor) and args[0].requires_grad:
        from .autograd.math_ops import SqueezeBackward

        return SqueezeBackward.apply(args[0], dim)
    t = args[0]._tensor if isinstance(args[0], Tensor) else args[0]
    res = get_op("Squeeze")()(t, axis=dim)
    return _wrap(res)


def stack(*args: OpArg | Sequence[Tensor], **kwargs: OpArg) -> Tensor:
    """Concatenates a sequence of tensors along a new dimension.

    Args:
        *args: Sequence of tensors to stack and optional dimension.
        **kwargs: Optional `tensors` and `dim` keyword arguments.

    Returns:
        Stacked tensor.
    """
    raw_t = args[0] if len(args) > 0 else kwargs.get("tensors")
    seq = raw_t if isinstance(raw_t, (list, tuple)) else []
    axis = args[1] if len(args) > 1 else kwargs.pop("dim", kwargs.pop("axis", 0))
    if _builtin_any(isinstance(t, Tensor) and t.requires_grad for t in seq):
        from .autograd.math_ops import StackBackward

        return StackBackward.apply(axis, *seq)
    unwrapped = [a._tensor if isinstance(a, Tensor) else a for a in seq]
    kwargs["axis"] = axis
    return _wrap(get_op("Stack")()(unwrapped, **kwargs))


def std(
    input: TensorLike,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Tensor:
    """Applies the std reduction operation.

    Args:
        input: Input tensor or scalar.
        dim: Optional dimension or sequence of dimensions to reduce.
        keepdim: Whether the output tensor has `dim` retained or not.
        axis: Optional alias for `dim`.
        keepdims: Optional alias for `keepdim`.
        dtype: Optional desired data type of returned tensor.

    Returns:
        Tensor: A new tensor with the std operation applied.
    """
    _ = dtype
    eff_axis = dim if dim is not None else axis
    eff_keepdims = keepdim or keepdims
    res = _reductions.std(_to_tensor(input), axis=eff_axis, keepdims=eff_keepdims)
    return _wrap(res)


def strided_slice(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the strided_slice operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the strided_slice operation applied.
    """
    res = get_op("StridedSlice")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def subtract(
    input: TensorLike, other: TensorLike, *, out: Tensor | None = None
) -> Tensor:
    """Applies the subtract operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the subtract operation applied.
    """
    _ = out
    res = _ops.subtract(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def sum(
    input: TensorLike,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Tensor:
    """Applies the sum reduction operation.

    Args:
        input: Input tensor or scalar.
        dim: Optional dimension or sequence of dimensions to reduce.
        keepdim: Whether the output tensor has `dim` retained or not.
        axis: Optional alias for `dim`.
        keepdims: Optional alias for `keepdim`.
        dtype: Optional desired data type of returned tensor.

    Returns:
        Tensor: A new tensor with the sum operation applied.
    """
    _ = dtype
    eff_axis = dim if dim is not None else axis
    eff_keepdims = keepdim or keepdims
    if isinstance(input, Tensor) and input.requires_grad:
        from .autograd.math_ops import SumBackward

        return SumBackward.apply(input, eff_axis, eff_keepdims)
    res = _reductions.sum(_to_tensor(input), axis=eff_axis, keepdims=eff_keepdims)
    return _wrap(res)


def svd(*args: OpArg, **kwargs: OpArg) -> tuple[Tensor, ...]:
    """Applies the svd operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tuple of `(U, S, Vh)` tensors.
    """
    res = get_op("Svd")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap_tuple(res)


def swapaxes(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the swapaxes operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the swapaxes operation applied.
    """
    res = _ops.swapaxes(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def take(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the take operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the take operation applied.
    """
    res = get_op("Take")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def tan(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the tan operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the tan operation applied.
    """
    _ = out
    res = _ops.tan(_to_tensor(input))
    return _wrap(res)


def tanh(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the tanh operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the tanh operation applied.
    """
    _ = out
    if isinstance(input, Tensor) and input.requires_grad:
        from .autograd.math_ops import TanhBackward

        return TanhBackward.apply(input)
    res = _ops.tanh(_to_tensor(input))
    return _wrap(res)


def tensordot(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the tensordot operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the tensordot operation applied.
    """
    res = _linalg_einsum.tensordot(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def tile(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the tile operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the tile operation applied.
    """
    res = get_op("Tile")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def transpose(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Returns a tensor that is a transposed version of input.

    Args:
        *args: Input tensor and optional `dim0`, `dim1` dimension indices.
        **kwargs: Optional `dim0`, `dim1`, or `axes` keyword arguments.

    Returns:
        Transposed tensor.
    """
    if (
        len(args) == 3
        and not kwargs
        and isinstance(args[0], Tensor)
        and args[0].requires_grad
        and isinstance(args[1], int)
        and isinstance(args[2], int)
    ):
        from .autograd.math_ops import TransposeBackward

        return TransposeBackward.apply(args[0], args[1], args[2])
    t = args[0]._tensor if isinstance(args[0], Tensor) else args[0]
    shape_len = len(t.shape) if isinstance(t, (_SwitcherooTensor, ProxyTensor)) else 2
    rank = _builtin_max(shape_len, 2)
    if len(args) == 3 and isinstance(args[1], int) and isinstance(args[2], int):
        dim0, dim1 = args[1], args[2]
        axes = list(range(rank))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        kwargs["axes"] = axes
    elif "dim0" in kwargs and "dim1" in kwargs:
        d0, d1 = kwargs.pop("dim0"), kwargs.pop("dim1")
        if isinstance(d0, int) and isinstance(d1, int):
            axes = list(range(rank))
            axes[d0], axes[d1] = axes[d1], axes[d0]
            kwargs["axes"] = axes

    res = get_op("Transpose")()(t, **kwargs)
    return _wrap(res)


def tril(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the tril operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the tril operation applied.
    """
    res = _shape_pad_and_tile.tril(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def triu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the triu operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the triu operation applied.
    """
    res = _shape_pad_and_tile.triu(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def trunc(input: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the trunc operation element-wise.

    Args:
        input: Input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the trunc operation applied.
    """
    _ = out
    res = _ops.trunc(_to_tensor(input))
    return _wrap(res)


def unsqueeze(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Returns a new tensor with a dimension of size one inserted at the specified position.

    Args:
        *args: Input tensor and dimension index `dim`.
        **kwargs: Optional `dim` keyword argument.

    Returns:
        Unsqueezed tensor.
    """
    t = args[0]
    assert isinstance(t, Tensor)
    raw_dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
    dim = int(raw_dim) if isinstance(raw_dim, int) else 0
    shape_list = list(t.shape)
    if dim < 0:
        dim += len(shape_list) + 1
    shape_list.insert(dim, 1)
    return reshape(t, tuple(shape_list))


def unstack(*args: OpArg, **kwargs: OpArg) -> list[Tensor]:
    """Applies the unstack operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        List of unstacked tensors.
    """
    res = _ops.unstack(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap_list(res)


def update_slice(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the update_slice operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the update_slice operation applied.
    """
    res = get_op("UpdateSlice")()(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def variance(
    input: TensorLike,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> Tensor:
    """Applies the variance reduction operation.

    Args:
        input: Input tensor or scalar.
        dim: Optional dimension or sequence of dimensions to reduce.
        keepdim: Whether the output tensor has `dim` retained or not.
        axis: Optional alias for `dim`.
        keepdims: Optional alias for `keepdim`.
        dtype: Optional desired data type of returned tensor.

    Returns:
        Tensor: A new tensor with the variance operation applied.
    """
    _ = dtype
    eff_axis = dim if dim is not None else axis
    eff_keepdims = keepdim or keepdims
    res = _reductions.variance(_to_tensor(input), axis=eff_axis, keepdims=eff_keepdims)
    return _wrap(res)


def vdot(input: TensorLike, other: TensorLike, *, out: Tensor | None = None) -> Tensor:
    """Applies the vdot operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the vdot operation applied.
    """
    _ = out
    res = _ops.vdot(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def where(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the where operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the where operation applied.
    """
    res = _shape_indexing.where(
        *[
            _to_tensor(a) if isinstance(a, (Tensor, int, float, bool)) else a
            for a in args
        ],
        **kwargs,
    )
    return _wrap(res)


def zeros(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the zeros operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the zeros operation applied.
    """
    res = _ops.zeros(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def zeros_like(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the zeros_like operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the zeros_like operation applied.
    """
    res = _ops.zeros_like(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def tensor(
    data: OpArg | _ArrayDataProtocol = None,
    requires_grad: bool = False,
    dtype: DTypeLike = None,
) -> Tensor:
    """Constructs a tensor with no autograd history by copying data.

    Args:
        data: Initial data for the tensor.
        requires_grad: If autograd should record operations on the returned tensor.
        dtype: Desired data type of returned tensor.

    Returns:
        Constructed `Tensor`.
    """
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


float32 = DType.Float32
int32 = DType.Int32


def logit(input: TensorLike, eps: float | None = None, **kwargs: OpArg) -> Tensor:
    """Returns a new tensor with the logit of the elements of `input`.

    Args:
        input: Input tensor.
        eps: Epsilon for clamping input to `[eps, 1 - eps]`.
        **kwargs: Optional keyword arguments.

    Returns:
        Logit tensor `log(x / (1 - x))`.
    """
    _ = kwargs
    input_t = _to_tensor(input)
    if eps is not None:
        input_t = _ops.clip(input_t, eps, 1.0 - eps)
    return _wrap(_ops.log(input_t / (1.0 - input_t)))


def signbit(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Tests if each element of `input` has its sign bit set (is less than zero).

    Args:
        input: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Boolean tensor indicating negative elements.
    """
    _ = kwargs
    return input < 0


def true_divide(
    input: TensorLike, other: TensorLike, *, out: Tensor | None = None
) -> Tensor:
    """Applies the true_divide operation element-wise.

    Args:
        input: First input tensor or scalar.
        other: Second input tensor or scalar.
        out: Optional output tensor.

    Returns:
        Tensor: A new tensor with the true_divide operation applied.
    """
    _ = out
    res = _ops.divide(_to_tensor(input), _to_tensor(other))
    return _wrap(res)


def xlogy(x: TensorLike, y: TensorLike, **kwargs: OpArg) -> Tensor:
    """Computes `x * log(y)` with `0` where `x == 0`.

    Args:
        x: Multiplier tensor or scalar.
        y: Argument to logarithm tensor or scalar.
        **kwargs: Optional keyword arguments.

    Returns:
        Computed `x * log(y)` tensor.
    """
    _ = kwargs
    x_t = _to_tensor(x)
    y_t = _to_tensor(y)
    res = _shape_indexing.where(x_t == 0, _to_tensor(0.0), x_t * _ops.log(y_t))
    return _wrap(res)


def mvlgamma(input: Tensor, p: int, **kwargs: OpArg) -> Tensor:
    """Computes the multivariate log-gamma function with dimension `p` element-wise.

    Args:
        input: Input tensor.
        p: Number of dimensions.
        **kwargs: Optional keyword arguments.

    Returns:
        Multivariate log-gamma tensor.
    """
    _ = kwargs
    res = input * 0.0
    for i in range(1, p + 1):
        shifted = input + (1 - i) / 2.0
        res = res + special.gammaln(shifted)
    res = res + (p * (p - 1) / 4.0) * math.log(math.pi)
    return res


def nan_to_num(
    input: TensorLike,
    nan: float | None = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
    **kwargs: OpArg,
) -> Tensor:
    """Replaces `NaN`, positive infinity, and negative infinity values in `input`.

    Args:
        input: Input tensor.
        nan: Value with which to replace `NaN`s.
        posinf: Value with which to replace positive infinity values.
        neginf: Value with which to replace negative infinity values.
        **kwargs: Optional keyword arguments.

    Returns:
        Sanitized tensor with finite values.
    """
    _ = kwargs
    res = _to_tensor(input)
    if nan is not None:
        res = _shape_indexing.where(_ops.isnan(res), _to_tensor(nan), res)

    pos_replacement = 3.402823466e38 if posinf is None else posinf
    res = _shape_indexing.where(
        _ops.logical_and(_ops.isinf(res), res > 0),
        _to_tensor(pos_replacement),
        res,
    )

    neg_replacement = -3.402823466e38 if neginf is None else neginf
    res = _shape_indexing.where(
        _ops.logical_and(_ops.isinf(res), res < 0),
        _to_tensor(neg_replacement),
        res,
    )

    return _wrap(res)


_RANDOM_SEED: int | None = None


def manual_seed(seed: int) -> int:
    """Sets the seed for generating random numbers.

    Args:
        seed: The desired seed integer.

    Returns:
        The applied seed integer.
    """
    global _RANDOM_SEED
    _RANDOM_SEED = seed
    random.seed(seed)
    return seed


def rand(*size: int | Sequence[int], **kwargs: OpArg) -> Tensor:
    """Returns a tensor filled with random numbers from a uniform distribution on `[0, 1)`.

    Args:
        *size: Sequence of integers defining the shape of the output tensor.
        **kwargs: Optional keyword arguments (`dtype`, `device`).

    Returns:
        Uniform random tensor.
    """
    resolved = (
        size[0] if len(size) == 1 and isinstance(size[0], (tuple, list)) else size
    )
    return _wrap(_creation.rand(*resolved, **kwargs))


def randn(*size: int | Sequence[int], **kwargs: OpArg) -> Tensor:
    """Returns a tensor filled with random numbers from a standard normal distribution.

    Args:
        *size: Sequence of integers defining the shape of the output tensor.
        **kwargs: Optional keyword arguments (`dtype`, `device`).

    Returns:
        Standard normal random tensor.
    """
    resolved = (
        size[0] if len(size) == 1 and isinstance(size[0], (tuple, list)) else size
    )
    return _wrap(_creation.randn(*resolved, **kwargs))


def randint(
    low: int,
    high: int | Sequence[int] | None = None,
    size: int | Sequence[int] | None = None,
    **kwargs: OpArg,
) -> Tensor:
    """Returns a tensor filled with random integers generated uniformly between `low` and `high`.

    Args:
        low: Lowest integer to be drawn from the distribution (or `high` when `high` is a shape).
        high: One above the highest integer to be drawn, or shape tuple when `size` is omitted.
        size: Tuple defining the shape of the output tensor.
        **kwargs: Optional keyword arguments (`dtype`, `device`).

    Returns:
        Random integer tensor.
    """
    resolved_size: Sequence[int]
    resolved_low: int = low
    resolved_high: int
    if size is None:
        if isinstance(high, (tuple, list)):
            resolved_size = high
            resolved_high = low
            resolved_low = 0
        else:
            resolved_size = ()
            resolved_high = int(high) if isinstance(high, int) else low
            if high is None:
                resolved_low = 0
    else:
        resolved_size = (size,) if isinstance(size, int) else size
        if isinstance(high, int):
            resolved_high = high
        else:
            resolved_high = low
            resolved_low = 0
    return _wrap(
        _creation.randint(resolved_low, resolved_high, resolved_size, **kwargs)
    )


def clip(
    input: Tensor,
    min: float | Tensor | None = None,
    max: float | Tensor | None = None,
    **kwargs: OpArg,
) -> Tensor:
    """Clamps all elements in `input` into the range `[min, max]`.

    Args:
        input: Input tensor.
        min: Lower-bound of the range to be clamped to.
        max: Upper-bound of the range to be clamped to.
        **kwargs: Optional keyword arguments.

    Returns:
        Clamped tensor.
    """
    _ = kwargs
    out = input
    if min is not None:
        out = maximum(out, min)
    if max is not None:
        out = minimum(out, max)
    return out


def clamp(
    input: Tensor,
    min: float | Tensor | None = None,
    max: float | Tensor | None = None,
    **kwargs: OpArg,
) -> Tensor:
    """Clamps all elements in `input` into the range `[min, max]`.

    Args:
        input: Input tensor.
        min: Lower-bound of the range to be clamped to.
        max: Upper-bound of the range to be clamped to.
        **kwargs: Optional keyword arguments.

    Returns:
        Clamped tensor.
    """
    return clip(input, min=min, max=max, **kwargs)


def sigmoid(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Applies the element-wise function `Sigmoid(x) = 1 / (1 + exp(-x))`.

    Args:
        input: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Sigmoid output tensor.
    """
    return nn.functional.sigmoid(input, **kwargs)


def softmax(input: Tensor, dim: int | None = None, **kwargs: OpArg) -> Tensor:
    """Applies a softmax function along `dim`.

    Args:
        input: Input tensor.
        dim: Dimension along which softmax will be computed.
        **kwargs: Optional keyword arguments (`dtype`).

    Returns:
        Softmax output tensor.
    """
    return nn.functional.softmax(input, dim=dim, **kwargs)
