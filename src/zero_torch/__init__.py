"""Zero-dependency PyTorch API frontend backed by ml-switcheroo-compiler."""

from __future__ import annotations

import math
import random
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


def abs(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the abs operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the abs operation applied.
    """
    res = _ops.abs(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def acos(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the acos operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the acos operation applied.
    """
    res = _ops.acos(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def acosh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the acosh operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the acosh operation applied.
    """
    res = _ops.acosh(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def add(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the add operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the add operation applied.
    """
    res = _ops.add(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def all(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the all operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the all operation applied.
    """
    res = _reductions.all(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def any(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the any operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the any operation applied.
    """
    res = _reductions.any(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def argmax(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the argmax operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the argmax operation applied.
    """
    res = _reductions.argmax(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def argmin(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the argmin operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the argmin operation applied.
    """
    res = _reductions.argmin(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def asin(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the asin operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the asin operation applied.
    """
    res = _ops.asin(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def asinh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the asinh operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the asinh operation applied.
    """
    res = _ops.asinh(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def atan(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the atan operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the atan operation applied.
    """
    res = _ops.atan(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def atan2(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the atan2 operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the atan2 operation applied.
    """
    res = _ops.atan2(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def atanh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the atanh operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the atanh operation applied.
    """
    res = _ops.atanh(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def binary(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the binary operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the binary operation applied.
    """
    res = _ops.binary(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def cbrt(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the cbrt operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the cbrt operation applied.
    """
    res = _ops.cbrt(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def ceil(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the ceil operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the ceil operation applied.
    """
    res = _ops.ceil(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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
        **kwargs: Arbitrary keyword arguments (`tensors`, `dim`).

    Returns:
        Tensor: Concatenated tensor.
    """
    raw_tensors = args[0] if len(args) > 0 else kwargs.get("tensors")
    dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
    seq = raw_tensors if isinstance(raw_tensors, (list, tuple)) else []
    tensors_t = [t._tensor if isinstance(t, Tensor) else t for t in seq]
    res = get_op("Concatenate")()(tensors_t, axis=dim)
    return _wrap(res)


def conj(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the conj operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the conj operation applied.
    """
    res = _ops.conj(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def copysign(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the copysign operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the copysign operation applied.
    """
    res = _ops.copysign(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def cos(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the cos operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the cos operation applied.
    """
    res = _ops.cos(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def cosh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the cosh operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the cosh operation applied.
    """
    res = _ops.cosh(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def count_nonzero(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the count_nonzero operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the count_nonzero operation applied.
    """
    res = _reductions.count_nonzero(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def creation(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the creation operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the creation operation applied.
    """
    res = _ops.creation(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def deg2rad(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the deg2rad operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the deg2rad operation applied.
    """
    res = _ops.deg2rad(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def digamma(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the digamma operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the digamma operation applied.
    """
    res = _ops.digamma(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def divide(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the divide operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the divide operation applied.
    """
    res = _ops.divide(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def divmod(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the divmod operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: Tuple of quotient and remainder tensors.
    """
    res = _ops.divmod(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def equal(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the equal operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the equal operation applied.
    """
    res = _ops.equal(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def erf(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the erf operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the erf operation applied.
    """
    res = _ops.erf(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def erfc(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the erfc operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the erfc operation applied.
    """
    res = _ops.erfc(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def erfinv(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the erfinv operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the erfinv operation applied.
    """
    res = _ops.erfinv(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def exp(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the exp operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the exp operation applied.
    """
    res = _ops.exp(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def exp2(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the exp2 operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the exp2 operation applied.
    """
    res = _ops.exp2(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def expm1(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the expm1 operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the expm1 operation applied.
    """
    res = _ops.expm1(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def fix(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the fix operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the fix operation applied.
    """
    res = _ops.fix(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
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


def float_power(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the float_power operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the float_power operation applied.
    """
    res = _ops.float_power(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def floor(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the floor operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the floor operation applied.
    """
    res = _ops.floor(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def floor_divide(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the floor_divide operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the floor_divide operation applied.
    """
    res = _ops.floor_divide(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def fmax(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the fmax operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the fmax operation applied.
    """
    res = _ops.fmax(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def fmin(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the fmin operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the fmin operation applied.
    """
    res = _ops.fmin(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def fmod(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the fmod operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the fmod operation applied.
    """
    res = _ops.fmod(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def greater(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the greater operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the greater operation applied.
    """
    res = _ops.greater(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def greater_equal(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the greater_equal operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the greater_equal operation applied.
    """
    res = _ops.greater_equal(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def heaviside(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the heaviside operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the heaviside operation applied.
    """
    res = _ops.heaviside(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def hypot(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the hypot operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the hypot operation applied.
    """
    res = _ops.hypot(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def imag(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the imag operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the imag operation applied.
    """
    res = _ops.imag(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def inner(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the inner operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the inner operation applied.
    """
    res = _ops.inner(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def isclose(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the isclose operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the isclose operation applied.
    """
    res = _ops.isclose(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def isfinite(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the isfinite operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the isfinite operation applied.
    """
    res = _ops.isfinite(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def isinf(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the isinf operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the isinf operation applied.
    """
    res = _ops.isinf(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def isnan(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the isnan operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the isnan operation applied.
    """
    res = _ops.isnan(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def ldexp(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the ldexp operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the ldexp operation applied.
    """
    res = _ops.ldexp(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def less(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the less operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the less operation applied.
    """
    res = _ops.less(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def less_equal(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the less_equal operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the less_equal operation applied.
    """
    res = _ops.less_equal(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def lgamma(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the lgamma operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the lgamma operation applied.
    """
    inp = args[0] if len(args) > 0 else kwargs.get("input", 1.0)
    t = inp if isinstance(inp, Tensor) else Tensor(inp)
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


def log(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the log operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the log operation applied.
    """
    res = _ops.log(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def log10(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the log10 operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the log10 operation applied.
    """
    res = _ops.log10(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def log1p(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the log1p operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the log1p operation applied.
    """
    res = _ops.log1p(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def log2(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the log2 operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the log2 operation applied.
    """
    res = _ops.log2(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def logaddexp(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the logaddexp operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the logaddexp operation applied.
    """
    res = _ops.logaddexp(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def logaddexp2(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the logaddexp2 operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the logaddexp2 operation applied.
    """
    res = _ops.logaddexp2(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def logsumexp(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the logsumexp operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the logsumexp operation applied.
    """
    res = _reductions.logsumexp(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def matmul(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the matmul operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the matmul operation applied.
    """
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


def max(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the max operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the max operation applied.
    """
    res = _reductions.max(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def maximum(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the maximum operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the maximum operation applied.
    """
    res = _ops.maximum(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def mean(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the mean operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the mean operation applied.
    """
    res = _reductions.mean(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def min(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the min operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the min operation applied.
    """
    res = _reductions.min(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def minimum(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the minimum operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the minimum operation applied.
    """
    res = _ops.minimum(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def mod(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the mod operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the mod operation applied.
    """
    res = _ops.mod(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
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


def multiply(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the multiply operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the multiply operation applied.
    """
    res = _ops.multiply(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def negative(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the negative operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the negative operation applied.
    """
    res = _ops.negative(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def nextafter(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the nextafter operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the nextafter operation applied.
    """
    res = _ops.nextafter(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def not_equal(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the not_equal operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the not_equal operation applied.
    """
    res = _ops.not_equal(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def outer(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the outer operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the outer operation applied.
    """
    res = _ops.outer(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def permute(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the permute operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the permute operation applied.
    """
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


def positive(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the positive operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the positive operation applied.
    """
    res = _ops.positive(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def power(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the power operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the power operation applied.
    """
    res = _ops.power(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def prod(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the prod operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the prod operation applied.
    """
    res = _reductions.prod(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def rad2deg(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the rad2deg operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the rad2deg operation applied.
    """
    res = _ops.rad2deg(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def real(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the real operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the real operation applied.
    """
    res = _ops.real(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def reciprocal(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the reciprocal operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the reciprocal operation applied.
    """
    res = _ops.reciprocal(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def reductions(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the reductions operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the reductions operation applied.
    """
    res = _ops.reductions(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def remainder(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the remainder operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the remainder operation applied.
    """
    res = _ops.remainder(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def round(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the round operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the round operation applied.
    """
    res = _ops.round(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def rsqrt(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the rsqrt operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the rsqrt operation applied.
    """
    res = _ops.rsqrt(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def scatter(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the scatter operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the scatter operation applied.
    """
    res = _ops.scatter(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def shape(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the shape operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the shape operation applied.
    """
    res = _ops.shape(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def sign(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the sign operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the sign operation applied.
    """
    res = _ops.sign(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def sin(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the sin operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the sin operation applied.
    """
    res = _ops.sin(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def sinc(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the sinc operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the sinc operation applied.
    """
    res = _ops.sinc(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def sinh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the sinh operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the sinh operation applied.
    """
    res = _ops.sinh(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def sqrt(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the sqrt operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the sqrt operation applied.
    """
    res = _ops.sqrt(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def square(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the square operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the square operation applied.
    """
    res = _ops.square(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def squeeze(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Returns a tensor with all specified dimensions of input of size 1 removed.

    Args:
        *args: Input tensor and optional dimension index.
        **kwargs: Optional `dim` or `axis` keyword arguments.

    Returns:
        Squeezed tensor.
    """
    t = args[0]._tensor if isinstance(args[0], Tensor) else args[0]
    dim = kwargs.get("dim", kwargs.get("axis"))
    if len(args) > 1:
        dim = args[1]
    res = get_op("Squeeze")()(t, axis=dim)
    return _wrap(res)


def stack(*args: OpArg | Sequence[Tensor], **kwargs: OpArg) -> Tensor:
    """Concatenates a sequence of tensors along a new dimension.

    Args:
        *args: Sequence of tensors to stack.
        **kwargs: Optional `tensors` and `dim` keyword arguments.

    Returns:
        Stacked tensor.
    """
    raw_t = args[0] if len(args) > 0 else kwargs.get("tensors")
    seq = raw_t if isinstance(raw_t, (list, tuple)) else []
    unwrapped = [a._tensor if isinstance(a, Tensor) else a for a in seq]
    kwargs["axis"] = kwargs.pop("dim", 0)
    return _wrap(get_op("Stack")()(unwrapped, **kwargs))


def std(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the std operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the std operation applied.
    """
    res = _reductions.std(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def subtract(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the subtract operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the subtract operation applied.
    """
    res = _ops.subtract(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def sum(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the sum operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the sum operation applied.
    """
    res = _reductions.sum(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def tan(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the tan operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the tan operation applied.
    """
    res = _ops.tan(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def tanh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the tanh operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the tanh operation applied.
    """
    res = _ops.tanh(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def trunc(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the trunc operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the trunc operation applied.
    """
    res = _ops.trunc(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def unary(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the unary operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the unary operation applied.
    """
    res = _ops.unary(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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


def variance(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the variance operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the variance operation applied.
    """
    res = _reductions.variance(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def vdot(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the vdot operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the vdot operation applied.
    """
    res = _ops.vdot(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
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
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
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


def true_divide(dividend: TensorLike, divisor: TensorLike, **kwargs: OpArg) -> Tensor:
    """Performs true division of `dividend` by `divisor`.

    Args:
        dividend: Numerator tensor or scalar.
        divisor: Denominator tensor or scalar.
        **kwargs: Optional keyword arguments.

    Returns:
        Element-wise quotient tensor.
    """
    res = _ops.true_divide(
        *[a._tensor if isinstance(a, Tensor) else a for a in (dividend, divisor)],
        **kwargs,
    )
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
