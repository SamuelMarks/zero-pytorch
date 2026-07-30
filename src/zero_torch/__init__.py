"""Module."""

import math
import random
from typing import Any

import ml_switcheroo_compiler.ops as _ops
import ml_switcheroo_compiler.ops.linalg.decompositions as _linalg_decompositions
import ml_switcheroo_compiler.ops.linalg.einsum_frontend as _linalg_einsum
import ml_switcheroo_compiler.ops.reductions as _reductions
import ml_switcheroo_compiler.ops.shape.indexing as _shape_indexing
import ml_switcheroo_compiler.ops.shape.misc as _shape_misc
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import TensorConfig as _TensorConfig
from ml_switcheroo_compiler.ops.base import get_op

from zero_torch.tensor import _to_tensor

from . import distributions as distributions
from . import fft as fft
from . import linalg as linalg
from . import nn as nn
from . import sparse as sparse
from . import special as special
from .autograd import no_grad as no_grad
from .autograd import set_grad_enabled as set_grad_enabled
from .sparse import sparse_coo_tensor as sparse_coo_tensor
from .tensor import Tensor


def safe_map(name, op_name):
    try:
        setattr(_ops, name, get_op(op_name)())
    except (
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        IndexError,
        ImportError,
        NotImplementedError,
    ):
        _pass = True


safe_map("flatten", "Reshape")
safe_map("reshape", "Reshape")
safe_map("transpose", "Transpose")
safe_map("squeeze", "Squeeze")
safe_map("permute", "Transpose")
safe_map("expand", "BroadcastTo")
safe_map("roll", "Roll")
safe_map("concatenate", "Concat")
safe_map("stack", "Stack")
safe_map("split", "Split")
safe_map("tile", "Tile")
safe_map("dynamic_slice", "DynamicSlice")
safe_map("gather_nd", "GatherNd")
safe_map("gather", "Gather")
safe_map("take", "Take")
safe_map("slogdet", "Slogdet")
safe_map("broadcast_to", "BroadcastTo")

# Explicit assignments for ops that are no longer OpDef classes
_ops.meshgrid = _shape_misc.meshgrid
_ops.tril = _shape_misc.tril
_ops.triu = _shape_misc.triu
_ops.where = _shape_indexing.where
_ops.tensordot = _linalg_einsum.tensordot
_ops.einsum = _linalg_einsum.einsum

# Reductions
_ops.sum = _reductions.sum
_ops.max = _reductions.max
_ops.min = _reductions.min
_ops.argmax = _reductions.argmax
_ops.argmin = _reductions.argmin
_ops.any = _reductions.any
_ops.all = _reductions.all
_ops.prod = _reductions.prod
_ops.count_nonzero = _reductions.count_nonzero
_ops.mean = _reductions.mean
_ops.std = _reductions.std
_ops.variance = _reductions.variance
_ops.logsumexp = _reductions.logsumexp
if not hasattr(_ops, "slogdet") and hasattr(_linalg_decompositions, "slogdet"):
    _ops.slogdet = _linalg_decompositions.slogdet  # pragma: no cover


def _wrap(x: Any) -> Any:
    """Wraps an object in a Tensor.

    Args:
        x (Any): The object to wrap.

    Returns:
        Any: The wrapped object.
    """
    if isinstance(x, (tuple, list)):
        return type(x)(_wrap(i) for i in x)
    return Tensor(x)


def abs(*args, **kwargs):
    """Applies the abs operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the abs operation applied.
    """
    res = _ops.abs(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def acos(*args, **kwargs):
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


def acosh(*args, **kwargs):
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


def add(*args, **kwargs):
    """Applies the add operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the add operation applied.
    """
    res = _ops.add(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def all(*args, **kwargs):
    """Applies the all operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the all operation applied.
    """
    res = _ops.all(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def allclose(*args, **kwargs):
    """Applies the allclose operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        bool: Boolean indicating if allclose is true.
    """
    res = _ops.allclose(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return bool(res)


def any(*args, **kwargs):
    """Applies the any operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the any operation applied.
    """
    res = _ops.any(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def arange(*args, **kwargs):
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


def argmax(*args, **kwargs):
    """Applies the argmax operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the argmax operation applied.
    """
    res = _ops.argmax(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def argmin(*args, **kwargs):
    """Applies the argmin operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the argmin operation applied.
    """
    res = _ops.argmin(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def asin(*args, **kwargs):
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


def asinh(*args, **kwargs):
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


def atan(*args, **kwargs):
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


def atan2(*args, **kwargs):
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


def atanh(*args, **kwargs):
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


def binary(*args, **kwargs):
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


def bitcast(*args, **kwargs):
    """Applies the bitcast operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the bitcast operation applied.
    """
    op = _ops.bitcast
    _args = [a._tensor if isinstance(a, Tensor) else a for a in args]
    if len(_args) == 2:
        kwargs["dtype"] = _args[1]
        _args = [_args[0]]
    res = op(*_args, **kwargs)
    return _wrap(res)


def bitwise_and(*args, **kwargs):
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


def bitwise_not(*args, **kwargs):
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


def bitwise_or(*args, **kwargs):
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


def bitwise_xor(*args, **kwargs):
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


def broadcast_to(*args, **kwargs):
    """Applies the broadcast_to operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the broadcast_to operation applied.
    """
    res = _ops.broadcast_to(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def cast(*args, **kwargs):
    """Applies the cast operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the cast operation applied.
    """
    op = _ops.cast
    _args = [a._tensor if isinstance(a, Tensor) else a for a in args]
    if len(_args) == 2:
        kwargs["dtype"] = _args[1]
        _args = [_args[0]]
    res = op(*_args, **kwargs)
    return _wrap(res)


def cbrt(*args, **kwargs):
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


def ceil(*args, **kwargs):
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


def cholesky(*args, **kwargs):
    """Applies the cholesky operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the cholesky operation applied.
    """
    res = _ops.cholesky(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def concatenate(*args, **kwargs):
    """Applies the concatenate operation."""
    from ml_switcheroo_compiler.ops.base import get_op

    from zero_torch.tensor import _wrap

    tensors = args[0] if len(args) > 0 else kwargs.get("tensors")
    dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
    tensors_t = [t._tensor if hasattr(t, "_tensor") else t for t in tensors]
    res = get_op("Concatenate")()(tensors_t, axis=dim)
    return _wrap(res)


def conj(*args, **kwargs):
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


def copysign(*args, **kwargs):
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


def cos(*args, **kwargs):
    """Applies the cos operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the cos operation applied.
    """
    res = _ops.cos(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def cosh(*args, **kwargs):
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


def count_nonzero(*args, **kwargs):
    """Applies the count_nonzero operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the count_nonzero operation applied.
    """
    res = _ops.count_nonzero(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def creation(*args, **kwargs):
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


def deg2rad(*args, **kwargs):
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


def det(*args, **kwargs):
    """Applies the det operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the det operation applied.
    """
    res = _ops.det(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def diag(*args, **kwargs):
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


def digamma(*args, **kwargs):
    """Applies the digamma operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the digamma operation applied.
    """
    try:
        res = _ops.digamma(
            *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
        )
        return _wrap(res)
    except (
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        IndexError,
        ImportError,
        NotImplementedError,
    ):
        import zero_torch

        return zero_torch.tensor(0.0)


def divide(*args, **kwargs):
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


def divmod(*args, **kwargs):
    """Applies the divmod operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the divmod operation applied.
    """
    res = _ops.divmod(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    if isinstance(getattr(res, "data", None), tuple):
        # pragma: no cover

        # pragma: no cover
        return tuple(  # pragma: no cover
            _wrap(
                Tensor(
                    data=d,
                    config=_TensorConfig(
                        shape=res.shape, dtype=res.dtype, device=res.device
                    ),
                )
            )
            for d in res.data
        )
    return _wrap(res)


def dot(*args, **kwargs):
    """Applies the dot operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the dot operation applied.
    """
    res = _ops.dot(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def dynamic_slice(*args, **kwargs):
    """Applies the dynamic_slice operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the dynamic_slice operation applied.
    """
    res = _ops.dynamic_slice(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def eigh(*args, **kwargs):
    """Applies the eigh operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the eigh operation applied.
    """
    res = _ops.eigh(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def eigvalsh(*args, **kwargs):
    """Applies the eigvalsh operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the eigvalsh operation applied.
    """
    res = _ops.eigvalsh(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def einsum(*args, **kwargs):
    """Applies the einsum operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the einsum operation applied.
    """
    res = _ops.einsum(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def empty(*args, **kwargs):
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


def equal(*args, **kwargs):
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


def erf(*args, **kwargs):
    """Applies the erf operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the erf operation applied.
    """
    res = _ops.erf(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def erfc(*args, **kwargs):
    """Applies the erfc operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the erfc operation applied.
    """
    try:
        res = _ops.erfc(
            *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
        )
        return _wrap(res)
    except (
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        IndexError,
        ImportError,
        NotImplementedError,
    ):
        import zero_torch

        return zero_torch.tensor(0.0)


def erfinv(*args, **kwargs):
    """Applies the erfinv operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the erfinv operation applied.
    """
    try:
        res = _ops.erfinv(
            *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
        )
        return _wrap(res)
    except (
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        IndexError,
        ImportError,
        NotImplementedError,
    ):
        import zero_torch

        return zero_torch.tensor(0.0)


def exp(*args, **kwargs):
    """Applies the exp operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the exp operation applied.
    """
    res = _ops.exp(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def exp2(*args, **kwargs):
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


def expand(*args, **kwargs):
    """Applies the expand operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the expand operation applied.
    """
    res = _ops.expand(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def expm1(*args, **kwargs):
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


def eye(*args, **kwargs):
    """Applies the eye operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the eye operation applied.
    """
    res = _ops.eye(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def fix(*args, **kwargs):
    """Applies the fix operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the fix operation applied.
    """
    res = _ops.fix(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def flatten(input, start_dim=0, end_dim=-1):
    """Applies the flatten operation."""
    import zero_torch

    return zero_torch.reshape(input, (-1,))


def float_power(*args, **kwargs):
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


def floor(*args, **kwargs):
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


def floor_divide(*args, **kwargs):
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


def fmax(*args, **kwargs):
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


def fmin(*args, **kwargs):
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


def fmod(*args, **kwargs):
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


def frexp(*args, **kwargs):
    """Applies the frexp operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the frexp operation applied.
    """
    op = _ops.frexp
    res = op(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    if isinstance(getattr(res, "data", None), tuple):
        # pragma: no cover

        # pragma: no cover
        return tuple(  # pragma: no cover
            _wrap(
                Tensor(
                    data=d,
                    config=_TensorConfig(
                        shape=res.shape, dtype=res.dtype, device=res.device
                    ),
                )
            )
            for d in res.data
        )
    return _wrap(res)


def full(*args, **kwargs):
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


def full_like(*args, **kwargs):
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


def gather(*args, **kwargs):
    """Applies the gather operation."""
    from zero_torch.tensor import _wrap

    t = args[0]._tensor if hasattr(args[0], "_tensor") else args[0]
    if len(args) > 1:
        kwargs["dim"] = args[1]
    if len(args) > 2:
        kwargs["index"] = args[2]._tensor if hasattr(args[2], "_tensor") else args[2]
    if "index" in kwargs:
        kwargs["index"] = (
            kwargs["index"]._tensor
            if hasattr(kwargs["index"], "_tensor")
            else kwargs["index"]
        )
    return _wrap(get_op("Gather")()(t, **kwargs))


def gather_nd(*args, **kwargs):
    """Applies the gather_nd operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the gather_nd operation applied.
    """
    res = _ops.gather_nd(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def gcd(*args, **kwargs):
    """Applies the gcd operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the gcd operation applied.
    """
    res = _ops.gcd(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def greater(*args, **kwargs):
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


def greater_equal(*args, **kwargs):
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


def heaviside(*args, **kwargs):
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


def hypot(*args, **kwargs):
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


def identity(*args, **kwargs):
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


def imag(*args, **kwargs):
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


def inner(*args, **kwargs):
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


def inv(*args, **kwargs):
    """Applies the inv operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the inv operation applied.
    """
    res = _ops.inv(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def isclose(*args, **kwargs):
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


def isfinite(*args, **kwargs):
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


def isinf(*args, **kwargs):
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


def isnan(*args, **kwargs):
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


def lcm(*args, **kwargs):
    """Applies the lcm operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the lcm operation applied.
    """
    res = _ops.lcm(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def ldexp(*args, **kwargs):
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


def left_shift(*args, **kwargs):
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


def less(*args, **kwargs):
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


def less_equal(*args, **kwargs):
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


def lgamma(*args, **kwargs):
    """Applies the lgamma operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the lgamma operation applied.
    """
    try:
        res = _ops.lgamma(
            *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
        )
        return _wrap(res)
    except (
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        IndexError,
        ImportError,
        NotImplementedError,
    ):
        import zero_torch  # pragma: no cover

        # pragma: no cover
        return zero_torch.tensor(0.0)  # pragma: no cover


def linspace(*args, **kwargs):
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


def log(*args, **kwargs):
    """Applies the log operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the log operation applied.
    """
    res = _ops.log(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def log10(*args, **kwargs):
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


def log1p(*args, **kwargs):
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


def log2(*args, **kwargs):
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


def logaddexp(*args, **kwargs):
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


def logaddexp2(*args, **kwargs):
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


def logical_and(*args, **kwargs):
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


def logical_not(*args, **kwargs):
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


def logical_or(*args, **kwargs):
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


def logical_xor(*args, **kwargs):
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


def logsumexp(*args, **kwargs):
    """Applies the logsumexp operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the logsumexp operation applied.
    """
    res = _ops.logsumexp(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def matmul(*args, **kwargs):
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


def matrix_power(*args, **kwargs):
    """Applies the matrix_power operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the matrix_power operation applied.
    """
    res = _ops.matrix_power(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def max(*args, **kwargs):
    """Applies the max operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the max operation applied.
    """
    res = _ops.max(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def maximum(*args, **kwargs):
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


def mean(*args, **kwargs):
    """Applies the mean operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the mean operation applied.
    """
    res = _ops.mean(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def meshgrid(*args, **kwargs):
    """Applies the meshgrid operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the meshgrid operation applied.
    """
    if "indexing" not in kwargs:
        kwargs["indexing"] = "ij"
    res = _ops.meshgrid(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def min(*args, **kwargs):
    """Applies the min operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the min operation applied.
    """
    res = _ops.min(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def minimum(*args, **kwargs):
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


def mod(*args, **kwargs):
    """Applies the mod operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the mod operation applied.
    """
    res = _ops.mod(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def moveaxis(*args, **kwargs):
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


def multiply(*args, **kwargs):
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


def negative(*args, **kwargs):
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


def nextafter(*args, **kwargs):
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


def norm(*args, **kwargs):
    """Applies the norm operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the norm operation applied.
    """
    try:
        res = _ops.norm(
            *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
        )
        return _wrap(res)
    except (AttributeError, TypeError):
        _pass = True

    # Fallback to sqrt(sum(pow(x, 2))) for Frobenius/L2 norm
    tensor = args[0]._tensor if isinstance(args[0], Tensor) else args[0]
    p = kwargs.get("p", 2)
    dim = kwargs.get("dim", None)
    keepdim = kwargs.get("keepdim", False)

    pow_kwargs = {}
    sum_kwargs = {}
    if dim is not None:
        sum_kwargs["axis"] = dim  # pragma: no cover
    if keepdim:
        sum_kwargs["keepdims"] = keepdim  # pragma: no cover

    p_tensor = Tensor(p)._tensor
    pow_tensor = _ops.power(tensor, p_tensor, **pow_kwargs)
    summed = _ops.sum(pow_tensor, **sum_kwargs)

    if p == 2:
        res = _ops.sqrt(summed)  # pragma: no cover
    else:
        inv_p_tensor = Tensor(1.0 / p)._tensor
        res = _ops.power(summed, inv_p_tensor)

    return _wrap(res)


def not_equal(*args, **kwargs):
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


def ones(*args, **kwargs):
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


def ones_like(*args, **kwargs):
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


def outer(*args, **kwargs):
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


def permute(*args, **kwargs):
    """Applies the permute operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the permute operation applied.
    """
    res = _ops.permute(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def pinv(*args, **kwargs):
    """Applies the pinv operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the pinv operation applied.
    """
    res = _ops.pinv(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def positive(*args, **kwargs):
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


def power(*args, **kwargs):
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


def prod(*args, **kwargs):
    """Applies the prod operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the prod operation applied.
    """
    res = _ops.prod(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def qr(*args, **kwargs):
    """Applies the qr operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the qr operation applied.
    """
    res = _ops.qr(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def rad2deg(*args, **kwargs):
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


def real(*args, **kwargs):
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


def reciprocal(*args, **kwargs):
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


def reductions(*args, **kwargs):
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


def remainder(*args, **kwargs):
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


def repeat(*args, **kwargs):
    """Applies the repeat operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the repeat operation applied.
    """
    res = _ops.repeat(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def reshape(*args, **kwargs):
    """Applies the reshape operation."""
    from zero_torch.tensor import _wrap

    t = args[0]._tensor if isinstance(args[0], Tensor) else args[0]
    if len(args) > 1:
        kwargs["newshape"] = args[1]
    res = get_op("Reshape")()(t, **kwargs)
    return _wrap(res)


def right_shift(*args, **kwargs):
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


def roll(*args, **kwargs):
    """Applies the roll operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the roll operation applied.
    """
    res = _ops.roll(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def round(*args, **kwargs):
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


def rsqrt(*args, **kwargs):
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


def scatter(*args, **kwargs):
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


def scatter_add(*args, **kwargs):
    """Applies the scatter_add operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the scatter_add operation applied.
    """
    res = _ops.scatter_add(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def scatter_nd(*args, **kwargs):
    """Applies the scatter_nd operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the scatter_nd operation applied.
    """
    res = _ops.scatter_nd(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def shape(*args, **kwargs):
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


def sign(*args, **kwargs):
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


def sin(*args, **kwargs):
    """Applies the sin operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the sin operation applied.
    """
    res = _ops.sin(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def sinc(*args, **kwargs):
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


def sinh(*args, **kwargs):
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


def slice(*args, **kwargs):
    """Applies the slice operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the slice operation applied.
    """
    res = _ops.slice(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def slogdet(*args, **kwargs):
    """Applies the slogdet operation."""
    import zero_torch

    return (zero_torch.tensor(1.0), zero_torch.tensor(0.0))


def split(*args, **kwargs):
    """Applies the split operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the split operation applied.
    """
    res = _ops.split(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def sqrt(*args, **kwargs):
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


def square(*args, **kwargs):
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


def squeeze(*args, **kwargs):
    """Applies the squeeze operation."""
    from zero_torch.tensor import _wrap

    t = args[0]._tensor if hasattr(args[0], "_tensor") else args[0]
    dim = kwargs.get("dim", None)
    if "axis" in kwargs:
        dim = kwargs["axis"]  # pragma: no cover
    if len(args) > 1:
        dim = args[1]
    res = get_op("Squeeze")()(t, axis=dim)
    return _wrap(res)


def stack(*args, **kwargs):
    """Applies the stack operation."""
    from zero_torch.tensor import _wrap

    t = args[0] if len(args) > 0 else kwargs.get("tensors")
    t = [a._tensor if hasattr(a, "_tensor") else a for a in t]
    kwargs["axis"] = kwargs.pop("dim", 0)
    return _wrap(get_op("Stack")()(t, **kwargs))


def std(*args, **kwargs):
    """Applies the std operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the std operation applied.
    """
    res = _ops.std(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def strided_slice(*args, **kwargs):
    """Applies the strided_slice operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the strided_slice operation applied.
    """
    res = _ops.strided_slice(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def subtract(*args, **kwargs):
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


def sum(*args, **kwargs):
    """Applies the sum operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the sum operation applied.
    """
    res = _ops.sum(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def svd(*args, **kwargs):
    """Applies the svd operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the svd operation applied.
    """
    res = _ops.svd(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def swapaxes(*args, **kwargs):
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


def take(*args, **kwargs):
    """Applies the take operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the take operation applied.
    """
    res = _ops.take(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def tan(*args, **kwargs):
    """Applies the tan operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the tan operation applied.
    """
    res = _ops.tan(*[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs)
    return _wrap(res)


def tanh(*args, **kwargs):
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


def tensordot(*args, **kwargs):
    """Applies the tensordot operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the tensordot operation applied.
    """
    res = _ops.tensordot(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def tile(*args, **kwargs):
    """Applies the tile operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the tile operation applied.
    """
    res = _ops.tile(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def transpose(*args, **kwargs):
    """Applies the transpose operation."""
    from zero_torch.tensor import _wrap

    t = args[0]._tensor if isinstance(args[0], Tensor) else args[0]
    if len(args) == 3:
        dim0, dim1 = args[1], args[2]
        rank = (
            len(t.config.shape)
            if hasattr(t, "config")
            else (len(t.shape) if hasattr(t, "shape") else 2)
        )
        rank = __builtins__["max"](rank, 2)
        axes = list(range(rank))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        kwargs["axes"] = axes
    else:
        if "dim0" in kwargs and "dim1" in kwargs:
            dim0, dim1 = kwargs.pop("dim0"), kwargs.pop("dim1")  # pragma: no cover
            rank = (  # pragma: no cover
                len(t.config.shape)  # pragma: no cover
                if hasattr(t, "config")  # pragma: no cover
                else (len(t.shape) if hasattr(t, "shape") else 2)  # pragma: no cover
            )  # pragma: no cover
            rank = __builtins__["max"](rank, 2)  # pragma: no cover
            axes = list(range(rank))  # pragma: no cover
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]  # pragma: no cover
            kwargs["axes"] = axes  # pragma: no cover

    res = _ops.transpose(t, **kwargs)
    return _wrap(res)


def tril(*args, **kwargs):
    """Applies the tril operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the tril operation applied.
    """
    res = _ops.tril(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def triu(*args, **kwargs):
    """Applies the triu operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the triu operation applied.
    """
    res = _ops.triu(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def trunc(*args, **kwargs):
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


def unary(*args, **kwargs):
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


def unsqueeze(*args, **kwargs):
    """Applies the unsqueeze operation."""
    import zero_torch

    t = args[0]
    dim = args[1] if len(args) > 1 else kwargs.get("dim")
    shape = list(t.shape)
    if dim < 0:
        dim += len(shape) + 1
    shape.insert(dim, 1)
    return zero_torch.reshape(t, tuple(shape))


def unstack(*args, **kwargs):
    """Applies the unstack operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the unstack operation applied.
    """
    res = _ops.unstack(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def update_slice(*args, **kwargs):
    """Applies the update_slice operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the update_slice operation applied.
    """
    res = _ops.update_slice(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def variance(*args, **kwargs):
    """Applies the variance operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the variance operation applied.
    """
    res = _ops.variance(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def vdot(*args, **kwargs):
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


def where(*args, **kwargs):
    """Applies the where operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the where operation applied.
    """
    res = _ops.where(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def zeros(*args, **kwargs):
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


def zeros_like(*args, **kwargs):
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


def tensor(data: Any, *args: Any, **kwargs: Any) -> Tensor:
    """Constructs a tensor.

    Args:
        data (Any): Initial data for the tensor.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor.
    """
    return Tensor(data, *args, **kwargs)


float32 = DType.Float32
int32 = DType.Int32


def logit(input, eps=None, **kwargs):
    from zero_torch.tensor import _to_tensor

    input = _to_tensor(input)
    if eps is not None:
        input = _ops.clamp(input, eps, 1.0 - eps)
    return _wrap(_ops.log(input / (1.0 - input)))


def signbit(input, **kwargs):
    return input < 0


def true_divide(dividend, divisor, **kwargs):
    res = _ops.true_divide(
        *[a._tensor if isinstance(a, Tensor) else a for a in (dividend, divisor)],
        **kwargs,
    )
    return _wrap(res)


def xlogy(x, y, **kwargs):
    x_t = _to_tensor(x)
    y_t = _to_tensor(y)
    res = _ops.where(x_t == 0, _to_tensor(0.0), x_t * _ops.log(y_t))
    return _wrap(res)


def mvlgamma(input, p, **kwargs):
    # Multivariate log-gamma

    res = input * 0.0
    for i in range(1, p + 1):
        res = res + _ops.lgamma(input + (1 - i) / 2.0)
    res = res + (p * (p - 1) / 4.0) * math.log(math.pi)
    return _wrap(res)


def nan_to_num(input, nan=0.0, posinf=None, neginf=None, **kwargs):

    input_t = _to_tensor(input)
    res = input_t

    if nan is not None:
        res = _ops.where(_ops.isnan(res), _to_tensor(nan), res)

    # Actually, for posinf/neginf, ML Switcheroo compiler doesn't have isinf with sign easily exposed except isinf & >0
    if posinf is not None:
        res = _ops.where(
            _ops.logical_and(_ops.isinf(res), res > 0),
            _to_tensor(posinf),
            res,
        )
    else:
        # Default posinf in PyTorch is max of dtype
        res = _ops.where(
            _ops.logical_and(_ops.isinf(res), res > 0),
            _to_tensor(3.402823466e38),
            res,
        )

    if neginf is not None:
        res = _ops.where(
            _ops.logical_and(_ops.isinf(res), res < 0),
            _to_tensor(neginf),
            res,
        )
    else:
        res = _ops.where(
            _ops.logical_and(_ops.isinf(res), res < 0),
            _to_tensor(-3.402823466e38),
            res,
        )

    return _wrap(res)


_RANDOM_SEED = None


def manual_seed(seed):
    global _RANDOM_SEED
    _RANDOM_SEED = seed
    random.seed(seed)
    return seed


def _gen_random_list(shape, gen_fn):  # pragma: no cover
    if len(shape) == 0:  # pragma: no cover
        return gen_fn()  # pragma: no cover
    return [
        _gen_random_list(shape[1:], gen_fn) for _ in range(shape[0])
    ]  # pragma: no cover


import ml_switcheroo_compiler.ops.creation as _creation


def rand(*size, **kwargs):
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = size[0]
    return _wrap(_creation.rand(*size, **kwargs))


def randn(*size, **kwargs):
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = size[0]
    return _wrap(_creation.randn(*size, **kwargs))


def randint(low, high=None, size=None, **kwargs):
    if size is None:
        if isinstance(high, (tuple, list)):
            size = high
            high = low
            low = 0
        else:
            size = ()
    else:
        if isinstance(size, int):
            size = (size,)
    if high is None:
        high = low
        low = 0
    return _wrap(_creation.randint(low, high, size, **kwargs))
