"""Functional normalization operations backed by ml-switcheroo-compiler."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.creation import ones, zeros
from ml_switcheroo_compiler.ops.nn.normalization import (
    BatchNormConfig,
    batch_normalization,
    l2_normalize,
    rms_normalization,
)
from ml_switcheroo_compiler.ops.nn.normalization import (
    instance_norm as _instance_norm,
)
from ml_switcheroo_compiler.ops.nn.normalization import (
    layer_norm as _layer_norm,
)

from zero_torch._types import OpArg, ShapeLike
from zero_torch.tensor import Tensor, _to_tensor, _wrap


def _norm_shape_tuple(shape: ShapeLike | OpArg) -> tuple[int, ...]:
    """Normalizes a `normalized_shape` argument into a tuple of ints.

    Args:
        shape: Integer, tuple, or list of normalized dimensions.

    Returns:
        Tuple of integer dimension sizes.
    """
    if isinstance(shape, int):
        return (shape,)
    if isinstance(shape, (tuple, list)):
        return tuple(int(s) for s in shape if isinstance(s, int))
    return (1,)


def batch_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies batch normalization across a mini-batch of inputs.

    Args:
        *args: Input tensor and optional positional parameters.
        **kwargs: Keyword arguments (`input`, `eps`).

    Returns:
        Batch-normalized tensor.
    """
    input_t = _to_tensor(args[0] if len(args) > 0 else kwargs.get("input"))
    eps_raw = kwargs.get("eps", 1e-5)
    eps = float(eps_raw) if isinstance(eps_raw, (int, float)) else 1e-5
    m = zeros(input_t.shape)
    v = ones(input_t.shape)
    return _wrap(
        batch_normalization(input_t, m, v, axis=1, config=BatchNormConfig(epsilon=eps))
    )


def layer_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies layer normalization for last certain number of dimensions.

    Args:
        *args: Input tensor, `normalized_shape`, and optional parameters.
        **kwargs: Keyword arguments (`input`, `normalized_shape`, `eps`).

    Returns:
        Layer-normalized tensor.
    """
    input_t = _to_tensor(args[0] if len(args) > 0 else kwargs.get("input"))
    raw_shape = (
        args[1] if len(args) > 1 else kwargs.get("normalized_shape", input_t.shape[-1:])
    )
    norm_shape = _norm_shape_tuple(raw_shape)
    eps_raw = kwargs.get("eps", 1e-5)
    eps = float(eps_raw) if isinstance(eps_raw, (int, float)) else 1e-5
    return _wrap(_layer_norm(input_t, normalized_shape=norm_shape, epsilon=eps))


def instance_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies instance normalization across each channel in each data sample.

    Args:
        *args: Input tensor and optional parameters.
        **kwargs: Keyword arguments (`input`, `eps`).

    Returns:
        Instance-normalized tensor.
    """
    input_t = _to_tensor(args[0] if len(args) > 0 else kwargs.get("input"))
    eps_raw = kwargs.get("eps", 1e-5)
    eps = float(eps_raw) if isinstance(eps_raw, (int, float)) else 1e-5
    return _wrap(_instance_norm(input_t, epsilon=eps))


def normalize(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Performs `L_p` normalization of inputs over specified dimension.

    Args:
        *args: Input tensor and optional parameters.
        **kwargs: Keyword arguments (`input`, `dim`, `eps`).

    Returns:
        Normalized tensor.
    """
    input_t = _to_tensor(args[0] if len(args) > 0 else kwargs.get("input"))
    dim_raw = kwargs.get("dim", 1)
    dim = int(dim_raw) if isinstance(dim_raw, int) else 1
    eps_raw = kwargs.get("eps", 1e-12)
    eps = float(eps_raw) if isinstance(eps_raw, (int, float)) else 1e-12
    return _wrap(l2_normalize(input_t, axis=dim, epsilon=eps))


def rms_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies Root Mean Square Layer Normalization.

    Args:
        *args: Input tensor and optional parameters.
        **kwargs: Keyword arguments (`input`, `eps`).

    Returns:
        RMS-normalized tensor.
    """
    input_t = _to_tensor(args[0] if len(args) > 0 else kwargs.get("input"))
    eps_raw = kwargs.get("eps", 1e-5)
    eps = float(eps_raw) if isinstance(eps_raw, (int, float)) else 1e-5
    scale = ones(input_t.shape[-1:])
    return _wrap(rms_normalization(input_t, scale=scale, epsilon=eps))
