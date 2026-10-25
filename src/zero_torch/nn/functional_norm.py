"""Functional normalization operations."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.nn.normalization import l2_normalize

import zero_torch
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


def _extract_opt_tensor(
    args: tuple[OpArg, ...],
    kwargs: dict[str, OpArg],
    index: int,
    name: str,
) -> Tensor | None:
    """Extracts an optional `Tensor` argument by positional index or keyword name.

    Args:
        args: Positional argument tuple.
        kwargs: Keyword argument dictionary.
        index: Positional index to check.
        name: Keyword argument name.

    Returns:
        Extracted `Tensor` or `None`.
    """
    raw = args[index] if len(args) > index else kwargs.get(name)
    return raw if isinstance(raw, Tensor) else None


def batch_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies batch normalization across a mini-batch of inputs.

    Args:
        *args: `input`, optional `running_mean`, `running_var`, `weight`, `bias`, `training`, `momentum`, `eps`.
        **kwargs: Keyword arguments (`input`, `running_mean`, `running_var`, `weight`, `bias`, `training`, `momentum`, `eps`).

    Returns:
        Batch-normalized tensor.
    """
    raw_in = args[0] if len(args) > 0 else kwargs.get("input", 0.0)
    x = raw_in if isinstance(raw_in, Tensor) else _wrap(_to_tensor(raw_in))
    running_mean = _extract_opt_tensor(args, kwargs, 1, "running_mean")
    running_var = _extract_opt_tensor(args, kwargs, 2, "running_var")
    weight = _extract_opt_tensor(args, kwargs, 3, "weight")
    bias = _extract_opt_tensor(args, kwargs, 4, "bias")
    training_raw = args[5] if len(args) > 5 else kwargs.get("training", False)
    training = bool(training_raw)
    mom_raw = args[6] if len(args) > 6 else kwargs.get("momentum", 0.1)
    momentum = float(mom_raw) if isinstance(mom_raw, (int, float)) else 0.1
    eps_raw = args[7] if len(args) > 7 else kwargs.get("eps", 1e-5)
    eps = float(eps_raw) if isinstance(eps_raw, (int, float)) else 1e-5

    ndim = len(x.shape)
    if ndim < 2:
        m = zero_torch.mean(x)
        diff = x - m
        v = zero_torch.mean(diff * diff)
        return diff * zero_torch.rsqrt(v + eps)

    c = x.shape[1]
    reduce_axes = (0,) + tuple(range(2, ndim))
    bcast_shape = (1, c) + (1,) * (ndim - 2)

    if training or running_mean is None or running_var is None:
        batch_mean = zero_torch.mean(x, axis=reduce_axes, keepdims=True)
        diff = x - batch_mean
        batch_var = zero_torch.mean(diff * diff, axis=reduce_axes, keepdims=True)
        if training and running_mean is not None and running_var is not None:
            mean_1d = zero_torch.reshape(batch_mean, (c,))
            var_1d = zero_torch.reshape(batch_var, (c,))
            n = x.numel() // c if c > 0 else 1
            corr = float(n) / float(n - 1) if n > 1 else 1.0
            running_mean.copy_((1.0 - momentum) * running_mean + momentum * mean_1d)
            running_var.copy_(
                (1.0 - momentum) * running_var + momentum * (var_1d * corr)
            )
        out = diff * zero_torch.rsqrt(batch_var + eps)
    else:
        m_b = zero_torch.reshape(running_mean, bcast_shape)
        v_b = zero_torch.reshape(running_var, bcast_shape)
        out = (x - m_b) * zero_torch.rsqrt(v_b + eps)

    if weight is not None:
        out = out * zero_torch.reshape(weight, bcast_shape)
    if bias is not None:
        out = out + zero_torch.reshape(bias, bcast_shape)
    return out


def layer_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies layer normalization over the last `len(normalized_shape)` dimensions.

    Args:
        *args: `input`, `normalized_shape`, optional `weight`, `bias`, `eps`.
        **kwargs: Keyword arguments (`input`, `normalized_shape`, `weight`, `bias`, `eps`).

    Returns:
        Layer-normalized tensor.
    """
    raw_in = args[0] if len(args) > 0 else kwargs.get("input", 0.0)
    x = raw_in if isinstance(raw_in, Tensor) else _wrap(_to_tensor(raw_in))
    raw_shape = (
        args[1] if len(args) > 1 else kwargs.get("normalized_shape", x.shape[-1:])
    )
    norm_shape = _norm_shape_tuple(raw_shape)
    weight = _extract_opt_tensor(args, kwargs, 2, "weight")
    bias = _extract_opt_tensor(args, kwargs, 3, "bias")
    eps_raw = args[4] if len(args) > 4 else kwargs.get("eps", 1e-5)
    eps = float(eps_raw) if isinstance(eps_raw, (int, float)) else 1e-5

    ndim = len(norm_shape)
    axes = tuple(range(len(x.shape) - ndim, len(x.shape)))
    m = zero_torch.mean(x, axis=axes, keepdims=True)
    diff = x - m
    v = zero_torch.mean(diff * diff, axis=axes, keepdims=True)
    out = diff * zero_torch.rsqrt(v + eps)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


def instance_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies instance normalization across spatial dimensions for each channel and sample.

    Args:
        *args: `input`, optional `running_mean`, `running_var`, `weight`, `bias`, `use_input_stats`, `momentum`, `eps`.
        **kwargs: Keyword arguments (`input`, `running_mean`, `running_var`, `weight`, `bias`, `use_input_stats`, `momentum`, `eps`).

    Returns:
        Instance-normalized tensor.
    """
    raw_in = args[0] if len(args) > 0 else kwargs.get("input", 0.0)
    x = raw_in if isinstance(raw_in, Tensor) else _wrap(_to_tensor(raw_in))
    running_mean = _extract_opt_tensor(args, kwargs, 1, "running_mean")
    running_var = _extract_opt_tensor(args, kwargs, 2, "running_var")
    weight = _extract_opt_tensor(args, kwargs, 3, "weight")
    bias = _extract_opt_tensor(args, kwargs, 4, "bias")
    use_stats_raw = args[5] if len(args) > 5 else kwargs.get("use_input_stats", True)
    use_input_stats = bool(use_stats_raw)
    mom_raw = args[6] if len(args) > 6 else kwargs.get("momentum", 0.1)
    momentum = float(mom_raw) if isinstance(mom_raw, (int, float)) else 0.1
    eps_raw = args[7] if len(args) > 7 else kwargs.get("eps", 1e-5)
    eps = float(eps_raw) if isinstance(eps_raw, (int, float)) else 1e-5

    ndim = len(x.shape)
    if ndim < 2:
        return x
    if ndim == 2:
        c = x.shape[0]
        m = zero_torch.mean(x, axis=1, keepdims=True)
        diff = x - m
        v = zero_torch.mean(diff * diff, axis=1, keepdims=True)
        out = diff * zero_torch.rsqrt(v + eps)
        if weight is not None:
            out = out * zero_torch.reshape(weight, (c, 1))
        if bias is not None:
            out = out + zero_torch.reshape(bias, (c, 1))
        return out

    c = x.shape[1]
    spatial_axes = tuple(range(2, ndim))
    bcast_shape = (1, c) + (1,) * (ndim - 2)

    if use_input_stats or running_mean is None or running_var is None:
        m = zero_torch.mean(x, axis=spatial_axes, keepdims=True)
        diff = x - m
        v = zero_torch.mean(diff * diff, axis=spatial_axes, keepdims=True)
        if use_input_stats and running_mean is not None and running_var is not None:
            mean_1d = zero_torch.reshape(zero_torch.mean(m, axis=0), (c,))
            var_1d = zero_torch.reshape(zero_torch.mean(v, axis=0), (c,))
            spatial_n = x.numel() // (x.shape[0] * c) if (x.shape[0] * c) > 0 else 1
            corr = float(spatial_n) / float(spatial_n - 1) if spatial_n > 1 else 1.0
            running_mean.copy_((1.0 - momentum) * running_mean + momentum * mean_1d)
            running_var.copy_(
                (1.0 - momentum) * running_var + momentum * (var_1d * corr)
            )
        out = diff * zero_torch.rsqrt(v + eps)
    else:
        m_b = zero_torch.reshape(running_mean, bcast_shape)
        v_b = zero_torch.reshape(running_var, bcast_shape)
        out = (x - m_b) * zero_torch.rsqrt(v_b + eps)

    if weight is not None:
        out = out * zero_torch.reshape(weight, bcast_shape)
    if bias is not None:
        out = out + zero_torch.reshape(bias, bcast_shape)
    return out


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
    """Applies Root Mean Square Layer Normalization over `normalized_shape`.

    Args:
        *args: `input`, optional `normalized_shape`, `weight`, `eps`.
        **kwargs: Keyword arguments (`input`, `normalized_shape`, `weight`, `eps`).

    Returns:
        RMS-normalized tensor.
    """
    raw_in = args[0] if len(args) > 0 else kwargs.get("input", 0.0)
    x = raw_in if isinstance(raw_in, Tensor) else _wrap(_to_tensor(raw_in))
    raw_shape = (
        args[1] if len(args) > 1 else kwargs.get("normalized_shape", x.shape[-1:])
    )
    norm_shape = _norm_shape_tuple(raw_shape)
    weight = _extract_opt_tensor(args, kwargs, 2, "weight")
    eps_raw = args[3] if len(args) > 3 else kwargs.get("eps", 1e-5)
    eps = float(eps_raw) if isinstance(eps_raw, (int, float)) else 1e-5

    ndim = len(norm_shape)
    axes = tuple(range(len(x.shape) - ndim, len(x.shape)))
    ms = zero_torch.mean(x * x, axis=axes, keepdims=True)
    out = x * zero_torch.rsqrt(ms + eps)
    if weight is not None:
        out = out * weight
    return out
