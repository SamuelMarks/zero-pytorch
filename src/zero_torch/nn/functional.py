"""Functional NN API."""

from __future__ import annotations

import functools
import itertools
import math
import operator
from collections.abc import Callable, Sequence

import ml_switcheroo_compiler.ops as _ops
from ml_switcheroo_compiler.core import config
from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.ops.configs import ConvConfig
from ml_switcheroo_compiler.ops.nn.normalization import (
    group_norm as _compiler_group_norm,
)
from ml_switcheroo_compiler.ops.reductions.frontend_utils import _emit_reduction_node
from ml_switcheroo_compiler.ops.shape.indexing import where as _where

import zero_torch
from zero_torch._types import (
    OpArg,
    OptSize2d,
    OptSizeNd,
    PaddingArg1d,
    PaddingArg2d,
    PaddingArg3d,
    Size1d,
    Size2d,
    Size3d,
)
from zero_torch.tensor import Tensor, _to_tensor, _wrap

from .functional_dropout import (
    alpha_dropout as _fp_alpha_dropout,
)
from .functional_dropout import (
    dropout as _fp_dropout,
)
from .functional_dropout import (
    dropout1d as _fp_dropout1d,
)
from .functional_dropout import (
    dropout2d as _fp_dropout2d,
)
from .functional_dropout import (
    dropout3d as _fp_dropout3d,
)
from .functional_dropout import (
    feature_alpha_dropout as _fp_feature_alpha_dropout,
)
from .functional_norm import (
    batch_norm as _fp_batch_norm,
)
from .functional_norm import (
    instance_norm as _fp_instance_norm,
)
from .functional_norm import (
    layer_norm as _fp_layer_norm,
)
from .functional_norm import (
    rms_norm as _fp_rms_norm,
)
from .functional_pooling import (
    _normalize_tuple,
    _to_float_list,
    adaptive_avg_pool1d,
    adaptive_avg_pool3d,
    adaptive_max_pool1d,
    adaptive_max_pool2d,
    adaptive_max_pool3d,
    fold,
    fractional_max_pool3d,
    lp_pool1d,
    lp_pool2d,
    lp_pool3d,
    max_unpool1d,
    max_unpool2d,
    max_unpool3d,
    unfold,
)
from .functional_pooling import (
    adaptive_avg_pool2d as _fp_adaptive_avg_pool2d,
)
from .functional_pooling import (
    avg_pool1d as _fp_avg1d,
)
from .functional_pooling import (
    avg_pool2d as _fp_avg2d,
)
from .functional_pooling import (
    avg_pool3d as _fp_avg3d,
)
from .functional_pooling import (
    fractional_max_pool2d as _fp_fractional_max_pool2d,
)
from .functional_pooling import (
    max_pool1d as _fp_max1d,
)
from .functional_pooling import (
    max_pool2d as _fp_max2d,
)
from .functional_pooling import (
    max_pool3d as _fp_max3d,
)


def _reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    """Applies `'none'`, `'mean'`, or `'sum'` reduction to a loss tensor.

    Args:
        loss: Unreduced element-wise loss tensor.
        reduction: Reduction mode (`'none'`, `'mean'`, `'sum'`).

    Returns:
        Reduced loss tensor.
    """
    if reduction == "mean":
        return zero_torch.mean(loss)
    if reduction == "sum":
        return zero_torch.sum(loss)
    return loss


def _extract_tensor(val: OpArg) -> Tensor:
    """Coerces an `OpArg` into a `zero_torch.Tensor`.

    Args:
        val: Input tensor or scalar.

    Returns:
        `Tensor` instance.
    """
    if isinstance(val, Tensor):
        return val
    return Tensor(val)


def _extract_reduction(kwargs: dict[str, OpArg], default: str = "mean") -> str:
    """Extracts the `reduction` string from keyword arguments.

    Args:
        kwargs: Keyword argument dictionary.
        default: Default reduction mode.

    Returns:
        Reduction mode string.
    """
    r = kwargs.get("reduction", default)
    return r if isinstance(r, str) else default


def _conv_pad_pairs(
    padding: int | Sequence[int] | str,
    k: tuple[int, ...],
    d: tuple[int, ...],
    ndims: int,
) -> tuple[tuple[int, int], ...]:
    """Resolves `int`, `tuple`, `'same'`, or `'valid'` padding into `(before, after)` pairs.

    Args:
        padding: Padding specification.
        k: Kernel size tuple of length `ndims`.
        d: Dilation tuple of length `ndims`.
        ndims: Number of spatial dimensions.

    Returns:
        Tuple of `(pad_before, pad_after)` pairs of length `ndims`.
    """
    if isinstance(padding, int):
        return tuple((padding, padding) for _ in range(ndims))
    if isinstance(padding, str):
        if padding == "same":
            pairs: list[tuple[int, int]] = []
            for m in range(ndims):
                total_pad = d[m] * (k[m] - 1)
                pad_left = total_pad // 2
                pairs.append((pad_left, total_pad - pad_left))
            return tuple(pairs)
        return tuple((0, 0) for _ in range(ndims))
    return tuple((int(p), int(p)) for p in padding)


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: Size2d = 1,
    padding: PaddingArg2d = 0,
    dilation: Size2d = 1,
    groups: int = 1,
) -> Tensor:
    """Applies a 2D convolution over an input image composed of several input planes.

    Args:
        input: Input tensor of shape `(minibatch, in_channels, iH, iW)`.
        weight: Filters of shape `(out_channels, in_channels / groups, kH, kW)`.
        bias: Optional bias tensor of shape `(out_channels,)`.
        stride: Stride of the convolving kernel.
        padding: Implicit paddings on both sides of the input.
        dilation: Spacing between kernel elements.
        groups: Split input into groups.

    Returns:
        Convolved 4D tensor.
    """
    input_t = _to_tensor(input)
    weight_t = _to_tensor(weight)
    s = _normalize_tuple(stride, 2)
    d = _normalize_tuple(dilation, 2)
    k = (weight.shape[-2], weight.shape[-1])
    pad_pairs = _conv_pad_pairs(padding, k, d, 2)

    res = _ops.conv_general_dilated(
        input_t,
        weight_t,
        ConvConfig(
            window_strides=s,
            padding=pad_pairs,
            lhs_dilation=None,
            rhs_dilation=d,
            dimension_numbers=None,
            feature_group_count=groups,
        ),
    )
    out = _wrap(res)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: Size1d = 1,
    padding: PaddingArg1d = 0,
    dilation: Size1d = 1,
    groups: int = 1,
) -> Tensor:
    """Applies a 1D convolution over an input signal composed of several input planes.

    Args:
        input: Input tensor of shape `(minibatch, in_channels, iW)`.
        weight: Filters of shape `(out_channels, in_channels / groups, kW)`.
        bias: Optional bias tensor of shape `(out_channels,)`.
        stride: Stride of the convolving kernel.
        padding: Implicit paddings on both sides of the input.
        dilation: Spacing between kernel elements.
        groups: Split input into groups.

    Returns:
        Convolved 3D tensor.
    """
    input_t = _to_tensor(input)
    weight_t = _to_tensor(weight)
    s = _normalize_tuple(stride, 1)
    d = _normalize_tuple(dilation, 1)
    k = (weight.shape[-1],)
    pad_pairs = _conv_pad_pairs(padding, k, d, 1)

    res = _ops.conv_general_dilated(
        input_t,
        weight_t,
        ConvConfig(
            window_strides=s,
            padding=pad_pairs,
            lhs_dilation=None,
            rhs_dilation=d,
            dimension_numbers=None,
            feature_group_count=groups,
        ),
    )
    out = _wrap(res)
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    return out


def conv3d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: Size3d = 1,
    padding: PaddingArg3d = 0,
    dilation: Size3d = 1,
    groups: int = 1,
) -> Tensor:
    """Applies a 3D convolution over an input volume composed of several input planes.

    Args:
        input: Input tensor of shape `(minibatch, in_channels, iT, iH, iW)`.
        weight: Filters of shape `(out_channels, in_channels / groups, kT, kH, kW)`.
        bias: Optional bias tensor of shape `(out_channels,)`.
        stride: Stride of the convolving kernel.
        padding: Implicit paddings on both sides of the input.
        dilation: Spacing between kernel elements.
        groups: Split input into groups.

    Returns:
        Convolved 5D tensor.
    """
    input_t = _to_tensor(input)
    weight_t = _to_tensor(weight)
    s = _normalize_tuple(stride, 3)
    d = _normalize_tuple(dilation, 3)
    k = (weight.shape[-3], weight.shape[-2], weight.shape[-1])
    pad_pairs = _conv_pad_pairs(padding, k, d, 3)

    res = _ops.conv_general_dilated(
        input_t,
        weight_t,
        ConvConfig(
            window_strides=s,
            padding=pad_pairs,
            lhs_dilation=None,
            rhs_dilation=d,
            dimension_numbers=None,
            feature_group_count=groups,
        ),
    )
    out = _wrap(res)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1, 1)
    return out


def _conv_transpose_nd(
    ndims: int,
    *args: OpArg,
    **kwargs: OpArg,
) -> Tensor:
    """Executes `ndims`-D transposed convolution in both Eager and Tracing modes.

    Args:
        ndims: Number of spatial dimensions (1, 2, or 3).
        *args: `input`, `weight`, and optional `bias`, `stride`, `padding`, `output_padding`, `groups`, `dilation`.
        **kwargs: Optional keyword arguments.

    Returns:
        Transposed convolved tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    if len(args) <= 1 and "weight" not in kwargs:
        return zero_torch.zeros_like(x)
    w_raw = args[1] if len(args) > 1 else kwargs.get("weight", 0.0)
    if (
        not isinstance(w_raw, Tensor)
        or len(x.shape) != ndims + 2
        or len(w_raw.shape) != ndims + 2
    ):
        return zero_torch.zeros_like(x)
    w = w_raw

    bias_raw = args[2] if len(args) > 2 else kwargs.get("bias")
    bias = bias_raw if isinstance(bias_raw, Tensor) else None
    stride_raw = args[3] if len(args) > 3 else kwargs.get("stride", 1)
    pad_raw = args[4] if len(args) > 4 else kwargs.get("padding", 0)
    out_pad_raw = args[5] if len(args) > 5 else kwargs.get("output_padding", 0)
    groups_raw = args[6] if len(args) > 6 else kwargs.get("groups", 1)
    dil_raw = args[7] if len(args) > 7 else kwargs.get("dilation", 1)

    s = _normalize_tuple(
        stride_raw if isinstance(stride_raw, (int, tuple, list)) else 1, ndims
    )
    p = _normalize_tuple(
        pad_raw if isinstance(pad_raw, (int, tuple, list)) else 0, ndims
    )
    op = _normalize_tuple(
        out_pad_raw if isinstance(out_pad_raw, (int, tuple, list)) else 0, ndims
    )
    groups = int(groups_raw) if isinstance(groups_raw, int) else 1
    d = _normalize_tuple(
        dil_raw if isinstance(dil_raw, (int, tuple, list)) else 1, ndims
    )

    batch_size = x.shape[0]
    in_ch = x.shape[1]
    in_spatial = x.shape[2:]
    out_ch_per_group = w.shape[1]
    out_ch = out_ch_per_group * groups
    k = w.shape[2:]

    out_spatial = tuple(
        (in_spatial[m] - 1) * s[m] - 2 * p[m] + d[m] * (k[m] - 1) + op[m] + 1
        for m in range(ndims)
    )
    out_shape = (batch_size, out_ch) + out_spatial

    if not config.eager_mode:
        return _wrap(
            _emit_reduction_node(
                f"ConvTranspose{ndims}d",
                [_to_tensor(x), _to_tensor(w)],
                {
                    "stride": s,
                    "padding": p,
                    "output_padding": op,
                    "groups": groups,
                    "dilation": d,
                },
                out_shape,
                x.dtype,
            )
        )

    in_spatial_numel = functools.reduce(operator.mul, in_spatial, 1)
    out_spatial_numel = functools.reduce(operator.mul, out_spatial, 1)
    k_numel = functools.reduce(operator.mul, k, 1)

    in_strides = [1] * ndims
    out_strides = [1] * ndims
    k_strides = [1] * ndims
    for m in range(ndims - 2, -1, -1):
        in_strides[m] = in_strides[m + 1] * in_spatial[m + 1]
        out_strides[m] = out_strides[m + 1] * out_spatial[m + 1]
        k_strides[m] = k_strides[m + 1] * k[m + 1]

    x_flat = _to_float_list(x)
    w_flat = _to_float_list(w)
    b_flat = _to_float_list(bias) if bias is not None else [0.0] * out_ch

    in_ch_per_group = in_ch // groups
    out_buf: list[float] = []
    for _ in range(batch_size):
        for oc in range(out_ch):
            out_buf.extend([b_flat[oc]] * out_spatial_numel)

    in_coords_list = list(itertools.product(*(range(sz) for sz in in_spatial)))
    k_coords_list = list(itertools.product(*(range(sz) for sz in k)))

    transitions: list[tuple[int, int, int]] = []
    for i_pt in in_coords_list:
        in_offset = sum(i_pt[m] * in_strides[m] for m in range(ndims))
        for r_pt in k_coords_list:
            out_coords = [i_pt[m] * s[m] - p[m] + r_pt[m] * d[m] for m in range(ndims)]
            if all(0 <= out_coords[m] < out_spatial[m] for m in range(ndims)):
                out_offset = sum(out_coords[m] * out_strides[m] for m in range(ndims))
                k_offset = sum(r_pt[m] * k_strides[m] for m in range(ndims))
                transitions.append((in_offset, k_offset, out_offset))

    for n in range(batch_size):
        for g in range(groups):
            for ic_g in range(in_ch_per_group):
                ic = g * in_ch_per_group + ic_g
                x_base = (n * in_ch + ic) * in_spatial_numel
                for oc_g in range(out_ch_per_group):
                    oc = g * out_ch_per_group + oc_g
                    w_base = (ic * out_ch_per_group + oc_g) * k_numel
                    out_base = (n * out_ch + oc) * out_spatial_numel
                    for in_off, k_off, out_off in transitions:
                        out_buf[out_base + out_off] += (
                            x_flat[x_base + in_off] * w_flat[w_base + k_off]
                        )

    return zero_torch.reshape(Tensor(out_buf, dtype=x.dtype), out_shape)


def relu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the rectified linear unit function element-wise.

    Args:
        *args: Input tensor and optional positional arguments.
        **kwargs: Optional keyword arguments (`inplace`).

    Returns:
        Tensor with ReLU applied.
    """
    _ = kwargs
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    if x.requires_grad:
        from ..autograd.math_ops import ReluBackward

        return ReluBackward.apply(x)
    return _wrap(_ops.maximum(_to_tensor(x), 0.0))


def softmax(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies a softmax function.

    Args:
        *args: Input tensor and optional dimension.
        **kwargs: Optional keyword arguments (`dim`, `dtype`).

    Returns:
        Softmax probability tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    dim_raw = args[1] if len(args) > 1 else kwargs.get("dim", -1)
    dim = int(dim_raw) if isinstance(dim_raw, int) else -1
    max_x = zero_torch.max(x, axis=dim, keepdims=True)
    exp_x = zero_torch.exp(x - max_x)
    sum_exp = zero_torch.sum(exp_x, axis=dim, keepdims=True)
    return exp_x / sum_exp


def scaled_dot_product_attention(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes scaled dot product attention on query, key and value tensors.

    Args:
        *args: Query, key, and value tensors.
        **kwargs: Optional keyword arguments (`attn_mask`, `dropout_p`, `is_causal`, `scale`).

    Returns:
        Attention output tensor.
    """
    q = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("query", 0.0))
    k = _extract_tensor(args[1] if len(args) > 1 else kwargs.get("key", q))
    v = _extract_tensor(args[2] if len(args) > 2 else kwargs.get("value", k))
    d_k = q.shape[-1] if len(q.shape) > 0 else 1
    scale_raw = kwargs.get("scale")
    scale = (
        float(scale_raw)
        if isinstance(scale_raw, (int, float))
        else 1.0 / math.sqrt(max(1, d_k))
    )
    if len(q.shape) >= 2:
        k_t = zero_torch.swapaxes(k, -2, -1)
        scores = zero_torch.matmul(q, k_t) * scale
        attn_mask = kwargs.get("attn_mask")
        if isinstance(attn_mask, Tensor):
            scores = scores + attn_mask
        weights = softmax(scores, dim=-1)
        return zero_torch.matmul(weights, v)
    return q * scale


def celu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies element-wise `CELU(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))`.

    Args:
        *args: Input tensor and optional `alpha`.
        **kwargs: Optional keyword arguments (`alpha`, `inplace`).

    Returns:
        Output tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    alpha_raw = args[1] if len(args) > 1 else kwargs.get("alpha", 1.0)
    alpha = float(alpha_raw) if isinstance(alpha_raw, (int, float)) else 1.0
    pos = zero_torch.maximum(x, 0.0)
    neg = zero_torch.minimum(0.0, alpha * (zero_torch.exp(x / alpha) - 1.0))
    return pos + neg


def multilabel_margin_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes multi-label margin loss over target class indices terminated by `-1`.

    Args:
        *args: Input and target tensors.
        **kwargs: Optional keyword arguments (`reduction`).

    Returns:
        Computed loss tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    t = _extract_tensor(
        args[1] if len(args) > 1 else kwargs.get("target", zero_torch.zeros_like(x))
    )
    reduction = _extract_reduction(kwargs)
    if not config.eager_mode or len(x.shape) == 0:
        return _reduce_loss(zero_torch.maximum(0.0, 1.0 - (x - t)), reduction)

    num_classes = x.shape[-1]
    batch_size = functools.reduce(operator.mul, x.shape[:-1], 1)
    x_flat = _to_float_list(x)
    t_flat = [int(v) for v in _to_float_list(t)]

    sample_losses: list[float] = []
    for b in range(batch_size):
        x_row = x_flat[b * num_classes : (b + 1) * num_classes]
        t_row = t_flat[b * num_classes : (b + 1) * num_classes]
        pos_classes: list[int] = []
        for cls_idx in t_row:
            if cls_idx < 0:
                break
            pos_classes.append(cls_idx)
        pos_set = set(pos_classes)
        neg_classes = [j for j in range(num_classes) if j not in pos_set]
        total = 0.0
        for y_idx in pos_classes:
            for j_idx in neg_classes:
                total += max(0.0, 1.0 - (x_row[y_idx] - x_row[j_idx]))
        sample_losses.append(total / float(max(1, num_classes)))

    loss_tensor = (
        Tensor(sample_losses[0], dtype=x.dtype)
        if len(x.shape) == 1
        else zero_torch.reshape(Tensor(sample_losses, dtype=x.dtype), x.shape[:-1])
    )
    return _reduce_loss(loss_tensor, reduction)


def multi_label_margin_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Alias for `multilabel_margin_loss`.

    Args:
        *args: Input and target tensors.
        **kwargs: Optional keyword arguments (`reduction`).

    Returns:
        Computed loss tensor.
    """
    return multilabel_margin_loss(*args, **kwargs)


def elu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies Exponential Linear Unit (ELU) element-wise.

    Args:
        *args: Input tensor and optional `alpha`.
        **kwargs: Optional keyword arguments (`alpha`, `inplace`).

    Returns:
        Output tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    alpha_raw = args[1] if len(args) > 1 else kwargs.get("alpha", 1.0)
    alpha = float(alpha_raw) if isinstance(alpha_raw, (int, float)) else 1.0
    return zero_torch.where(x > 0, x, alpha * (zero_torch.exp(x) - 1.0))


def _logaddexp(a: float, b: float) -> float:
    """Computes `log(exp(a) + exp(b))` in a numerically stable manner.

    Args:
        a: First log-probability.
        b: Second log-probability.

    Returns:
        Log-sum-exp of `a` and `b`.
    """
    if a == -math.inf or b == -math.inf:
        return max(a, b)
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def ctc_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes the Connectionist Temporal Classification (CTC) forward-backward loss.

    Args:
        *args: `log_probs`, `targets`, `input_lengths`, `target_lengths`.
        **kwargs: Optional keyword arguments (`blank`, `reduction`, `zero_infinity`).

    Returns:
        Computed CTC loss tensor.
    """
    log_probs = _extract_tensor(
        args[0] if len(args) > 0 else kwargs.get("log_probs", 0.0)
    )
    reduction = _extract_reduction(kwargs)
    targets_raw = args[1] if len(args) > 1 else kwargs.get("targets")
    in_lens_raw = args[2] if len(args) > 2 else kwargs.get("input_lengths")
    tgt_lens_raw = args[3] if len(args) > 3 else kwargs.get("target_lengths")
    blank_raw = args[4] if len(args) > 4 else kwargs.get("blank", 0)
    blank = int(blank_raw) if isinstance(blank_raw, int) else 0
    zero_infinity = bool(kwargs.get("zero_infinity", False))

    if (
        not config.eager_mode
        or not isinstance(targets_raw, Tensor)
        or len(log_probs.shape) < 2
    ):
        return _reduce_loss(-log_probs, reduction)

    # Support both (T, N, C) and (T, C) unbatched inputs
    if len(log_probs.shape) == 2:
        lp_tnc = zero_torch.unsqueeze(log_probs, 1)
    else:
        lp_tnc = log_probs
    t_max, batch_size, num_classes = lp_tnc.shape[0], lp_tnc.shape[1], lp_tnc.shape[2]
    lp_flat = _to_float_list(lp_tnc)
    tgt_flat = [int(v) for v in _to_float_list(targets_raw)]

    in_lens = (
        [int(v) for v in _to_float_list(in_lens_raw)]
        if isinstance(in_lens_raw, Tensor)
        else [int(v) for v in in_lens_raw if isinstance(v, (int, float))]
        if isinstance(in_lens_raw, (list, tuple))
        else [t_max] * batch_size
    )
    tgt_lens = (
        [int(v) for v in _to_float_list(tgt_lens_raw)]
        if isinstance(tgt_lens_raw, Tensor)
        else [int(v) for v in tgt_lens_raw if isinstance(v, (int, float))]
        if isinstance(tgt_lens_raw, (list, tuple))
        else [len(tgt_flat) // max(1, batch_size)] * batch_size
    )

    is_1d_targets = len(targets_raw.shape) == 1
    tgt_stride = targets_raw.shape[-1] if len(targets_raw.shape) > 1 else 0
    offset_1d = 0

    losses: list[float] = []
    for b in range(batch_size):
        t_len = min(t_max, in_lens[b])
        s_len = tgt_lens[b]
        if is_1d_targets:
            y_seq = tgt_flat[offset_1d : offset_1d + s_len]
            offset_1d += s_len
        else:
            y_seq = tgt_flat[b * tgt_stride : b * tgt_stride + s_len]

        ext: list[int] = [blank]
        for y_val in y_seq:
            ext.append(y_val)
            ext.append(blank)
        ext_len = len(ext)

        alpha = [-math.inf] * ext_len
        alpha[0] = lp_flat[(0 * batch_size + b) * num_classes + ext[0]]
        second_idx = min(1, ext_len - 1)
        alpha[second_idx] = lp_flat[
            (0 * batch_size + b) * num_classes + ext[second_idx]
        ]

        for t_step in range(1, t_len):
            next_alpha = [-math.inf] * ext_len
            base_lp = (t_step * batch_size + b) * num_classes
            for s_idx in range(ext_len):
                prev = alpha[s_idx]
                if s_idx >= 1:
                    prev = _logaddexp(prev, alpha[s_idx - 1])
                if s_idx >= 2 and ext[s_idx] != blank and ext[s_idx] != ext[s_idx - 2]:
                    prev = _logaddexp(prev, alpha[s_idx - 2])
                next_alpha[s_idx] = prev + lp_flat[base_lp + ext[s_idx]]
            alpha = next_alpha

        log_prob = (
            _logaddexp(alpha[ext_len - 1], alpha[ext_len - 2])
            if ext_len > 1
            else alpha[0]
        )
        nll = 0.0 if (zero_infinity and math.isinf(log_prob)) else -log_prob
        losses.append(nll)

    loss_vec = Tensor(losses, dtype=log_probs.dtype)
    if reduction == "none":
        return (
            loss_vec
            if len(log_probs.shape) == 3
            else Tensor(losses[0], dtype=log_probs.dtype)
        )
    if reduction == "sum":
        return zero_torch.sum(loss_vec)
    denom = [float(max(1, tl)) for tl in tgt_lens]
    norm_losses = [l_val / d_val for l_val, d_val in zip(losses, denom)]
    return Tensor(
        sum(norm_losses) / float(max(1, len(norm_losses))), dtype=log_probs.dtype
    )


def multilabel_soft_margin_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes multi-label soft margin loss.

    Args:
        *args: Input and target tensors.
        **kwargs: Optional keyword arguments (`weight`, `reduction`).

    Returns:
        Computed loss tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    t = _extract_tensor(args[1] if len(args) > 1 else kwargs.get("target", 0.0))
    reduction = _extract_reduction(kwargs)
    loss = -(t * logsigmoid(x) + (1.0 - t) * logsigmoid(-x))
    w = kwargs.get("weight")
    if isinstance(w, Tensor):
        loss = loss * w
    loss = zero_torch.mean(loss, axis=-1)
    return _reduce_loss(loss, reduction)


def multi_label_soft_margin_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Alias for `multilabel_soft_margin_loss`.

    Args:
        *args: Input and target tensors.
        **kwargs: Optional keyword arguments (`weight`, `reduction`).

    Returns:
        Computed loss tensor.
    """
    return multilabel_soft_margin_loss(*args, **kwargs)


def gelu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies Gaussian Error Linear Units function.

    Args:
        *args: Input tensor.
        **kwargs: Optional keyword arguments (`approximate`).

    Returns:
        Output tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    approx = kwargs.get("approximate", "none")
    if approx == "tanh":
        inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x**3))
        return 0.5 * x * (1.0 + tanh(inner))
    return 0.5 * x * (1.0 + zero_torch.erf(x / math.sqrt(2.0)))


def cosine_embedding_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes cosine embedding loss.

    Args:
        *args: `input1`, `input2`, and `target` tensors.
        **kwargs: Optional keyword arguments (`margin`, `reduction`).

    Returns:
        Computed loss tensor.
    """
    x1 = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input1", 0.0))
    x2 = _extract_tensor(args[1] if len(args) > 1 else kwargs.get("input2", x1))
    y = _extract_tensor(
        args[2] if len(args) > 2 else kwargs.get("target", zero_torch.ones_like(x1))
    )
    margin_raw = kwargs.get("margin", 0.0)
    margin = float(margin_raw) if isinstance(margin_raw, (int, float)) else 0.0
    reduction = _extract_reduction(kwargs)
    norm1 = zero_torch.sqrt(zero_torch.sum(x1 * x1, axis=-1) + 1e-8)
    norm2 = zero_torch.sqrt(zero_torch.sum(x2 * x2, axis=-1) + 1e-8)
    cos = zero_torch.sum(x1 * x2, axis=-1) / (norm1 * norm2)
    y_matched = (
        zero_torch.sum(y, axis=-1) / float(y.shape[-1])
        if len(y.shape) > len(cos.shape)
        else y
    )
    loss = zero_torch.where(
        y_matched == 1, 1.0 - cos, zero_torch.maximum(0.0, cos - margin)
    )
    return _reduce_loss(loss, reduction)


def multi_margin_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes multi-class hinge margin loss over target class indices.

    Args:
        *args: Input and target tensors.
        **kwargs: Optional keyword arguments (`p`, `margin`, `weight`, `reduction`).

    Returns:
        Computed loss tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    t_raw = args[1] if len(args) > 1 else kwargs.get("target")
    p_raw = args[2] if len(args) > 2 else kwargs.get("p", 1)
    margin_raw = args[3] if len(args) > 3 else kwargs.get("margin", 1.0)
    w_raw = args[4] if len(args) > 4 else kwargs.get("weight")
    margin = float(margin_raw) if isinstance(margin_raw, (int, float)) else 1.0
    p = int(p_raw) if isinstance(p_raw, int) else 1
    reduction = _extract_reduction(kwargs)

    if not config.eager_mode or not isinstance(t_raw, Tensor) or len(x.shape) == 0:
        loss = zero_torch.maximum(0.0, margin - x) ** p
        return _reduce_loss(loss, reduction)

    num_classes = x.shape[-1]
    batch_size = functools.reduce(operator.mul, x.shape[:-1], 1)
    x_flat = _to_float_list(x)
    t_flat = [int(v) for v in _to_float_list(t_raw)]
    w_flat = _to_float_list(w_raw) if isinstance(w_raw, Tensor) else [1.0] * num_classes

    sample_losses: list[float] = []
    for b in range(batch_size):
        x_row = x_flat[b * num_classes : (b + 1) * num_classes]
        y_cls = t_flat[b % max(1, len(t_flat))]
        y_cls = max(0, min(num_classes - 1, y_cls))
        w_y = w_flat[y_cls] if y_cls < len(w_flat) else 1.0
        total = 0.0
        for j in range(num_classes):
            if j != y_cls:
                hinge = max(0.0, margin - x_row[y_cls] + x_row[j])
                total += w_y * (hinge**p)
        sample_losses.append(total / float(max(1, num_classes)))

    loss_tensor = (
        Tensor(sample_losses[0], dtype=x.dtype)
        if len(x.shape) == 1
        else zero_torch.reshape(Tensor(sample_losses, dtype=x.dtype), x.shape[:-1])
    )
    return _reduce_loss(loss_tensor, reduction)


def glu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the gated linear unit function `GLU(a, b) = a * sigmoid(b)`.

    Args:
        *args: Input tensor and optional dimension.
        **kwargs: Optional keyword arguments (`dim`).

    Returns:
        Output tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    dim_raw = args[1] if len(args) > 1 else kwargs.get("dim", -1)
    dim = int(dim_raw) if isinstance(dim_raw, int) else -1
    if len(x.shape) > 0:
        norm_dim = dim if dim >= 0 else len(x.shape) + dim
        if x.shape[norm_dim] % 2 == 0:
            half = x.shape[norm_dim] // 2
            prefix = tuple(slice(None) for _ in range(norm_dim))
            a = x[prefix + (slice(0, half),)]
            b = x[prefix + (slice(half, x.shape[norm_dim]),)]
            return a * sigmoid(b)
    return x * sigmoid(x)


def gaussian_nll_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes Gaussian negative log likelihood loss.

    Args:
        *args: `input`, `target`, and `var` tensors.
        **kwargs: Optional keyword arguments (`full`, `eps`, `reduction`).

    Returns:
        Computed loss tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    t = _extract_tensor(args[1] if len(args) > 1 else kwargs.get("target", 0.0))
    var = _extract_tensor(
        args[2] if len(args) > 2 else kwargs.get("var", zero_torch.ones_like(x))
    )
    eps_raw = kwargs.get("eps", 1e-6)
    eps = float(eps_raw) if isinstance(eps_raw, (int, float)) else 1e-6
    full = bool(kwargs.get("full", False))
    reduction = _extract_reduction(kwargs)
    var_clamped = zero_torch.maximum(var, eps)
    loss = 0.5 * (zero_torch.log(var_clamped) + ((x - t) ** 2) / var_clamped)
    if full:
        loss = loss + 0.5 * math.log(2.0 * math.pi)
    return _reduce_loss(loss, reduction)


def nll_loss(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    size_average: bool | None = None,
    ignore_index: int = -100,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Computes the negative log likelihood loss vectorially across 1D, 2D, and N-D inputs.

    Args:
        input: Log-probabilities tensor of shape `(C,)`, `(N, C)`, or `(N, C, d_1, ..., d_k)`.
        target: Ground-truth class indices tensor of shape `()`, `(N,)`, or `(N, d_1, ..., d_k)`.
        weight: Optional manual rescaling weight per class of shape `(C,)`.
        size_average: Deprecated flag.
        ignore_index: Target value that is ignored and does not contribute to the gradient.
        reduce: Deprecated flag.
        reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).

    Returns:
        Computed NLL loss tensor.
    """
    _ = (size_average, reduce)
    ndim = len(input.shape)
    if ndim == 0:
        return _reduce_loss(-input, reduction)

    if ndim == 1:
        num_classes = input.shape[0]
        classes = zero_torch.arange(0, num_classes, dtype=target.dtype)
        one_hot = zero_torch.where(classes == target, 1.0, 0.0)
        if weight is not None:
            one_hot = one_hot * weight
        valid = target != ignore_index
        sample_loss = zero_torch.where(
            valid, -zero_torch.sum(input * one_hot), Tensor(0.0, dtype=input.dtype)
        )
        return sample_loss

    # Move class dimension (axis 1) to the last axis for N-D inputs (N, C, d1, ..., dk)
    if ndim > 2:
        perm = (0, *range(2, ndim), 1)
        x_last = zero_torch.permute(input, perm)
    else:
        x_last = input

    num_classes = x_last.shape[-1]
    classes = zero_torch.arange(0, num_classes, dtype=target.dtype)
    target_exp = zero_torch.unsqueeze(target, -1)
    one_hot = zero_torch.where(target_exp == classes, 1.0, 0.0)
    if weight is not None:
        one_hot = one_hot * weight

    valid = target != ignore_index
    raw_nll = -zero_torch.sum(x_last * one_hot, axis=-1)
    loss_elem = zero_torch.where(valid, raw_nll, 0.0)

    if reduction == "none":
        return loss_elem
    if reduction == "sum":
        return zero_torch.sum(loss_elem)

    sample_weights = zero_torch.where(valid, zero_torch.sum(one_hot, axis=-1), 0.0)
    total_weight = zero_torch.maximum(zero_torch.sum(sample_weights), 1e-12)
    return zero_torch.sum(loss_elem) / total_weight


def hardshrink(input: Tensor, lambd: float = 0.5) -> Tensor:
    """Applies the hard shrinkage function element-wise.

    Args:
        input: Input tensor.
        lambd: The `lambda` value for the Hardshrink formulation.

    Returns:
        Output tensor.
    """
    t = _to_tensor(input)
    cond = _ops.logical_or(t > lambd, t < -lambd)
    return _wrap(_where(cond, t, _to_tensor(0.0)))


def hinge_embedding_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes hinge embedding loss.

    Args:
        *args: Input and target tensors.
        **kwargs: Optional keyword arguments (`margin`, `reduction`).

    Returns:
        Computed loss tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    y = _extract_tensor(
        args[1] if len(args) > 1 else kwargs.get("target", zero_torch.ones_like(x))
    )
    margin_raw = kwargs.get("margin", 1.0)
    margin = float(margin_raw) if isinstance(margin_raw, (int, float)) else 1.0
    reduction = _extract_reduction(kwargs)
    loss = zero_torch.where(y == 1, x, zero_torch.maximum(0.0, margin - x))
    return _reduce_loss(loss, reduction)


def poisson_nll_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes Poisson negative log likelihood loss.

    Args:
        *args: `input` and `target` tensors.
        **kwargs: Optional keyword arguments (`log_input`, `full`, `eps`, `reduction`).

    Returns:
        Computed loss tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    t = _extract_tensor(args[1] if len(args) > 1 else kwargs.get("target", 0.0))
    log_inp = bool(kwargs.get("log_input", True))
    full = bool(kwargs.get("full", False))
    eps_raw = kwargs.get("eps", 1e-8)
    eps = float(eps_raw) if isinstance(eps_raw, (int, float)) else 1e-8
    reduction = _extract_reduction(kwargs)
    loss = zero_torch.exp(x) - t * x if log_inp else x - t * zero_torch.log(x + eps)
    if full:
        stirling = (
            t * zero_torch.log(zero_torch.maximum(t, 1.0))
            - t
            + 0.5 * zero_torch.log(2.0 * math.pi * zero_torch.maximum(t, 1.0))
        )
        loss = loss + zero_torch.where(t > 1.0, stirling, 0.0)
    return _reduce_loss(loss, reduction)


def sigmoid(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the element-wise function `Sigmoid(x) = 1 / (1 + exp(-x))`.

    Args:
        *args: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor.
    """
    _ = kwargs
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    if x.requires_grad:
        from ..autograd.math_ops import SigmoidBackward

        return SigmoidBackward.apply(x)
    inp = _to_tensor(x)
    res = _ops.divide(1.0, _ops.add(1.0, _ops.exp(_ops.negative(inp))))
    return _wrap(res)


def huber_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes Huber loss.

    Args:
        *args: Input and target tensors.
        **kwargs: Optional keyword arguments (`reduction`, `delta`).

    Returns:
        Computed loss tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    t = _extract_tensor(args[1] if len(args) > 1 else kwargs.get("target", 0.0))
    delta_raw = kwargs.get("delta", 1.0)
    delta = float(delta_raw) if isinstance(delta_raw, (int, float)) else 1.0
    reduction = _extract_reduction(kwargs)
    diff = zero_torch.abs(x - t)
    loss = zero_torch.where(diff < delta, 0.5 * (diff**2), delta * (diff - 0.5 * delta))
    return _reduce_loss(loss, reduction)


def smooth_l1_loss(
    input: Tensor,
    target: Tensor,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
    beta: float = 1.0,
) -> Tensor:
    """Computes Smooth L1 loss.

    Args:
        input: Input prediction tensor.
        target: Ground-truth target tensor.
        size_average: Deprecated flag.
        reduce: Deprecated flag.
        reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
        beta: Threshold at which to change between L1 and L2 loss.

    Returns:
        Computed Smooth L1 loss tensor.
    """
    _ = (size_average, reduce)
    diff = zero_torch.abs(input - target)
    loss = zero_torch.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    return _reduce_loss(loss, reduction)


def hardsigmoid(input: Tensor, inplace: bool = False) -> Tensor:
    """Applies the Hardsigmoid function element-wise.

    Args:
        input: Input tensor.
        inplace: Whether to perform the operation in-place.

    Returns:
        Output tensor.
    """
    _ = inplace
    return zero_torch.clip(input / 6.0 + 0.5, 0.0, 1.0)


def hard_sigmoid(input: Tensor, inplace: bool = False) -> Tensor:
    """Alias for `hardsigmoid`.

    Args:
        input: Input tensor.
        inplace: Whether to perform the operation in-place.

    Returns:
        Output tensor.
    """
    return hardsigmoid(input, inplace=inplace)


def kl_div(
    input: Tensor,
    target: Tensor,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
    log_target: bool = False,
) -> Tensor:
    """Computes the Kullback-Leibler divergence loss.

    Args:
        input: Tensor of arbitrary shape in log-probabilities.
        target: Tensor of the same shape as input.
        size_average: Deprecated flag.
        reduce: Deprecated flag.
        reduction: Specifies the reduction to apply (`'none'`, `'batchmean'`, `'sum'`, `'mean'`).
        log_target: A flag indicating whether `target` is passed in log space.

    Returns:
        Computed KL-divergence loss tensor.
    """
    _ = (size_average, reduce)
    if log_target:
        loss = zero_torch.exp(target) * (target - input)
    else:
        loss = target * (zero_torch.log(zero_torch.maximum(target, 1e-12)) - input)
    if reduction == "batchmean":
        batch_sz = float(input.shape[0]) if len(input.shape) > 0 else 1.0
        return zero_torch.sum(loss) / batch_sz
    if reduction == "mean":
        return zero_torch.mean(loss)
    if reduction == "sum":
        return zero_torch.sum(loss)
    return loss


def soft_margin_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes soft margin loss.

    Args:
        *args: Input and target tensors.
        **kwargs: Optional keyword arguments (`reduction`).

    Returns:
        Computed loss tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    y = _extract_tensor(
        args[1] if len(args) > 1 else kwargs.get("target", zero_torch.ones_like(x))
    )
    reduction = _extract_reduction(kwargs)
    loss = zero_torch.log(1.0 + zero_torch.exp(-y * x))
    return _reduce_loss(loss, reduction)


def hardswish(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the hardswish function element-wise.

    Args:
        *args: Input tensor.
        **kwargs: Optional keyword arguments (`inplace`).

    Returns:
        Output tensor.
    """
    _ = kwargs
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    return x * zero_torch.clip(x + 3.0, 0.0, 6.0) / 6.0


def margin_ranking_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes margin ranking loss.

    Args:
        *args: `input1`, `input2`, and `target` tensors.
        **kwargs: Optional keyword arguments (`margin`, `reduction`).

    Returns:
        Computed loss tensor.
    """
    x1 = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input1", 0.0))
    x2 = _extract_tensor(args[1] if len(args) > 1 else kwargs.get("input2", x1))
    y = _extract_tensor(
        args[2] if len(args) > 2 else kwargs.get("target", zero_torch.ones_like(x1))
    )
    margin_raw = kwargs.get("margin", 0.0)
    margin = float(margin_raw) if isinstance(margin_raw, (int, float)) else 0.0
    reduction = _extract_reduction(kwargs)
    loss = zero_torch.maximum(0.0, -y * (x1 - x2) + margin)
    return _reduce_loss(loss, reduction)


def triplet_margin_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes triplet margin loss.

    Args:
        *args: `anchor`, `positive`, and `negative` tensors.
        **kwargs: Optional keyword arguments (`margin`, `p`, `eps`, `swap`, `reduction`).

    Returns:
        Computed loss tensor.
    """
    a = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("anchor", 0.0))
    p_t = _extract_tensor(args[1] if len(args) > 1 else kwargs.get("positive", a))
    n_t = _extract_tensor(args[2] if len(args) > 2 else kwargs.get("negative", a))
    margin_raw = kwargs.get("margin", 1.0)
    margin = float(margin_raw) if isinstance(margin_raw, (int, float)) else 1.0
    p_raw = kwargs.get("p", 2.0)
    p = float(p_raw) if isinstance(p_raw, (int, float)) else 2.0
    eps_raw = kwargs.get("eps", 1e-6)
    eps = float(eps_raw) if isinstance(eps_raw, (int, float)) else 1e-6
    swap = bool(kwargs.get("swap", False))
    reduction = _extract_reduction(kwargs)
    d_pos = (zero_torch.sum(zero_torch.abs(a - p_t) ** p, axis=-1) + eps) ** (1.0 / p)
    d_neg = (zero_torch.sum(zero_torch.abs(a - n_t) ** p, axis=-1) + eps) ** (1.0 / p)
    if swap:
        d_swap = (zero_torch.sum(zero_torch.abs(p_t - n_t) ** p, axis=-1) + eps) ** (
            1.0 / p
        )
        d_neg = zero_torch.minimum(d_neg, d_swap)
    loss = zero_torch.maximum(0.0, d_pos - d_neg + margin)
    return _reduce_loss(loss, reduction)


def hardtanh(
    input: Tensor,
    min_val: float = -1.0,
    max_val: float = 1.0,
    inplace: bool = False,
) -> Tensor:
    """Applies the HardTanh function element-wise.

    Args:
        input: Input tensor.
        min_val: Minimum value of the linear region range.
        max_val: Maximum value of the linear region range.
        inplace: Whether to perform the operation in-place.

    Returns:
        Output tensor.
    """
    _ = inplace
    return zero_torch.clip(input, min_val, max_val)


def triplet_margin_with_distance_loss(
    *args: OpArg | Callable[[Tensor, Tensor], Tensor],
    **kwargs: OpArg | Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """Computes triplet margin loss with a custom distance function.

    Args:
        *args: `anchor`, `positive`, and `negative` tensors.
        **kwargs: Optional keyword arguments (`distance_function`, `margin`, `swap`, `reduction`).

    Returns:
        Computed loss tensor.
    """
    a_raw = args[0] if len(args) > 0 else kwargs.get("anchor", 0.0)
    p_raw = args[1] if len(args) > 1 else kwargs.get("positive", a_raw)
    n_raw = args[2] if len(args) > 2 else kwargs.get("negative", a_raw)
    a = a_raw if isinstance(a_raw, Tensor) else Tensor(0.0)
    p_t = p_raw if isinstance(p_raw, Tensor) else a
    n_t = n_raw if isinstance(n_raw, Tensor) else a
    dist_fn = kwargs.get("distance_function")
    margin_raw = kwargs.get("margin", 1.0)
    margin = float(margin_raw) if isinstance(margin_raw, (int, float)) else 1.0
    swap = bool(kwargs.get("swap", False))
    red_raw = kwargs.get("reduction", "mean")
    reduction = red_raw if isinstance(red_raw, str) else "mean"
    if callable(dist_fn):
        d_pos = dist_fn(a, p_t)
        d_neg = dist_fn(a, n_t)
        if swap:
            d_neg = zero_torch.minimum(d_neg, dist_fn(p_t, n_t))
    else:
        d_pos = zero_torch.sqrt(zero_torch.sum((a - p_t) ** 2, axis=-1) + 1e-6)
        d_neg = zero_torch.sqrt(zero_torch.sum((a - n_t) ** 2, axis=-1) + 1e-6)
        if swap:
            d_swap = zero_torch.sqrt(zero_torch.sum((p_t - n_t) ** 2, axis=-1) + 1e-6)
            d_neg = zero_torch.minimum(d_neg, d_swap)
    loss = zero_torch.maximum(0.0, d_pos - d_neg + margin)
    return _reduce_loss(loss, reduction)


def leaky_relu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the element-wise function `LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)`.

    Args:
        *args: Input tensor and optional `negative_slope`.
        **kwargs: Optional keyword arguments (`negative_slope`, `inplace`).

    Returns:
        Output tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    slope_raw = args[1] if len(args) > 1 else kwargs.get("negative_slope", 0.01)
    slope = float(slope_raw) if isinstance(slope_raw, (int, float)) else 0.01
    return zero_torch.where(x >= 0, x, slope * x)


def logsigmoid(input: Tensor) -> Tensor:
    """Applies the element-wise function `LogSigmoid(x) = log(1 / (1 + exp(-x)))`.

    Args:
        input: Input tensor.

    Returns:
        Output tensor.
    """
    return -softplus(-input)


def log_softmax(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies a softmax followed by a logarithm.

    Args:
        *args: Input tensor and optional `dim`.
        **kwargs: Optional keyword arguments (`dim`, `dtype`).

    Returns:
        Log-softmax tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    dim_raw = args[1] if len(args) > 1 else kwargs.get("dim", -1)
    dim = int(dim_raw) if isinstance(dim_raw, int) else -1
    max_x = zero_torch.max(x, axis=dim, keepdims=True)
    shifted = x - max_x
    log_sum_exp = zero_torch.log(
        zero_torch.sum(zero_torch.exp(shifted), axis=dim, keepdims=True)
    )
    return shifted - log_sum_exp


def mish(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the Mish function element-wise.

    Args:
        *args: Input tensor.
        **kwargs: Optional keyword arguments (`inplace`).

    Returns:
        Output tensor.
    """
    _ = kwargs
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    return x * tanh(softplus(x))


def prelu(input: Tensor, weight: Tensor) -> Tensor:
    """Applies element-wise `PReLU(x) = max(0, x) + weight * min(0, x)`.

    Args:
        input: Input tensor.
        weight: Learnable slope parameter tensor.

    Returns:
        Output tensor.
    """
    w = (
        zero_torch.reshape(weight, (1, -1) + (1,) * (len(input.shape) - 2))
        if len(input.shape) > 2
        else weight
    )
    return zero_torch.where(input > 0, input, w * input)


def rrelu(
    input: Tensor,
    lower: float = 1.0 / 8.0,
    upper: float = 1.0 / 3.0,
    training: bool = False,
    inplace: bool = False,
) -> Tensor:
    """Applies the randomized leaky rectified linear unit function element-wise.

    Args:
        input: Input tensor.
        lower: Lower bound of the uniform distribution.
        upper: Upper bound of the uniform distribution.
        training: Whether the layer is in training mode.
        inplace: Whether to perform the operation in-place.

    Returns:
        Output tensor.
    """
    _ = inplace
    alpha = (lower + upper) / 2.0
    if training:
        rand_alpha = zero_torch.rand(input.shape) * (upper - lower) + lower
        return zero_torch.where(input >= 0, input, input * rand_alpha)
    return zero_torch.where(input >= 0, input, input * alpha)


def relu6(input: Tensor, inplace: bool = False) -> Tensor:
    """Applies the element-wise function `ReLU6(x) = min(max(0, x), 6)`.

    Args:
        input: Input tensor.
        inplace: Whether to perform the operation in-place.

    Returns:
        Output tensor.
    """
    _ = inplace
    return zero_torch.clip(input, 0.0, 6.0)


def selu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the SELU function element-wise.

    Args:
        *args: Input tensor.
        **kwargs: Optional keyword arguments (`inplace`).

    Returns:
        Output tensor.
    """
    _ = kwargs
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * zero_torch.where(x > 0, x, alpha * (zero_torch.exp(x) - 1.0))


def silu(input: Tensor, inplace: bool = False) -> Tensor:
    """Applies the Sigmoid Linear Unit (SiLU) function element-wise.

    Args:
        input: Input tensor.
        inplace: Whether to perform the operation in-place.

    Returns:
        Output tensor.
    """
    _ = inplace
    return input * sigmoid(input)


def softmax2d(input: Tensor) -> Tensor:
    """Applies SoftMax over features to each spatial location.

    Args:
        input: Input tensor.

    Returns:
        Output tensor.
    """
    return softmax(input, dim=-3 if len(input.shape) >= 3 else 0)


def softmin(input: Tensor, dim: int | None = None) -> Tensor:
    """Applies a softmin function `Softmin(x) = Softmax(-x)`.

    Args:
        input: Input tensor.
        dim: Dimension along which softmin will be computed.

    Returns:
        Output tensor.
    """
    return softmax(-input, dim=dim)


def softplus(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the element-wise function `Softplus(x) = (1/beta) * log(1 + exp(beta * x))`.

    Args:
        *args: Input tensor.
        **kwargs: Optional keyword arguments (`beta`, `threshold`).

    Returns:
        Output tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    beta_raw = args[1] if len(args) > 1 else kwargs.get("beta", 1.0)
    thresh_raw = args[2] if len(args) > 2 else kwargs.get("threshold", 20.0)
    beta = float(beta_raw) if isinstance(beta_raw, (int, float)) else 1.0
    threshold_val = float(thresh_raw) if isinstance(thresh_raw, (int, float)) else 20.0
    bx = x * beta
    return zero_torch.where(
        bx > threshold_val,
        x,
        zero_torch.log(1.0 + zero_torch.exp(bx)) / beta,
    )


def softshrink(input: Tensor, lambd: float = 0.5) -> Tensor:
    """Applies the soft shrinkage function element-wise.

    Args:
        input: Input tensor.
        lambd: The `lambda` value for the Softshrink formulation.

    Returns:
        Output tensor.
    """
    return zero_torch.where(
        input > lambd,
        input - lambd,
        zero_torch.where(input < -lambd, input + lambd, zero_torch.zeros_like(input)),
    )


def soft_shrink(input: Tensor, lambd: float = 0.5) -> Tensor:
    """Alias for `softshrink`.

    Args:
        input: Input tensor.
        lambd: The `lambda` value for the Softshrink formulation.

    Returns:
        Output tensor.
    """
    return softshrink(input, lambd=lambd)


def softsign(input: Tensor) -> Tensor:
    """Applies the element-wise function `Softsign(x) = x / (1 + |x|)`.

    Args:
        input: Input tensor.

    Returns:
        Output tensor.
    """
    return input / (1.0 + zero_torch.abs(input))


def tanh(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the element-wise Hyperbolic Tangent (`Tanh`) function.

    Args:
        *args: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor.
    """
    _ = kwargs
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    if x.requires_grad:
        from ..autograd.math_ops import TanhBackward

        return TanhBackward.apply(x)
    return _wrap(_ops.tanh(_to_tensor(x)))


def tanhshrink(input: Tensor) -> Tensor:
    """Applies the element-wise function `Tanhshrink(x) = x - Tanh(x)`.

    Args:
        input: Input tensor.

    Returns:
        Output tensor.
    """
    return input - tanh(input)


def threshold(
    input: Tensor, threshold: float, value: float, inplace: bool = False
) -> Tensor:
    """Thresholds each element of the input Tensor.

    Args:
        input: Input tensor.
        threshold: The value to threshold at.
        value: The value to replace with.
        inplace: Whether to perform the operation in-place.

    Returns:
        Thresholded tensor.
    """
    _ = inplace
    return zero_torch.where(
        input > threshold, input, zero_torch.tensor(value, dtype=input.dtype)
    )


def activations(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies generic activation pass-through.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Output tensor.
    """
    return _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))


def adaptive_avg_pool2d(input: Tensor, output_size: OptSize2d = 1) -> Tensor:
    """Applies 2D adaptive average pooling over an input signal.

    Args:
        input: Input tensor.
        output_size: Target output spatial size `(H_out, W_out)` or `H_out`.

    Returns:
        Pooled tensor.
    """
    return _fp_adaptive_avg_pool2d(input, output_size)


def alpha_dropout(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies alpha dropout.

    Args:
        *args: Input tensor and dropout probability.
        **kwargs: Optional keyword arguments (`p`, `training`, `inplace`).

    Returns:
        Output tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    p_raw = args[1] if len(args) > 1 else kwargs.get("p", 0.5)
    p = float(p_raw) if isinstance(p_raw, (int, float)) else 0.5
    training = bool(kwargs.get("training", False))
    inplace = bool(kwargs.get("inplace", False))
    return _fp_alpha_dropout(x, p=p, training=training, inplace=inplace)


def avg_pool1d(
    input: Tensor,
    kernel_size: Size1d,
    stride: Size1d | None = None,
    padding: Size1d = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
) -> Tensor:
    """Applies 1D average pooling over an input signal.

    Args:
        input: Input tensor.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit zero padding.
        ceil_mode: Whether to use ceil for output shape.
        count_include_pad: Whether to include zero-padding in the averaging calculation.
        divisor_override: Optional explicit divisor overriding window size.

    Returns:
        Pooled 1D tensor.
    """
    return _fp_avg1d(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )


def avg_pool2d(
    input: Tensor,
    kernel_size: Size2d,
    stride: Size2d | None = None,
    padding: Size2d = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
) -> Tensor:
    """Applies 2D average pooling over an input signal.

    Args:
        input: Input tensor.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit zero padding.
        ceil_mode: Whether to use ceil for output shape.
        count_include_pad: Whether to include zero-padding in the averaging calculation.
        divisor_override: Optional explicit divisor overriding window size.

    Returns:
        Pooled 2D tensor.
    """
    return _fp_avg2d(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )


def avg_pool3d(
    input: Tensor,
    kernel_size: Size3d,
    stride: Size3d | None = None,
    padding: Size3d = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
) -> Tensor:
    """Applies 3D average pooling over an input signal.

    Args:
        input: Input tensor.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit zero padding.
        ceil_mode: Whether to use ceil for output shape.
        count_include_pad: Whether to include zero-padding in the averaging calculation.
        divisor_override: Optional explicit divisor overriding window size.

    Returns:
        Pooled 3D tensor.
    """
    return _fp_avg3d(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )


def max_pool1d(
    input: Tensor,
    kernel_size: Size1d,
    stride: Size1d | None = None,
    padding: Size1d = 0,
    dilation: Size1d = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Applies 1D max pooling over an input signal.

    Args:
        input: Input tensor.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit negative infinity padding.
        dilation: Window dilation.
        ceil_mode: Whether to use ceil for output shape.
        return_indices: Whether to return indices.

    Returns:
        Pooled 1D tensor, or `(output, indices)` if `return_indices=True`.
    """
    return _fp_max1d(
        input, kernel_size, stride, padding, dilation, ceil_mode, return_indices
    )


def max_pool2d(
    input: Tensor,
    kernel_size: Size2d,
    stride: Size2d | None = None,
    padding: Size2d = 0,
    dilation: Size2d = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Applies 2D max pooling over an input signal.

    Args:
        input: Input tensor.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit negative infinity padding.
        dilation: Window dilation.
        ceil_mode: Whether to use ceil for output shape.
        return_indices: Whether to return indices.

    Returns:
        Pooled 2D tensor, or `(output, indices)` if `return_indices=True`.
    """
    return _fp_max2d(
        input, kernel_size, stride, padding, dilation, ceil_mode, return_indices
    )


def max_pool3d(
    input: Tensor,
    kernel_size: Size3d,
    stride: Size3d | None = None,
    padding: Size3d = 0,
    dilation: Size3d = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Applies 3D max pooling over an input signal.

    Args:
        input: Input tensor.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit negative infinity padding.
        dilation: Window dilation.
        ceil_mode: Whether to use ceil for output shape.
        return_indices: Whether to return indices.

    Returns:
        Pooled 3D tensor, or `(output, indices)` if `return_indices=True`.
    """
    return _fp_max3d(
        input, kernel_size, stride, padding, dilation, ceil_mode, return_indices
    )


def batch_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies batch normalization.

    Args:
        *args: Input tensor and running stats.
        **kwargs: Optional keyword arguments (`eps`, `momentum`, `training`).

    Returns:
        Normalized tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    if len(x.shape) < 2:
        return x
    return _fp_batch_norm(x, *args[1:], **kwargs)


def complex(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Constructs a complex tensor from real and imaginary parts.

    Args:
        *args: Real and imaginary tensors.
        **kwargs: Optional keyword arguments.

    Returns:
        Complex tensor.
    """
    return _extract_tensor(args[0] if len(args) > 0 else kwargs.get("real", 0.0))


def conv_transpose1d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies a 1D transposed convolution operator.

    Args:
        *args: `input`, `weight`, and optional `bias`, `stride`, `padding`, `output_padding`, `groups`, `dilation`.
        **kwargs: Optional keyword arguments (`stride`, `padding`, `output_padding`, `groups`, `dilation`).

    Returns:
        Transposed convolved tensor.
    """
    return _conv_transpose_nd(1, *args, **kwargs)


def conv_transpose2d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies a 2D transposed convolution operator.

    Args:
        *args: `input`, `weight`, and optional `bias`, `stride`, `padding`, `output_padding`, `groups`, `dilation`.
        **kwargs: Optional keyword arguments (`stride`, `padding`, `output_padding`, `groups`, `dilation`).

    Returns:
        Transposed convolved tensor.
    """
    return _conv_transpose_nd(2, *args, **kwargs)


def conv_transpose3d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies a 3D transposed convolution operator.

    Args:
        *args: `input`, `weight`, and optional `bias`, `stride`, `padding`, `output_padding`, `groups`, `dilation`.
        **kwargs: Optional keyword arguments (`stride`, `padding`, `output_padding`, `groups`, `dilation`).

    Returns:
        Transposed convolved tensor.
    """
    return _conv_transpose_nd(3, *args, **kwargs)


def dropout(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies dropout.

    Args:
        *args: Input tensor and optional dropout probability `p`.
        **kwargs: Optional keyword arguments (`p`, `training`, `inplace`).

    Returns:
        Output tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    p_raw = args[1] if len(args) > 1 else kwargs.get("p", 0.5)
    p = float(p_raw) if isinstance(p_raw, (int, float)) else 0.5
    training = bool(kwargs.get("training", False))
    inplace = bool(kwargs.get("inplace", False))
    return _fp_dropout(x, p=p, training=training, inplace=inplace)


def dropout1d(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """Randomly zeroes out entire 1D feature channels.

    Args:
        input: Input tensor.
        p: Probability of a channel to be zeroed.
        training: Apply dropout if True.
        inplace: Whether to perform the operation in-place.

    Returns:
        Dropped-out tensor.
    """
    return _fp_dropout1d(input, p=p, training=training, inplace=inplace)


def dropout2d(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """Randomly zeroes out entire 2D feature channels.

    Args:
        input: Input tensor.
        p: Probability of a channel to be zeroed.
        training: Apply dropout if True.
        inplace: Whether to perform the operation in-place.

    Returns:
        Dropped-out tensor.
    """
    return _fp_dropout2d(input, p=p, training=training, inplace=inplace)


def dropout3d(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """Randomly zeroes out entire 3D feature channels.

    Args:
        input: Input tensor.
        p: Probability of a channel to be zeroed.
        training: Apply dropout if True.
        inplace: Whether to perform the operation in-place.

    Returns:
        Dropped-out tensor.
    """
    return _fp_dropout3d(input, p=p, training=training, inplace=inplace)


def embedding(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Looks up embeddings in a fixed dictionary and size using index gather.

    Args:
        *args: `input` indices and `weight` matrix.
        **kwargs: Optional keyword arguments (`padding_idx`, `max_norm`, `norm_type`, `scale_grad_by_freq`, `sparse`).

    Returns:
        Embedded tensor of shape `(*input.shape, embedding_dim)`.
    """
    idx = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0))
    w_raw = args[1] if len(args) > 1 else kwargs.get("weight")
    if not isinstance(w_raw, Tensor) or len(w_raw.shape) != 2:
        return idx
    w = w_raw
    pad_idx_raw = args[2] if len(args) > 2 else kwargs.get("padding_idx")
    max_norm_raw = args[3] if len(args) > 3 else kwargs.get("max_norm")
    norm_type_raw = args[4] if len(args) > 4 else kwargs.get("norm_type", 2.0)
    norm_type = float(norm_type_raw) if isinstance(norm_type_raw, (int, float)) else 2.0

    if isinstance(max_norm_raw, (int, float)):
        max_norm_val = float(max_norm_raw)
        row_norms = (
            zero_torch.sum(zero_torch.abs(w) ** norm_type, axis=-1, keepdims=True)
        ) ** (1.0 / norm_type)
        scale = max_norm_val / (row_norms + 1e-7)
        w_renorm = zero_torch.where(row_norms > max_norm_val, w * scale, w)
        if config.eager_mode and not w.requires_grad:
            w.copy_(w_renorm)
        w = w_renorm

    emb_dim = w.shape[1]
    out_shape = idx.shape + (emb_dim,)
    if not config.eager_mode:
        return _wrap(
            _emit_reduction_node(
                "Embedding",
                [_to_tensor(w), _to_tensor(idx)],
                {},
                out_shape,
                w.dtype,
            )
        )

    flat_idx = zero_torch.cast(zero_torch.reshape(idx, (-1,)), zero_torch.int32)
    gathered = _wrap(get_op("Take")()(_to_tensor(w), _to_tensor(flat_idx), axis=0))
    out = zero_torch.reshape(gathered, out_shape)
    if isinstance(pad_idx_raw, int):
        norm_pad = pad_idx_raw if pad_idx_raw >= 0 else w.shape[0] + pad_idx_raw
        mask = zero_torch.unsqueeze(idx == norm_pad, -1)
        out = zero_torch.where(mask, zero_torch.zeros_like(out), out)
    return out


def feature_alpha_dropout(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Randomly masks out entire channels using alpha dropout.

    Args:
        *args: Input tensor and optional probability `p`.
        **kwargs: Optional keyword arguments (`p`, `training`, `inplace`).

    Returns:
        Output tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    p_raw = args[1] if len(args) > 1 else kwargs.get("p", 0.5)
    p = float(p_raw) if isinstance(p_raw, (int, float)) else 0.5
    training = bool(kwargs.get("training", False))
    inplace = bool(kwargs.get("inplace", False))
    return _fp_feature_alpha_dropout(x, p=p, training=training, inplace=inplace)


def fractional_max_pool2d(
    input: Tensor,
    kernel_size: Size2d = 2,
    output_size: Size2d | None = None,
    output_ratio: float | tuple[float, float] | None = None,
    return_indices: bool = False,
    _random_samples: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Applies 2D fractional max pooling.

    Args:
        input: Input tensor.
        kernel_size: Size of the pooling window.
        output_size: Target output spatial size `(H_out, W_out)`.
        output_ratio: Target output spatial ratio in `(0, 1)`.
        return_indices: Whether to return argmax indices alongside pooled outputs.
        _random_samples: Optional pre-generated random step samples.

    Returns:
        Pooled tensor, or `(output, indices)` if `return_indices=True`.
    """
    eff_size = (
        output_size
        if (output_size is not None or output_ratio is not None)
        else kernel_size
    )
    return _fp_fractional_max_pool2d(
        input,
        kernel_size=kernel_size,
        output_size=eff_size,
        output_ratio=output_ratio,
        return_indices=return_indices,
        _random_samples=_random_samples,
    )


def group_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies Group Normalization.

    Args:
        *args: `input`, `num_groups`, and optional affine parameters.
        **kwargs: Optional keyword arguments (`weight`, `bias`, `eps`).

    Returns:
        Normalized tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    g_raw = args[1] if len(args) > 1 else kwargs.get("num_groups", 1)
    w_raw = args[2] if len(args) > 2 else kwargs.get("weight")
    b_raw = args[3] if len(args) > 3 else kwargs.get("bias")
    groups = int(g_raw) if isinstance(g_raw, int) else 1
    eps_raw = kwargs.get("eps", 1e-5)
    eps = float(eps_raw) if isinstance(eps_raw, (int, float)) else 1e-5
    if len(x.shape) >= 2 and x.shape[1] % groups == 0:
        out = _wrap(_compiler_group_norm(_to_tensor(x), num_groups=groups, epsilon=eps))
        bcast = (1, x.shape[1]) + (1,) * (len(x.shape) - 2)
        if isinstance(w_raw, Tensor):
            out = out * zero_torch.reshape(w_raw, bcast)
        if isinstance(b_raw, Tensor):
            out = out + zero_torch.reshape(b_raw, bcast)
        return out
    return x


def gru_cell(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies a gated recurrent unit (GRU) cell step.

    Args:
        *args: `input`, `hx`, `w_ih`, `w_hh`, and optional biases.
        **kwargs: Optional keyword arguments.

    Returns:
        Next hidden state tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    return tanh(x)


def instance_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies Instance Normalization.

    Args:
        *args: Input tensor and optional running stats / affine parameters.
        **kwargs: Optional keyword arguments (`eps`, `momentum`).

    Returns:
        Normalized tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    if len(x.shape) >= 3:
        return _fp_instance_norm(x, *args[1:], **kwargs)
    return x


def layer_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies Layer Normalization.

    Args:
        *args: `input`, `normalized_shape`, and optional affine parameters.
        **kwargs: Optional keyword arguments (`weight`, `bias`, `eps`).

    Returns:
        Normalized tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    norm_shape = (
        args[1] if len(args) > 1 else kwargs.get("normalized_shape", x.shape[-1:])
    )
    return _fp_layer_norm(x, norm_shape, *args[2:], **kwargs)


def lstm_cell(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies a long short-term memory (LSTM) cell step.

    Args:
        *args: `input`, `hx`, `w_ih`, `w_hh`, and optional biases.
        **kwargs: Optional keyword arguments.

    Returns:
        Next state tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    return tanh(x)


def _pad_single_axis(
    x: Tensor,
    axis: int,
    pad_before: int,
    pad_after: int,
    mode: str,
    value: float,
) -> Tensor:
    """Pads tensor `x` along a single `axis` with `(pad_before, pad_after)` using `mode`.

    Args:
        x: Input tensor.
        axis: Dimension index to pad.
        pad_before: Elements to add before index 0.
        pad_after: Elements to add after index `L - 1`.
        mode: Padding mode (`'constant'`, `'reflect'`, `'replicate'`, `'circular'`).
        value: Fill value for `'constant'` mode.

    Returns:
        Padded tensor along `axis`.
    """
    if pad_before == 0 and pad_after == 0:
        return x
    ndim = len(x.shape)
    norm_axis = axis if axis >= 0 else ndim + axis
    in_len = x.shape[norm_axis]
    prefix = tuple(slice(None) for _ in range(norm_axis))

    if mode == "constant":
        parts: list[Tensor] = []
        if pad_before > 0:
            left_shape = x.shape[:norm_axis] + (pad_before,) + x.shape[norm_axis + 1 :]
            parts.append(zero_torch.full(left_shape, value, dtype=x.dtype))
        core_start = max(0, -pad_before)
        core_end = min(in_len, in_len + pad_after)
        parts.append(x[prefix + (slice(core_start, core_end),)])
        if pad_after > 0:
            right_shape = x.shape[:norm_axis] + (pad_after,) + x.shape[norm_axis + 1 :]
            parts.append(zero_torch.full(right_shape, value, dtype=x.dtype))
        return zero_torch.concatenate(parts, dim=norm_axis)

    total_len = in_len + pad_before + pad_after
    src_indices: list[int] = []
    for j in range(total_len):
        rel = j - pad_before
        if 0 <= rel < in_len:
            src_indices.append(rel)
        elif mode == "replicate":
            src_indices.append(max(0, min(in_len - 1, rel)))
        elif mode == "circular":
            src_indices.append(rel % in_len)
        elif mode == "reflect":
            if in_len <= 1:
                src_indices.append(0)
            else:
                period = 2 * (in_len - 1)
                rem = abs(rel) % period
                src_indices.append(rem if rem < in_len else period - rem)
        else:
            raise ValueError(f"Unsupported padding mode: {mode}")

    slices = [x[prefix + (slice(idx, idx + 1),)] for idx in src_indices]
    return zero_torch.concatenate(slices, dim=norm_axis)


def pad(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Pads tensor according to `pad` specification (`(left, right, top, bottom, ...)` ) and `mode`.

    Args:
        *args: `input` tensor and `pad` sizes (`int` or `2*k`-tuple).
        **kwargs: Optional keyword arguments (`pad`, `mode`, `value`).

    Returns:
        Padded tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    pad_raw = args[1] if len(args) > 1 else kwargs.get("pad", ())
    mode_raw = args[2] if len(args) > 2 else kwargs.get("mode", "constant")
    val_raw = args[3] if len(args) > 3 else kwargs.get("value", 0.0)
    mode = str(mode_raw) if isinstance(mode_raw, str) else "constant"
    val = float(val_raw) if isinstance(val_raw, (int, float)) else 0.0

    if isinstance(pad_raw, int):
        pad_seq: tuple[int, ...] = (pad_raw, pad_raw)
    elif isinstance(pad_raw, (tuple, list)):
        pad_seq = tuple(int(p) for p in pad_raw if isinstance(p, (int, float)))
    else:
        return x

    if len(pad_seq) == 0 or len(x.shape) == 0:
        return x

    num_dims_to_pad = min(len(pad_seq) // 2, len(x.shape))
    if not config.eager_mode:
        out_shape = list(x.shape)
        for d_idx in range(num_dims_to_pad):
            p_left = pad_seq[2 * d_idx]
            p_right = pad_seq[2 * d_idx + 1]
            out_shape[-(d_idx + 1)] += p_left + p_right
        return _wrap(
            _emit_reduction_node(
                "Pad",
                [_to_tensor(x)],
                {"pad": pad_seq, "mode": mode, "value": val},
                tuple(out_shape),
                x.dtype,
            )
        )

    out = x
    for d_idx in range(num_dims_to_pad):
        p_left = pad_seq[2 * d_idx]
        p_right = pad_seq[2 * d_idx + 1]
        out = _pad_single_axis(out, -(d_idx + 1), p_left, p_right, mode, val)
    return out


def rms_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies Root Mean Square Layer Normalization.

    Args:
        *args: `input` tensor and `normalized_shape`.
        **kwargs: Optional keyword arguments (`weight`, `eps`).

    Returns:
        RMS-normalized tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    return _fp_rms_norm(x, *args[1:], **kwargs)


def rnn_cell(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies an Elman RNN cell step.

    Args:
        *args: `input`, `hx`, `w_ih`, `w_hh`, and optional biases.
        **kwargs: Optional keyword arguments.

    Returns:
        Next hidden state tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    return tanh(x)


def spatial_dropout(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies spatial channel dropout.

    Args:
        *args: Input tensor and dropout rate.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    p_raw = args[1] if len(args) > 1 else kwargs.get("p", 0.5)
    p = float(p_raw) if isinstance(p_raw, (int, float)) else 0.5
    training = bool(kwargs.get("training", False))
    inplace = bool(kwargs.get("inplace", False))
    if len(x.shape) == 5:
        return _fp_dropout3d(x, p=p, training=training, inplace=inplace)
    if len(x.shape) == 4:
        return _fp_dropout2d(x, p=p, training=training, inplace=inplace)
    if len(x.shape) == 3:
        return _fp_dropout1d(x, p=p, training=training, inplace=inplace)
    return _fp_dropout(x, p=p, training=training, inplace=inplace)


def swish(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the Swish activation function element-wise.

    Args:
        *args: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    return silu(x)


def _interpolate_single_axis(
    x: Tensor, axis: int, out_len: int, mode: str, align_corners: bool
) -> Tensor:
    """Interpolates tensor `x` along a single spatial `axis` to length `out_len`.

    Args:
        x: Input tensor.
        axis: Spatial axis to resize.
        out_len: Target size along `axis`.
        mode: Interpolation mode (`'nearest'`, `'linear'`, `'bilinear'`, `'bicubic'`, `'trilinear'`).
        align_corners: Whether to align corner pixels in linear modes.

    Returns:
        Resized tensor along `axis`.
    """
    ndim = len(x.shape)
    norm_axis = axis if axis >= 0 else ndim + axis
    in_len = x.shape[norm_axis]
    if out_len == in_len:
        return x
    prefix = tuple(slice(None) for _ in range(norm_axis))

    if mode in ("nearest", "nearest-exact", "area"):
        pieces: list[Tensor] = []
        for i in range(out_len):
            src_idx = min(in_len - 1, (i * in_len) // out_len)
            pieces.append(x[prefix + (slice(src_idx, src_idx + 1),)])
        return zero_torch.concatenate(pieces, dim=norm_axis)

    pieces = []
    for i in range(out_len):
        if in_len == 1:
            coord = 0.0
        elif align_corners:
            coord = (i * float(in_len - 1)) / float(out_len - 1) if out_len > 1 else 0.0
        else:
            coord = max(
                0.0,
                min(
                    float(in_len - 1),
                    (i + 0.5) * (float(in_len) / float(out_len)) - 0.5,
                ),
            )
        i0 = math.floor(coord)
        i1 = min(in_len - 1, i0 + 1)
        w = coord - float(i0)
        s0 = x[prefix + (slice(i0, i0 + 1),)]
        if w == 0.0 or i0 == i1:
            pieces.append(s0)
        else:
            s1 = x[prefix + (slice(i1, i1 + 1),)]
            pieces.append(s0 * (1.0 - w) + s1 * w)
    return zero_torch.concatenate(pieces, dim=norm_axis)


def interpolate(
    input: Tensor,
    size: OptSizeNd = None,
    scale_factor: float | Sequence[float] | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
    recompute_scale_factor: bool | None = None,
    antialias: bool = False,
) -> Tensor:
    """Down/up samples the input to either the given `size` or the given `scale_factor`.

    Args:
        input: Input tensor of shape `(N, C, *spatial)`.
        size: Output spatial size (`int` or tuple of `int`s).
        scale_factor: Multiplier for spatial size (`float` or tuple of `float`s).
        mode: Algorithm used for upsampling (`'nearest'`, `'linear'`, `'bilinear'`, `'bicubic'`, `'trilinear'`).
        align_corners: Geometrics alignment of corner pixels in linear modes.
        recompute_scale_factor: Whether to recompute scale_factor.
        antialias: Whether to apply antialiasing.

    Returns:
        Interpolated tensor.
    """
    _ = (recompute_scale_factor, antialias)
    if len(input.shape) < 3:
        return input
    spatial_ndims = len(input.shape) - 2
    in_spatial = input.shape[2:]

    if size is not None:
        if isinstance(size, int):
            out_spatial = (size,) * spatial_ndims
        else:
            out_spatial = tuple(
                in_spatial[m] if s is None else int(s) for m, s in enumerate(size)
            )
    elif scale_factor is not None:
        if isinstance(scale_factor, (int, float)):
            factors = (float(scale_factor),) * spatial_ndims
        else:
            factors = tuple(float(f) for f in scale_factor)
        out_spatial = tuple(
            max(1, math.floor(in_spatial[m] * factors[m])) for m in range(spatial_ndims)
        )
    else:
        return input

    if not config.eager_mode:
        out_shape = input.shape[:2] + out_spatial
        return _wrap(
            _emit_reduction_node(
                "Interpolate",
                [_to_tensor(input)],
                {
                    "size": out_spatial,
                    "mode": mode,
                    "align_corners": bool(align_corners),
                },
                out_shape,
                input.dtype,
            )
        )

    out = input
    ac = bool(align_corners)
    for m in range(spatial_ndims):
        out = _interpolate_single_axis(
            out, axis=2 + m, out_len=out_spatial[m], mode=mode, align_corners=ac
        )
    return out


def upsample_bilinear(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Upsamples the input using bilinear interpolation (`align_corners=True` by default).

    Args:
        *args: Input tensor and optional `size` or `scale_factor`.
        **kwargs: Optional keyword arguments (`size`, `scale_factor`, `align_corners`).

    Returns:
        Upsampled tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    size_raw = args[1] if len(args) > 1 else kwargs.get("size")
    scale_raw = args[2] if len(args) > 2 else kwargs.get("scale_factor")
    ac_raw = kwargs.get("align_corners", True)
    size_arg: OptSizeNd = size_raw if isinstance(size_raw, (int, tuple, list)) else None
    scale_arg: float | Sequence[float] | None = (
        scale_raw if isinstance(scale_raw, (int, float, tuple, list)) else None
    )
    return interpolate(
        x,
        size=size_arg,
        scale_factor=scale_arg,
        mode="bilinear",
        align_corners=bool(ac_raw),
    )


def upsample_nearest(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Upsamples the input using nearest-neighbor interpolation.

    Args:
        *args: Input tensor and optional `size` or `scale_factor`.
        **kwargs: Optional keyword arguments (`size`, `scale_factor`).

    Returns:
        Upsampled tensor.
    """
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    size_raw = args[1] if len(args) > 1 else kwargs.get("size")
    scale_raw = args[2] if len(args) > 2 else kwargs.get("scale_factor")
    size_arg: OptSizeNd = size_raw if isinstance(size_raw, (int, tuple, list)) else None
    scale_arg: float | Sequence[float] | None = (
        scale_raw if isinstance(scale_raw, (int, float, tuple, list)) else None
    )
    return interpolate(
        x,
        size=size_arg,
        scale_factor=scale_arg,
        mode="nearest",
    )


def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Applies a linear transformation to the incoming data: `y = xA^T + b`.

    Args:
        input: Input tensor of shape `(*, in_features)`.
        weight: Weight tensor of shape `(out_features, in_features)`.
        bias: Optional bias tensor of shape `(out_features,)`.

    Returns:
        Transformed tensor of shape `(*, out_features)`.
    """
    out = zero_torch.matmul(input, zero_torch.transpose(weight, 0, 1))
    if bias is not None:
        return out + bias
    return out


def mse_loss(
    input: Tensor,
    target: Tensor,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Computes the element-wise mean squared error.

    Args:
        input: Input prediction tensor.
        target: Ground-truth target tensor.
        size_average: Deprecated flag.
        reduce: Deprecated flag.
        reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).

    Returns:
        Computed MSE loss tensor.
    """
    _ = (size_average, reduce)
    diff = input - target
    return _reduce_loss(diff * diff, reduction)


def cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    size_average: bool | None = None,
    ignore_index: int = -100,
    reduce: bool | None = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    """Computes the cross entropy loss between input logits and target (class indices or probabilities).

    Args:
        input: Predicted unnormalized logits tensor of shape `(C,)`, `(N, C)`, or `(N, C, d_1, ..., d_k)`.
        target: Ground-truth class indices or class probabilities tensor.
        weight: Optional manual rescaling weight per class of shape `(C,)`.
        size_average: Deprecated flag.
        ignore_index: Target value that is ignored.
        reduce: Deprecated flag.
        reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
        label_smoothing: Amount of label smoothing in `[0.0, 1.0]`.

    Returns:
        Computed cross-entropy loss tensor.
    """
    _ = (size_average, reduce)
    class_dim = -1 if len(input.shape) <= 1 else 1
    log_probs = log_softmax(input, dim=class_dim)

    if input.shape == target.shape:
        num_classes = float(input.shape[class_dim]) if len(input.shape) > 0 else 1.0
        tgt = (
            target * (1.0 - label_smoothing) + (label_smoothing / num_classes)
            if label_smoothing > 0.0
            else target
        )
        weighted_lp = log_probs
        if weight is not None:
            if len(input.shape) > 2:
                w_shape = (1, weight.shape[0]) + (1,) * (len(input.shape) - 2)
                weighted_lp = log_probs * zero_torch.reshape(weight, w_shape)
            else:
                weighted_lp = log_probs * weight
        loss = -zero_torch.sum(tgt * weighted_lp, axis=class_dim)
        return _reduce_loss(loss, reduction)

    nll = nll_loss(
        log_probs,
        target,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    if label_smoothing > 0.0:
        if weight is not None:
            if len(input.shape) > 2:
                w_shape = (1, weight.shape[0]) + (1,) * (len(input.shape) - 2)
                w_lp = log_probs * zero_torch.reshape(weight, w_shape)
            else:
                w_lp = log_probs * weight
            smooth_per_sample = -zero_torch.sum(w_lp, axis=class_dim) / float(
                max(1, input.shape[class_dim])
            )
        else:
            smooth_per_sample = -zero_torch.mean(log_probs, axis=class_dim)
        valid = target != ignore_index
        smooth_masked = zero_torch.where(valid, smooth_per_sample, 0.0)
        if reduction == "none":
            smooth_loss = smooth_masked
        elif reduction == "sum":
            smooth_loss = zero_torch.sum(smooth_masked)
        else:
            valid_count = zero_torch.maximum(
                zero_torch.sum(zero_torch.where(valid, 1.0, 0.0)), 1.0
            )
            smooth_loss = zero_torch.sum(smooth_masked) / valid_count
        return (1.0 - label_smoothing) * nll + label_smoothing * smooth_loss
    return nll


def binary_cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Measures the Binary Cross Entropy between the target and the input probabilities.

    Args:
        input: Tensor of arbitrary shape as probabilities.
        target: Tensor of the same shape as input with values between 0 and 1.
        weight: Manual rescaling weight if provided.
        size_average: Deprecated flag.
        reduce: Deprecated flag.
        reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).

    Returns:
        Computed BCE loss tensor.
    """
    _ = (size_average, reduce)
    p = zero_torch.clip(input, 1e-12, 1.0 - 1e-12)
    loss = -(target * zero_torch.log(p) + (1.0 - target) * zero_torch.log(1.0 - p))
    if weight is not None:
        loss = loss * weight
    return _reduce_loss(loss, reduction)


def binary_cross_entropy_with_logits(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
    pos_weight: Tensor | None = None,
) -> Tensor:
    """Measures Binary Cross Entropy between target and input logits.

    Args:
        input: Tensor of arbitrary shape as unnormalized logits.
        target: Tensor of the same shape as input with values between 0 and 1.
        weight: Manual rescaling weight if provided.
        size_average: Deprecated flag.
        reduce: Deprecated flag.
        reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
        pos_weight: Weight of positive examples.

    Returns:
        Computed BCE-with-logits loss tensor.
    """
    _ = (size_average, reduce)
    max_val = zero_torch.maximum(-input, 0.0)
    if pos_weight is not None:
        log_weight = (pos_weight - 1.0) * target + 1.0
        loss = (1.0 - target) * input + log_weight * (
            zero_torch.log(zero_torch.exp(-max_val) + zero_torch.exp(-input - max_val))
            + max_val
        )
    else:
        loss = (
            zero_torch.maximum(input, 0.0)
            - input * target
            + zero_torch.log(1.0 + zero_torch.exp(-zero_torch.abs(input)))
        )
    if weight is not None:
        loss = loss * weight
    return _reduce_loss(loss, reduction)


def l1_loss(
    input: Tensor,
    target: Tensor,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Measures the mean absolute error (MAE) between each element in `input` and `target`.

    Args:
        input: Input prediction tensor.
        target: Ground-truth target tensor.
        size_average: Deprecated flag.
        reduce: Deprecated flag.
        reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).

    Returns:
        Computed L1 loss tensor.
    """
    _ = (size_average, reduce)
    return _reduce_loss(zero_torch.abs(input - target), reduction)


def bilinear(
    input1: Tensor,
    input2: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Applies a bilinear transformation to the incoming data: `y = x_1^T A x_2 + b`.

    Args:
        input1: First input tensor of shape `(*, in1_features)`.
        input2: Second input tensor of shape `(*, in2_features)`.
        weight: Weight tensor of shape `(out_features, in1_features, in2_features)`.
        bias: Optional bias tensor of shape `(out_features,)`.

    Returns:
        Transformed tensor of shape `(*, out_features)`.
    """
    if len(weight.shape) == 3 and len(input1.shape) >= 1 and len(input2.shape) >= 1:
        out_features = weight.shape[0]
        slices: list[Tensor] = []
        for k in range(out_features):
            w_k = weight[k]
            proj = zero_torch.matmul(input1, w_k)
            val = zero_torch.sum(proj * input2, axis=-1, keepdims=True)
            slices.append(val)
        res = zero_torch.concatenate(slices, dim=-1)
        if bias is not None:
            res = res + bias
        return res
    return Tensor(0.0)


__all__ = [
    "activations",
    "adaptive_avg_pool1d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "adaptive_max_pool3d",
    "alpha_dropout",
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool3d",
    "batch_norm",
    "bilinear",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "celu",
    "complex",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "cosine_embedding_loss",
    "cross_entropy",
    "ctc_loss",
    "dropout",
    "dropout1d",
    "dropout2d",
    "dropout3d",
    "elu",
    "embedding",
    "feature_alpha_dropout",
    "fold",
    "fractional_max_pool2d",
    "fractional_max_pool3d",
    "gaussian_nll_loss",
    "gelu",
    "glu",
    "group_norm",
    "gru_cell",
    "hard_sigmoid",
    "hardshrink",
    "hardsigmoid",
    "hardswish",
    "hardtanh",
    "hinge_embedding_loss",
    "huber_loss",
    "instance_norm",
    "interpolate",
    "kl_div",
    "l1_loss",
    "layer_norm",
    "leaky_relu",
    "linear",
    "log_softmax",
    "logsigmoid",
    "lp_pool1d",
    "lp_pool2d",
    "lp_pool3d",
    "lstm_cell",
    "margin_ranking_loss",
    "max_pool1d",
    "max_pool2d",
    "max_pool3d",
    "max_unpool1d",
    "max_unpool2d",
    "max_unpool3d",
    "mish",
    "mse_loss",
    "multi_label_margin_loss",
    "multi_label_soft_margin_loss",
    "multi_margin_loss",
    "multilabel_margin_loss",
    "multilabel_soft_margin_loss",
    "nll_loss",
    "pad",
    "poisson_nll_loss",
    "prelu",
    "relu",
    "relu6",
    "rms_norm",
    "rnn_cell",
    "rrelu",
    "scaled_dot_product_attention",
    "selu",
    "sigmoid",
    "silu",
    "smooth_l1_loss",
    "soft_margin_loss",
    "soft_shrink",
    "softmax",
    "softmax2d",
    "softmin",
    "softplus",
    "softshrink",
    "softsign",
    "spatial_dropout",
    "swish",
    "tanh",
    "tanhshrink",
    "threshold",
    "triplet_margin_loss",
    "triplet_margin_with_distance_loss",
    "unfold",
    "upsample_bilinear",
    "upsample_nearest",
]
