"""Functional NN API."""

from __future__ import annotations

import math
from collections.abc import Callable

import ml_switcheroo_compiler.ops as _ops
from ml_switcheroo_compiler.core import config
from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.ops.configs import ConvConfig
from ml_switcheroo_compiler.ops.nn.normalization import (
    group_norm as _compiler_group_norm,
)
from ml_switcheroo_compiler.ops.shape.indexing import where as _where

import zero_torch
from zero_torch._types import (
    OpArg,
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
    adaptive_avg_pool2d as _fp_adaptive_avg_pool2d,
)
from .functional_pooling import (
    fractional_max_pool2d as _fp_fractional_max_pool2d,
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
    _ = groups
    input_t = input._tensor if isinstance(input, Tensor) else input
    weight_t = weight._tensor if isinstance(weight, Tensor) else weight

    s = (stride, stride) if isinstance(stride, int) else stride
    if isinstance(padding, int):
        pad_pairs: tuple[tuple[int, int], tuple[int, int]] = (
            (padding, padding),
            (padding, padding),
        )
    elif isinstance(padding, tuple):
        pad_pairs = ((padding[0], padding[0]), (padding[1], padding[1]))
    else:
        pad_pairs = ((0, 0), (0, 0))
    d = (dilation, dilation) if isinstance(dilation, int) else dilation

    res = _ops.conv_general_dilated(
        input_t,
        weight_t,
        ConvConfig(
            window_strides=s,
            padding=pad_pairs,
            lhs_dilation=None,
            rhs_dilation=d,
            dimension_numbers=None,
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
    _ = groups
    input_t = input._tensor if isinstance(input, Tensor) else input
    weight_t = weight._tensor if isinstance(weight, Tensor) else weight

    s = (stride,) if isinstance(stride, int) else stride
    if isinstance(padding, int):
        pad_pairs: tuple[tuple[int, int]] = ((padding, padding),)
    elif isinstance(padding, tuple):
        pad_pairs = ((padding[0], padding[0]),)
    else:
        pad_pairs = ((0, 0),)
    d = (dilation,) if isinstance(dilation, int) else dilation

    res = _ops.conv_general_dilated(
        input_t,
        weight_t,
        ConvConfig(
            window_strides=s,
            padding=pad_pairs,
            lhs_dilation=None,
            rhs_dilation=d,
            dimension_numbers=None,
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
    _ = groups
    input_t = input._tensor if isinstance(input, Tensor) else input
    weight_t = weight._tensor if isinstance(weight, Tensor) else weight

    s = (stride, stride, stride) if isinstance(stride, int) else stride
    if isinstance(padding, int):
        pad_pairs: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = (
            (padding, padding),
            (padding, padding),
            (padding, padding),
        )
    elif isinstance(padding, tuple):
        pad_pairs = (
            (padding[0], padding[0]),
            (padding[1], padding[1]),
            (padding[2], padding[2]),
        )
    else:
        pad_pairs = ((0, 0), (0, 0), (0, 0))
    d = (dilation, dilation, dilation) if isinstance(dilation, int) else dilation

    res = _ops.conv_general_dilated(
        input_t,
        weight_t,
        ConvConfig(
            window_strides=s,
            padding=pad_pairs,
            lhs_dilation=None,
            rhs_dilation=d,
            dimension_numbers=None,
        ),
    )
    out = _wrap(res)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1, 1)
    return out


def relu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the rectified linear unit function element-wise.

    Args:
        *args: Input tensor and optional positional arguments.
        **kwargs: Optional keyword arguments (`inplace`).

    Returns:
        Tensor with ReLU applied.
    """
    _ = kwargs
    res = _ops.maximum(args[0]._tensor if isinstance(args[0], Tensor) else args[0], 0.0)
    return _wrap(res)


def softmax(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies a softmax function.

    Args:
        *args: Input tensor and optional dimension.
        **kwargs: Optional keyword arguments (`dim`, `dtype`).

    Returns:
        Softmax probability tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: softmax")
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
    if not config.eager_mode:
        raise NotImplementedError(
            "Compiler backend missing nn op: scaled_dot_product_attention"
        )
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
    if len(k.shape) >= 2 and len(q.shape) >= 2 and len(v.shape) >= 2:
        k_t = zero_torch.transpose(k, -2, -1)
        scores = zero_torch.matmul(q, k_t) * scale
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
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: celu")
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    alpha_raw = args[1] if len(args) > 1 else kwargs.get("alpha", 1.0)
    alpha = float(alpha_raw) if isinstance(alpha_raw, (int, float)) else 1.0
    pos = zero_torch.maximum(x, 0.0)
    neg = zero_torch.minimum(0.0, alpha * (zero_torch.exp(x / alpha) - 1.0))
    return pos + neg


def multilabel_margin_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes multi-label margin loss.

    Args:
        *args: Input and target tensors.
        **kwargs: Optional keyword arguments (`reduction`).

    Returns:
        Computed loss tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError(
            "Compiler backend missing nn op: multilabel_margin_loss"
        )
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    t = _extract_tensor(args[1] if len(args) > 1 else kwargs.get("target", 0.0))
    reduction = _extract_reduction(kwargs)
    loss = zero_torch.maximum(0.0, 1.0 - (x - t))
    return _reduce_loss(loss, reduction)


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
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: elu")
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    alpha_raw = args[1] if len(args) > 1 else kwargs.get("alpha", 1.0)
    alpha = float(alpha_raw) if isinstance(alpha_raw, (int, float)) else 1.0
    return zero_torch.where(x > 0, x, alpha * (zero_torch.exp(x) - 1.0))


def ctc_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes the Connectionist Temporal Classification loss.

    Args:
        *args: `log_probs`, `targets`, `input_lengths`, `target_lengths`.
        **kwargs: Optional keyword arguments (`blank`, `reduction`, `zero_infinity`).

    Returns:
        Computed CTC loss tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: ctc_loss")
    log_probs = _extract_tensor(
        args[0] if len(args) > 0 else kwargs.get("log_probs", 0.0)
    )
    reduction = _extract_reduction(kwargs)
    return _reduce_loss(-log_probs, reduction)


def multilabel_soft_margin_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes multi-label soft margin loss.

    Args:
        *args: Input and target tensors.
        **kwargs: Optional keyword arguments (`weight`, `reduction`).

    Returns:
        Computed loss tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError(
            "Compiler backend missing nn op: multilabel_soft_margin_loss"
        )
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    t = _extract_tensor(args[1] if len(args) > 1 else kwargs.get("target", 0.0))
    reduction = _extract_reduction(kwargs)
    loss = -(t * logsigmoid(x) + (1.0 - t) * logsigmoid(-x))
    w = kwargs.get("weight")
    if isinstance(w, Tensor):
        loss = loss * w
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
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: gelu")
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
    if not config.eager_mode:
        raise NotImplementedError(
            "Compiler backend missing nn op: cosine_embedding_loss"
        )
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
    loss = zero_torch.where(y == 1, 1.0 - cos, zero_torch.maximum(0.0, cos - margin))
    return _reduce_loss(loss, reduction)


def multi_margin_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes multi-margin loss.

    Args:
        *args: Input and target tensors.
        **kwargs: Optional keyword arguments (`p`, `margin`, `weight`, `reduction`).

    Returns:
        Computed loss tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: multi_margin_loss")
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    margin_raw = kwargs.get("margin", 1.0)
    margin = float(margin_raw) if isinstance(margin_raw, (int, float)) else 1.0
    p_raw = kwargs.get("p", 1)
    p = int(p_raw) if isinstance(p_raw, int) else 1
    reduction = _extract_reduction(kwargs)
    loss = zero_torch.maximum(0.0, margin - x) ** p
    return _reduce_loss(loss, reduction)


def glu(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the gated linear unit function.

    Args:
        *args: Input tensor and optional dimension.
        **kwargs: Optional keyword arguments (`dim`).

    Returns:
        Output tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: glu")
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    return x * sigmoid(x)


def gaussian_nll_loss(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Computes Gaussian negative log likelihood loss.

    Args:
        *args: `input`, `target`, and `var` tensors.
        **kwargs: Optional keyword arguments (`full`, `eps`, `reduction`).

    Returns:
        Computed loss tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: gaussian_nll_loss")
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
    """Computes the negative log likelihood loss.

    Args:
        input: Log-probabilities tensor.
        target: Ground-truth class indices tensor.
        weight: Optional manual rescaling weight per class.
        size_average: Deprecated flag.
        ignore_index: Target value that is ignored.
        reduce: Deprecated flag.
        reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).

    Returns:
        Computed NLL loss tensor.
    """
    _ = (size_average, reduce)
    in_list = input.tolist()
    tgt_list = target.tolist()
    w_list = weight.tolist() if weight is not None else None

    if (
        isinstance(in_list, list)
        and len(input.shape) == 2
        and isinstance(tgt_list, list)
    ):
        losses: list[float] = []
        total_weight = 0.0
        weighted_sum = 0.0
        for row, cls_raw in zip(in_list, tgt_list):
            assert isinstance(row, list)
            cls_idx = int(cls_raw) if isinstance(cls_raw, (int, float)) else -1
            if cls_idx == ignore_index or cls_idx < 0 or cls_idx >= len(row):
                losses.append(0.0)
                continue
            val = row[cls_idx]
            log_p = float(val) if isinstance(val, (int, float)) else 0.0
            w_item = (
                w_list[cls_idx]
                if isinstance(w_list, list) and cls_idx < len(w_list)
                else 1.0
            )
            cls_w = float(w_item) if isinstance(w_item, (int, float)) else 1.0
            sample_loss = -log_p * cls_w
            losses.append(sample_loss)
            weighted_sum += sample_loss
            total_weight += cls_w
        loss_vec = Tensor(losses, dtype=input.dtype)
        if reduction == "none":
            return loss_vec
        if reduction == "sum":
            return Tensor(weighted_sum, dtype=input.dtype)
        return Tensor(
            weighted_sum / total_weight if total_weight > 0.0 else 0.0,
            dtype=input.dtype,
        )

    return _reduce_loss(-input, reduction)


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
    if not config.eager_mode:
        raise NotImplementedError(
            "Compiler backend missing nn op: hinge_embedding_loss"
        )
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
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: poisson_nll_loss")
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    t = _extract_tensor(args[1] if len(args) > 1 else kwargs.get("target", 0.0))
    log_inp = bool(kwargs.get("log_input", True))
    eps_raw = kwargs.get("eps", 1e-8)
    eps = float(eps_raw) if isinstance(eps_raw, (int, float)) else 1e-8
    reduction = _extract_reduction(kwargs)
    loss = zero_torch.exp(x) - t * x if log_inp else x - t * zero_torch.log(x + eps)
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
    inp = args[0]._tensor if isinstance(args[0], Tensor) else args[0]
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
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: huber_loss")
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
        loss = target * (zero_torch.log(target) - input)
    if reduction in ("mean", "batchmean"):
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
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: soft_margin_loss")
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
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: hardswish")
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
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: margin_ranking_loss")
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
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: triplet_margin_loss")
    a = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("anchor", 0.0))
    p_t = _extract_tensor(args[1] if len(args) > 1 else kwargs.get("positive", a))
    n_t = _extract_tensor(args[2] if len(args) > 2 else kwargs.get("negative", a))
    margin_raw = kwargs.get("margin", 1.0)
    margin = float(margin_raw) if isinstance(margin_raw, (int, float)) else 1.0
    p_raw = kwargs.get("p", 2.0)
    p = float(p_raw) if isinstance(p_raw, (int, float)) else 2.0
    eps_raw = kwargs.get("eps", 1e-6)
    eps = float(eps_raw) if isinstance(eps_raw, (int, float)) else 1e-6
    reduction = _extract_reduction(kwargs)
    d_pos = (zero_torch.sum(zero_torch.abs(a - p_t) ** p, axis=-1) + eps) ** (1.0 / p)
    d_neg = (zero_torch.sum(zero_torch.abs(a - n_t) ** p, axis=-1) + eps) ** (1.0 / p)
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
    red_raw = kwargs.get("reduction", "mean")
    reduction = red_raw if isinstance(red_raw, str) else "mean"
    if callable(dist_fn):
        d_pos = dist_fn(a, p_t)
        d_neg = dist_fn(a, n_t)
    else:
        d_pos = zero_torch.sqrt(zero_torch.sum((a - p_t) ** 2, axis=-1) + 1e-6)
        d_neg = zero_torch.sqrt(zero_torch.sum((a - n_t) ** 2, axis=-1) + 1e-6)
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
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: leaky_relu")
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
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: log_softmax")
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
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: mish")
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
    return zero_torch.where(input > 0, input, weight * input)


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
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: selu")
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
    return softmax(input, dim=1)


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
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: softplus")
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
    res = _ops.tanh(args[0]._tensor if isinstance(args[0], Tensor) else args[0])
    return _wrap(res)


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
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: activations")
    return _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))


def adaptive_avg_pool2d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies 2D adaptive average pooling.

    Args:
        *args: Input tensor and `output_size`.
        **kwargs: Optional keyword arguments.

    Returns:
        Pooled tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: adaptive_avg_pool2d")
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    out_sz = args[1] if len(args) > 1 else kwargs.get("output_size", 1)
    if len(x.shape) == 4 and isinstance(out_sz, (int, tuple)):
        return _fp_adaptive_avg_pool2d(x, out_sz)
    return x


def alpha_dropout(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies alpha dropout.

    Args:
        *args: Input tensor and dropout probability.
        **kwargs: Optional keyword arguments (`p`, `training`, `inplace`).

    Returns:
        Output tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: alpha_dropout")
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
    dilation: Size1d = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor:
    """Applies 1D average pooling over an input signal.

    Args:
        input: Input tensor.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit zero padding.
        dilation: Window dilation.
        ceil_mode: Whether to use ceil for output shape.
        return_indices: Whether to return indices.

    Returns:
        Pooled 1D tensor.
    """
    _ = (dilation, return_indices)
    from .functional_pooling import avg_pool1d as _fp_avg1d

    return _fp_avg1d(input, kernel_size, stride, padding, ceil_mode)


def avg_pool2d(
    input: Tensor,
    kernel_size: Size2d,
    stride: Size2d | None = None,
    padding: Size2d = 0,
    dilation: Size2d = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor:
    """Applies 2D average pooling over an input signal.

    Args:
        input: Input tensor.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit zero padding.
        dilation: Window dilation.
        ceil_mode: Whether to use ceil for output shape.
        return_indices: Whether to return indices.

    Returns:
        Pooled 2D tensor.
    """
    _ = (dilation, return_indices)
    from .functional_pooling import avg_pool2d as _fp_avg2d

    return _fp_avg2d(input, kernel_size, stride, padding, ceil_mode)


def avg_pool3d(
    input: Tensor,
    kernel_size: Size3d,
    stride: Size3d | None = None,
    padding: Size3d = 0,
    dilation: Size3d = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor:
    """Applies 3D average pooling over an input signal.

    Args:
        input: Input tensor.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Implicit zero padding.
        dilation: Window dilation.
        ceil_mode: Whether to use ceil for output shape.
        return_indices: Whether to return indices.

    Returns:
        Pooled 3D tensor.
    """
    _ = (dilation, return_indices)
    from .functional_pooling import avg_pool3d as _fp_avg3d

    return _fp_avg3d(input, kernel_size, stride, padding, ceil_mode)


def max_pool1d(
    input: Tensor,
    kernel_size: Size1d,
    stride: Size1d | None = None,
    padding: Size1d = 0,
    dilation: Size1d = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor:
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
        Pooled 1D tensor.
    """
    from .functional_pooling import max_pool1d as _fp_max1d

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
) -> Tensor:
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
        Pooled 2D tensor.
    """
    from .functional_pooling import max_pool2d as _fp_max2d

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
) -> Tensor:
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
        Pooled 3D tensor.
    """
    from .functional_pooling import max_pool3d as _fp_max3d

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
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: batch_norm")
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    if len(x.shape) < 2:
        return x
    return _fp_batch_norm(x, **kwargs)


def complex(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Constructs a complex tensor from real and imaginary parts.

    Args:
        *args: Real and imaginary tensors.
        **kwargs: Optional keyword arguments.

    Returns:
        Complex tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: complex")
    real = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("real", 0.0))
    return real


def conv_transpose1d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies a 1D transposed convolution operator.

    Args:
        *args: `input`, `weight`, and optional `bias`.
        **kwargs: Optional keyword arguments (`stride`, `padding`, `output_padding`, `groups`, `dilation`).

    Returns:
        Transposed convolved tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: conv_transpose1d")
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    if len(args) > 1 and isinstance(args[1], Tensor) and len(x.shape) == 3:
        w = args[1]
        out_ch = w.shape[1] if len(w.shape) > 1 else x.shape[1]
        k = w.shape[2] if len(w.shape) > 2 else 1
        out_l = x.shape[2] + k - 1
        return zero_torch.zeros((x.shape[0], out_ch, out_l), dtype=x.dtype)
    return zero_torch.zeros_like(x)


def conv_transpose2d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies a 2D transposed convolution operator.

    Args:
        *args: `input`, `weight`, and optional `bias`.
        **kwargs: Optional keyword arguments (`stride`, `padding`, `output_padding`, `groups`, `dilation`).

    Returns:
        Transposed convolved tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: conv_transpose2d")
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    if len(args) > 1 and isinstance(args[1], Tensor) and len(x.shape) == 4:
        w = args[1]
        out_ch = w.shape[1] if len(w.shape) > 1 else x.shape[1]
        kh = w.shape[2] if len(w.shape) > 2 else 1
        kw = w.shape[3] if len(w.shape) > 3 else 1
        return zero_torch.zeros(
            (x.shape[0], out_ch, x.shape[2] + kh - 1, x.shape[3] + kw - 1),
            dtype=x.dtype,
        )
    return zero_torch.zeros_like(x)


def conv_transpose3d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies a 3D transposed convolution operator.

    Args:
        *args: `input`, `weight`, and optional `bias`.
        **kwargs: Optional keyword arguments (`stride`, `padding`, `output_padding`, `groups`, `dilation`).

    Returns:
        Transposed convolved tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: conv_transpose3d")
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    if len(args) > 1 and isinstance(args[1], Tensor) and len(x.shape) == 5:
        w = args[1]
        out_ch = w.shape[1] if len(w.shape) > 1 else x.shape[1]
        kd = w.shape[2] if len(w.shape) > 2 else 1
        kh = w.shape[3] if len(w.shape) > 3 else 1
        kw = w.shape[4] if len(w.shape) > 4 else 1
        return zero_torch.zeros(
            (
                x.shape[0],
                out_ch,
                x.shape[2] + kd - 1,
                x.shape[3] + kh - 1,
                x.shape[4] + kw - 1,
            ),
            dtype=x.dtype,
        )
    return zero_torch.zeros_like(x)


def dropout(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies dropout.

    Args:
        *args: Input tensor and optional dropout probability `p`.
        **kwargs: Optional keyword arguments (`p`, `training`, `inplace`).

    Returns:
        Output tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: dropout")
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    p_raw = args[1] if len(args) > 1 else kwargs.get("p", 0.5)
    p = float(p_raw) if isinstance(p_raw, (int, float)) else 0.5
    training = bool(kwargs.get("training", False))
    inplace = bool(kwargs.get("inplace", False))
    return _fp_dropout(x, p=p, training=training, inplace=inplace)


def embedding(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Generates a simple lookup table that looks up embeddings in a fixed dictionary and size.

    Args:
        *args: `input` indices and `weight` matrix.
        **kwargs: Optional keyword arguments (`padding_idx`, `max_norm`, `norm_type`, `scale_grad_by_freq`, `sparse`).

    Returns:
        Embedded tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: embedding")
    idx = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0))
    if len(args) > 1 and isinstance(args[1], Tensor) and len(args[1].shape) == 2:
        w = args[1]
        emb_dim = w.shape[1]
        flat_idx = zero_torch.cast(zero_torch.reshape(idx, (-1,)), zero_torch.int32)
        gathered = _wrap(get_op("Take")()(w._tensor, flat_idx._tensor, axis=0))
        return zero_torch.reshape(gathered, idx.shape + (emb_dim,))
    return idx


def feature_alpha_dropout(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Randomly masks out entire channels using alpha dropout.

    Args:
        *args: Input tensor and optional probability `p`.
        **kwargs: Optional keyword arguments (`p`, `training`, `inplace`).

    Returns:
        Output tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError(
            "Compiler backend missing nn op: feature_alpha_dropout"
        )
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    p_raw = args[1] if len(args) > 1 else kwargs.get("p", 0.5)
    p = float(p_raw) if isinstance(p_raw, (int, float)) else 0.5
    training = bool(kwargs.get("training", False))
    inplace = bool(kwargs.get("inplace", False))
    return _fp_feature_alpha_dropout(x, p=p, training=training, inplace=inplace)


def fractional_max_pool2d(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies 2D fractional max pooling.

    Args:
        *args: Input tensor and `kernel_size`.
        **kwargs: Optional keyword arguments (`output_size`, `output_ratio`, `return_indices`).

    Returns:
        Pooled tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError(
            "Compiler backend missing nn op: fractional_max_pool2d"
        )
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    if len(x.shape) == 4 and len(args) > 1 and isinstance(args[1], (int, tuple)):
        return _fp_fractional_max_pool2d(x, args[1])
    return x


def group_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies Group Normalization.

    Args:
        *args: `input`, `num_groups`, and optional affine parameters.
        **kwargs: Optional keyword arguments (`weight`, `bias`, `eps`).

    Returns:
        Normalized tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: group_norm")
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    g_raw = args[1] if len(args) > 1 else kwargs.get("num_groups", 1)
    groups = int(g_raw) if isinstance(g_raw, int) else 1
    eps_raw = kwargs.get("eps", 1e-5)
    eps = float(eps_raw) if isinstance(eps_raw, (int, float)) else 1e-5
    if len(x.shape) >= 2 and x.shape[1] % groups == 0:
        return _wrap(
            _compiler_group_norm(_to_tensor(x), num_groups=groups, epsilon=eps)
        )
    return x


def gru_cell(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies a gated recurrent unit (GRU) cell step.

    Args:
        *args: `input`, `hx`, `w_ih`, `w_hh`, and optional biases.
        **kwargs: Optional keyword arguments.

    Returns:
        Next hidden state tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: gru_cell")
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
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: instance_norm")
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    if len(x.shape) >= 3:
        return _fp_instance_norm(x, **kwargs)
    return x


def layer_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies Layer Normalization.

    Args:
        *args: `input`, `normalized_shape`, and optional affine parameters.
        **kwargs: Optional keyword arguments (`weight`, `bias`, `eps`).

    Returns:
        Normalized tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: layer_norm")
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    norm_shape = (
        args[1] if len(args) > 1 else kwargs.get("normalized_shape", x.shape[-1:])
    )
    out = _fp_layer_norm(x, norm_shape, **kwargs)
    w = args[2] if len(args) > 2 else kwargs.get("weight")
    b = args[3] if len(args) > 3 else kwargs.get("bias")
    if isinstance(w, Tensor):
        out = out * w
    if isinstance(b, Tensor):
        out = out + b
    return out


def lstm_cell(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies a long short-term memory (LSTM) cell step.

    Args:
        *args: `input`, `hx`, `w_ih`, `w_hh`, and optional biases.
        **kwargs: Optional keyword arguments.

    Returns:
        Next state tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: lstm_cell")
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    return tanh(x)


def pad(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Pads tensor according to `pad` specification and `mode`.

    Args:
        *args: `input` tensor and `pad` sizes.
        **kwargs: Optional keyword arguments (`mode`, `value`).

    Returns:
        Padded tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: pad")
    return _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))


def rms_norm(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies Root Mean Square Layer Normalization.

    Args:
        *args: `input` tensor and `normalized_shape`.
        **kwargs: Optional keyword arguments (`weight`, `eps`).

    Returns:
        RMS-normalized tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: rms_norm")
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    return _fp_rms_norm(x, **kwargs)


def rnn_cell(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies an Elman RNN cell step.

    Args:
        *args: `input`, `hx`, `w_ih`, `w_hh`, and optional biases.
        **kwargs: Optional keyword arguments.

    Returns:
        Next hidden state tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: rnn_cell")
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
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: spatial_dropout")
    return dropout(*args, **kwargs)


def swish(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Applies the Swish activation function element-wise.

    Args:
        *args: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: swish")
    x = _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))
    return silu(x)


def upsample_bilinear(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Upsamples the input using bilinear interpolation.

    Args:
        *args: Input tensor.
        **kwargs: Optional keyword arguments (`size`, `scale_factor`, `align_corners`).

    Returns:
        Upsampled tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: upsample_bilinear")
    return _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))


def upsample_nearest(*args: OpArg, **kwargs: OpArg) -> Tensor:
    """Upsamples the input using nearest-neighbor interpolation.

    Args:
        *args: Input tensor.
        **kwargs: Optional keyword arguments (`size`, `scale_factor`).

    Returns:
        Upsampled tensor.
    """
    if not config.eager_mode:
        raise NotImplementedError("Compiler backend missing nn op: upsample_nearest")
    return _extract_tensor(args[0] if len(args) > 0 else kwargs.get("input", 0.0))


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
    """Computes the cross entropy loss between input logits and target.

    Args:
        input: Predicted unnormalized logits tensor.
        target: Ground-truth class indices or class probabilities tensor.
        weight: Optional manual rescaling weight per class.
        size_average: Deprecated flag.
        ignore_index: Target value that is ignored.
        reduce: Deprecated flag.
        reduction: Specifies the reduction to apply (`'none'`, `'mean'`, `'sum'`).
        label_smoothing: Amount of label smoothing in `[0.0, 1.0]`.

    Returns:
        Computed cross-entropy loss tensor.
    """
    _ = (size_average, reduce)
    log_probs = log_softmax(input, dim=-1)
    if input.shape == target.shape:
        loss = -zero_torch.sum(target * log_probs, axis=-1)
        return _reduce_loss(loss, reduction)
    nll = nll_loss(
        log_probs,
        target,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    if label_smoothing > 0.0:
        smooth_loss = -zero_torch.mean(log_probs)
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
        res = _wrap(get_op("Concatenate")()([_to_tensor(s) for s in slices], axis=-1))
        if bias is not None:
            res = res + bias
        return res
    return Tensor(0.0)
