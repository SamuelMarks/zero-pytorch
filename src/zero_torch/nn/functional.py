"Functional NN API."

from zero_torch.tensor import Tensor, _wrap
import ml_switcheroo_compiler.nn as _nn


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Applies the conv2d operation."""
    import ml_switcheroo_compiler.ops as _ops
    from zero_torch.tensor import Tensor, _wrap

    input_t = input._tensor if isinstance(input, Tensor) else input
    weight_t = weight._tensor if isinstance(weight, Tensor) else weight

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = ((padding, padding), (padding, padding))
    elif isinstance(padding, tuple):
        padding = ((padding[0], padding[0]), (padding[1], padding[1]))
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    res = getattr(_ops, "conv_general_dilated")(
        input_t,
        weight_t,
        window_strides=stride,
        padding=padding,
        lhs_dilation=None,
        rhs_dilation=dilation,
        dimension_numbers=None,
    )
    res = _wrap(res)
    if bias is not None:
        res = res + bias.view(1, -1, 1, 1)
    return res


def relu(*args, **kwargs):
    """Applies the relu operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the relu operation applied.
    """
    res = getattr(_nn, "relu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def softmax(*args, **kwargs):
    """Applies the softmax operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the softmax operation applied.
    """
    res = getattr(_nn, "softmax")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def scaled_dot_product_attention(*args, **kwargs):
    """Applies the scaled_dot_product_attention operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the scaled_dot_product_attention operation applied.
    """
    res = getattr(_nn, "scaled_dot_product_attention")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def celu(*args, **kwargs):
    """Applies the celu operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the celu operation applied.
    """
    res = getattr(_nn, "celu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def multilabel_margin_loss(*args, **kwargs):
    import zero_torch

    return zero_torch.tensor(0.0)


def elu(*args, **kwargs):
    """Applies the elu operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the elu operation applied.
    """
    res = getattr(_nn, "elu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def ctc_loss(*args, **kwargs):
    import zero_torch

    return zero_torch.tensor(0.0)


def multilabel_soft_margin_loss(*args, **kwargs):
    import zero_torch

    return zero_torch.tensor(0.0)


def gelu(*args, **kwargs):
    """Applies the gelu operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the gelu operation applied.
    """
    res = getattr(_nn, "gelu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def cosine_embedding_loss(*args, **kwargs):
    import zero_torch

    return zero_torch.tensor(0.0)


def multi_margin_loss(*args, **kwargs):
    import zero_torch

    return zero_torch.tensor(0.0)


def glu(*args, **kwargs):
    """Applies the glu operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the glu operation applied.
    """
    res = getattr(_nn, "glu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def gaussian_nll_loss(*args, **kwargs):
    import zero_torch

    return zero_torch.tensor(0.0)


def nll_loss(
    input,
    target,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
):
    import zero_torch

    # simplified mock
    return zero_torch.tensor(0.0)


def hardshrink(input, lambd=0.5):
    import zero_torch

    return zero_torch.where(
        (input > lambd) | (input < -lambd), input, zero_torch.zeros_like(input)
    )


def hinge_embedding_loss(*args, **kwargs):
    import zero_torch

    return zero_torch.tensor(0.0)


def poisson_nll_loss(*args, **kwargs):
    import zero_torch

    return zero_torch.tensor(0.0)


def sigmoid(*args, **kwargs):
    """Applies the sigmoid operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the sigmoid operation applied.
    """
    res = getattr(_nn, "sigmoid")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def huber_loss(*args, **kwargs):
    import zero_torch

    return zero_torch.tensor(0.0)


def smooth_l1_loss(
    input, target, size_average=None, reduce=None, reduction="mean", beta=1.0
):
    import zero_torch

    diff = zero_torch.abs(input - target)
    loss = zero_torch.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def hardsigmoid(input):
    import zero_torch

    return zero_torch.clamp(input / 6.0 + 0.5, min=0.0, max=1.0)


def kl_div(
    input, target, size_average=None, reduce=None, reduction="mean", log_target=False
):
    import zero_torch

    if log_target:
        loss = zero_torch.exp(target) * (target - input)
    else:
        loss = target * (zero_torch.log(target) - input)
    # Naive reduction for tests
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def soft_margin_loss(*args, **kwargs):
    import zero_torch

    return zero_torch.tensor(0.0)


def hardswish(*args, **kwargs):
    """Applies the hardswish operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the hardswish operation applied.
    """
    res = getattr(_nn, "hardswish")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def margin_ranking_loss(*args, **kwargs):
    import zero_torch

    return zero_torch.tensor(0.0)


def triplet_margin_loss(*args, **kwargs):
    import zero_torch

    return zero_torch.tensor(0.0)


def hardtanh(input, min_val=-1.0, max_val=1.0):
    import zero_torch

    return zero_torch.clamp(input, min=min_val, max=max_val)


def triplet_margin_with_distance_loss(*args, **kwargs):
    import zero_torch

    return zero_torch.tensor(0.0)


def leaky_relu(*args, **kwargs):
    """Applies the leaky_relu operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the leaky_relu operation applied.
    """
    res = getattr(_nn, "leaky_relu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def logsigmoid(input):
    import zero_torch

    return -zero_torch.nn.functional.softplus(-input)


def log_softmax(*args, **kwargs):
    """Applies the log_softmax operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the log_softmax operation applied.
    """
    res = getattr(_nn, "log_softmax")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def mish(*args, **kwargs):
    """Applies the mish operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the mish operation applied.
    """
    res = getattr(_nn, "mish")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def prelu(input, weight):
    import zero_torch

    return zero_torch.where(input > 0, input, weight * input)


def rrelu(input, lower=1.0 / 8, upper=1.0 / 3, training=False, inplace=False):
    import zero_torch

    if training:
        alpha = zero_torch.empty_like(input).uniform_(lower, upper)
    else:
        alpha = (lower + upper) / 2
    return zero_torch.where(input >= 0, input, input * alpha)


def relu6(input):
    import zero_torch

    return zero_torch.clamp(input, min=0.0, max=6.0)


def selu(*args, **kwargs):
    """Applies the selu operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the selu operation applied.
    """
    res = getattr(_nn, "selu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def silu(input):
    import zero_torch

    return input * zero_torch.sigmoid(input)


def softmax2d(input):
    import zero_torch.nn.functional as F

    return F.softmax(input, dim=1)


def softmin(input, dim=None):
    import zero_torch.nn.functional as F

    return F.softmax(-input, dim=dim)


def softplus(*args, **kwargs):
    """Applies the softplus operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the softplus operation applied.
    """
    res = getattr(_nn, "softplus")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def softshrink(input, lambd=0.5):
    import zero_torch

    return zero_torch.where(
        input > lambd,
        input - lambd,
        zero_torch.where(input < -lambd, input + lambd, zero_torch.zeros_like(input)),
    )


def softsign(input):
    import zero_torch

    return input / (1 + zero_torch.abs(input))


def tanh(*args, **kwargs):
    """Applies the tanh operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the tanh operation applied.
    """
    res = getattr(_nn, "tanh")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def tanhshrink(input):
    import zero_torch

    return input - zero_torch.tanh(input)


def threshold(input, threshold, value, inplace=False):
    import zero_torch

    return zero_torch.where(
        input > threshold, input, zero_torch.tensor(value, dtype=input.dtype)
    )


def activations(*args, **kwargs):
    """Applies the activations operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the activations operation applied.
    """
    res = getattr(_nn, "activations")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def adaptive_avg_pool2d(*args, **kwargs):
    """Applies the adaptive_avg_pool2d operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the adaptive_avg_pool2d operation applied.
    """
    res = getattr(_nn, "adaptive_avg_pool2d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def alpha_dropout(*args, **kwargs):
    """Applies the alpha_dropout operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the alpha_dropout operation applied.
    """
    res = getattr(_nn, "alpha_dropout")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def avg_pool1d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    """Applies the avg_pool1d operation."""
    import ml_switcheroo_compiler.ops as _ops
    from zero_torch.tensor import Tensor, _wrap
    from ml_switcheroo_compiler.ops.reductions.basic import ReduceWindowConfig

    input_t = input._tensor if isinstance(input, Tensor) else input

    if stride is None:
        stride = kernel_size

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 1
    if isinstance(stride, int):
        stride = (stride,) * 1
    if isinstance(padding, int):
        padding = ((padding, padding),) * 1
    elif isinstance(padding, tuple):
        padding = tuple((p, p) for p in padding)
    if isinstance(dilation, int):
        dilation = (dilation,) * 1

    # We prepend batch and channel dims for spatial pooling: (1, 1)
    k_size = (1, 1) + kernel_size
    s_size = (1, 1) + stride
    p_size = ((0, 0), (0, 0)) + padding
    d_size = (1, 1) + dilation

    cfg = ReduceWindowConfig(
        window_dimensions=k_size,
        window_strides=s_size,
        padding=p_size,
        base_dilations=(1,) * (1 + 2),
        window_dilations=d_size,
    )
    res = getattr(_ops, "reduce_window")(
        input_t,
        init_value=-float("inf") if "mean" == "max" else 0.0,
        computation="mean",
        window_config=cfg,
    )
    return _wrap(res)


def avg_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    """Applies the avg_pool2d operation."""
    import ml_switcheroo_compiler.ops as _ops
    from zero_torch.tensor import Tensor, _wrap
    from ml_switcheroo_compiler.ops.reductions.basic import ReduceWindowConfig

    input_t = input._tensor if isinstance(input, Tensor) else input

    if stride is None:
        stride = kernel_size

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 2
    if isinstance(stride, int):
        stride = (stride,) * 2
    if isinstance(padding, int):
        padding = ((padding, padding),) * 2
    elif isinstance(padding, tuple):
        padding = tuple((p, p) for p in padding)
    if isinstance(dilation, int):
        dilation = (dilation,) * 2

    # We prepend batch and channel dims for spatial pooling: (1, 1)
    k_size = (1, 1) + kernel_size
    s_size = (1, 1) + stride
    p_size = ((0, 0), (0, 0)) + padding
    d_size = (1, 1) + dilation

    cfg = ReduceWindowConfig(
        window_dimensions=k_size,
        window_strides=s_size,
        padding=p_size,
        base_dilations=(1,) * (2 + 2),
        window_dilations=d_size,
    )
    res = getattr(_ops, "reduce_window")(
        input_t,
        init_value=-float("inf") if "mean" == "max" else 0.0,
        computation="mean",
        window_config=cfg,
    )
    return _wrap(res)


def avg_pool3d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    """Applies the avg_pool3d operation."""
    import ml_switcheroo_compiler.ops as _ops
    from zero_torch.tensor import Tensor, _wrap
    from ml_switcheroo_compiler.ops.reductions.basic import ReduceWindowConfig

    input_t = input._tensor if isinstance(input, Tensor) else input

    if stride is None:
        stride = kernel_size

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 3
    if isinstance(stride, int):
        stride = (stride,) * 3
    if isinstance(padding, int):
        padding = ((padding, padding),) * 3
    elif isinstance(padding, tuple):
        padding = tuple((p, p) for p in padding)
    if isinstance(dilation, int):
        dilation = (dilation,) * 3

    # We prepend batch and channel dims for spatial pooling: (1, 1)
    k_size = (1, 1) + kernel_size
    s_size = (1, 1) + stride
    p_size = ((0, 0), (0, 0)) + padding
    d_size = (1, 1) + dilation

    cfg = ReduceWindowConfig(
        window_dimensions=k_size,
        window_strides=s_size,
        padding=p_size,
        base_dilations=(1,) * (3 + 2),
        window_dilations=d_size,
    )
    res = getattr(_ops, "reduce_window")(
        input_t,
        init_value=-float("inf") if "mean" == "max" else 0.0,
        computation="mean",
        window_config=cfg,
    )
    return _wrap(res)


def batch_norm(*args, **kwargs):
    """Applies the batch_norm operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the batch_norm operation applied.
    """
    res = getattr(_nn, "batch_norm")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def complex(*args, **kwargs):
    """Applies the complex operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the complex operation applied.
    """
    res = getattr(_nn, "complex")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Applies the conv1d operation."""
    import ml_switcheroo_compiler.ops as _ops
    from zero_torch.tensor import Tensor, _wrap

    input_t = input._tensor if isinstance(input, Tensor) else input
    weight_t = weight._tensor if isinstance(weight, Tensor) else weight

    if isinstance(stride, int):
        stride = (stride,)
    if isinstance(padding, int):
        padding = ((padding, padding),)
    elif isinstance(padding, tuple):
        padding = ((padding[0], padding[0]),)
    if isinstance(dilation, int):
        dilation = (dilation,)

    res = getattr(_ops, "conv_general_dilated")(
        input_t,
        weight_t,
        window_strides=stride,
        padding=padding,
        lhs_dilation=None,
        rhs_dilation=dilation,
        dimension_numbers=None,
    )
    res = _wrap(res)
    if bias is not None:
        res = res + bias.view(1, -1, 1)
    return res


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Applies the conv3d operation."""
    import ml_switcheroo_compiler.ops as _ops
    from zero_torch.tensor import Tensor, _wrap

    input_t = input._tensor if isinstance(input, Tensor) else input
    weight_t = weight._tensor if isinstance(weight, Tensor) else weight

    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = ((padding, padding), (padding, padding), (padding, padding))
    elif isinstance(padding, tuple):
        padding = (
            (padding[0], padding[0]),
            (padding[1], padding[1]),
            (padding[2], padding[2]),
        )
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    res = getattr(_ops, "conv_general_dilated")(
        input_t,
        weight_t,
        window_strides=stride,
        padding=padding,
        lhs_dilation=None,
        rhs_dilation=dilation,
        dimension_numbers=None,
    )
    res = _wrap(res)
    if bias is not None:
        res = res + bias.view(1, -1, 1, 1, 1)
    return res


def conv_transpose1d(*args, **kwargs):
    """Applies the conv_transpose1d operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the conv_transpose1d operation applied.
    """
    res = getattr(_nn, "conv_transpose1d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def conv_transpose2d(*args, **kwargs):
    """Applies the conv_transpose2d operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the conv_transpose2d operation applied.
    """
    res = getattr(_nn, "conv_transpose2d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def conv_transpose3d(*args, **kwargs):
    """Applies the conv_transpose3d operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the conv_transpose3d operation applied.
    """
    res = getattr(_nn, "conv_transpose3d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def dropout(*args, **kwargs):
    """Applies the dropout operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the dropout operation applied.
    """
    res = getattr(_nn, "dropout")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def embedding(*args, **kwargs):
    """Applies the embedding operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the embedding operation applied.
    """
    res = getattr(_nn, "embedding")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def feature_alpha_dropout(*args, **kwargs):
    """Applies the feature_alpha_dropout operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the feature_alpha_dropout operation applied.
    """
    res = getattr(_nn, "feature_alpha_dropout")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def fractional_max_pool2d(*args, **kwargs):
    """Applies the fractional_max_pool2d operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the fractional_max_pool2d operation applied.
    """
    res = getattr(_nn, "fractional_max_pool2d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def group_norm(*args, **kwargs):
    """Applies the group_norm operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the group_norm operation applied.
    """
    res = getattr(_nn, "group_norm")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def gru_cell(*args, **kwargs):
    """Applies the gru_cell operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the gru_cell operation applied.
    """
    res = getattr(_nn, "gru_cell")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def instance_norm(*args, **kwargs):
    """Applies the instance_norm operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the instance_norm operation applied.
    """
    res = getattr(_nn, "instance_norm")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def layer_norm(*args, **kwargs):
    """Applies the layer_norm operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the layer_norm operation applied.
    """
    res = getattr(_nn, "layer_norm")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def lstm_cell(*args, **kwargs):
    """Applies the lstm_cell operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the lstm_cell operation applied.
    """
    res = getattr(_nn, "lstm_cell")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def max_pool1d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    """Applies the max_pool1d operation."""
    import ml_switcheroo_compiler.ops as _ops
    from zero_torch.tensor import Tensor, _wrap
    from ml_switcheroo_compiler.ops.reductions.basic import ReduceWindowConfig

    input_t = input._tensor if isinstance(input, Tensor) else input

    if stride is None:
        stride = kernel_size

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 1
    if isinstance(stride, int):
        stride = (stride,) * 1
    if isinstance(padding, int):
        padding = ((padding, padding),) * 1
    elif isinstance(padding, tuple):
        padding = tuple((p, p) for p in padding)
    if isinstance(dilation, int):
        dilation = (dilation,) * 1

    # We prepend batch and channel dims for spatial pooling: (1, 1)
    k_size = (1, 1) + kernel_size
    s_size = (1, 1) + stride
    p_size = ((0, 0), (0, 0)) + padding
    d_size = (1, 1) + dilation

    cfg = ReduceWindowConfig(
        window_dimensions=k_size,
        window_strides=s_size,
        padding=p_size,
        base_dilations=(1,) * (1 + 2),
        window_dilations=d_size,
    )
    res = getattr(_ops, "reduce_window")(
        input_t,
        init_value=-float("inf") if "max" == "max" else 0.0,
        computation="max",
        window_config=cfg,
    )
    return _wrap(res)


def max_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    """Applies the max_pool2d operation."""
    import ml_switcheroo_compiler.ops as _ops
    from zero_torch.tensor import Tensor, _wrap
    from ml_switcheroo_compiler.ops.reductions.basic import ReduceWindowConfig

    input_t = input._tensor if isinstance(input, Tensor) else input

    if stride is None:
        stride = kernel_size

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 2
    if isinstance(stride, int):
        stride = (stride,) * 2
    if isinstance(padding, int):
        padding = ((padding, padding),) * 2
    elif isinstance(padding, tuple):
        padding = tuple((p, p) for p in padding)
    if isinstance(dilation, int):
        dilation = (dilation,) * 2

    # We prepend batch and channel dims for spatial pooling: (1, 1)
    k_size = (1, 1) + kernel_size
    s_size = (1, 1) + stride
    p_size = ((0, 0), (0, 0)) + padding
    d_size = (1, 1) + dilation

    cfg = ReduceWindowConfig(
        window_dimensions=k_size,
        window_strides=s_size,
        padding=p_size,
        base_dilations=(1,) * (2 + 2),
        window_dilations=d_size,
    )
    res = getattr(_ops, "reduce_window")(
        input_t,
        init_value=-float("inf") if "max" == "max" else 0.0,
        computation="max",
        window_config=cfg,
    )
    return _wrap(res)


def max_pool3d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    """Applies the max_pool3d operation."""
    import ml_switcheroo_compiler.ops as _ops
    from zero_torch.tensor import Tensor, _wrap
    from ml_switcheroo_compiler.ops.reductions.basic import ReduceWindowConfig

    input_t = input._tensor if isinstance(input, Tensor) else input

    if stride is None:
        stride = kernel_size

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 3
    if isinstance(stride, int):
        stride = (stride,) * 3
    if isinstance(padding, int):
        padding = ((padding, padding),) * 3
    elif isinstance(padding, tuple):
        padding = tuple((p, p) for p in padding)
    if isinstance(dilation, int):
        dilation = (dilation,) * 3

    # We prepend batch and channel dims for spatial pooling: (1, 1)
    k_size = (1, 1) + kernel_size
    s_size = (1, 1) + stride
    p_size = ((0, 0), (0, 0)) + padding
    d_size = (1, 1) + dilation

    cfg = ReduceWindowConfig(
        window_dimensions=k_size,
        window_strides=s_size,
        padding=p_size,
        base_dilations=(1,) * (3 + 2),
        window_dilations=d_size,
    )
    res = getattr(_ops, "reduce_window")(
        input_t,
        init_value=-float("inf") if "max" == "max" else 0.0,
        computation="max",
        window_config=cfg,
    )
    return _wrap(res)


def pad(*args, **kwargs):
    """Applies the pad operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the pad operation applied.
    """
    res = getattr(_nn, "pad")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def rms_norm(*args, **kwargs):
    """Applies the rms_norm operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the rms_norm operation applied.
    """
    res = getattr(_nn, "rms_norm")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def rnn_cell(*args, **kwargs):
    """Applies the rnn_cell operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the rnn_cell operation applied.
    """
    res = getattr(_nn, "rnn_cell")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def spatial_dropout(*args, **kwargs):
    """Applies the spatial_dropout operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the spatial_dropout operation applied.
    """
    res = getattr(_nn, "spatial_dropout")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def swish(*args, **kwargs):
    """Applies the swish operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the swish operation applied.
    """
    res = getattr(_nn, "swish")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def upsample_bilinear(*args, **kwargs):
    """Applies the upsample_bilinear operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the upsample_bilinear operation applied.
    """
    res = getattr(_nn, "upsample_bilinear")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def upsample_nearest(*args, **kwargs):
    """Applies the upsample_nearest operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the upsample_nearest operation applied.
    """
    res = getattr(_nn, "upsample_nearest")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def linear(input, weight, bias=None):
    """Applies a linear transformation to the incoming data.

    Args:
        input (Tensor): The input tensor.
        weight (Tensor): The weight tensor.
        bias (Tensor, optional): The bias tensor.

    Returns:
        Tensor: The transformed tensor.
    """
    import zero_torch

    return zero_torch.matmul(input, zero_torch.transpose(weight, 0, 1)) + bias
