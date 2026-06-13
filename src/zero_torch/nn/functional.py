"Functional NN API."

from zero_torch.tensor import Tensor, _wrap
import ml_switcheroo_compiler.nn as _nn


def conv2d(*args, **kwargs):
    """Applies the conv2d operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the conv2d operation applied.
    """
    res = getattr(_nn, "conv2d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


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
    """Applies the multilabel_margin_loss operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the multilabel_margin_loss operation applied.
    """
    pass


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
    """Applies the ctc_loss operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the ctc_loss operation applied.
    """
    pass


def multilabel_soft_margin_loss(*args, **kwargs):
    """Applies the multilabel_soft_margin_loss operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the multilabel_soft_margin_loss operation applied.
    """
    pass


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
    """Applies the cosine_embedding_loss operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the cosine_embedding_loss operation applied.
    """
    pass


def multi_margin_loss(*args, **kwargs):
    """Applies the multi_margin_loss operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the multi_margin_loss operation applied.
    """
    pass


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
    """Applies the gaussian_nll_loss operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the gaussian_nll_loss operation applied.
    """
    pass


def nll_loss(*args, **kwargs):
    """Applies the nll_loss operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the nll_loss operation applied.
    """
    pass


def hardshrink(*args, **kwargs):
    """Applies the hardshrink operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the hardshrink operation applied.
    """
    pass


def hinge_embedding_loss(*args, **kwargs):
    """Applies the hinge_embedding_loss operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the hinge_embedding_loss operation applied.
    """
    pass


def poisson_nll_loss(*args, **kwargs):
    """Applies the poisson_nll_loss operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the poisson_nll_loss operation applied.
    """
    pass


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
    """Applies the huber_loss operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the huber_loss operation applied.
    """
    pass


def smooth_l1_loss(*args, **kwargs):
    """Applies the smooth_l1_loss operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the smooth_l1_loss operation applied.
    """
    pass


def hardsigmoid(*args, **kwargs):
    """Applies the hardsigmoid operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the hardsigmoid operation applied.
    """
    pass


def kl_div(*args, **kwargs):
    """Applies the kl_div operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the kl_div operation applied.
    """
    pass


def soft_margin_loss(*args, **kwargs):
    """Applies the soft_margin_loss operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the soft_margin_loss operation applied.
    """
    pass


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
    """Applies the margin_ranking_loss operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the margin_ranking_loss operation applied.
    """
    pass


def triplet_margin_loss(*args, **kwargs):
    """Applies the triplet_margin_loss operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the triplet_margin_loss operation applied.
    """
    pass


def hardtanh(*args, **kwargs):
    """Applies the hardtanh operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the hardtanh operation applied.
    """
    pass


def triplet_margin_with_distance_loss(*args, **kwargs):
    """Applies the triplet_margin_with_distance_loss operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the triplet_margin_with_distance_loss operation applied.
    """
    pass


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


def logsigmoid(*args, **kwargs):
    """Applies the logsigmoid operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the logsigmoid operation applied.
    """
    pass


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


def prelu(*args, **kwargs):
    """Applies the prelu operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the prelu operation applied.
    """
    pass


def rrelu(*args, **kwargs):
    """Applies the rrelu operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the rrelu operation applied.
    """
    pass


def relu6(*args, **kwargs):
    """Applies the relu6 operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the relu6 operation applied.
    """
    pass


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


def silu(*args, **kwargs):
    """Applies the silu operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the silu operation applied.
    """
    pass


def softmax2d(*args, **kwargs):
    """Applies the softmax2d operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the softmax2d operation applied.
    """
    pass


def softmin(*args, **kwargs):
    """Applies the softmin operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the softmin operation applied.
    """
    pass


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


def softshrink(*args, **kwargs):
    """Applies the softshrink operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the softshrink operation applied.
    """
    pass


def softsign(*args, **kwargs):
    """Applies the softsign operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the softsign operation applied.
    """
    pass


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


def tanhshrink(*args, **kwargs):
    """Applies the tanhshrink operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the tanhshrink operation applied.
    """
    pass


def threshold(*args, **kwargs):
    """Applies the threshold operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the threshold operation applied.
    """
    pass


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


def avg_pool1d(*args, **kwargs):
    """Applies the avg_pool1d operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the avg_pool1d operation applied.
    """
    res = getattr(_nn, "avg_pool1d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def avg_pool2d(*args, **kwargs):
    """Applies the avg_pool2d operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the avg_pool2d operation applied.
    """
    res = getattr(_nn, "avg_pool2d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def avg_pool3d(*args, **kwargs):
    """Applies the avg_pool3d operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the avg_pool3d operation applied.
    """
    res = getattr(_nn, "avg_pool3d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
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


def conv1d(*args, **kwargs):
    """Applies the conv1d operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the conv1d operation applied.
    """
    res = getattr(_nn, "conv1d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def conv3d(*args, **kwargs):
    """Applies the conv3d operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the conv3d operation applied.
    """
    res = getattr(_nn, "conv3d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


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


def max_pool1d(*args, **kwargs):
    """Applies the max_pool1d operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the max_pool1d operation applied.
    """
    res = getattr(_nn, "max_pool1d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def max_pool2d(*args, **kwargs):
    """Applies the max_pool2d operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the max_pool2d operation applied.
    """
    res = getattr(_nn, "max_pool2d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def max_pool3d(*args, **kwargs):
    """Applies the max_pool3d operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: A new tensor with the max_pool3d operation applied.
    """
    res = getattr(_nn, "max_pool3d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
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
