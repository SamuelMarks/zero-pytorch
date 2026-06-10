"Functional NN API."

from zero_torch.tensor import Tensor, _wrap
import ml_switcheroo.nn as _nn


def conv2d(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "conv2d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def relu(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "relu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def softmax(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "softmax")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def scaled_dot_product_attention(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "scaled_dot_product_attention")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def celu(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "celu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def multilabel_margin_loss(*args, **kwargs):
    """Function."""
    pass


def elu(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "elu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def ctc_loss(*args, **kwargs):
    """Function."""
    pass


def multilabel_soft_margin_loss(*args, **kwargs):
    """Function."""
    pass


def gelu(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "gelu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def cosine_embedding_loss(*args, **kwargs):
    """Function."""
    pass


def multi_margin_loss(*args, **kwargs):
    """Function."""
    pass


def glu(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "glu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def gaussian_nll_loss(*args, **kwargs):
    """Function."""
    pass


def nll_loss(*args, **kwargs):
    """Function."""
    pass


def hardshrink(*args, **kwargs):
    """Function."""
    pass


def hinge_embedding_loss(*args, **kwargs):
    """Function."""
    pass


def poisson_nll_loss(*args, **kwargs):
    """Function."""
    pass


def sigmoid(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "sigmoid")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def huber_loss(*args, **kwargs):
    """Function."""
    pass


def smooth_l1_loss(*args, **kwargs):
    """Function."""
    pass


def hardsigmoid(*args, **kwargs):
    """Function."""
    pass


def kl_div(*args, **kwargs):
    """Function."""
    pass


def soft_margin_loss(*args, **kwargs):
    """Function."""
    pass


def hardswish(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "hardswish")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def margin_ranking_loss(*args, **kwargs):
    """Function."""
    pass


def triplet_margin_loss(*args, **kwargs):
    """Function."""
    pass


def hardtanh(*args, **kwargs):
    """Function."""
    pass


def triplet_margin_with_distance_loss(*args, **kwargs):
    """Function."""
    pass


def leaky_relu(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "leaky_relu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def logsigmoid(*args, **kwargs):
    """Function."""
    pass


def log_softmax(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "log_softmax")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def mish(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "mish")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def prelu(*args, **kwargs):
    """Function."""
    pass


def rrelu(*args, **kwargs):
    """Function."""
    pass


def relu6(*args, **kwargs):
    """Function."""
    pass


def selu(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "selu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def silu(*args, **kwargs):
    """Function."""
    pass


def softmax2d(*args, **kwargs):
    """Function."""
    pass


def softmin(*args, **kwargs):
    """Function."""
    pass


def softplus(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "softplus")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def softshrink(*args, **kwargs):
    """Function."""
    pass


def softsign(*args, **kwargs):
    """Function."""
    pass


def tanh(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "tanh")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def tanhshrink(*args, **kwargs):
    """Function."""
    pass


def threshold(*args, **kwargs):
    """Function."""
    pass


def activations(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "activations")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def adaptive_avg_pool2d(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "adaptive_avg_pool2d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def alpha_dropout(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "alpha_dropout")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def avg_pool1d(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "avg_pool1d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def avg_pool2d(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "avg_pool2d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def avg_pool3d(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "avg_pool3d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def batch_norm(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "batch_norm")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def complex(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "complex")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def conv1d(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "conv1d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def conv3d(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "conv3d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def conv_transpose1d(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "conv_transpose1d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def conv_transpose2d(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "conv_transpose2d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def conv_transpose3d(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "conv_transpose3d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def dropout(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "dropout")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def embedding(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "embedding")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def feature_alpha_dropout(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "feature_alpha_dropout")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def fractional_max_pool2d(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "fractional_max_pool2d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def group_norm(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "group_norm")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def gru_cell(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "gru_cell")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def instance_norm(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "instance_norm")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def layer_norm(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "layer_norm")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def lstm_cell(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "lstm_cell")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def max_pool1d(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "max_pool1d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def max_pool2d(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "max_pool2d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def max_pool3d(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "max_pool3d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def pad(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "pad")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def rms_norm(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "rms_norm")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def rnn_cell(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "rnn_cell")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def spatial_dropout(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "spatial_dropout")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def swish(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "swish")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def upsample_bilinear(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "upsample_bilinear")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def upsample_nearest(*args, **kwargs):
    """Function."""
    res = getattr(_nn, "upsample_nearest")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)  # pragma: no cover


def linear(input, weight, bias=None):
    """Function."""
    import zero_torch

    return zero_torch.matmul(input, zero_torch.transpose(weight, 0, 1)) + bias
