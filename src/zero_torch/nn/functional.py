"Functional NN API."

from zero_torch.tensor import Tensor, _wrap
import ml_switcheroo.nn as _nn


def conv2d(*args, **kwargs):
    res = getattr(_nn, "conv2d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def relu(*args, **kwargs):
    res = getattr(_nn, "relu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def softmax(*args, **kwargs):
    res = getattr(_nn, "softmax")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def scaled_dot_product_attention(*args, **kwargs):
    res = getattr(_nn, "scaled_dot_product_attention")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def celu(*args, **kwargs):
    res = getattr(_nn, "celu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def multilabel_margin_loss(*args, **kwargs):
    pass


def elu(*args, **kwargs):
    res = getattr(_nn, "elu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def ctc_loss(*args, **kwargs):
    pass


def multilabel_soft_margin_loss(*args, **kwargs):
    pass


def gelu(*args, **kwargs):
    res = getattr(_nn, "gelu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def cosine_embedding_loss(*args, **kwargs):
    pass


def multi_margin_loss(*args, **kwargs):
    pass


def glu(*args, **kwargs):
    res = getattr(_nn, "glu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def gaussian_nll_loss(*args, **kwargs):
    pass


def nll_loss(*args, **kwargs):
    pass


def hardshrink(*args, **kwargs):
    pass


def hinge_embedding_loss(*args, **kwargs):
    pass


def poisson_nll_loss(*args, **kwargs):
    pass


def sigmoid(*args, **kwargs):
    res = getattr(_nn, "sigmoid")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def huber_loss(*args, **kwargs):
    pass


def smooth_l1_loss(*args, **kwargs):
    pass


def hardsigmoid(*args, **kwargs):
    pass


def kl_div(*args, **kwargs):
    pass


def soft_margin_loss(*args, **kwargs):
    pass


def hardswish(*args, **kwargs):
    res = getattr(_nn, "hardswish")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def margin_ranking_loss(*args, **kwargs):
    pass


def triplet_margin_loss(*args, **kwargs):
    pass


def hardtanh(*args, **kwargs):
    pass


def triplet_margin_with_distance_loss(*args, **kwargs):
    pass


def leaky_relu(*args, **kwargs):
    res = getattr(_nn, "leaky_relu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def logsigmoid(*args, **kwargs):
    pass


def log_softmax(*args, **kwargs):
    res = getattr(_nn, "log_softmax")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def mish(*args, **kwargs):
    res = getattr(_nn, "mish")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def prelu(*args, **kwargs):
    pass


def rrelu(*args, **kwargs):
    pass


def relu6(*args, **kwargs):
    pass


def selu(*args, **kwargs):
    res = getattr(_nn, "selu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def silu(*args, **kwargs):
    pass


def softmax2d(*args, **kwargs):
    pass


def softmin(*args, **kwargs):
    pass


def softplus(*args, **kwargs):
    res = getattr(_nn, "softplus")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def softshrink(*args, **kwargs):
    pass


def softsign(*args, **kwargs):
    pass


def tanh(*args, **kwargs):
    res = getattr(_nn, "tanh")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def tanhshrink(*args, **kwargs):
    pass


def threshold(*args, **kwargs):
    pass


def activations(*args, **kwargs):
    res = getattr(_nn, "activations")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def adaptive_avg_pool2d(*args, **kwargs):
    res = getattr(_nn, "adaptive_avg_pool2d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def alpha_dropout(*args, **kwargs):
    res = getattr(_nn, "alpha_dropout")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def avg_pool1d(*args, **kwargs):
    res = getattr(_nn, "avg_pool1d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def avg_pool2d(*args, **kwargs):
    res = getattr(_nn, "avg_pool2d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def avg_pool3d(*args, **kwargs):
    res = getattr(_nn, "avg_pool3d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def batch_norm(*args, **kwargs):
    res = getattr(_nn, "batch_norm")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def complex(*args, **kwargs):
    res = getattr(_nn, "complex")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def conv1d(*args, **kwargs):
    res = getattr(_nn, "conv1d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def conv3d(*args, **kwargs):
    res = getattr(_nn, "conv3d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def conv_transpose1d(*args, **kwargs):
    res = getattr(_nn, "conv_transpose1d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def conv_transpose2d(*args, **kwargs):
    res = getattr(_nn, "conv_transpose2d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def conv_transpose3d(*args, **kwargs):
    res = getattr(_nn, "conv_transpose3d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def dropout(*args, **kwargs):
    res = getattr(_nn, "dropout")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def embedding(*args, **kwargs):
    res = getattr(_nn, "embedding")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def feature_alpha_dropout(*args, **kwargs):
    res = getattr(_nn, "feature_alpha_dropout")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def fractional_max_pool2d(*args, **kwargs):
    res = getattr(_nn, "fractional_max_pool2d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def group_norm(*args, **kwargs):
    res = getattr(_nn, "group_norm")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def gru_cell(*args, **kwargs):
    res = getattr(_nn, "gru_cell")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def instance_norm(*args, **kwargs):
    res = getattr(_nn, "instance_norm")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def layer_norm(*args, **kwargs):
    res = getattr(_nn, "layer_norm")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def lstm_cell(*args, **kwargs):
    res = getattr(_nn, "lstm_cell")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def max_pool1d(*args, **kwargs):
    res = getattr(_nn, "max_pool1d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def max_pool2d(*args, **kwargs):
    res = getattr(_nn, "max_pool2d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def max_pool3d(*args, **kwargs):
    res = getattr(_nn, "max_pool3d")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def pad(*args, **kwargs):
    res = getattr(_nn, "pad")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def rms_norm(*args, **kwargs):
    res = getattr(_nn, "rms_norm")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def rnn_cell(*args, **kwargs):
    res = getattr(_nn, "rnn_cell")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def spatial_dropout(*args, **kwargs):
    res = getattr(_nn, "spatial_dropout")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def swish(*args, **kwargs):
    res = getattr(_nn, "swish")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def upsample_bilinear(*args, **kwargs):
    res = getattr(_nn, "upsample_bilinear")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def upsample_nearest(*args, **kwargs):
    res = getattr(_nn, "upsample_nearest")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)
