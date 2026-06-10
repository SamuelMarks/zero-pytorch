"Package init."

__all__ = [
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AdaptiveLogSoftmaxWithLoss",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "AlphaDropout",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "BCELoss",
    "BCEWithLogitsLoss",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "Bilinear",
    "CELU",
    "CTCLoss",
    "ChannelShuffle",
    "CircularPad1d",
    "CircularPad2d",
    "CircularPad3d",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "Container",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "CosineEmbeddingLoss",
    "CosineSimilarity",
    "CrossEntropyLoss",
    "CrossMapLRN2d",
    "DataParallel",
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "ELU",
    "Embedding",
    "EmbeddingBag",
    "F",
    "FeatureAlphaDropout",
    "Flatten",
    "Fold",
    "FractionalMaxPool2d",
    "FractionalMaxPool3d",
    "GELU",
    "GLU",
    "GRU",
    "GRUCell",
    "GaussianNLLLoss",
    "GroupNorm",
    "Hardshrink",
    "Hardsigmoid",
    "Hardswish",
    "Hardtanh",
    "HingeEmbeddingLoss",
    "HuberLoss",
    "Identity",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "KLDivLoss",
    "L1Loss",
    "LPPool1d",
    "LPPool2d",
    "LPPool3d",
    "LSTM",
    "LSTMCell",
    "LayerNorm",
    "LazyBatchNorm1d",
    "LazyBatchNorm2d",
    "LazyBatchNorm3d",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "LazyConvTranspose1d",
    "LazyConvTranspose2d",
    "LazyConvTranspose3d",
    "LazyInstanceNorm1d",
    "LazyInstanceNorm2d",
    "LazyInstanceNorm3d",
    "LazyLinear",
    "LeakyReLU",
    "Linear",
    "LocalResponseNorm",
    "LogSigmoid",
    "LogSoftmax",
    "MSELoss",
    "MarginRankingLoss",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "MaxUnpool1d",
    "MaxUnpool2d",
    "MaxUnpool3d",
    "Mish",
    "Module",
    "ModuleDict",
    "ModuleList",
    "MultiLabelMarginLoss",
    "MultiLabelSoftMarginLoss",
    "MultiMarginLoss",
    "MultiheadAttention",
    "NLLLoss",
    "NLLLoss2d",
    "PReLU",
    "PairwiseDistance",
    "Parameter",
    "ParameterDict",
    "ParameterList",
    "PixelShuffle",
    "PixelUnshuffle",
    "PoissonNLLLoss",
    "RMSNorm",
    "RNN",
    "RNNBase",
    "RNNCell",
    "RNNCellBase",
    "RReLU",
    "ReLU",
    "ReLU6",
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReflectionPad3d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "SELU",
    "Sequential",
    "SiLU",
    "Sigmoid",
    "SmoothL1Loss",
    "SoftMarginLoss",
    "Softmax",
    "Softmax2d",
    "Softmin",
    "Softplus",
    "Softshrink",
    "Softsign",
    "SyncBatchNorm",
    "Tanh",
    "Tanhshrink",
    "Threshold",
    "Transformer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
    "Unflatten",
    "Unfold",
    "Upsample",
    "UpsamplingBilinear2d",
    "UpsamplingNearest2d",
    "ZeroPad1d",
    "ZeroPad2d",
    "ZeroPad3d",
]
from typing import Any
from .module import Module, Parameter, ModuleList, ParameterList
from . import functional as F
from .identity import Identity
from .linear import Linear
from .relu import ReLU, Sequential
from .parallel import DataParallel
from .loss import (
    BCELoss,
    BCEWithLogitsLoss,
    CTCLoss,
    CosineEmbeddingLoss,
    CrossEntropyLoss,
    GaussianNLLLoss,
    HingeEmbeddingLoss,
    HuberLoss,
    KLDivLoss,
    L1Loss,
    MSELoss,
    MarginRankingLoss,
    MultiLabelMarginLoss,
    MultiLabelSoftMarginLoss,
    MultiMarginLoss,
    NLLLoss,
    NLLLoss2d,
    PoissonNLLLoss,
    SmoothL1Loss,
    SoftMarginLoss,
    TripletMarginLoss,
    TripletMarginWithDistanceLoss,
)
from .dropout import (
    Dropout,
    Dropout1d,
    Dropout2d,
    Dropout3d,
    AlphaDropout,
    FeatureAlphaDropout,
)
from .pooling import (
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    AdaptiveMaxPool1d,
    AdaptiveMaxPool2d,
    AdaptiveMaxPool3d,
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
)


class AdaptiveLogSoftmaxWithLoss:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class BatchNorm1d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class BatchNorm2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class BatchNorm3d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Bilinear:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class CELU:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class ChannelShuffle:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class CircularPad1d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class CircularPad2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class CircularPad3d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class ConstantPad1d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class ConstantPad2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class ConstantPad3d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Container:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Conv1d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Conv2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Conv3d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class ConvTranspose1d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class ConvTranspose2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class ConvTranspose3d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class CosineSimilarity:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class CrossMapLRN2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class ELU:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Embedding:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class EmbeddingBag:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Flatten:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Fold:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class FractionalMaxPool2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class FractionalMaxPool3d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class GELU:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class GLU:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class GRU:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class GRUCell:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class GroupNorm:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Hardshrink:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Hardsigmoid:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Hardswish:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Hardtanh:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class InstanceNorm1d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class InstanceNorm2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class InstanceNorm3d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LPPool1d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LPPool2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LPPool3d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LSTM:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LSTMCell:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LayerNorm:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LazyBatchNorm1d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LazyBatchNorm2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LazyBatchNorm3d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LazyConv1d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LazyConv2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LazyConv3d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LazyConvTranspose1d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LazyConvTranspose2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LazyConvTranspose3d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LazyInstanceNorm1d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LazyInstanceNorm2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LazyInstanceNorm3d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LazyLinear:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LeakyReLU:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LocalResponseNorm:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LogSigmoid:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class LogSoftmax:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class MaxPool1d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class MaxPool2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class MaxPool3d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class MaxUnpool1d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class MaxUnpool2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class MaxUnpool3d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Mish:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class ModuleDict:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class MultiheadAttention:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class PReLU:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class PairwiseDistance:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class ParameterDict:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class PixelShuffle:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class PixelUnshuffle:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class RMSNorm:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class RNN:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class RNNBase:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class RNNCell:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class RNNCellBase:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class RReLU:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class ReLU6:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class ReflectionPad1d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class ReflectionPad2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class ReflectionPad3d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class ReplicationPad1d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class ReplicationPad2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class ReplicationPad3d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class SELU:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class SiLU:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Sigmoid:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Softmax:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Softmax2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Softmin:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Softplus:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Softshrink:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Softsign:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class SyncBatchNorm:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Tanh:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Tanhshrink:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Threshold:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Transformer:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class TransformerDecoder:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class TransformerDecoderLayer:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class TransformerEncoder:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class TransformerEncoderLayer:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Unflatten:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Unfold:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class Upsample:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class UpsamplingBilinear2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class UpsamplingNearest2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class ZeroPad1d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class ZeroPad2d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass


class ZeroPad3d:
    """Stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Stub."""
        pass
