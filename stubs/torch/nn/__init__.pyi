from typing import Any, Optional

_Optional = Optional
from collections.abc import Iterable
from typing import Callable, Optional

import torch
from torch import Tensor

class GELU:
    def __init__(
        self, approximate: str = "none", __constants__: list = ["approximate"]
    ) -> None: ...

class LeakyReLU:
    def __init__(
        self,
        negative_slope: float = 0.01,
        inplace: bool = False,
        __constants__: list = ["inplace", "negative_slope"],
    ) -> None: ...

class ReLU:
    def __init__(
        self, inplace: bool = False, __constants__: list = ["inplace"]
    ) -> Any: ...

class SiLU:
    def __init__(
        self, __constants__: list = ["inplace"], inplace: bool = False
    ) -> Any: ...

class Sigmoid:
    def __init__(self, *args, **kwargs) -> None: ...

class Softmax:
    def __init__(
        self, dim: int | None = "```(None)```", __constants__: list = ["dim"]
    ) -> None: ...

class Tanh:
    def __init__(self, *args, **kwargs) -> None: ...

class AdaptiveAvgPool1d:
    def __init__(self, output_size: Any = ...) -> None: ...

class AdaptiveAvgPool2d:
    def __init__(self, output_size: Any = ...) -> None: ...

class AdaptiveAvgPool3d:
    def __init__(self, output_size: Any = ...) -> None: ...

class AdaptiveMaxPool1d:
    def __init__(
        self, output_size: Any = ..., return_indices: bool = False
    ) -> None: ...

class AdaptiveMaxPool2d:
    def __init__(
        self, output_size: Any = ..., return_indices: bool = False
    ) -> None: ...

class AdaptiveMaxPool3d:
    def __init__(
        self, output_size: Any = ..., return_indices: bool = False
    ) -> None: ...

class AlphaDropout:
    def __init__(
        self, p: float = 0.5, inplace: bool | None = "```(None)```"
    ) -> None: ...

class AvgPool1d:
    def __init__(
        self,
        kernel_size: Any = ...,
        stride: Any = ":attr:`kernel_size",
        padding: Any = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None: ...

class AvgPool2d:
    def __init__(
        self,
        kernel_size: Any = ...,
        stride: Any = ":attr:`kernel_size",
        padding: Any = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = "```(None)```",
        __constants__: list = [
            "kernel_size",
            "stride",
            "padding",
            "ceil_mode",
            "count_include_pad",
            "divisor_override",
        ],
    ) -> None: ...

class AvgPool3d:
    def __init__(
        self,
        kernel_size: Any = ...,
        stride: Any = ":attr:`kernel_size",
        padding: Any = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = "```(None)```",
        __constants__: list = [
            "kernel_size",
            "stride",
            "padding",
            "ceil_mode",
            "count_include_pad",
            "divisor_override",
        ],
    ) -> None: ...

class BatchNorm1d:
    def __init__(
        self,
        num_features: int = ...,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...

class BatchNorm2d:
    def __init__(
        self,
        num_features: int = ...,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...

class BatchNorm3d:
    def __init__(
        self,
        num_features: int = ...,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...

class Bilinear:
    def __init__(
        self,
        in1_features: int = ...,
        in2_features: int = ...,
        out_features: int = ...,
        bias: bool = True,
        __constants__: list = ["in1_features", "in2_features", "out_features"],
        weight: Tensor = ...,
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
    ) -> None: ...

class CELU:
    def __init__(
        self,
        alpha: float = 1.0,
        inplace: bool = False,
        __constants__: list = ["alpha", "inplace"],
    ) -> None: ...

class ChannelShuffle:
    def __init__(self, groups: int = ..., __constants__: list = ["groups"]) -> None: ...

class CircularPad1d:
    def __init__(self, padding: tuple[int, int] = ...) -> None: ...

class CircularPad2d:
    def __init__(self, padding: tuple[int, int, int, int] = ...) -> None: ...

class CircularPad3d:
    def __init__(self, padding: tuple[int, int, int, int, int, int] = ...) -> None: ...

class ConstantPad1d:
    def __init__(self, padding: tuple[int, int] = ..., value: float = ...) -> Any: ...

class ConstantPad2d:
    def __init__(
        self,
        padding: tuple[int, int, int, int] = ...,
        __constants__: list = ["padding", "value"],
        value: float = ...,
    ) -> None: ...

class ConstantPad3d:
    def __init__(
        self, padding: tuple[int, int, int, int, int, int] = ..., value: float = ...
    ) -> None: ...

class Container:
    def __init__(self, **kwargs) -> None: ...

class Conv1d:
    def __init__(
        self,
        __doc__: Any = "```('Applies a 1D convolution over an input signal composed of several input\\n    planes.\\n\\n    In the simplest case, the output value of the layer with input size\\n    :math:`(N, C_{\\\\text{in}}, L)` and output :math:`(N, C_{\\\\text{out}}, L_{\\\\text{out}})` can be\\n    precisely described as:\\n\\n    .. math::\\n        \\\\text{out}(N_i, C_{\\\\text{out}_j}) = \\\\text{bias}(C_{\\\\text{out}_j}) +\\n        \\\\sum_{k = 0}^{C_{in} - 1} \\\\text{weight}(C_{\\\\text{out}_j}, k)\\n        \\\\star \\\\text{input}(N_i, k)\\n\\n    where :math:`\\\\star` is the valid `cross-correlation`_ operator,\\n    :math:`N` is a batch size, :math:`C` denotes a number of channels,\\n    :math:`L` is a length of signal sequence.\\n    ' + \"\\n\\n    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.\\n\\n    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.\\n\\n    * :attr:`stride` controls the stride for the cross-correlation, a single\\n      number or a one-element tuple.\\n\\n    * :attr:`padding` controls the amount of padding applied to the input. It\\n      can be either a string {{'valid', 'same'}} or a tuple of ints giving the\\n      amount of implicit padding applied on both sides.\\n\\n    * :attr:`dilation` controls the spacing between the kernel points; also\\n      known as the à trous algorithm. It is harder to describe, but this `link`_\\n      has a nice visualization of what :attr:`dilation` does.\\n\\n    {groups_note}\\n\\n    Note:\\n        {depthwise_separable_note}\\n    Note:\\n        {cudnn_reproducibility_note}\\n\\n    Note:\\n        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads\\n        the input so the output has the shape as the input. However, this mode\\n        doesn't support any stride values other than 1.\\n\\n    Note:\\n        This module supports complex data types i.e. ``complex32, complex64, complex128``.\\n\\n    Args:\\n        in_channels (int): Number of channels in the input image\\n        out_channels (int): Number of channels produced by the convolution\\n        kernel_size (int or tuple): Size of the convolving kernel\\n        stride (int or tuple, Any): Stride of the convolution. Default: 1\\n        padding (int, tuple or str, Any): Padding added to both sides of\\n            the input. Default: 0\\n        dilation (int or tuple, Any): Spacing between kernel\\n            elements. Default: 1\\n        groups (int, Any): Number of blocked connections from input\\n            channels to output channels. Default: 1\\n        bias (bool, Any): If ``True``, adds a learnable bias to the\\n            output. Default: ``True``\\n        padding_mode (str, Any): ``'zeros'``, ``'reflect'``,\\n            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``\\n\\n    \".format(**reproducibility_notes, **convolution_notes) + '\\n\\n    Shape:\\n        - Input: :math:`(N, C_{in}, L_{in})` or :math:`(C_{in}, L_{in})`\\n        - Output: :math:`(N, C_{out}, L_{out})` or :math:`(C_{out}, L_{out})`, where\\n\\n          .. math::\\n              L_{out} = \\\\left\\\\lfloor\\\\frac{L_{in} + 2 \\\\times \\\\text{padding} - \\\\text{dilation}\\n                        \\\\times (\\\\text{kernel\\\\Any} - 1) - 1}{\\\\text{stride}} + 1\\\\right\\\\rfloor\\n\\n    Attributes:\\n        weight (Tensor): the learnable weights of the module of shape\\n            :math:`(\\\\text{out\\\\_channels},\\n            \\\\frac{\\\\text{in\\\\_channels}}{\\\\text{groups}}, \\\\text{kernel\\\\Any})`.\\n            The values of these weights are sampled from\\n            :math:`\\\\mathcal{U}(-\\\\sqrt{k}, \\\\sqrt{k})` where\\n            :math:`k = \\\\frac{groups}{C_\\\\text{in} * \\\\text{kernel\\\\Any}}`\\n        bias (Tensor):   the learnable bias of the module of shape\\n            (out_channels). If :attr:`bias` is ``True``, then the values of these weights are\\n            sampled from :math:`\\\\mathcal{U}(-\\\\sqrt{k}, \\\\sqrt{k})` where\\n            :math:`k = \\\\frac{groups}{C_\\\\text{in} * \\\\text{kernel\\\\Any}}`\\n\\n    Examples::\\n\\n        >>> m = nn.Conv1d(16, 33, 3, stride=2)\\n        >>> input = torch.randn(20, 16, 50)\\n        >>> output = m(input)\\n\\n    .. _cross-correlation:\\n        https://en.wikipedia.org/wiki/Cross-correlation\\n\\n    .. _link:\\n        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md\\n    ')```",
        device: Any = "```(None)```",
        padding: str | Any = 0,
        out_channels: int = ...,
        groups: int = 1,
        dilation: Any = 1,
        in_channels: int = ...,
        bias: bool = True,
        padding_mode: str = "zeros",
        dtype: Any = "```(None)```",
        kernel_size: Any = ...,
        stride: Any = 1,
        transposed: bool = ...,
        output_padding: tuple = ...,
    ) -> None: ...

class Conv2d:
    def __init__(
        self,
        __doc__: Any = "```('Applies a 2D convolution over an input signal composed of several input\\n    planes.\\n\\n    In the simplest case, the output value of the layer with input size\\n    :math:`(N, C_{\\\\text{in}}, H, W)` and output :math:`(N, C_{\\\\text{out}}, H_{\\\\text{out}}, W_{\\\\text{out}})`\\n    can be precisely described as:\\n\\n    .. math::\\n        \\\\text{out}(N_i, C_{\\\\text{out}_j}) = \\\\text{bias}(C_{\\\\text{out}_j}) +\\n        \\\\sum_{k = 0}^{C_{\\\\text{in}} - 1} \\\\text{weight}(C_{\\\\text{out}_j}, k) \\\\star \\\\text{input}(N_i, k)\\n\\n\\n    where :math:`\\\\star` is the valid 2D `cross-correlation`_ operator,\\n    :math:`N` is a batch size, :math:`C` denotes a number of channels,\\n    :math:`H` is a height of input planes in pixels, and :math:`W` is\\n    width in pixels.\\n    ' + \"\\n\\n    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.\\n\\n    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.\\n\\n    * :attr:`stride` controls the stride for the cross-correlation, a single\\n      number or a tuple.\\n\\n    * :attr:`padding` controls the amount of padding applied to the input. It\\n      can be either a string {{'valid', 'same'}} or an int / a tuple of ints giving the\\n      amount of implicit padding applied on both sides.\\n\\n    * :attr:`dilation` controls the spacing between the kernel points; also\\n      known as the à trous algorithm. It is harder to describe, but this `link`_\\n      has a nice visualization of what :attr:`dilation` does.\\n\\n\\n    {groups_note}\\n\\n    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:\\n\\n        - a single ``int`` -- in which case the same value is used for the height and width dimension\\n        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,\\n          and the second `int` for the width dimension\\n\\n    Note:\\n        {depthwise_separable_note}\\n\\n    Note:\\n        {cudnn_reproducibility_note}\\n\\n    Note:\\n        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads\\n        the input so the output has the shape as the input. However, this mode\\n        doesn't support any stride values other than 1.\\n\\n    Note:\\n        This module supports complex data types i.e. ``complex32, complex64, complex128``.\\n\\n    Args:\\n        in_channels (int): Number of channels in the input image\\n        out_channels (int): Number of channels produced by the convolution\\n        kernel_size (int or tuple): Size of the convolving kernel\\n        stride (int or tuple, Any): Stride of the convolution. Default: 1\\n        padding (int, tuple or str, Any): Padding added to all four sides of\\n            the input. Default: 0\\n        dilation (int or tuple, Any): Spacing between kernel elements. Default: 1\\n        groups (int, Any): Number of blocked connections from input\\n            channels to output channels. Default: 1\\n        bias (bool, Any): If ``True``, adds a learnable bias to the\\n            output. Default: ``True``\\n        padding_mode (str, Any): ``'zeros'``, ``'reflect'``,\\n            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``\\n    \".format(**reproducibility_notes, **convolution_notes) + '\\n\\n    Shape:\\n        - Input: :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in})`\\n        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`, where\\n\\n          .. math::\\n              H_{out} = \\\\left\\\\lfloor\\\\frac{H_{in}  + 2 \\\\times \\\\text{padding}[0] - \\\\text{dilation}[0]\\n                        \\\\times (\\\\text{kernel\\\\Any}[0] - 1) - 1}{\\\\text{stride}[0]} + 1\\\\right\\\\rfloor\\n\\n          .. math::\\n              W_{out} = \\\\left\\\\lfloor\\\\frac{W_{in}  + 2 \\\\times \\\\text{padding}[1] - \\\\text{dilation}[1]\\n                        \\\\times (\\\\text{kernel\\\\Any}[1] - 1) - 1}{\\\\text{stride}[1]} + 1\\\\right\\\\rfloor\\n\\n    Attributes:\\n        weight (Tensor): the learnable weights of the module of shape\\n            :math:`(\\\\text{out\\\\_channels}, \\\\frac{\\\\text{in\\\\_channels}}{\\\\text{groups}},`\\n            :math:`\\\\text{kernel\\\\Any[0]}, \\\\text{kernel\\\\Any[1]})`.\\n            The values of these weights are sampled from\\n            :math:`\\\\mathcal{U}(-\\\\sqrt{k}, \\\\sqrt{k})` where\\n            :math:`k = \\\\frac{groups}{C_\\\\text{in} * \\\\prod_{i=0}^{1}\\\\text{kernel\\\\Any}[i]}`\\n        bias (Tensor):   the learnable bias of the module of shape\\n            (out_channels). If :attr:`bias` is ``True``,\\n            then the values of these weights are\\n            sampled from :math:`\\\\mathcal{U}(-\\\\sqrt{k}, \\\\sqrt{k})` where\\n            :math:`k = \\\\frac{groups}{C_\\\\text{in} * \\\\prod_{i=0}^{1}\\\\text{kernel\\\\Any}[i]}`\\n\\n    Examples:\\n\\n        >>> # With square kernels and equal stride\\n        >>> m = nn.Conv2d(16, 33, 3, stride=2)\\n        >>> # non-square kernels and unequal stride and with padding\\n        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))\\n        >>> # non-square kernels and unequal stride and with padding and dilation\\n        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))\\n        >>> input = torch.randn(20, 16, 50, 100)\\n        >>> output = m(input)\\n\\n    .. _cross-correlation:\\n        https://en.wikipedia.org/wiki/Cross-correlation\\n\\n    .. _link:\\n        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md\\n    ')```",
        device: Any = "```(None)```",
        padding: str | Any = 0,
        out_channels: int = ...,
        groups: int = 1,
        dilation: Any = 1,
        in_channels: int = ...,
        bias: bool = True,
        padding_mode: str = "zeros",
        dtype: Any = "```(None)```",
        kernel_size: Any = ...,
        stride: Any = 1,
        transposed: bool = ...,
        output_padding: tuple = ...,
    ) -> None: ...

class Conv3d:
    def __init__(
        self,
        __doc__: Any = "```('Applies a 3D convolution over an input signal composed of several input\\n    planes.\\n\\n    In the simplest case, the output value of the layer with input size :math:`(N, C_{in}, D, H, W)`\\n    and output :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` can be precisely described as:\\n\\n    .. math::\\n        out(N_i, C_{out_j}) = bias(C_{out_j}) +\\n                                \\\\sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k) \\\\star input(N_i, k)\\n\\n    where :math:`\\\\star` is the valid 3D `cross-correlation`_ operator\\n    ' + \"\\n\\n    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.\\n\\n    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.\\n\\n    * :attr:`stride` controls the stride for the cross-correlation.\\n\\n    * :attr:`padding` controls the amount of padding applied to the input. It\\n      can be either a string {{'valid', 'same'}} or a tuple of ints giving the\\n      amount of implicit padding applied on both sides.\\n\\n    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.\\n      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.\\n\\n\\n    {groups_note}\\n\\n    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:\\n\\n        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension\\n        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,\\n          the second `int` for the height dimension and the third `int` for the width dimension\\n\\n    Note:\\n        {depthwise_separable_note}\\n\\n    Note:\\n        {cudnn_reproducibility_note}\\n\\n    Note:\\n        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads\\n        the input so the output has the shape as the input. However, this mode\\n        doesn't support any stride values other than 1.\\n\\n    Note:\\n        This module supports complex data types i.e. ``complex32, complex64, complex128``.\\n\\n    Args:\\n        in_channels (int): Number of channels in the input image\\n        out_channels (int): Number of channels produced by the convolution\\n        kernel_size (int or tuple): Size of the convolving kernel\\n        stride (int or tuple, Any): Stride of the convolution. Default: 1\\n        padding (int, tuple or str, Any): Padding added to all six sides of\\n            the input. Default: 0\\n        dilation (int or tuple, Any): Spacing between kernel elements. Default: 1\\n        groups (int, Any): Number of blocked connections from input channels to output channels. Default: 1\\n        bias (bool, Any): If ``True``, adds a learnable bias to the output. Default: ``True``\\n        padding_mode (str, Any): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``\\n    \".format(**reproducibility_notes, **convolution_notes) + '\\n\\n    Shape:\\n        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` or :math:`(C_{in}, D_{in}, H_{in}, W_{in})`\\n        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` or :math:`(C_{out}, D_{out}, H_{out}, W_{out})`,\\n          where\\n\\n          .. math::\\n              D_{out} = \\\\left\\\\lfloor\\\\frac{D_{in} + 2 \\\\times \\\\text{padding}[0] - \\\\text{dilation}[0]\\n                    \\\\times (\\\\text{kernel\\\\Any}[0] - 1) - 1}{\\\\text{stride}[0]} + 1\\\\right\\\\rfloor\\n\\n          .. math::\\n              H_{out} = \\\\left\\\\lfloor\\\\frac{H_{in} + 2 \\\\times \\\\text{padding}[1] - \\\\text{dilation}[1]\\n                    \\\\times (\\\\text{kernel\\\\Any}[1] - 1) - 1}{\\\\text{stride}[1]} + 1\\\\right\\\\rfloor\\n\\n          .. math::\\n              W_{out} = \\\\left\\\\lfloor\\\\frac{W_{in} + 2 \\\\times \\\\text{padding}[2] - \\\\text{dilation}[2]\\n                    \\\\times (\\\\text{kernel\\\\Any}[2] - 1) - 1}{\\\\text{stride}[2]} + 1\\\\right\\\\rfloor\\n\\n    Attributes:\\n        weight (Tensor): the learnable weights of the module of shape\\n                         :math:`(\\\\text{out\\\\_channels}, \\\\frac{\\\\text{in\\\\_channels}}{\\\\text{groups}},`\\n                         :math:`\\\\text{kernel\\\\Any[0]}, \\\\text{kernel\\\\Any[1]}, \\\\text{kernel\\\\Any[2]})`.\\n                         The values of these weights are sampled from\\n                         :math:`\\\\mathcal{U}(-\\\\sqrt{k}, \\\\sqrt{k})` where\\n                         :math:`k = \\\\frac{groups}{C_\\\\text{in} * \\\\prod_{i=0}^{2}\\\\text{kernel\\\\Any}[i]}`\\n        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,\\n                         then the values of these weights are\\n                         sampled from :math:`\\\\mathcal{U}(-\\\\sqrt{k}, \\\\sqrt{k})` where\\n                         :math:`k = \\\\frac{groups}{C_\\\\text{in} * \\\\prod_{i=0}^{2}\\\\text{kernel\\\\Any}[i]}`\\n\\n    Examples::\\n\\n        >>> # With square kernels and equal stride\\n        >>> m = nn.Conv3d(16, 33, 3, stride=2)\\n        >>> # non-square kernels and unequal stride and with padding\\n        >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))\\n        >>> input = torch.randn(20, 16, 10, 50, 100)\\n        >>> output = m(input)\\n\\n    .. _cross-correlation:\\n        https://en.wikipedia.org/wiki/Cross-correlation\\n\\n    .. _link:\\n        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md\\n    ')```",
        device: Any = "```(None)```",
        padding: str | Any = 0,
        out_channels: int = ...,
        groups: int = 1,
        dilation: Any = 1,
        in_channels: int = ...,
        bias: bool = True,
        padding_mode: str = "zeros",
        dtype: Any = "```(None)```",
        kernel_size: Any = ...,
        stride: Any = 1,
        transposed: bool = ...,
        output_padding: tuple = ...,
    ) -> None: ...

class ConvTranspose1d:
    def __init__(
        self,
        __doc__: Any = "```('Applies a 1D transposed convolution operator over an input image\\n    composed of several input planes.\\n\\n    This module can be seen as the gradient of Conv1d with respect to its input.\\n    It is also known as a fractionally-strided convolution or\\n    a deconvolution (although it is not an actual deconvolution operation as it does\\n    not compute a true inverse of convolution). For more information, see the visualizations\\n    `here`_ and the `Deconvolutional Networks`_ paper.\\n\\n    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.\\n\\n    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.\\n\\n    * :attr:`stride` controls the stride for the cross-correlation.\\n\\n    * :attr:`padding` controls the amount of implicit zero padding on both\\n      sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note\\n      below for details.\\n\\n    * :attr:`output_padding` controls the additional size added to one side\\n      of the output shape. See note below for details.\\n\\n    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.\\n      It is harder to describe, but the link `here`_ has a nice visualization of what :attr:`dilation` does.\\n\\n    {groups_note}\\n\\n    Note:\\n        The :attr:`padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``\\n        amount of zero padding to both sizes of the input. This is set so that\\n        when a :class:`~torch.nn.Conv1d` and a :class:`~torch.nn.ConvTranspose1d`\\n        are initialized with same parameters, they are inverses of each other in\\n        regard to the input and output shapes. However, when ``stride > 1``,\\n        :class:`~torch.nn.Conv1d` maps multiple input shapes to the same output\\n        shape. :attr:`output_padding` is provided to resolve this ambiguity by\\n        effectively increasing the calculated output shape on one side. Note\\n        that :attr:`output_padding` is only used to find output shape, but does\\n        not actually add zero-padding to output.\\n\\n    Note:\\n        In some circumstances when using the CUDA backend with CuDNN, this operator\\n        may select a nondeterministic algorithm to increase performance. If this is\\n        undesirable, you can try to make the operation deterministic (potentially at\\n        a performance cost) by setting ``torch.backends.cudnn.deterministic =\\n        True``.\\n        Please see the notes on :doc:`/notes/randomness` for background.\\n\\n\\n    Args:\\n        in_channels (int): Number of channels in the input image\\n        out_channels (int): Number of channels produced by the convolution\\n        kernel_size (int or tuple): Size of the convolving kernel\\n        stride (int or tuple, Any): Stride of the convolution. Default: 1\\n        padding (int or tuple, Any): ``dilation * (kernel_size - 1) - padding`` zero-padding\\n            will be added to both sides of the input. Default: 0\\n        output_padding (int or tuple, Any): Additional size added to one side\\n            of the output shape. Default: 0\\n        groups (int, Any): Number of blocked connections from input channels to output channels. Default: 1\\n        bias (bool, Any): If ``True``, adds a learnable bias to the output. Default: ``True``\\n        dilation (int or tuple, Any): Spacing between kernel elements. Default: 1\\n    '.format(**reproducibility_notes, **convolution_notes) + '\\n\\n    Shape:\\n        - Input: :math:`(N, C_{in}, L_{in})` or :math:`(C_{in}, L_{in})`\\n        - Output: :math:`(N, C_{out}, L_{out})` or :math:`(C_{out}, L_{out})`, where\\n\\n          .. math::\\n              L_{out} = (L_{in} - 1) \\\\times \\\\text{stride} - 2 \\\\times \\\\text{padding} + \\\\text{dilation}\\n                        \\\\times (\\\\text{kernel\\\\Any} - 1) + \\\\text{output\\\\_padding} + 1\\n\\n    Attributes:\\n        weight (Tensor): the learnable weights of the module of shape\\n                         :math:`(\\\\text{in\\\\_channels}, \\\\frac{\\\\text{out\\\\_channels}}{\\\\text{groups}},`\\n                         :math:`\\\\text{kernel\\\\Any})`.\\n                         The values of these weights are sampled from\\n                         :math:`\\\\mathcal{U}(-\\\\sqrt{k}, \\\\sqrt{k})` where\\n                         :math:`k = \\\\frac{groups}{C_\\\\text{out} * \\\\text{kernel\\\\Any}}`\\n        bias (Tensor):   the learnable bias of the module of shape (out_channels).\\n                         If :attr:`bias` is ``True``, then the values of these weights are\\n                         sampled from :math:`\\\\mathcal{U}(-\\\\sqrt{k}, \\\\sqrt{k})` where\\n                         :math:`k = \\\\frac{groups}{C_\\\\text{out} * \\\\text{kernel\\\\Any}}`\\n\\n    .. _`here`:\\n        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md\\n\\n    .. _`Deconvolutional Networks`:\\n        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf\\n    ')```",
        output_padding: Any = 0,
        device: Any = "```(None)```",
        padding: Any = 0,
        out_channels: int = ...,
        groups: int = 1,
        dilation: Any = 1,
        in_channels: int = ...,
        bias: bool = True,
        padding_mode: str = "zeros",
        dtype: Any = "```(None)```",
        kernel_size: Any = ...,
        stride: Any = 1,
        transposed: Any = ...,
    ) -> None: ...

class ConvTranspose2d:
    def __init__(
        self,
        __doc__: Any = "```('Applies a 2D transposed convolution operator over an input image\\n    composed of several input planes.\\n\\n    This module can be seen as the gradient of Conv2d with respect to its input.\\n    It is also known as a fractionally-strided convolution or\\n    a deconvolution (although it is not an actual deconvolution operation as it does\\n    not compute a true inverse of convolution). For more information, see the visualizations\\n    `here`_ and the `Deconvolutional Networks`_ paper.\\n\\n    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.\\n\\n    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.\\n\\n    * :attr:`stride` controls the stride for the cross-correlation. When stride > 1, ConvTranspose2d inserts zeros between input\\n      elements along the spatial dimensions before applying the convolution kernel. This zero-insertion operation is the standard\\n      behavior of transposed convolutions, which can increase the spatial resolution and is equivalent to a learnable\\n      upsampling operation.\\n\\n    * :attr:`padding` controls the amount of implicit zero padding on both\\n      sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note\\n      below for details.\\n\\n    * :attr:`output_padding` controls the additional size added to one side\\n      of the output shape. See note below for details.\\n\\n    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.\\n      It is harder to describe, but the link `here`_ has a nice visualization of what :attr:`dilation` does.\\n\\n    {groups_note}\\n\\n    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`\\n    can either be:\\n\\n        - a single ``int`` -- in which case the same value is used for the height and width dimensions\\n        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,\\n          and the second `int` for the width dimension\\n\\n    Note:\\n        The :attr:`padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``\\n        amount of zero padding to both sizes of the input. This is set so that\\n        when a :class:`~torch.nn.Conv2d` and a :class:`~torch.nn.ConvTranspose2d`\\n        are initialized with same parameters, they are inverses of each other in\\n        regard to the input and output shapes. However, when ``stride > 1``,\\n        :class:`~torch.nn.Conv2d` maps multiple input shapes to the same output\\n        shape. :attr:`output_padding` is provided to resolve this ambiguity by\\n        effectively increasing the calculated output shape on one side. Note\\n        that :attr:`output_padding` is only used to find output shape, but does\\n        not actually add zero-padding to output.\\n\\n    Note:\\n        {cudnn_reproducibility_note}\\n\\n    Args:\\n        in_channels (int): Number of channels in the input image\\n        out_channels (int): Number of channels produced by the convolution\\n        kernel_size (int or tuple): Size of the convolving kernel\\n        stride (int or tuple, Any): Stride of the convolution. Default: 1\\n        padding (int or tuple, Any): ``dilation * (kernel_size - 1) - padding`` zero-padding\\n            will be added to both sides of each dimension in the input. Default: 0\\n        output_padding (int or tuple, Any): Additional size added to one side\\n            of each dimension in the output shape. Default: 0\\n        groups (int, Any): Number of blocked connections from input channels to output channels. Default: 1\\n        bias (bool, Any): If ``True``, adds a learnable bias to the output. Default: ``True``\\n        dilation (int or tuple, Any): Spacing between kernel elements. Default: 1\\n    '.format(**reproducibility_notes, **convolution_notes) + '\\n\\n    Shape:\\n        - Input: :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in})`\\n        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`, where\\n\\n        .. math::\\n              H_{out} = (H_{in} - 1) \\\\times \\\\text{stride}[0] - 2 \\\\times \\\\text{padding}[0] + \\\\text{dilation}[0]\\n                        \\\\times (\\\\text{kernel\\\\Any}[0] - 1) + \\\\text{output\\\\_padding}[0] + 1\\n        .. math::\\n              W_{out} = (W_{in} - 1) \\\\times \\\\text{stride}[1] - 2 \\\\times \\\\text{padding}[1] + \\\\text{dilation}[1]\\n                        \\\\times (\\\\text{kernel\\\\Any}[1] - 1) + \\\\text{output\\\\_padding}[1] + 1\\n\\n    Attributes:\\n        weight (Tensor): the learnable weights of the module of shape\\n                         :math:`(\\\\text{in\\\\_channels}, \\\\frac{\\\\text{out\\\\_channels}}{\\\\text{groups}},`\\n                         :math:`\\\\text{kernel\\\\Any[0]}, \\\\text{kernel\\\\Any[1]})`.\\n                         The values of these weights are sampled from\\n                         :math:`\\\\mathcal{U}(-\\\\sqrt{k}, \\\\sqrt{k})` where\\n                         :math:`k = \\\\frac{groups}{C_\\\\text{out} * \\\\prod_{i=0}^{1}\\\\text{kernel\\\\Any}[i]}`\\n        bias (Tensor):   the learnable bias of the module of shape (out_channels)\\n                         If :attr:`bias` is ``True``, then the values of these weights are\\n                         sampled from :math:`\\\\mathcal{U}(-\\\\sqrt{k}, \\\\sqrt{k})` where\\n                         :math:`k = \\\\frac{groups}{C_\\\\text{out} * \\\\prod_{i=0}^{1}\\\\text{kernel\\\\Any}[i]}`\\n\\n    Examples::\\n\\n        >>> # With square kernels and equal stride\\n        >>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)\\n        >>> # non-square kernels and unequal stride and with padding\\n        >>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))\\n        >>> input = torch.randn(20, 16, 50, 100)\\n        >>> output = m(input)\\n        >>> # exact output size can be also specified as an argument\\n        >>> input = torch.randn(1, 16, 12, 12)\\n        >>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)\\n        >>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)\\n        >>> h = downsample(input)\\n        >>> h.size()\\n        torch.Size([1, 16, 6, 6])\\n        >>> output = upsample(h, output_size=input.size())\\n        >>> output.size()\\n        torch.Size([1, 16, 12, 12])\\n\\n    .. _`here`:\\n        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md\\n\\n    .. _`Deconvolutional Networks`:\\n        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf\\n    ')```",
        output_padding: Any = 0,
        device: Any = "```(None)```",
        padding: Any = 0,
        out_channels: int = ...,
        groups: int = 1,
        dilation: Any = 1,
        in_channels: int = ...,
        bias: bool = True,
        padding_mode: str = "zeros",
        dtype: Any = "```(None)```",
        kernel_size: Any = ...,
        stride: Any = 1,
        transposed: Any = ...,
    ) -> None: ...

class ConvTranspose3d:
    def __init__(
        self,
        __doc__: Any = "```('Applies a 3D transposed convolution operator over an input image composed of several input\\n    planes.\\n    The transposed convolution operator multiplies each input value element-wise by a learnable kernel,\\n    and sums over the outputs from all input feature planes.\\n\\n    This module can be seen as the gradient of Conv3d with respect to its input.\\n    It is also known as a fractionally-strided convolution or\\n    a deconvolution (although it is not an actual deconvolution operation as it does\\n    not compute a true inverse of convolution). For more information, see the visualizations\\n    `here`_ and the `Deconvolutional Networks`_ paper.\\n\\n    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.\\n\\n    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.\\n\\n    * :attr:`stride` controls the stride for the cross-correlation.\\n\\n    * :attr:`padding` controls the amount of implicit zero padding on both\\n      sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note\\n      below for details.\\n\\n    * :attr:`output_padding` controls the additional size added to one side\\n      of the output shape. See note below for details.\\n\\n    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.\\n      It is harder to describe, but the link `here`_ has a nice visualization of what :attr:`dilation` does.\\n\\n    {groups_note}\\n\\n    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`\\n    can either be:\\n\\n        - a single ``int`` -- in which case the same value is used for the depth, height and width dimensions\\n        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,\\n          the second `int` for the height dimension and the third `int` for the width dimension\\n\\n    Note:\\n        The :attr:`padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``\\n        amount of zero padding to both sizes of the input. This is set so that\\n        when a :class:`~torch.nn.Conv3d` and a :class:`~torch.nn.ConvTranspose3d`\\n        are initialized with same parameters, they are inverses of each other in\\n        regard to the input and output shapes. However, when ``stride > 1``,\\n        :class:`~torch.nn.Conv3d` maps multiple input shapes to the same output\\n        shape. :attr:`output_padding` is provided to resolve this ambiguity by\\n        effectively increasing the calculated output shape on one side. Note\\n        that :attr:`output_padding` is only used to find output shape, but does\\n        not actually add zero-padding to output.\\n\\n    Note:\\n        {cudnn_reproducibility_note}\\n\\n    Args:\\n        in_channels (int): Number of channels in the input image\\n        out_channels (int): Number of channels produced by the convolution\\n        kernel_size (int or tuple): Size of the convolving kernel\\n        stride (int or tuple, Any): Stride of the convolution. Default: 1\\n        padding (int or tuple, Any): ``dilation * (kernel_size - 1) - padding`` zero-padding\\n            will be added to both sides of each dimension in the input. Default: 0\\n        output_padding (int or tuple, Any): Additional size added to one side\\n            of each dimension in the output shape. Default: 0\\n        groups (int, Any): Number of blocked connections from input channels to output channels. Default: 1\\n        bias (bool, Any): If ``True``, adds a learnable bias to the output. Default: ``True``\\n        dilation (int or tuple, Any): Spacing between kernel elements. Default: 1\\n    '.format(**reproducibility_notes, **convolution_notes) + '\\n\\n    Shape:\\n        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` or :math:`(C_{in}, D_{in}, H_{in}, W_{in})`\\n        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` or\\n          :math:`(C_{out}, D_{out}, H_{out}, W_{out})`, where\\n\\n        .. math::\\n              D_{out} = (D_{in} - 1) \\\\times \\\\text{stride}[0] - 2 \\\\times \\\\text{padding}[0] + \\\\text{dilation}[0]\\n                        \\\\times (\\\\text{kernel\\\\Any}[0] - 1) + \\\\text{output\\\\_padding}[0] + 1\\n        .. math::\\n              H_{out} = (H_{in} - 1) \\\\times \\\\text{stride}[1] - 2 \\\\times \\\\text{padding}[1] + \\\\text{dilation}[1]\\n                        \\\\times (\\\\text{kernel\\\\Any}[1] - 1) + \\\\text{output\\\\_padding}[1] + 1\\n        .. math::\\n              W_{out} = (W_{in} - 1) \\\\times \\\\text{stride}[2] - 2 \\\\times \\\\text{padding}[2] + \\\\text{dilation}[2]\\n                        \\\\times (\\\\text{kernel\\\\Any}[2] - 1) + \\\\text{output\\\\_padding}[2] + 1\\n\\n\\n    Attributes:\\n        weight (Tensor): the learnable weights of the module of shape\\n                         :math:`(\\\\text{in\\\\_channels}, \\\\frac{\\\\text{out\\\\_channels}}{\\\\text{groups}},`\\n                         :math:`\\\\text{kernel\\\\Any[0]}, \\\\text{kernel\\\\Any[1]}, \\\\text{kernel\\\\Any[2]})`.\\n                         The values of these weights are sampled from\\n                         :math:`\\\\mathcal{U}(-\\\\sqrt{k}, \\\\sqrt{k})` where\\n                         :math:`k = \\\\frac{groups}{C_\\\\text{out} * \\\\prod_{i=0}^{2}\\\\text{kernel\\\\Any}[i]}`\\n        bias (Tensor):   the learnable bias of the module of shape (out_channels)\\n                         If :attr:`bias` is ``True``, then the values of these weights are\\n                         sampled from :math:`\\\\mathcal{U}(-\\\\sqrt{k}, \\\\sqrt{k})` where\\n                         :math:`k = \\\\frac{groups}{C_\\\\text{out} * \\\\prod_{i=0}^{2}\\\\text{kernel\\\\Any}[i]}`\\n\\n    Examples::\\n\\n        >>> # With square kernels and equal stride\\n        >>> m = nn.ConvTranspose3d(16, 33, 3, stride=2)\\n        >>> # non-square kernels and unequal stride and with padding\\n        >>> m = nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))\\n        >>> input = torch.randn(20, 16, 10, 50, 100)\\n        >>> output = m(input)\\n\\n    .. _`here`:\\n        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md\\n\\n    .. _`Deconvolutional Networks`:\\n        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf\\n    ')```",
        output_padding: Any = 0,
        device: Any = "```(None)```",
        padding: Any = 0,
        out_channels: int = ...,
        groups: int = 1,
        dilation: Any = 1,
        in_channels: int = ...,
        bias: bool = True,
        padding_mode: str = "zeros",
        dtype: Any = "```(None)```",
        kernel_size: Any = ...,
        stride: Any = 1,
        transposed: Any = ...,
    ) -> None: ...

class CosineSimilarity:
    def __init__(
        self, dim: int = 1, eps: float = 1e-08, __constants__: list = ["dim", "eps"]
    ) -> None: ...

class CrossMapLRN2d:
    def __init__(
        self, size: int = ..., alpha: float = 0.0001, beta: float = 0.75, k: float = 1
    ) -> None: ...

class Dropout:
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...

class Dropout1d:
    def __init__(self, p: float | None = ..., inplace: bool | None = ...) -> None: ...

class Dropout2d:
    def __init__(self, p: float | None = ..., inplace: bool | None = ...) -> None: ...

class Dropout3d:
    def __init__(self, p: float | None = ..., inplace: bool | None = ...) -> None: ...

class ELU:
    def __init__(
        self,
        alpha: float = 1.0,
        inplace: bool = False,
        __constants__: list = ["alpha", "inplace"],
    ) -> None: ...

class Embedding:
    def __init__(
        self,
        num_embeddings: int = ...,
        embedding_dim: int = ...,
        padding_idx: int | None = "```(None)```",
        max_norm: float | None = "```(None)```",
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        __constants__: list = [
            "num_embeddings",
            "embedding_dim",
            "padding_idx",
            "max_norm",
            "norm_type",
            "scale_grad_by_freq",
            "sparse",
        ],
        weight: Tensor = ...,
        freeze: bool = ...,
        device: Any = "```(None)```",
        _freeze: bool = False,
        dtype: Any = "```(None)```",
        _weight: Tensor | None = "```(None)```",
    ) -> None: ...

class EmbeddingBag:
    def __init__(
        self,
        num_embeddings: int = ...,
        embedding_dim: int = ...,
        max_norm: float | None = "```(None)```",
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        include_last_offset: bool = False,
        padding_idx: int | None = "```(None)```",
        __constants__: list = [
            "num_embeddings",
            "embedding_dim",
            "max_norm",
            "norm_type",
            "scale_grad_by_freq",
            "mode",
            "sparse",
            "include_last_offset",
            "padding_idx",
        ],
        weight: Tensor = ...,
        device: Any = "```(None)```",
        _weight: Tensor | None = "```(None)```",
        dtype: Any = "```(None)```",
    ) -> None: ...

class FeatureAlphaDropout:
    def __init__(
        self, p: float | None = 0.5, inplace: bool | None = "```(None)```"
    ) -> None: ...

class Flatten:
    def __init__(
        self,
        start_dim: int = 1,
        end_dim: int = -1,
        __constants__: list = ["start_dim", "end_dim"],
    ) -> None: ...

class Fold:
    def __init__(
        self,
        output_size: Any = ...,
        kernel_size: Any = ...,
        dilation: Any = 1,
        padding: Any = 0,
        stride: Any = 1,
        __constants__: list = [
            "output_size",
            "kernel_size",
            "dilation",
            "padding",
            "stride",
        ],
    ) -> None: ...

class FractionalMaxPool2d:
    def __init__(
        self,
        kernel_size: Any = ...,
        output_size: Any = "```(None)```",
        output_ratio: Any = "```(None)```",
        return_indices: bool = False,
        __constants__: list = [
            "kernel_size",
            "return_indices",
            "output_size",
            "output_ratio",
        ],
        _random_samples: Any = "```(None)```",
    ) -> None: ...

class FractionalMaxPool3d:
    def __init__(
        self,
        kernel_size: Any = ...,
        output_size: Any = "```(None)```",
        output_ratio: Any = "```(None)```",
        return_indices: bool = False,
        __constants__: list = [
            "kernel_size",
            "return_indices",
            "output_size",
            "output_ratio",
        ],
        _random_samples: Any = "```(None)```",
    ) -> None: ...

class GLU:
    def __init__(self, dim: int = -1, __constants__: list = ["dim"]) -> None: ...

class GRU:
    def __init__(
        self,
        input_size: int = ...,
        hidden_size: int = ...,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: int = 0,
        bidirectional: bool = False,
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
        mode: str = ...,
        proj_size: int = 0,
    ) -> Any: ...

class GRUCell:
    def __init__(
        self,
        input_size: int = ...,
        hidden_size: int = ...,
        bias: bool = True,
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
        num_chunks: int = ...,
    ) -> None: ...

class GroupNorm:
    def __init__(
        self,
        num_groups: int = ...,
        num_channels: int = ...,
        eps: float = 1e-05,
        affine: bool = True,
        __constants__: list = ["num_groups", "num_channels", "eps", "affine"],
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
    ) -> None: ...

class Hardshrink:
    def __init__(self, lambd: float = 0.5, __constants__: list = ["lambd"]) -> None: ...

class Hardsigmoid:
    def __init__(
        self, inplace: bool = False, __constants__: list = ["inplace"]
    ) -> None: ...

class Hardswish:
    def __init__(
        self, inplace: bool = False, __constants__: list = ["inplace"]
    ) -> None: ...

class Hardtanh:
    def __init__(
        self,
        min_val: float = -1.0,
        max_val: float = 1,
        inplace: bool = False,
        __constants__: list = ["min_val", "max_val", "inplace"],
        min_value: float | None = "```(None)```",
        max_value: float | None = "```(None)```",
    ) -> None: ...

class Identity:
    def __init__(self, *args, **kwargs) -> None: ...

class InstanceNorm1d:
    def __init__(
        self,
        num_features: int = ...,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...

class InstanceNorm2d:
    def __init__(
        self,
        num_features: int = ...,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...

class InstanceNorm3d:
    def __init__(
        self,
        num_features: int = ...,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...

class LPPool1d:
    def __init__(
        self,
        kernel_size: Any = ...,
        stride: Any = ":attr:`kernel_size",
        ceil_mode: bool = "```(None)```",
        norm_type: float = ...,
    ) -> None: ...

class LPPool2d:
    def __init__(
        self,
        kernel_size: Any = ...,
        stride: Any = ":attr:`kernel_size",
        ceil_mode: bool = "```(None)```",
        norm_type: float = ...,
    ) -> None: ...

class LPPool3d:
    def __init__(
        self,
        kernel_size: Any = ...,
        stride: Any = ":attr:`kernel_size",
        ceil_mode: bool = "```(None)```",
        norm_type: float = ...,
    ) -> None: ...

class LSTM:
    def __init__(
        self,
        input_size: int = ...,
        hidden_size: int = ...,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: int = 0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
        mode: str = ...,
    ) -> Any: ...

class LSTMCell:
    def __init__(
        self,
        input_size: int = ...,
        hidden_size: int = ...,
        bias: bool = True,
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
        num_chunks: int = ...,
    ) -> None: ...

class LayerNorm:
    def __init__(
        self,
        normalized_shape: tuple[int, ...] = ...,
        __constants__: list = ["normalized_shape", "eps", "elementwise_affine"],
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        device: Any = "```(None)```",
        bias: bool = True,
        dtype: Any = "```(None)```",
    ) -> None: ...

class LazyBatchNorm1d:
    def __init__(
        self,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        cls_to_become: str = "BatchNorm1d",
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...

class LazyBatchNorm2d:
    def __init__(
        self,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        cls_to_become: str = "BatchNorm2d",
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...

class LazyBatchNorm3d:
    def __init__(
        self,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        cls_to_become: str = "BatchNorm3d",
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...

class LazyConv1d:
    def __init__(
        self,
        out_channels: int = ...,
        kernel_size: int | tuple = ...,
        stride: int | tuple | Any = 1,
        padding: int | tuple | Any = 0,
        dilation: int | tuple | Any = 1,
        groups: int | None = 1,
        bias: bool | None = True,
        padding_mode: str | None = "zeros",
        cls_to_become: str = "Conv1d",
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
        in_channels: int = ...,
        transposed: bool = ...,
        output_padding: tuple = ...,
    ) -> None: ...

class LazyConv2d:
    def __init__(
        self,
        out_channels: int = ...,
        kernel_size: int | tuple = ...,
        stride: int | tuple | Any = 1,
        padding: int | tuple | Any = 0,
        dilation: int | tuple | Any = 1,
        groups: int | None = 1,
        bias: bool | None = True,
        padding_mode: str | None = "zeros",
        cls_to_become: str = "Conv2d",
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
        in_channels: int = ...,
        transposed: bool = ...,
        output_padding: tuple = ...,
    ) -> None: ...

class LazyConv3d:
    def __init__(
        self,
        out_channels: int = ...,
        kernel_size: int | tuple = ...,
        stride: int | tuple | Any = 1,
        padding: int | tuple | Any = 0,
        dilation: int | tuple | Any = 1,
        groups: int | None = 1,
        bias: bool | None = True,
        padding_mode: str | None = "zeros",
        cls_to_become: str = "Conv3d",
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
        in_channels: int = ...,
        transposed: bool = ...,
        output_padding: tuple = ...,
    ) -> None: ...

class LazyConvTranspose1d:
    def __init__(
        self,
        out_channels: int = ...,
        kernel_size: int | tuple = ...,
        stride: int | tuple | Any = 1,
        padding: int | tuple | Any = 0,
        output_padding: int | tuple | Any = 0,
        groups: int | None = 1,
        bias: bool | None = True,
        dilation: int | tuple | Any = 1,
        cls_to_become: str = "ConvTranspose1d",
        dtype: Any = "```(None)```",
        device: Any = "```(None)```",
        padding_mode: str = "zeros",
        in_channels: int = ...,
        transposed: Any = ...,
    ) -> None: ...

class LazyConvTranspose2d:
    def __init__(
        self,
        out_channels: int = ...,
        kernel_size: int | tuple = ...,
        stride: int | tuple | Any = 1,
        padding: int | tuple | Any = 0,
        output_padding: int | tuple | Any = 0,
        groups: int | None = 1,
        bias: bool | None = True,
        dilation: int | tuple | Any = 1,
        cls_to_become: str = "ConvTranspose2d",
        dtype: Any = "```(None)```",
        device: Any = "```(None)```",
        padding_mode: str = "zeros",
        in_channels: int = ...,
        transposed: Any = ...,
    ) -> None: ...

class LazyConvTranspose3d:
    def __init__(
        self,
        out_channels: int = ...,
        kernel_size: int | tuple = ...,
        stride: int | tuple | Any = 1,
        padding: int | tuple | Any = 0,
        output_padding: int | tuple | Any = 0,
        groups: int | None = 1,
        bias: bool | None = True,
        dilation: int | tuple | Any = 1,
        cls_to_become: str = "ConvTranspose3d",
        dtype: Any = "```(None)```",
        device: Any = "```(None)```",
        padding_mode: str = "zeros",
        in_channels: int = ...,
        transposed: Any = ...,
    ) -> None: ...

class LazyInstanceNorm1d:
    def __init__(
        self,
        num_features: Any = ...,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        cls_to_become: str = "InstanceNorm1d",
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...

class LazyInstanceNorm2d:
    def __init__(
        self,
        num_features: Any = ...,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        cls_to_become: str = "InstanceNorm2d",
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...

class LazyInstanceNorm3d:
    def __init__(
        self,
        num_features: Any = ...,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        cls_to_become: str = "InstanceNorm3d",
        device: Any = None,
        dtype: Any = None,
    ) -> None: ...

class LazyLinear:
    def __init__(
        self,
        out_features: int = ...,
        bias: Any = True,
        cls_to_become: str = "Linear",
        weight: Any = ...,
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
    ) -> None: ...

class Linear:
    def __init__(
        self,
        in_features: int = ...,
        out_features: int = ...,
        bias: bool = True,
        __constants__: list = ["in_features", "out_features"],
        weight: Tensor = ...,
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
    ) -> None: ...

class LocalResponseNorm:
    def __init__(
        self,
        size: int = ...,
        alpha: float = 0.0001,
        beta: float = 0.75,
        k: float = 1,
        __constants__: list = ["size", "alpha", "beta", "k"],
    ) -> None: ...

class LogSigmoid:
    def __init__(self, *args, **kwargs) -> None: ...

class LogSoftmax:
    def __init__(
        self, dim: int | None = "```(None)```", __constants__: list = ["dim"]
    ) -> None: ...

class MaxPool1d:
    def __init__(
        self,
        kernel_size: Any = ...,
        stride: Any = ":attr:`kernel_size",
        padding: Any = "```(None)```",
        dilation: Any = "```(None)```",
        return_indices: bool = "```(None)```",
        ceil_mode: bool = "```(None)```",
    ) -> None: ...

class MaxPool2d:
    def __init__(
        self,
        kernel_size: Any = ...,
        stride: Any = ":attr:`kernel_size",
        padding: Any = "```(None)```",
        dilation: Any = "```(None)```",
        return_indices: bool = "```(None)```",
        ceil_mode: bool = "```(None)```",
    ) -> None: ...

class MaxPool3d:
    def __init__(
        self,
        kernel_size: Any = ...,
        stride: Any = ":attr:`kernel_size",
        padding: Any = "```(None)```",
        dilation: Any = "```(None)```",
        return_indices: bool = "```(None)```",
        ceil_mode: bool = "```(None)```",
    ) -> None: ...

class MaxUnpool1d:
    def __init__(
        self,
        kernel_size: Any = ...,
        stride: Any = "```(None)```",
        padding: Any = 0,
    ) -> None: ...

class MaxUnpool2d:
    def __init__(
        self,
        kernel_size: Any = ...,
        stride: Any = "```(None)```",
        padding: Any = 0,
    ) -> None: ...

class MaxUnpool3d:
    def __init__(
        self,
        kernel_size: Any = ...,
        stride: Any = "```(None)```",
        padding: Any = 0,
    ) -> None: ...

class Mish:
    def __init__(
        self, __constants__: list = ["inplace"], inplace: bool = False
    ) -> Any: ...

class Module:
    def __init__(
        self,
        training: bool = ...,
        dump_patches: bool = False,
        _version: int = 1,
        _parameters: dict[str, Any | None] = ...,
        _buffers: dict[str, Tensor | None] = ...,
        _non_persistent_buffers_set: set[str] = ...,
        _backward_pre_hooks: dict[int, Callable] = ...,
        _backward_hooks: dict[int, Callable] = ...,
        _is_full_backward_hook: bool | None = ...,
        _forward_hooks: dict[int, Callable] = ...,
        _forward_hooks_with_kwargs: dict[int, bool] = ...,
        _forward_hooks_always_called: dict[int, bool] = ...,
        _forward_pre_hooks: dict[int, Callable] = ...,
        _forward_pre_hooks_with_kwargs: dict[int, bool] = ...,
        _state_dict_hooks: dict[int, Callable] = ...,
        _load_state_dict_pre_hooks: dict[int, Callable] = ...,
        _state_dict_pre_hooks: dict[int, Callable] = ...,
        _load_state_dict_post_hooks: dict[int, Callable] = ...,
        _modules: dict[str, Module | None] = ...,
        call_super_init: bool = False,
        _compiled_call_impl: Callable | None = "```(None)```",
        forward: Callable[..., Any] = "_forward_unimplemented",
        __call__: Callable[..., Any] = "_wrapped_call_impl",
        T_destination: Any = "```(TypeVar('T_destination', bound=dict[str, Any]))```",
        *args,
        **kwargs,
    ) -> None: ...

class ModuleDict:
    def __init__(
        self,
        modules: Iterable | None = "```(None)```",
        _modules: dict[str, Module] = ...,
    ) -> Any: ...

class ModuleList:
    def __init__(
        self,
        modules: Iterable | None = "```(None)```",
        _modules: dict[str, Module] = ...,
    ) -> Any: ...

class MultiheadAttention:
    def __init__(
        self,
        embed_dim: Any = ...,
        num_heads: Any = ...,
        dropout: str = "0.0`` (no dropout).",
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: str = "None`` (uses ``kdim=embed_dim``).",
        vdim: str = "None`` (uses ``vdim=embed_dim``).",
        batch_first: str = "False`` (seq, batch, feature).",
        __constants__: list = ["batch_first"],
        bias_k: torch.Tensor | None = ...,
        bias_v: torch.Tensor | None = ...,
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
    ) -> None: ...

class NLLLoss2d:
    def __init__(
        self,
        weight: Tensor | None = "```(None)```",
        size_average: bool | None = "```(None)```",
        ignore_index: int = -100,
        reduce: bool | None = "```(None)```",
        reduction: str = "mean",
    ) -> None: ...

class PReLU:
    def __init__(
        self,
        num_parameters: int = 1,
        init: float = 0.25,
        __constants__: list = ["num_parameters"],
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
    ) -> None: ...

class PairwiseDistance:
    def __init__(
        self,
        p: Any | None = 2,
        eps: float = 1e-06,
        keepdim: bool = False,
        __constants__: list = ["norm", "eps", "keepdim"],
        norm: float = ...,
    ) -> None: ...

class ParameterDict:
    def __init__(
        self, values: Iterable | None = ..., parameters: Any = "```(None)```"
    ) -> None: ...

class ParameterList:
    def __init__(
        self,
        parameters: Iterable | None = ...,
        values: Iterable[Any] | None = "```(None)```",
    ) -> Any: ...

class PixelShuffle:
    def __init__(
        self, upscale_factor: int = ..., __constants__: list = ["upscale_factor"]
    ) -> None: ...

class PixelUnshuffle:
    def __init__(
        self, downscale_factor: int = ..., __constants__: list = ["downscale_factor"]
    ) -> None: ...

class RMSNorm:
    def __init__(
        self,
        normalized_shape: tuple[int, ...] = ...,
        __constants__: list = ["normalized_shape", "eps", "elementwise_affine"],
        eps: float | None = "```(None)```",
        elementwise_affine: bool = True,
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
    ) -> None: ...

class RNN:
    def __init__(
        self,
        input_size: int = ...,
        hidden_size: int = ...,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: int = 0,
        bidirectional: bool = False,
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
        mode: str = ...,
        proj_size: int = 0,
    ) -> Any: ...

class RNNBase:
    def __init__(
        self,
        __constants__: list = [
            "mode",
            "input_size",
            "hidden_size",
            "num_layers",
            "bias",
            "batch_first",
            "dropout",
            "bidirectional",
            "proj_size",
        ],
        __jit_unused_properties__: list = ["all_weights"],
        mode: str = ...,
        input_size: int = ...,
        hidden_size: int = ...,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
    ) -> None: ...

class RNNCell:
    def __init__(
        self,
        input_size: int = ...,
        hidden_size: int = ...,
        bias: bool = True,
        nonlinearity: str = "tanh",
        __constants__: list = ["input_size", "hidden_size", "bias", "nonlinearity"],
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
        num_chunks: int = ...,
    ) -> None: ...

class RNNCellBase:
    def __init__(
        self,
        __constants__: list = ["input_size", "hidden_size", "bias"],
        input_size: int = ...,
        hidden_size: int = ...,
        bias: bool = ...,
        weight_ih: Tensor = ...,
        weight_hh: Tensor = ...,
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
        num_chunks: int = ...,
    ) -> None: ...

class RReLU:
    def __init__(
        self,
        lower: float = ":math:`\\frac{1}{8}",
        upper: float = ":math:`\\frac{1}{3}",
        inplace: bool = False,
        __constants__: list = ["lower", "upper", "inplace"],
    ) -> Any: ...

class ReLU6:
    def __init__(self, inplace: bool = False) -> Any: ...

class ReflectionPad1d:
    def __init__(self, padding: tuple[int, int] = ...) -> None: ...

class ReflectionPad2d:
    def __init__(self, padding: tuple[int, int, int, int] = ...) -> None: ...

class ReflectionPad3d:
    def __init__(self, padding: tuple[int, int, int, int, int, int] = ...) -> None: ...

class ReplicationPad1d:
    def __init__(self, padding: tuple[int, int] = ...) -> None: ...

class ReplicationPad2d:
    def __init__(self, padding: tuple[int, int, int, int] = ...) -> None: ...

class ReplicationPad3d:
    def __init__(self, padding: tuple[int, int, int, int, int, int] = ...) -> None: ...

class SELU:
    def __init__(
        self, inplace: bool = False, __constants__: list = ["inplace"]
    ) -> None: ...

class Sequential:
    def __init__(self, _modules: dict[str, Module] = ..., *args) -> Any: ...

class Softmax2d:
    def __init__(self, *args, **kwargs) -> None: ...

class Softmin:
    def __init__(
        self, dim: int | None = "```(None)```", __constants__: list = ["dim"]
    ) -> None: ...

class Softplus:
    def __init__(
        self,
        beta: float = 1,
        threshold: float = 20,
        __constants__: list = ["beta", "threshold"],
    ) -> None: ...

class Softshrink:
    def __init__(self, lambd: float = 0.5, __constants__: list = ["lambd"]) -> None: ...

class Softsign:
    def __init__(self, *args, **kwargs) -> None: ...

class SyncBatchNorm:
    def __init__(
        self,
        num_features: int = ...,
        eps: float = 1e-05,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        process_group: Any | None = "```(None)```",
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
    ) -> None: ...

class Tanhshrink:
    def __init__(self, *args, **kwargs) -> None: ...

class Threshold:
    def __init__(
        self,
        threshold: float = ...,
        value: float = ...,
        inplace: bool = False,
        __constants__: list = ["threshold", "value", "inplace"],
    ) -> None: ...

class Transformer:
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = "relu",
        custom_encoder: Any | None = "```(None)```",
        custom_decoder: Any | None = "```(None)```",
        layer_norm_eps: float = 1e-05,
        batch_first: str = "False`` (seq, batch, feature).",
        norm_first: str = "False`` (after).",
        bias: bool = True,
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
    ) -> None: ...

class TransformerDecoder:
    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer = ...,
        num_layers: int = ...,
        norm: Module | None = "```(None)```",
        __constants__: list = ["norm"],
    ) -> None: ...

class TransformerDecoderLayer:
    def __init__(
        self,
        d_model: int = ...,
        nhead: int = ...,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = "relu",
        layer_norm_eps: float = 1e-05,
        batch_first: str = "False`` (seq, batch, feature).",
        norm_first: str = "False`` (after).",
        bias: bool = True,
        __constants__: list = ["norm_first"],
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
    ) -> None: ...

class TransformerEncoder:
    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer = ...,
        num_layers: int = ...,
        norm: Module | None = "```(None)```",
        enable_nested_tensor: str = "True`` (enabled).",
        __constants__: list = ["norm"],
        mask_check: bool = True,
    ) -> None: ...

class TransformerEncoderLayer:
    def __init__(
        self,
        d_model: int = ...,
        nhead: int = ...,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = "relu",
        layer_norm_eps: float = 1e-05,
        batch_first: str = "False`` (seq, batch, feature).",
        norm_first: str = "False`` (after).",
        bias: bool = True,
        __constants__: list = ["norm_first"],
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
    ) -> None: ...

class Unflatten:
    def __init__(
        self,
        dim: int | str = ...,
        unflattened_size: Any = ...,
        Any: Any = "```(tuple)```",
        __constants__: list = ["dim", "unflattened_size"],
    ) -> None: ...

class Unfold:
    def __init__(
        self,
        kernel_size: Any = ...,
        dilation: Any = 1,
        padding: Any = 0,
        stride: Any = 1,
        __constants__: list = ["kernel_size", "dilation", "padding", "stride"],
    ) -> None: ...

class Upsample:
    def __init__(
        self,
        __constants__: list = [
            "size",
            "scale_factor",
            "mode",
            "align_corners",
            "name",
            "recompute_scale_factor",
        ],
        name: str = ...,
        size: Any | None = "```(None)```",
        scale_factor: Any | None = "```(None)```",
        mode: str = "nearest",
        align_corners: bool | None = "```(None)```",
        recompute_scale_factor: bool | None = "```(None)```",
    ) -> None: ...

class UpsamplingBilinear2d:
    def __init__(
        self,
        size: int | tuple[int, int] | Any = "```(None)```",
        scale_factor: float | tuple[float, float] | Any = "```(None)```",
    ) -> None: ...

class UpsamplingNearest2d:
    def __init__(
        self,
        size: int | tuple[int, int] | Any = "```(None)```",
        scale_factor: float | tuple[float, float] | Any = "```(None)```",
    ) -> None: ...

class ZeroPad1d:
    def __init__(self, padding: tuple[int, int] = ...) -> None: ...

class ZeroPad2d:
    def __init__(self, padding: tuple[int, int, int, int] = ...) -> None: ...

class ZeroPad3d:
    def __init__(self, padding: tuple[int, int, int, int, int, int] = ...) -> None: ...

class AdaptiveLogSoftmaxWithLoss:
    def __init__(
        self,
        in_features: int = ...,
        n_classes: int = ...,
        cutoffs: list[int] = ...,
        div_value: float = 4.0,
        head_bias: bool = False,
        head: Linear = ...,
        tail: ModuleList = ...,
        device: Any = "```(None)```",
        dtype: Any = "```(None)```",
    ) -> None: ...

class BCELoss:
    def __init__(
        self,
        weight: Tensor | None = "```(None)```",
        size_average: bool | None = True,
        reduce: bool | None = True,
        reduction: str | None = "mean",
        __constants__: list = ["reduction"],
    ) -> None: ...

class BCEWithLogitsLoss:
    def __init__(
        self,
        weight: Tensor | None = "```(None)```",
        size_average: bool | None = True,
        reduce: bool | None = True,
        reduction: str | None = "mean",
        pos_weight: Tensor | None = "```(None)```",
    ) -> None: ...

class CTCLoss:
    def __init__(
        self,
        blank: int = 0,
        reduction: str | None = "mean",
        __constants__: list = ["blank", "reduction"],
        zero_infinity: bool = False,
    ) -> Any: ...

class CosineEmbeddingLoss:
    def __init__(
        self,
        margin: float = ":math:`0",
        size_average: bool | None = True,
        reduce: bool | None = True,
        reduction: str | None = "mean",
        __constants__: list = ["margin", "reduction"],
    ) -> None: ...

class CrossEntropyLoss:
    def __init__(
        self,
        weight: Tensor | None = "```(None)```",
        size_average: bool | None = True,
        ignore_index: int = -100,
        reduce: bool | None = True,
        reduction: str | None = "mean",
        label_smoothing: float = ":math:`0.0",
        __constants__: list = ["ignore_index", "reduction", "label_smoothing"],
    ) -> None: ...

class GaussianNLLLoss:
    def __init__(
        self,
        full: bool = False,
        eps: float = 1e-06,
        reduction: str | None = "mean",
        __constants__: list = ["full", "eps", "reduction"],
    ) -> None: ...

class HingeEmbeddingLoss:
    def __init__(
        self,
        margin: float = 1.0,
        size_average: bool | None = True,
        reduce: bool | None = True,
        reduction: str | None = "mean",
        __constants__: list = ["margin", "reduction"],
    ) -> None: ...

class HuberLoss:
    def __init__(
        self,
        reduction: str | None = "mean",
        delta: float | None = 1.0,
        __constants__: list = ["reduction", "delta"],
    ) -> None: ...

class KLDivLoss:
    def __init__(
        self,
        size_average: bool | None = True,
        reduce: bool | None = True,
        reduction: str | None = "mean",
        log_target: bool | None = False,
        __constants__: list = ["reduction"],
    ) -> None: ...

class L1Loss:
    def __init__(
        self,
        size_average: bool | None = True,
        reduce: bool | None = True,
        reduction: str | None = "mean",
        __constants__: list = ["reduction"],
    ) -> None: ...

class MSELoss:
    def __init__(
        self,
        size_average: bool | None = True,
        reduce: bool | None = True,
        reduction: str | None = "mean",
        __constants__: list = ["reduction"],
    ) -> None: ...

class MarginRankingLoss:
    def __init__(
        self,
        margin: float = 0.0,
        size_average: bool | None = True,
        reduce: bool | None = True,
        reduction: str | None = "mean",
        __constants__: list = ["margin", "reduction"],
    ) -> None: ...

class MultiLabelMarginLoss:
    def __init__(
        self,
        size_average: bool | None = True,
        reduce: bool | None = True,
        reduction: str | None = "mean",
        __constants__: list = ["reduction"],
    ) -> None: ...

class MultiLabelSoftMarginLoss:
    def __init__(
        self,
        weight: Tensor | None = "```(None)```",
        size_average: bool | None = True,
        reduce: bool | None = True,
        reduction: str | None = "mean",
        __constants__: list = ["reduction"],
    ) -> None: ...

class MultiMarginLoss:
    def __init__(
        self,
        p: int = 1,
        margin: float = 1.0,
        weight: Tensor | None = "```(None)```",
        size_average: bool | None = True,
        reduce: bool | None = True,
        reduction: str | None = "mean",
        __constants__: list = ["p", "margin", "reduction"],
    ) -> None: ...

class NLLLoss:
    def __init__(
        self,
        weight: Tensor | None = "```(None)```",
        size_average: bool | None = "```(None)```",
        ignore_index: int = -100,
        reduce: bool | None = "```(None)```",
        reduction: str | None = "mean",
        __constants__: list = ["ignore_index", "reduction"],
    ) -> None: ...

class PoissonNLLLoss:
    def __init__(
        self,
        log_input: bool = True,
        full: bool = False,
        __constants__: list = ["log_input", "full", "eps", "reduction"],
        eps: float = 1e-08,
        reduce: Any = "```(None)```",
        reduction: str = "mean",
        size_average: Any = "```(None)```",
    ) -> None: ...

class SmoothL1Loss:
    def __init__(
        self,
        size_average: bool | None = True,
        reduce: bool | None = True,
        reduction: str | None = "mean",
        beta: float | None = 1.0,
        __constants__: list = ["reduction"],
    ) -> None: ...

class SoftMarginLoss:
    def __init__(
        self,
        size_average: bool | None = True,
        reduce: bool | None = True,
        reduction: str | None = "mean",
        __constants__: list = ["reduction"],
    ) -> None: ...

class TripletMarginLoss:
    def __init__(
        self,
        margin: float = ":math:`1",
        p: float = ":math:`2",
        eps: float = ":math:`1e-6",
        swap: bool = False,
        size_average: bool | None = True,
        reduce: bool | None = True,
        reduction: str | None = "mean",
        __constants__: list = ["margin", "p", "eps", "swap", "reduction"],
    ) -> Any: ...

class TripletMarginWithDistanceLoss:
    def __init__(
        self,
        distance_function: Callable | None = "```(None)```",
        margin: float = ":math:`1",
        swap: bool = False,
        reduction: str | None = "mean",
        __constants__: list = ["margin", "swap", "reduction"],
    ) -> Any: ...
