import ml_switcheroo

from .tensor import Tensor
from . import nn
from .autograd import no_grad, set_grad_enabled

import ml_switcheroo.ops as _ops
from typing import Any


def _wrap(x):
    return Tensor(x) if isinstance(x, ml_switcheroo.Tensor) else Tensor(x)


def abs(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "abs")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def acos(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "acos")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def acosh(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "acosh")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def add(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "add")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def all(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "all")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def allclose(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "allclose")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def any(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "any")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def arange(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "arange")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def argmax(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "argmax")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def argmin(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "argmin")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def asin(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "asin")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def asinh(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "asinh")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def atan(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "atan")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def atan2(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "atan2")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def atanh(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "atanh")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def binary(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "binary")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def bitcast(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "bitcast")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def bitwise_and(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "bitwise_and")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def bitwise_not(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "bitwise_not")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def bitwise_or(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "bitwise_or")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def bitwise_xor(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "bitwise_xor")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def broadcast_to(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "broadcast_to")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def cast(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "cast")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def cbrt(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "cbrt")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def ceil(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "ceil")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def cholesky(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "cholesky")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def concatenate(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "concatenate")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def conj(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "conj")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def copysign(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "copysign")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def cos(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "cos")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def cosh(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "cosh")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def count_nonzero(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "count_nonzero")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def creation(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "creation")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def deg2rad(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "deg2rad")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def det(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "det")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def diag(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "diag")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def digamma(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "digamma")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def divide(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "divide")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def divmod(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "divmod")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def dot(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "dot")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def dynamic_slice(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "dynamic_slice")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def eigh(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "eigh")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def eigvalsh(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "eigvalsh")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def einsum(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "einsum")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def empty(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "empty")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def equal(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "equal")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def erf(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "erf")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def erfc(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "erfc")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def erfinv(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "erfinv")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def exp(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "exp")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def exp2(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "exp2")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def expand(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "expand")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def expm1(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "expm1")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def eye(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "eye")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def fix(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "fix")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def flatten(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "flatten")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def float_power(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "float_power")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def floor(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "floor")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def floor_divide(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "floor_divide")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def fmax(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "fmax")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def fmin(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "fmin")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def fmod(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "fmod")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def frexp(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "frexp")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def full(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "full")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def full_like(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "full_like")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def gather(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "gather")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def gather_nd(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "gather_nd")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def gcd(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "gcd")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def greater(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "greater")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def greater_equal(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "greater_equal")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def heaviside(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "heaviside")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def hypot(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "hypot")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def identity(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "identity")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def imag(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "imag")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def inner(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "inner")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def inv(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "inv")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def isclose(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "isclose")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def isfinite(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "isfinite")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def isinf(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "isinf")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def isnan(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "isnan")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def lcm(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "lcm")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def ldexp(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "ldexp")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def left_shift(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "left_shift")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def less(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "less")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def less_equal(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "less_equal")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def lgamma(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "lgamma")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def linalg(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "linalg")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def linspace(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "linspace")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def log(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "log")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def log10(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "log10")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def log1p(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "log1p")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def log2(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "log2")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def logaddexp(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "logaddexp")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def logaddexp2(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "logaddexp2")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def logical_and(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "logical_and")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def logical_not(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "logical_not")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def logical_or(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "logical_or")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def logical_xor(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "logical_xor")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def logsumexp(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "logsumexp")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def matmul(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "matmul")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def matrix_power(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "matrix_power")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def max(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "max")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def maximum(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "maximum")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def mean(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "mean")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def meshgrid(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "meshgrid")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def min(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "min")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def minimum(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "minimum")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def mod(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "mod")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def moveaxis(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "moveaxis")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def multiply(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "multiply")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def negative(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "negative")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def nextafter(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "nextafter")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def norm(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "norm")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def not_equal(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "not_equal")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def ones(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "ones")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def ones_like(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "ones_like")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def outer(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "outer")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def permute(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "permute")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def pinv(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "pinv")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def positive(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "positive")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def power(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "power")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def prod(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "prod")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def qr(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "qr")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def rad2deg(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "rad2deg")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def real(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "real")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def reciprocal(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "reciprocal")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def reductions(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "reductions")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def remainder(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "remainder")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def repeat(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "repeat")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def reshape(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "reshape")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def right_shift(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "right_shift")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def roll(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "roll")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def round(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "round")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def rsqrt(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "rsqrt")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def scatter(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "scatter")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def scatter_add(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "scatter_add")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def scatter_nd(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "scatter_nd")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def shape(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "shape")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def sign(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "sign")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def sin(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "sin")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def sinc(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "sinc")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def sinh(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "sinh")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def slice(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "slice")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def slogdet(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "slogdet")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def split(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "split")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def sqrt(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "sqrt")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def square(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "square")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def squeeze(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "squeeze")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def stack(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "stack")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def std(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "std")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def strided_slice(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "strided_slice")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def subtract(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "subtract")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def sum(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "sum")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def svd(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "svd")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def swapaxes(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "swapaxes")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def take(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "take")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def tan(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "tan")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def tanh(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "tanh")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def tensordot(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "tensordot")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def tile(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "tile")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def transpose(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "transpose")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def tril(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "tril")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def triu(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "triu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def trunc(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "trunc")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def unary(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "unary")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def unsqueeze(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "unsqueeze")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def unstack(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "unstack")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def update_slice(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "update_slice")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def variance(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "variance")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def vdot(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "vdot")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def where(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "where")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def zeros(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "zeros")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res


def zeros_like(*args, **kwargs):
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "zeros_like")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res) if isinstance(res, ml_switcheroo.Tensor) else res
