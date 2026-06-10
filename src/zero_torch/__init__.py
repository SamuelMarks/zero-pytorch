"""Module."""

import ml_switcheroo

from .tensor import Tensor
from . import nn
from .autograd import no_grad, set_grad_enabled

import ml_switcheroo.ops as _ops
from typing import Any


def _wrap(x):
    """Function."""
    if isinstance(x, (tuple, list)):
        return type(x)(_wrap(i) for i in x)
    return Tensor(x)


def abs(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "abs")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def acos(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "acos")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def acosh(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "acosh")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def add(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "add")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def all(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "all")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def allclose(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "allclose")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return res


def any(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "any")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def arange(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "arange")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def argmax(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "argmax")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def argmin(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "argmin")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def asin(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "asin")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def asinh(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "asinh")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def atan(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "atan")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def atan2(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "atan2")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def atanh(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "atanh")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def binary(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "binary")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def bitcast(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "bitcast")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def bitwise_and(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "bitwise_and")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def bitwise_not(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "bitwise_not")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def bitwise_or(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "bitwise_or")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def bitwise_xor(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "bitwise_xor")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def broadcast_to(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "broadcast_to")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def cast(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "cast")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def cbrt(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "cbrt")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def ceil(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "ceil")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def cholesky(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "cholesky")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def concatenate(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "concatenate")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def conj(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "conj")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def copysign(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "copysign")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def cos(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "cos")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def cosh(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "cosh")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def count_nonzero(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "count_nonzero")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def creation(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "creation")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def deg2rad(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "deg2rad")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def det(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "det")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def diag(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "diag")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def digamma(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "digamma")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def divide(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "divide")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def divmod(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "divmod")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def dot(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "dot")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def dynamic_slice(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "dynamic_slice")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def eigh(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "eigh")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def eigvalsh(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "eigvalsh")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def einsum(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "einsum")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def empty(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "empty")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def equal(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "equal")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def erf(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "erf")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def erfc(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "erfc")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def erfinv(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "erfinv")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def exp(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "exp")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def exp2(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "exp2")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def expand(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "expand")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def expm1(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "expm1")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def eye(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "eye")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def fix(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "fix")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def flatten(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "flatten")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def float_power(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "float_power")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def floor(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "floor")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def floor_divide(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "floor_divide")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def fmax(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "fmax")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def fmin(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "fmin")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def fmod(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "fmod")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def frexp(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "frexp")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def full(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "full")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def full_like(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "full_like")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def gather(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "gather")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def gather_nd(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "gather_nd")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def gcd(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "gcd")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def greater(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "greater")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def greater_equal(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "greater_equal")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def heaviside(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "heaviside")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def hypot(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "hypot")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def identity(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "identity")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def imag(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "imag")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def inner(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "inner")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def inv(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "inv")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def isclose(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "isclose")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def isfinite(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "isfinite")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def isinf(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "isinf")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def isnan(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "isnan")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def lcm(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "lcm")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def ldexp(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "ldexp")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def left_shift(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "left_shift")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def less(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "less")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def less_equal(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "less_equal")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def lgamma(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "lgamma")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def linalg(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "linalg")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def linspace(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "linspace")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def log(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "log")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def log10(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "log10")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def log1p(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "log1p")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def log2(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "log2")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def logaddexp(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "logaddexp")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def logaddexp2(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "logaddexp2")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def logical_and(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "logical_and")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def logical_not(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "logical_not")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def logical_or(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "logical_or")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def logical_xor(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "logical_xor")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def logsumexp(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "logsumexp")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def matmul(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "matmul")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def matrix_power(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "matrix_power")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def max(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "max")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def maximum(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "maximum")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def mean(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "mean")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def meshgrid(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "meshgrid")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def min(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "min")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def minimum(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "minimum")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def mod(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "mod")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def moveaxis(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "moveaxis")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def multiply(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "multiply")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def negative(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "negative")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def nextafter(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "nextafter")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def norm(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "norm")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def not_equal(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "not_equal")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def ones(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "ones")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def ones_like(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "ones_like")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def outer(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "outer")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def permute(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "permute")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def pinv(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "pinv")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def positive(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "positive")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def power(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "power")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def prod(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "prod")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def qr(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "qr")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def rad2deg(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "rad2deg")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def real(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "real")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def reciprocal(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "reciprocal")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def reductions(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "reductions")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def remainder(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "remainder")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def repeat(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "repeat")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def reshape(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "reshape")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def right_shift(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "right_shift")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def roll(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "roll")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def round(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "round")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def rsqrt(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "rsqrt")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def scatter(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "scatter")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def scatter_add(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "scatter_add")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def scatter_nd(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "scatter_nd")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def shape(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "shape")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def sign(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "sign")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def sin(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "sin")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def sinc(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "sinc")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def sinh(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "sinh")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def slice(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "slice")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def slogdet(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "slogdet")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def split(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "split")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def sqrt(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "sqrt")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def square(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "square")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def squeeze(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "squeeze")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def stack(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "stack")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def std(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "std")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def strided_slice(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "strided_slice")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def subtract(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "subtract")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def sum(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "sum")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def svd(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "svd")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def swapaxes(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "swapaxes")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def take(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "take")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def tan(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "tan")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def tanh(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "tanh")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def tensordot(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "tensordot")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def tile(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "tile")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def transpose(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "transpose")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def tril(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "tril")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def triu(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "triu")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def trunc(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "trunc")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def unary(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "unary")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def unsqueeze(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "unsqueeze")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def unstack(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "unstack")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def update_slice(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "update_slice")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def variance(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "variance")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def vdot(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "vdot")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def where(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "where")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def zeros(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "zeros")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def zeros_like(*args, **kwargs):
    """Function."""
    if "dim" in kwargs:
        pass
    res = getattr(_ops, "zeros_like")(
        *[a._tensor if isinstance(a, Tensor) else a for a in args], **kwargs
    )
    return _wrap(res)


def tensor(data, *args, **kwargs):
    """Function."""
    return Tensor(data, *args, **kwargs)


from ml_switcheroo.core.dtype import DType

float32 = DType.Float32
int32 = DType.Int32
