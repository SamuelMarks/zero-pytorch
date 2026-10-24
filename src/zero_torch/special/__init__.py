"""Special Functions frontend module."""

from __future__ import annotations

import math

import ml_switcheroo_compiler.ops as _ops
from ml_switcheroo_compiler.ops.base import get_op

import zero_torch
from zero_torch._types import OpArg, Scalar
from zero_torch.tensor import Tensor, _flatten_nested, _wrap


def bessel_j0(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the Bessel function of the first kind of order 0.

    Args:
        input: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `J_0(input)`.
    """
    return _wrap(get_op("BesselJ0")()(input._tensor, **kwargs))


def bessel_j1(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the Bessel function of the first kind of order 1.

    Args:
        input: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `J_1(input)`.
    """
    return _wrap(get_op("BesselJ1")()(input._tensor, **kwargs))


def bessel_y0(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the Bessel function of the second kind of order 0.

    Args:
        input: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `Y_0(input)`.
    """
    return _wrap(get_op("BesselY0")()(input._tensor, **kwargs))


def bessel_y1(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the Bessel function of the second kind of order 1.

    Args:
        input: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `Y_1(input)`.
    """
    return _wrap(get_op("BesselY1")()(input._tensor, **kwargs))


def modified_bessel_i0(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the modified Bessel function of the first kind of order 0.

    Args:
        input: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `I_0(input)`.
    """
    _ = kwargs
    quarter_x2 = (input * input) / 4.0
    res = zero_torch.ones_like(input)
    term = zero_torch.ones_like(input)
    for k in range(1, 15):
        term = term * quarter_x2 / float(k * k)
        res = res + term
    return res


def modified_bessel_i1(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the modified Bessel function of the first kind of order 1.

    Args:
        input: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `I_1(input)`.
    """
    _ = kwargs
    quarter_x2 = (input * input) / 4.0
    half_x = input / 2.0
    res = half_x
    term = half_x
    for k in range(1, 15):
        term = term * quarter_x2 / float(k * (k + 1))
        res = res + term
    return res


def modified_bessel_k0(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the modified Bessel function of the second kind of order 0.

    Args:
        input: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `K_0(input)`.
    """
    _ = kwargs
    return zero_torch.exp(-input) * zero_torch.sqrt(math.pi / (2.0 * input))


def modified_bessel_k1(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the modified Bessel function of the second kind of order 1.

    Args:
        input: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `K_1(input)`.
    """
    _ = kwargs
    return (
        zero_torch.exp(-input)
        * zero_torch.sqrt(math.pi / (2.0 * input))
        * (1.0 + 3.0 / (8.0 * input))
    )


def erfcx(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the scaled complementary error function `exp(x^2) * erfc(x)`.

    Args:
        input: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `erfcx(input)`.
    """
    _ = kwargs
    return zero_torch.exp(input * input) * zero_torch.erfc(input)


def ndtr(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the area under the standard Gaussian probability density function.

    Args:
        input: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `ndtr(input)`.
    """
    _ = kwargs
    return 0.5 * (1.0 + zero_torch.erf(input / math.sqrt(2.0)))


def ndtri(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the argument `x` for which the area under the Gaussian PDF is equal to `input`.

    Args:
        input: Input probability tensor in `(0, 1)`.
        **kwargs: Optional keyword arguments.

    Returns:
        Output quantile tensor with `ndtri(input)`.
    """
    _ = kwargs
    return zero_torch.erfinv(2.0 * input - 1.0) * math.sqrt(2.0)


def gammaln(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the natural logarithm of the absolute value of the gamma function.

    Args:
        input: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `ln(|Gamma(input)|)`.
    """
    _ = kwargs
    flat: list[Scalar] = _flatten_nested(input.tolist())
    vals = [
        math.lgamma(float(v)) if isinstance(v, (int, float)) and float(v) > 0.0 else 0.0
        for v in flat
    ]
    return zero_torch.reshape(Tensor(vals, dtype=input.dtype), input.shape)


def polygamma(n: int, input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the `n`-th derivative of the digamma function on `input`.

    Args:
        n: Non-negative derivative order.
        input: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `psi^(n)(input)`.
    """
    return _wrap(_ops.polygamma(n, input._tensor, **kwargs))


def multigammaln(input: Tensor, p: int, **kwargs: OpArg) -> Tensor:
    """Computes the multivariate log-gamma function of dimension `p`.

    Args:
        input: Input tensor.
        p: Number of dimensions.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with multivariate log-gamma values.
    """
    return zero_torch.mvlgamma(input, p, **kwargs)


def zeta(input: Tensor, other: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the Hurwitz zeta function.

    Args:
        input: Exponent tensor `x`.
        other: Shift tensor `q`.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `zeta(input, other)`.
    """
    return _wrap(_ops.zeta(input._tensor, other._tensor, **kwargs))


def chebyshev_polynomial_t(input: Tensor, n: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the Chebyshev polynomial of the first kind `T_n(input)`.

    Args:
        input: Input evaluation points tensor.
        n: Polynomial degree tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Evaluated Chebyshev polynomial tensor.
    """
    return _wrap(_ops.chebyshev_polynomial_t(input._tensor, n._tensor, **kwargs))


def hermite_polynomial_h(input: Tensor, n: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the physicist's Hermite polynomial `H_n(input)`.

    Args:
        input: Input evaluation points tensor.
        n: Polynomial degree tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Evaluated Hermite polynomial tensor.
    """
    return _wrap(_ops.hermite_polynomial_h(input._tensor, n._tensor, **kwargs))


def laguerre_polynomial_l(input: Tensor, n: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the Laguerre polynomial `L_n(input)`.

    Args:
        input: Input evaluation points tensor.
        n: Polynomial degree tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Evaluated Laguerre polynomial tensor.
    """
    return _wrap(_ops.laguerre_polynomial_l(input._tensor, n._tensor, **kwargs))


def legendre_polynomial_p(input: Tensor, n: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the Legendre polynomial of the first kind `P_n(input)`.

    Args:
        input: Input evaluation points tensor.
        n: Polynomial degree tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Evaluated Legendre polynomial tensor.
    """
    return _wrap(_ops.legendre_polynomial_p(input._tensor, n._tensor, **kwargs))
