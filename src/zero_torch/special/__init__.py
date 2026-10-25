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
    euler_gamma = 0.5772156649015328606
    i0 = modified_bessel_i0(input)
    quarter_x2 = (input * input) / 4.0
    small_res = -(zero_torch.log(input / 2.0) + euler_gamma) * i0
    term = zero_torch.ones_like(input)
    harmonic = 0.0
    for k in range(1, 15):
        term = term * quarter_x2 / float(k * k)
        harmonic += 1.0 / float(k)
        small_res = small_res + term * harmonic
    large_res = (
        zero_torch.exp(-input)
        * zero_torch.sqrt(math.pi / (2.0 * input))
        * (
            1.0
            - 1.0 / (8.0 * input)
            + 9.0 / (128.0 * input * input)
            - 225.0 / (3072.0 * input * input * input)
        )
    )
    return zero_torch.where(input <= 2.0, small_res, large_res)


def modified_bessel_k1(input: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the modified Bessel function of the second kind of order 1.

    Args:
        input: Input tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `K_1(input)`.
    """
    _ = kwargs
    euler_gamma = 0.5772156649015328606
    i1 = modified_bessel_i1(input)
    quarter_x2 = (input * input) / 4.0
    half_x = input / 2.0
    small_res = (1.0 / input) + zero_torch.log(half_x) * i1
    term = half_x
    psi_k1 = -euler_gamma
    psi_k2 = -euler_gamma + 1.0
    small_res = small_res - 0.5 * (psi_k1 + psi_k2) * term
    for k in range(1, 15):
        term = term * quarter_x2 / float(k * (k + 1))
        psi_k1 += 1.0 / float(k)
        psi_k2 += 1.0 / float(k + 1)
        small_res = small_res - 0.5 * (psi_k1 + psi_k2) * term
    large_res = (
        zero_torch.exp(-input)
        * zero_torch.sqrt(math.pi / (2.0 * input))
        * (
            1.0
            + 3.0 / (8.0 * input)
            - 15.0 / (128.0 * input * input)
            + 315.0 / (3072.0 * input * input * input)
        )
    )
    return zero_torch.where(input <= 2.0, small_res, large_res)


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


def _safe_lgamma(val: Scalar) -> float:
    """Computes `math.lgamma(float(val))` safely, returning `inf` at non-positive integer poles.

    Args:
        val: Scalar input value.

    Returns:
        Evaluated log-gamma float.
    """
    x = float(val) if isinstance(val, (int, float)) else 1.0
    if x <= 0.0 and x == math.floor(x):
        return float("inf")
    return math.lgamma(x)


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
    vals = [_safe_lgamma(v) for v in flat]
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


def _eval_chebyshev_t(x: float, deg: int) -> float:
    """Evaluates Chebyshev polynomial of the first kind `T_deg(x)` via recurrence.

    Args:
        x: Evaluation point.
        deg: Non-negative integer degree.

    Returns:
        Evaluated polynomial scalar.
    """
    if deg <= 0:
        return 1.0
    if deg == 1:
        return x
    t0, t1 = 1.0, x
    for _ in range(2, deg + 1):
        t0, t1 = t1, 2.0 * x * t1 - t0
    return t1


def _eval_hermite_h(x: float, deg: int) -> float:
    """Evaluates physicist's Hermite polynomial `H_deg(x)` via recurrence.

    Args:
        x: Evaluation point.
        deg: Non-negative integer degree.

    Returns:
        Evaluated polynomial scalar.
    """
    if deg <= 0:
        return 1.0
    if deg == 1:
        return 2.0 * x
    h0, h1 = 1.0, 2.0 * x
    for k in range(1, deg):
        h0, h1 = h1, 2.0 * x * h1 - 2.0 * float(k) * h0
    return h1


def _eval_laguerre_l(x: float, deg: int) -> float:
    """Evaluates Laguerre polynomial `L_deg(x)` via recurrence.

    Args:
        x: Evaluation point.
        deg: Non-negative integer degree.

    Returns:
        Evaluated polynomial scalar.
    """
    if deg <= 0:
        return 1.0
    if deg == 1:
        return 1.0 - x
    l0, l1 = 1.0, 1.0 - x
    for k in range(1, deg):
        l0, l1 = (
            l1,
            ((2.0 * float(k) + 1.0 - x) * l1 - float(k) * l0) / float(k + 1),
        )
    return l1


def _eval_legendre_p(x: float, deg: int) -> float:
    """Evaluates Legendre polynomial `P_deg(x)` via recurrence.

    Args:
        x: Evaluation point.
        deg: Non-negative integer degree.

    Returns:
        Evaluated polynomial scalar.
    """
    if deg <= 0:
        return 1.0
    if deg == 1:
        return x
    p0, p1 = 1.0, x
    for k in range(1, deg):
        p0, p1 = (
            p1,
            ((2.0 * float(k) + 1.0) * x * p1 - float(k) * p0) / float(k + 1),
        )
    return p1


def chebyshev_polynomial_t(input: Tensor, n: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the Chebyshev polynomial of the first kind `T_n(input)`.

    Args:
        input: Input evaluation points tensor.
        n: Polynomial degree tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Evaluated Chebyshev polynomial tensor.
    """
    _ = kwargs
    b_n = zero_torch.broadcast_to(n, input.shape)
    xs = _flatten_nested(input.tolist())
    ns = _flatten_nested(b_n.tolist())
    vals = [
        _eval_chebyshev_t(
            float(x) if isinstance(x, (int, float)) else 0.0,
            int(float(d)) if isinstance(d, (int, float)) else 0,
        )
        for x, d in zip(xs, ns)
    ]
    return zero_torch.reshape(Tensor(vals, dtype=input.dtype), input.shape)


def hermite_polynomial_h(input: Tensor, n: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the physicist's Hermite polynomial `H_n(input)`.

    Args:
        input: Input evaluation points tensor.
        n: Polynomial degree tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Evaluated Hermite polynomial tensor.
    """
    _ = kwargs
    b_n = zero_torch.broadcast_to(n, input.shape)
    xs = _flatten_nested(input.tolist())
    ns = _flatten_nested(b_n.tolist())
    vals = [
        _eval_hermite_h(
            float(x) if isinstance(x, (int, float)) else 0.0,
            int(float(d)) if isinstance(d, (int, float)) else 0,
        )
        for x, d in zip(xs, ns)
    ]
    return zero_torch.reshape(Tensor(vals, dtype=input.dtype), input.shape)


def laguerre_polynomial_l(input: Tensor, n: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the Laguerre polynomial `L_n(input)`.

    Args:
        input: Input evaluation points tensor.
        n: Polynomial degree tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Evaluated Laguerre polynomial tensor.
    """
    _ = kwargs
    b_n = zero_torch.broadcast_to(n, input.shape)
    xs = _flatten_nested(input.tolist())
    ns = _flatten_nested(b_n.tolist())
    vals = [
        _eval_laguerre_l(
            float(x) if isinstance(x, (int, float)) else 0.0,
            int(float(d)) if isinstance(d, (int, float)) else 0,
        )
        for x, d in zip(xs, ns)
    ]
    return zero_torch.reshape(Tensor(vals, dtype=input.dtype), input.shape)


def legendre_polynomial_p(input: Tensor, n: Tensor, **kwargs: OpArg) -> Tensor:
    """Computes the Legendre polynomial of the first kind `P_n(input)`.

    Args:
        input: Input evaluation points tensor.
        n: Polynomial degree tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Evaluated Legendre polynomial tensor.
    """
    _ = kwargs
    b_n = zero_torch.broadcast_to(n, input.shape)
    xs = _flatten_nested(input.tolist())
    ns = _flatten_nested(b_n.tolist())
    vals = [
        _eval_legendre_p(
            float(x) if isinstance(x, (int, float)) else 0.0,
            int(float(d)) if isinstance(d, (int, float)) else 0,
        )
        for x, d in zip(xs, ns)
    ]
    return zero_torch.reshape(Tensor(vals, dtype=input.dtype), input.shape)
