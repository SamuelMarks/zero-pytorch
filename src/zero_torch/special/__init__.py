"""Special Functions frontend module."""

from __future__ import annotations

import math

import zero_torch
from zero_torch._types import DTypeLike, OpArg, Scalar
from zero_torch.tensor import Tensor, _flatten_nested


def _copy_if_out(res: Tensor, out: Tensor | None) -> Tensor:
    """Copies `res` into `out` if `out` is provided, and returns the resulting tensor.

    Args:
        res: Computed result tensor.
        out: Optional destination tensor.

    Returns:
        Result tensor (`out` if provided, otherwise `res`).
    """
    if out is not None:
        return out.copy_(res)
    return res


def bessel_j0(input: Tensor, *, out: Tensor | None = None, **kwargs: OpArg) -> Tensor:
    """Computes the Bessel function of the first kind of order 0.

    Args:
        input: Input tensor.
        out: Optional output tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `J_0(input)`.
    """
    _ = kwargs
    quarter_x2 = (input * input) / 4.0
    res = zero_torch.ones_like(input)
    term = zero_torch.ones_like(input)
    for k in range(1, 15):
        term = -term * quarter_x2 / float(k * k)
        res = res + term
    return _copy_if_out(res, out)


def bessel_j1(input: Tensor, *, out: Tensor | None = None, **kwargs: OpArg) -> Tensor:
    """Computes the Bessel function of the first kind of order 1.

    Args:
        input: Input tensor.
        out: Optional output tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `J_1(input)`.
    """
    _ = kwargs
    quarter_x2 = (input * input) / 4.0
    half_x = input / 2.0
    res = half_x
    term = half_x
    for k in range(1, 15):
        term = -term * quarter_x2 / float(k * (k + 1))
        res = res + term
    return _copy_if_out(res, out)


def bessel_y0(input: Tensor, *, out: Tensor | None = None, **kwargs: OpArg) -> Tensor:
    """Computes the Bessel function of the second kind of order 0.

    Args:
        input: Input tensor.
        out: Optional output tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `Y_0(input)`.
    """
    _ = kwargs
    euler_gamma = 0.5772156649015328606
    j0 = bessel_j0(input)
    quarter_x2 = (input * input) / 4.0
    series = (zero_torch.log(input / 2.0) + euler_gamma) * j0
    term = zero_torch.ones_like(input)
    harmonic = 0.0
    for k in range(1, 15):
        term = -term * quarter_x2 / float(k * k)
        harmonic += 1.0 / float(k)
        series = series - term * harmonic
    return _copy_if_out((2.0 / math.pi) * series, out)


def bessel_y1(input: Tensor, *, out: Tensor | None = None, **kwargs: OpArg) -> Tensor:
    """Computes the Bessel function of the second kind of order 1.

    Args:
        input: Input tensor.
        out: Optional output tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `Y_1(input)`.
    """
    _ = kwargs
    euler_gamma = 0.5772156649015328606
    j1 = bessel_j1(input)
    quarter_x2 = (input * input) / 4.0
    half_x = input / 2.0
    series = (zero_torch.log(half_x) + euler_gamma) * j1 - (1.0 / input) - 0.5 * half_x
    term = half_x
    h_k = 0.0
    h_k1 = 1.0
    for k in range(1, 15):
        term = -term * quarter_x2 / float(k * (k + 1))
        h_k += 1.0 / float(k)
        h_k1 += 1.0 / float(k + 1)
        series = series - 0.5 * (h_k + h_k1) * term
    return _copy_if_out((2.0 / math.pi) * series, out)


def modified_bessel_i0(
    input: Tensor, *, out: Tensor | None = None, **kwargs: OpArg
) -> Tensor:
    """Computes the modified Bessel function of the first kind of order 0.

    Args:
        input: Input tensor.
        out: Optional output tensor.
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
    return _copy_if_out(res, out)


def modified_bessel_i1(
    input: Tensor, *, out: Tensor | None = None, **kwargs: OpArg
) -> Tensor:
    """Computes the modified Bessel function of the first kind of order 1.

    Args:
        input: Input tensor.
        out: Optional output tensor.
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
    return _copy_if_out(res, out)


def modified_bessel_k0(
    input: Tensor, *, out: Tensor | None = None, **kwargs: OpArg
) -> Tensor:
    """Computes the modified Bessel function of the second kind of order 0.

    Args:
        input: Input tensor.
        out: Optional output tensor.
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
    return _copy_if_out(zero_torch.where(input <= 2.0, small_res, large_res), out)


def modified_bessel_k1(
    input: Tensor, *, out: Tensor | None = None, **kwargs: OpArg
) -> Tensor:
    """Computes the modified Bessel function of the second kind of order 1.

    Args:
        input: Input tensor.
        out: Optional output tensor.
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
    return _copy_if_out(zero_torch.where(input <= 2.0, small_res, large_res), out)


def i0(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Computes the zeroth-order modified Bessel function of the first kind.

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Output tensor with `I_0(input)`.
    """
    return modified_bessel_i0(input, out=out)


def i0e(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Computes the exponentially scaled zeroth-order modified Bessel function of the first kind.

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Output tensor with `exp(-|x|) * I_0(x)`.
    """
    return _copy_if_out(
        zero_torch.exp(-zero_torch.abs(input)) * modified_bessel_i0(input), out
    )


def i1(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Computes the first-order modified Bessel function of the first kind.

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Output tensor with `I_1(input)`.
    """
    return modified_bessel_i1(input, out=out)


def i1e(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Computes the exponentially scaled first-order modified Bessel function of the first kind.

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Output tensor with `exp(-|x|) * I_1(x)`.
    """
    return _copy_if_out(
        zero_torch.exp(-zero_torch.abs(input)) * modified_bessel_i1(input), out
    )


def scaled_modified_bessel_k0(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Computes the scaled modified Bessel function of the second kind of order 0 (`exp(x) * K_0(x)`).

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Output tensor with `exp(input) * K_0(input)`.
    """
    return _copy_if_out(zero_torch.exp(input) * modified_bessel_k0(input), out)


def scaled_modified_bessel_k1(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Computes the scaled modified Bessel function of the second kind of order 1 (`exp(x) * K_1(x)`).

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Output tensor with `exp(input) * K_1(input)`.
    """
    return _copy_if_out(zero_torch.exp(input) * modified_bessel_k1(input), out)


def spherical_bessel_j0(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Computes the spherical Bessel function of the first kind of order 0 (`sin(x) / x`).

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Output tensor with `j_0(input)`.
    """
    safe_x = zero_torch.where(input == 0.0, zero_torch.ones_like(input), input)
    res = zero_torch.where(
        input == 0.0, zero_torch.ones_like(input), zero_torch.sin(safe_x) / safe_x
    )
    return _copy_if_out(res, out)


def sinc(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Computes the normalized sinc function `sin(pi * x) / (pi * x)`.

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Output tensor with `sinc(input)`.
    """
    return spherical_bessel_j0(input * math.pi, out=out)


def airy_ai(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Computes the Airy function `Ai(input)`.

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Output tensor with `Ai(input)`.
    """
    c1 = 0.35502805388781724
    c2 = 0.2588194037928068
    x2 = input * input
    x3 = x2 * input
    f_part = 1.0 + x3 / 6.0 + (x3 * x3) / 180.0
    g_part = input + (x2 * x2) / 12.0 + (x3 * x3 * input) / 504.0
    return _copy_if_out(f_part * c1 - g_part * c2, out)


def erf(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Computes the error function of `input`.

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Output tensor with `erf(input)`.
    """
    return _copy_if_out(zero_torch.erf(input), out)


def erfc(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Computes the complementary error function of `input`.

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Output tensor with `erfc(input)`.
    """
    return _copy_if_out(zero_torch.erfc(input), out)


def erfcx(input: Tensor, *, out: Tensor | None = None, **kwargs: OpArg) -> Tensor:
    """Computes the scaled complementary error function `exp(x^2) * erfc(x)`.

    Args:
        input: Input tensor.
        out: Optional output tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `erfcx(input)`.
    """
    _ = kwargs
    return _copy_if_out(zero_torch.exp(input * input) * zero_torch.erfc(input), out)


def erfinv(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Computes the inverse error function of `input`.

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Output tensor with `erfinv(input)`.
    """
    a = 0.147
    y = zero_torch.clamp(input, min=-0.999999, max=0.999999)
    ln_term = zero_torch.log(1.0 - y * y)
    t1 = 2.0 / (math.pi * a) + 0.5 * ln_term
    t2 = ln_term / a
    rad = zero_torch.maximum(
        zero_torch.sqrt(t1 * t1 - t2) - t1, zero_torch.zeros_like(y)
    )
    x = zero_torch.sign(y) * zero_torch.sqrt(rad)
    two_over_sqrt_pi = 2.0 / math.sqrt(math.pi)
    for _ in range(2):
        x = x - (zero_torch.erf(x) - y) / (two_over_sqrt_pi * zero_torch.exp(-(x * x)))
    return _copy_if_out(x, out)


def exp2(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Computes the base-2 exponential `2^input`.

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Output tensor with `2^input`.
    """
    return _copy_if_out(zero_torch.exp2(input), out)


def expm1(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Computes `exp(input) - 1`.

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Output tensor with `exp(input) - 1`.
    """
    return _copy_if_out(zero_torch.expm1(input), out)


def expit(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Computes the expit (logistic sigmoid) function `1 / (1 + exp(-input))`.

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Output tensor with `expit(input)`.
    """
    return _copy_if_out(zero_torch.sigmoid(input), out)


def logit(
    input: Tensor,
    eps: float | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """Computes the logit function `log(x / (1 - x))`.

    Args:
        input: Input probability tensor.
        eps: Optional clamping epsilon for `[eps, 1 - eps]`.
        out: Optional output tensor.

    Returns:
        Output tensor with `logit(input)`.
    """
    x = zero_torch.clamp(input, min=eps, max=1.0 - eps) if eps is not None else input
    return _copy_if_out(zero_torch.log(x / (1.0 - x)), out)


def log1p(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Computes `log(1 + input)`.

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Output tensor with `log(1 + input)`.
    """
    return _copy_if_out(zero_torch.log1p(input), out)


def entr(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Computes the element-wise entropy `-input * log(input)` for `input > 0`.

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Entropy tensor.
    """
    safe_x = zero_torch.where(input > 0.0, input, zero_torch.ones_like(input))
    pos_val = -safe_x * zero_torch.log(safe_x)
    res = zero_torch.where(
        input > 0.0,
        pos_val,
        zero_torch.where(
            input == 0.0,
            zero_torch.zeros_like(input),
            zero_torch.full(input.shape, float("-inf"), dtype=input.dtype),
        ),
    )
    return _copy_if_out(res, out)


def xlogy(
    input: Tensor | float,
    other: Tensor | float,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """Computes `input * log(other)` with `0 * log(0) = 0`.

    Args:
        input: Multiplier tensor or scalar.
        other: Logarithm argument tensor or scalar.
        out: Optional output tensor.

    Returns:
        Output tensor with `input * log(other)`.
    """
    x = input if isinstance(input, Tensor) else Tensor(input)
    y = other if isinstance(other, Tensor) else Tensor(other)
    safe_y = zero_torch.where(x == 0.0, zero_torch.ones_like(y), y)
    res = zero_torch.where(
        x == 0.0, zero_torch.zeros_like(x), x * zero_torch.log(safe_y)
    )
    return _copy_if_out(res, out)


def xlog1py(
    input: Tensor | float,
    other: Tensor | float,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """Computes `input * log1p(other)` with `0 * log1p(other) = 0`.

    Args:
        input: Multiplier tensor or scalar.
        other: `log1p` argument tensor or scalar.
        out: Optional output tensor.

    Returns:
        Output tensor with `input * log1p(other)`.
    """
    x = input if isinstance(input, Tensor) else Tensor(input)
    y = other if isinstance(other, Tensor) else Tensor(other)
    safe_y = zero_torch.where(x == 0.0, zero_torch.zeros_like(y), y)
    res = zero_torch.where(
        x == 0.0, zero_torch.zeros_like(x), x * zero_torch.log1p(safe_y)
    )
    return _copy_if_out(res, out)


def digamma(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Computes the logarithmic derivative of the gamma function on `input`.

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Output tensor with `psi(input)`.
    """
    z = input
    shift_sum = zero_torch.zeros_like(input)
    for _ in range(8):
        shift_sum = shift_sum - 1.0 / z
        z = z + 1.0
    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z
    asymp = (
        zero_torch.log(z)
        - 0.5 * inv_z
        - inv_z2 * (1.0 / 12.0 - inv_z2 * (1.0 / 120.0 - inv_z2 / 252.0))
    )
    return _copy_if_out(shift_sum + asymp, out)


def psi(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Alias for `torch.special.digamma`.

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Output tensor with `psi(input)`.
    """
    return digamma(input, out=out)


def gammainc(
    input: Tensor,
    other: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """Computes the regularized lower incomplete gamma function `P(input, other)`.

    Args:
        input: Positive shape parameter tensor `a`.
        other: Non-negative upper integration limit tensor `x`.
        out: Optional output tensor.

    Returns:
        Output tensor with `gammainc(input, other)`.
    """
    a = input
    x = zero_torch.maximum(other, 1e-12)
    log_prefactor = a * zero_torch.log(x) - x - gammaln(a)
    prefactor = zero_torch.exp(log_prefactor)

    # Power series for x < a + 1
    term = 1.0 / a
    total = term
    for k in range(1, 30):
        term = term * x / (a + float(k))
        total = total + term
    p_series = zero_torch.clamp(prefactor * total, min=0.0, max=1.0)

    # Lentz continued fraction for Q(a, x) when x >= a + 1
    tiny = 1e-30
    b = x + 1.0 - a
    c = zero_torch.full(x.shape, 1.0 / tiny, dtype=x.dtype)
    d = 1.0 / zero_torch.where(zero_torch.abs(b) < tiny, tiny, b)
    h = d
    for i in range(1, 30):
        an = -float(i) * (float(i) - a)
        b = b + 2.0
        d = an * d + b
        d = 1.0 / zero_torch.where(zero_torch.abs(d) < tiny, tiny, d)
        c = b + an / zero_torch.where(zero_torch.abs(c) < tiny, tiny, c)
        h = h * d * c
    q_cf = zero_torch.clamp(prefactor * h, min=0.0, max=1.0)
    res = zero_torch.where(
        other <= 0.0,
        zero_torch.zeros_like(other),
        zero_torch.where(x < a + 1.0, p_series, 1.0 - q_cf),
    )
    return _copy_if_out(res, out)


def gammaincc(
    input: Tensor,
    other: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """Computes the regularized upper incomplete gamma function `Q(input, other)`.

    Args:
        input: Positive shape parameter tensor `a`.
        other: Non-negative lower integration limit tensor `x`.
        out: Optional output tensor.

    Returns:
        Output tensor with `gammaincc(input, other)`.
    """
    return _copy_if_out(1.0 - gammainc(input, other), out)


def ndtr(input: Tensor, *, out: Tensor | None = None, **kwargs: OpArg) -> Tensor:
    """Computes the area under the standard Gaussian probability density function.

    Args:
        input: Input tensor.
        out: Optional output tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `ndtr(input)`.
    """
    _ = kwargs
    return _copy_if_out(0.5 * (1.0 + zero_torch.erf(input / math.sqrt(2.0))), out)


def log_ndtr(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Computes the natural logarithm of the area under the standard Gaussian PDF.

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Output tensor with `log(ndtr(input))`.
    """
    return _copy_if_out(zero_torch.log(ndtr(input)), out)


def ndtri(input: Tensor, *, out: Tensor | None = None, **kwargs: OpArg) -> Tensor:
    """Computes the argument `x` for which the area under the Gaussian PDF is equal to `input`.

    Args:
        input: Input probability tensor in `(0, 1)`.
        out: Optional output tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output quantile tensor with `ndtri(input)`.
    """
    _ = kwargs
    return _copy_if_out(zero_torch.erfinv(2.0 * input - 1.0) * math.sqrt(2.0), out)


def log_softmax(
    input: Tensor,
    dim: int,
    *,
    dtype: DTypeLike = None,
) -> Tensor:
    """Computes `softmax(input)` followed by `log`.

    Args:
        input: Input tensor.
        dim: Dimension along which `log_softmax` will be computed.
        dtype: Optional desired data type of returned tensor.

    Returns:
        Output tensor after `log_softmax`.
    """
    return zero_torch.nn.functional.log_softmax(input, dim=dim, dtype=dtype)


def softmax(
    input: Tensor,
    dim: int,
    *,
    dtype: DTypeLike = None,
) -> Tensor:
    """Computes the softmax function along `dim`.

    Args:
        input: Input tensor.
        dim: Dimension along which `softmax` will be computed.
        dtype: Optional desired data type of returned tensor.

    Returns:
        Output probability tensor.
    """
    return zero_torch.nn.functional.softmax(input, dim=dim, dtype=dtype)


def logsumexp(
    input: Tensor,
    dim: int | tuple[int, ...],
    keepdim: bool = False,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """Computes the logarithm of summed exponentials of each row of `input` in `dim`.

    Args:
        input: Input tensor.
        dim: Dimension or tuple of dimensions to reduce.
        keepdim: Whether the output tensor has `dim` retained or not.
        out: Optional output tensor.

    Returns:
        Reduced `logsumexp` tensor.
    """
    return _copy_if_out(zero_torch.logsumexp(input, dim=dim, keepdim=keepdim), out)


def round(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """Rounds elements of `input` to the nearest integer.

    Args:
        input: Input tensor.
        out: Optional output tensor.

    Returns:
        Rounded tensor.
    """
    return _copy_if_out(zero_torch.round(input), out)


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


def gammaln(input: Tensor, *, out: Tensor | None = None, **kwargs: OpArg) -> Tensor:
    """Computes the natural logarithm of the absolute value of the gamma function.

    Args:
        input: Input tensor.
        out: Optional output tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `ln(|Gamma(input)|)`.
    """
    _ = kwargs
    flat: list[Scalar] = _flatten_nested(input.tolist())
    vals = [_safe_lgamma(v) for v in flat]
    return _copy_if_out(
        zero_torch.reshape(Tensor(vals, dtype=input.dtype), input.shape), out
    )


def polygamma(
    n: int, input: Tensor, *, out: Tensor | None = None, **kwargs: OpArg
) -> Tensor:
    """Computes the `n`-th derivative of the digamma function on `input`.

    Args:
        n: Non-negative derivative order.
        input: Input tensor.
        out: Optional output tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `psi^(n)(input)`.
    """
    _ = kwargs
    s = float(n + 1)
    scale = ((-1.0) ** (n + 1)) * math.gamma(s)
    return _copy_if_out(scale * zeta(Tensor(s, dtype=input.dtype), input), out)


def multigammaln(
    input: Tensor, p: int, *, out: Tensor | None = None, **kwargs: OpArg
) -> Tensor:
    """Computes the multivariate log-gamma function of dimension `p`.

    Args:
        input: Input tensor.
        p: Number of dimensions.
        out: Optional output tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with multivariate log-gamma values.
    """
    _ = kwargs
    return _copy_if_out(zero_torch.mvlgamma(input, p, out=out), out)


def zeta(
    input: Tensor, other: Tensor, *, out: Tensor | None = None, **kwargs: OpArg
) -> Tensor:
    """Computes the Hurwitz zeta function.

    Args:
        input: Exponent tensor `x`.
        other: Shift tensor `q`.
        out: Optional output tensor.
        **kwargs: Optional keyword arguments.

    Returns:
        Output tensor with `zeta(input, other)`.
    """
    _ = kwargs
    total = zero_torch.zeros_like(other)
    for k in range(8):
        total = total + (other + float(k)) ** (-input)
    qn = other + 8.0
    safe_sm1 = zero_torch.where(input == 1.0, zero_torch.ones_like(input), input - 1.0)
    tail = (
        (qn ** (1.0 - input)) / safe_sm1
        + 0.5 * (qn ** (-input))
        + (input / 12.0) * (qn ** (-(input + 1.0)))
    )
    return _copy_if_out(total + tail, out)


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


def _eval_chebyshev_u(x: float, deg: int) -> float:
    """Evaluates Chebyshev polynomial of the second kind `U_deg(x)` via recurrence.

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
    u0, u1 = 1.0, 2.0 * x
    for _ in range(2, deg + 1):
        u0, u1 = u1, 2.0 * x * u1 - u0
    return u1


def _eval_chebyshev_v(x: float, deg: int) -> float:
    """Evaluates Chebyshev polynomial of the third kind `V_deg(x)` via recurrence.

    Args:
        x: Evaluation point.
        deg: Non-negative integer degree.

    Returns:
        Evaluated polynomial scalar.
    """
    if deg <= 0:
        return 1.0
    if deg == 1:
        return 2.0 * x - 1.0
    v0, v1 = 1.0, 2.0 * x - 1.0
    for _ in range(2, deg + 1):
        v0, v1 = v1, 2.0 * x * v1 - v0
    return v1


def _eval_chebyshev_w(x: float, deg: int) -> float:
    """Evaluates Chebyshev polynomial of the fourth kind `W_deg(x)` via recurrence.

    Args:
        x: Evaluation point.
        deg: Non-negative integer degree.

    Returns:
        Evaluated polynomial scalar.
    """
    if deg <= 0:
        return 1.0
    if deg == 1:
        return 2.0 * x + 1.0
    w0, w1 = 1.0, 2.0 * x + 1.0
    for _ in range(2, deg + 1):
        w0, w1 = w1, 2.0 * x * w1 - w0
    return w1


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


def _eval_hermite_he(x: float, deg: int) -> float:
    """Evaluates probabilist's Hermite polynomial `He_deg(x)` via recurrence.

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
    h0, h1 = 1.0, x
    for k in range(1, deg):
        h0, h1 = h1, x * h1 - float(k) * h0
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


def chebyshev_polynomial_t(
    input: Tensor, n: Tensor, *, out: Tensor | None = None, **kwargs: OpArg
) -> Tensor:
    """Computes the Chebyshev polynomial of the first kind `T_n(input)`.

    Args:
        input: Input evaluation points tensor.
        n: Polynomial degree tensor.
        out: Optional output tensor.
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
    return _copy_if_out(
        zero_torch.reshape(Tensor(vals, dtype=input.dtype), input.shape), out
    )


def chebyshev_polynomial_u(
    input: Tensor, n: Tensor, *, out: Tensor | None = None
) -> Tensor:
    """Computes the Chebyshev polynomial of the second kind `U_n(input)`.

    Args:
        input: Input evaluation points tensor.
        n: Polynomial degree tensor.
        out: Optional output tensor.

    Returns:
        Evaluated Chebyshev polynomial tensor.
    """
    b_n = zero_torch.broadcast_to(n, input.shape)
    xs = _flatten_nested(input.tolist())
    ns = _flatten_nested(b_n.tolist())
    vals = [
        _eval_chebyshev_u(
            float(x) if isinstance(x, (int, float)) else 0.0,
            int(float(d)) if isinstance(d, (int, float)) else 0,
        )
        for x, d in zip(xs, ns)
    ]
    return _copy_if_out(
        zero_torch.reshape(Tensor(vals, dtype=input.dtype), input.shape), out
    )


def chebyshev_polynomial_v(
    input: Tensor, n: Tensor, *, out: Tensor | None = None
) -> Tensor:
    """Computes the Chebyshev polynomial of the third kind `V_n(input)`.

    Args:
        input: Input evaluation points tensor.
        n: Polynomial degree tensor.
        out: Optional output tensor.

    Returns:
        Evaluated Chebyshev polynomial tensor.
    """
    b_n = zero_torch.broadcast_to(n, input.shape)
    xs = _flatten_nested(input.tolist())
    ns = _flatten_nested(b_n.tolist())
    vals = [
        _eval_chebyshev_v(
            float(x) if isinstance(x, (int, float)) else 0.0,
            int(float(d)) if isinstance(d, (int, float)) else 0,
        )
        for x, d in zip(xs, ns)
    ]
    return _copy_if_out(
        zero_torch.reshape(Tensor(vals, dtype=input.dtype), input.shape), out
    )


def chebyshev_polynomial_w(
    input: Tensor, n: Tensor, *, out: Tensor | None = None
) -> Tensor:
    """Computes the Chebyshev polynomial of the fourth kind `W_n(input)`.

    Args:
        input: Input evaluation points tensor.
        n: Polynomial degree tensor.
        out: Optional output tensor.

    Returns:
        Evaluated Chebyshev polynomial tensor.
    """
    b_n = zero_torch.broadcast_to(n, input.shape)
    xs = _flatten_nested(input.tolist())
    ns = _flatten_nested(b_n.tolist())
    vals = [
        _eval_chebyshev_w(
            float(x) if isinstance(x, (int, float)) else 0.0,
            int(float(d)) if isinstance(d, (int, float)) else 0,
        )
        for x, d in zip(xs, ns)
    ]
    return _copy_if_out(
        zero_torch.reshape(Tensor(vals, dtype=input.dtype), input.shape), out
    )


def shifted_chebyshev_polynomial_t(
    input: Tensor, n: Tensor, *, out: Tensor | None = None
) -> Tensor:
    """Computes the shifted Chebyshev polynomial of the first kind `T_n^*(input) = T_n(2 * input - 1)`.

    Args:
        input: Input evaluation points tensor.
        n: Polynomial degree tensor.
        out: Optional output tensor.

    Returns:
        Evaluated shifted Chebyshev polynomial tensor.
    """
    return chebyshev_polynomial_t(2.0 * input - 1.0, n, out=out)


def shifted_chebyshev_polynomial_u(
    input: Tensor, n: Tensor, *, out: Tensor | None = None
) -> Tensor:
    """Computes the shifted Chebyshev polynomial of the second kind `U_n^*(input) = U_n(2 * input - 1)`.

    Args:
        input: Input evaluation points tensor.
        n: Polynomial degree tensor.
        out: Optional output tensor.

    Returns:
        Evaluated shifted Chebyshev polynomial tensor.
    """
    return chebyshev_polynomial_u(2.0 * input - 1.0, n, out=out)


def shifted_chebyshev_polynomial_v(
    input: Tensor, n: Tensor, *, out: Tensor | None = None
) -> Tensor:
    """Computes the shifted Chebyshev polynomial of the third kind `V_n^*(input) = V_n(2 * input - 1)`.

    Args:
        input: Input evaluation points tensor.
        n: Polynomial degree tensor.
        out: Optional output tensor.

    Returns:
        Evaluated shifted Chebyshev polynomial tensor.
    """
    return chebyshev_polynomial_v(2.0 * input - 1.0, n, out=out)


def shifted_chebyshev_polynomial_w(
    input: Tensor, n: Tensor, *, out: Tensor | None = None
) -> Tensor:
    """Computes the shifted Chebyshev polynomial of the fourth kind `W_n^*(input) = W_n(2 * input - 1)`.

    Args:
        input: Input evaluation points tensor.
        n: Polynomial degree tensor.
        out: Optional output tensor.

    Returns:
        Evaluated shifted Chebyshev polynomial tensor.
    """
    return chebyshev_polynomial_w(2.0 * input - 1.0, n, out=out)


def hermite_polynomial_h(
    input: Tensor, n: Tensor, *, out: Tensor | None = None, **kwargs: OpArg
) -> Tensor:
    """Computes the physicist's Hermite polynomial `H_n(input)`.

    Args:
        input: Input evaluation points tensor.
        n: Polynomial degree tensor.
        out: Optional output tensor.
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
    return _copy_if_out(
        zero_torch.reshape(Tensor(vals, dtype=input.dtype), input.shape), out
    )


def hermite_polynomial_he(
    input: Tensor, n: Tensor, *, out: Tensor | None = None
) -> Tensor:
    """Computes the probabilist's Hermite polynomial `He_n(input)`.

    Args:
        input: Input evaluation points tensor.
        n: Polynomial degree tensor.
        out: Optional output tensor.

    Returns:
        Evaluated probabilist's Hermite polynomial tensor.
    """
    b_n = zero_torch.broadcast_to(n, input.shape)
    xs = _flatten_nested(input.tolist())
    ns = _flatten_nested(b_n.tolist())
    vals = [
        _eval_hermite_he(
            float(x) if isinstance(x, (int, float)) else 0.0,
            int(float(d)) if isinstance(d, (int, float)) else 0,
        )
        for x, d in zip(xs, ns)
    ]
    return _copy_if_out(
        zero_torch.reshape(Tensor(vals, dtype=input.dtype), input.shape), out
    )


def laguerre_polynomial_l(
    input: Tensor, n: Tensor, *, out: Tensor | None = None, **kwargs: OpArg
) -> Tensor:
    """Computes the Laguerre polynomial `L_n(input)`.

    Args:
        input: Input evaluation points tensor.
        n: Polynomial degree tensor.
        out: Optional output tensor.
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
    return _copy_if_out(
        zero_torch.reshape(Tensor(vals, dtype=input.dtype), input.shape), out
    )


def legendre_polynomial_p(
    input: Tensor, n: Tensor, *, out: Tensor | None = None, **kwargs: OpArg
) -> Tensor:
    """Computes the Legendre polynomial of the first kind `P_n(input)`.

    Args:
        input: Input evaluation points tensor.
        n: Polynomial degree tensor.
        out: Optional output tensor.
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
    return _copy_if_out(
        zero_torch.reshape(Tensor(vals, dtype=input.dtype), input.shape), out
    )
