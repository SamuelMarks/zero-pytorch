"""Special Functions frontend module."""

from zero_torch.tensor import Tensor, _wrap

try:
    import ml_switcheroo_compiler.ops.binary.special as _binary_ops
    import ml_switcheroo_compiler.ops.unary.special as _ops
except ImportError:

    class _MockSpecial:
        pass

    _ops = _MockSpecial()
    _binary_ops = _MockSpecial()


def _get_op(name, binary=False):
    target_ops = _binary_ops if binary else _ops
    if not hasattr(target_ops, name):
        raise NotImplementedError(f"Compiler backend missing special op: {name}")
    return getattr(target_ops, name)


# Bessel Functions
def bessel_j0(input: Tensor, **kwargs) -> Tensor:
    """Computes the Bessel function of the first kind of order 0."""
    return _wrap(_get_op("bessel_j0")(input._tensor, **kwargs))


def bessel_j1(input: Tensor, **kwargs) -> Tensor:
    """Computes the Bessel function of the first kind of order 1."""
    return _wrap(_get_op("bessel_j1")(input._tensor, **kwargs))


def bessel_y0(input: Tensor, **kwargs) -> Tensor:
    """Computes the Bessel function of the second kind of order 0."""
    return _wrap(_get_op("bessel_y0")(input._tensor, **kwargs))


def bessel_y1(input: Tensor, **kwargs) -> Tensor:
    """Computes the Bessel function of the second kind of order 1."""
    return _wrap(_get_op("bessel_y1")(input._tensor, **kwargs))


def modified_bessel_i0(input: Tensor, **kwargs) -> Tensor:
    """Computes the modified Bessel function of the first kind of order 0."""
    return _wrap(_get_op("modified_bessel_i0")(input._tensor, **kwargs))


def modified_bessel_i1(input: Tensor, **kwargs) -> Tensor:
    """Computes the modified Bessel function of the first kind of order 1."""
    return _wrap(_get_op("modified_bessel_i1")(input._tensor, **kwargs))


def modified_bessel_k0(input: Tensor, **kwargs) -> Tensor:
    """Computes the modified Bessel function of the second kind of order 0."""
    return _wrap(_get_op("modified_bessel_k0")(input._tensor, **kwargs))


def modified_bessel_k1(input: Tensor, **kwargs) -> Tensor:
    """Computes the modified Bessel function of the second kind of order 1."""
    return _wrap(_get_op("modified_bessel_k1")(input._tensor, **kwargs))


# Gamma and Error Functions
def erfcx(input: Tensor, **kwargs) -> Tensor:
    """Computes the scaled complementary error function."""
    return _wrap(_get_op("erfcx")(input._tensor, **kwargs))


def ndtr(input: Tensor, **kwargs) -> Tensor:
    """Computes the area under the standard normal cumulative distribution function."""
    return _wrap(_get_op("ndtr")(input._tensor, **kwargs))


def ndtri(input: Tensor, **kwargs) -> Tensor:
    """Computes the inverse of the standard normal cumulative distribution function."""
    return _wrap(_get_op("ndtri")(input._tensor, **kwargs))


def gammaln(input: Tensor, **kwargs) -> Tensor:
    """Computes the natural logarithm of the absolute value of the gamma function."""
    return _wrap(_get_op("gammaln")(input._tensor, **kwargs))


def polygamma(n: int, input: Tensor, **kwargs) -> Tensor:
    """Computes the n-th derivative of the digamma function."""
    return _wrap(_get_op("polygamma", binary=True)(n, input._tensor, **kwargs))


def multigammaln(input: Tensor, p: int, **kwargs) -> Tensor:
    """Computes the multivariate log-gamma function."""
    return _wrap(_get_op("multigammaln")(input._tensor, p, **kwargs))


def zeta(input: Tensor, other: Tensor, **kwargs) -> Tensor:
    """Computes the Hurwitz zeta function."""
    return _wrap(_get_op("zeta", binary=True)(input._tensor, other._tensor, **kwargs))


# Legendre and Polynomials
def chebyshev_polynomial_t(input: Tensor, n: Tensor, **kwargs) -> Tensor:
    """Computes the Chebyshev polynomial of the first kind."""
    return _wrap(
        _get_op("chebyshev_polynomial_t", binary=True)(
            input._tensor, n._tensor, **kwargs
        )
    )


def hermite_polynomial_h(input: Tensor, n: Tensor, **kwargs) -> Tensor:
    """Computes the physicist's Hermite polynomial."""
    return _wrap(
        _get_op("hermite_polynomial_h", binary=True)(input._tensor, n._tensor, **kwargs)
    )


def laguerre_polynomial_l(input: Tensor, n: Tensor, **kwargs) -> Tensor:
    """Computes the Laguerre polynomial."""
    return _wrap(
        _get_op("laguerre_polynomial_l", binary=True)(
            input._tensor, n._tensor, **kwargs
        )
    )


def legendre_polynomial_p(input: Tensor, n: Tensor, **kwargs) -> Tensor:
    """Computes the Legendre polynomial."""
    return _wrap(
        _get_op("legendre_polynomial_p", binary=True)(
            input._tensor, n._tensor, **kwargs
        )
    )
