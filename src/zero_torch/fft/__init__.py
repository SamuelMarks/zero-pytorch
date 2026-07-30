"""Fast Fourier Transforms (FFT) frontend module."""

from __future__ import annotations

from zero_torch.tensor import Tensor, _wrap

try:
    import ml_switcheroo_compiler.ops.generated.missing_ops_misc as _missing_ops_misc
    import ml_switcheroo_compiler.ops.signal as _ops
except ImportError:  # pragma: no cover

    class _MockFFT:  # pragma: no cover
        pass

    _ops = _MockFFT()  # pragma: no cover
    _missing_ops_misc = _MockFFT()  # pragma: no cover


def _get_op(name):
    if hasattr(_ops, name):
        return getattr(_ops, name)
    if hasattr(_missing_ops_misc, name):
        return getattr(_missing_ops_misc, name)
    raise NotImplementedError(f"Compiler backend missing fft op: {name}")


def fft(
    input: Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
    **kwargs,
) -> Tensor:
    """Computes the 1D discrete Fourier transform."""
    return _wrap(_get_op("fft")(input._tensor, n=n, axis=dim, norm=norm, **kwargs))


def ifft(
    input: Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
    **kwargs,
) -> Tensor:
    """Computes the 1D inverse discrete Fourier transform."""
    return _wrap(_get_op("ifft")(input._tensor, n=n, axis=dim, norm=norm, **kwargs))


def rfft(
    input: Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
    **kwargs,
) -> Tensor:
    """Computes the 1D real-to-complex discrete Fourier transform."""
    return _wrap(_get_op("rfft")(input._tensor, n=n, axis=dim, norm=norm, **kwargs))


def irfft(
    input: Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
    **kwargs,
) -> Tensor:
    """Computes the 1D complex-to-real inverse discrete Fourier transform."""
    return _wrap(_get_op("irfft")(input._tensor, n=n, axis=dim, norm=norm, **kwargs))


def fft2(
    input: Tensor,
    s: tuple | None = None,
    dim: tuple = (-2, -1),
    norm: str | None = None,
    **kwargs,
) -> Tensor:
    """Computes the 2D discrete Fourier transform."""
    return _wrap(_get_op("fft2")(input._tensor, s=s, axes=dim, norm=norm, **kwargs))


def ifft2(
    input: Tensor,
    s: tuple | None = None,
    dim: tuple = (-2, -1),
    norm: str | None = None,
    **kwargs,
) -> Tensor:
    """Computes the 2D inverse discrete Fourier transform."""
    return _wrap(_get_op("ifft2")(input._tensor, s=s, axes=dim, norm=norm, **kwargs))


def rfft2(
    input: Tensor,
    s: tuple | None = None,
    dim: tuple = (-2, -1),
    norm: str | None = None,
    **kwargs,
) -> Tensor:
    """Computes the 2D real-to-complex discrete Fourier transform."""
    return _wrap(_get_op("rfft2")(input._tensor, s=s, axes=dim, norm=norm, **kwargs))


def irfft2(
    input: Tensor,
    s: tuple | None = None,
    dim: tuple = (-2, -1),
    norm: str | None = None,
    **kwargs,
) -> Tensor:
    """Computes the 2D complex-to-real inverse discrete Fourier transform."""
    return _wrap(_get_op("irfft2")(input._tensor, s=s, axes=dim, norm=norm, **kwargs))


def fftn(
    input: Tensor,
    s: tuple | None = None,
    dim: tuple | None = None,
    norm: str | None = None,
    **kwargs,
) -> Tensor:
    """Computes the N dimensional discrete Fourier transform."""
    return _wrap(_get_op("fftn")(input._tensor, s=s, axes=dim, norm=norm, **kwargs))


def ifftn(
    input: Tensor,
    s: tuple | None = None,
    dim: tuple | None = None,
    norm: str | None = None,
    **kwargs,
) -> Tensor:
    """Computes the N dimensional inverse discrete Fourier transform."""
    return _wrap(_get_op("ifftn")(input._tensor, s=s, axes=dim, norm=norm, **kwargs))


def rfftn(
    input: Tensor,
    s: tuple | None = None,
    dim: tuple | None = None,
    norm: str | None = None,
    **kwargs,
) -> Tensor:
    """Computes the N dimensional real-to-complex discrete Fourier transform."""
    return _wrap(_get_op("rfftn")(input._tensor, s=s, axes=dim, norm=norm, **kwargs))


def irfftn(
    input: Tensor,
    s: tuple | None = None,
    dim: tuple | None = None,
    norm: str | None = None,
    **kwargs,
) -> Tensor:
    """Computes the N dimensional inverse of a real-to-complex discrete Fourier transform."""
    return _wrap(_get_op("irfftn")(input._tensor, s=s, axes=dim, norm=norm, **kwargs))


def fftfreq(n: int, d: float = 1.0, **kwargs) -> Tensor:
    """Computes the discrete Fourier Transform sample frequencies."""
    return _wrap(_get_op("fftfreq")(n, d=d, **kwargs))


def rfftfreq(n: int, d: float = 1.0, **kwargs) -> Tensor:
    """Computes the discrete Fourier Transform sample frequencies for rfft."""
    return _wrap(_get_op("rfftfreq")(n, d=d, **kwargs))


def fftshift(input: Tensor, dim=None) -> Tensor:
    """Shifts the zero-frequency component to the center of the spectrum."""
    return _wrap(_get_op("fftshift")(input._tensor, axes=dim))


def ifftshift(input: Tensor, dim=None) -> Tensor:
    """The inverse of fftshift."""
    return _wrap(_get_op("ifftshift")(input._tensor, axes=dim))
