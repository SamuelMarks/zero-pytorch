"""Fast Fourier Transforms (FFT) frontend module."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.base import get_op

from zero_torch.tensor import Tensor, _wrap


def fft(
    input: Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
) -> Tensor:
    """Computes the one-dimensional discrete Fourier transform of input.

    Args:
        input: The input tensor.
        n: Signal length. If given, the input will either be zero-padded or trimmed to this length.
        dim: The dimension along which to take the one dimensional FFT.
        norm: Normalization mode ('forward', 'backward', or 'ortho').

    Returns:
        The complex-valued 1D FFT output tensor.
    """
    return _wrap(get_op("Fft")()(input._tensor, n=n, axis=dim, norm=norm))


def ifft(
    input: Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
) -> Tensor:
    """Computes the one-dimensional inverse discrete Fourier transform of input.

    Args:
        input: The input tensor.
        n: Signal length.
        dim: The dimension along which to take the one dimensional IFFT.
        norm: Normalization mode ('forward', 'backward', or 'ortho').

    Returns:
        The 1D inverse FFT output tensor.
    """
    return _wrap(get_op("Ifft")()(input._tensor, n=n, axis=dim, norm=norm))


def rfft(
    input: Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
) -> Tensor:
    """Computes the one-dimensional discrete Fourier transform of real-valued input.

    Args:
        input: The real-valued input tensor.
        n: Signal length.
        dim: The dimension along which to take the one dimensional real FFT.
        norm: Normalization mode ('forward', 'backward', or 'ortho').

    Returns:
        The complex-valued 1D real FFT output tensor.
    """
    return _wrap(get_op("Rfft")()(input._tensor, n=n, axis=dim, norm=norm))


def irfft(
    input: Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
) -> Tensor:
    """Computes the inverse of rfft.

    Args:
        input: The input tensor representing a half-Hermitian spectrum.
        n: Output signal length.
        dim: The dimension along which to take the one dimensional inverse real FFT.
        norm: Normalization mode ('forward', 'backward', or 'ortho').

    Returns:
        The real-valued 1D inverse FFT output tensor.
    """
    return _wrap(get_op("Irfft")()(input._tensor, n=n, axis=dim, norm=norm))


def fft2(
    input: Tensor,
    s: tuple[int, ...] | None = None,
    dim: tuple[int, ...] = (-2, -1),
    norm: str | None = None,
) -> Tensor:
    """Computes the 2-dimensional discrete Fourier transform of input.

    Args:
        input: The input tensor.
        s: Signal size in the transformed dimensions.
        dim: Dimensions to be transformed.
        norm: Normalization mode ('forward', 'backward', or 'ortho').

    Returns:
        The 2D FFT output tensor.
    """
    return _wrap(get_op("Fft2")()(input._tensor, s=s, axes=dim, norm=norm))


def ifft2(
    input: Tensor,
    s: tuple[int, ...] | None = None,
    dim: tuple[int, ...] = (-2, -1),
    norm: str | None = None,
) -> Tensor:
    """Computes the 2-dimensional inverse discrete Fourier transform of input.

    Args:
        input: The input tensor.
        s: Signal size in the transformed dimensions.
        dim: Dimensions to be transformed.
        norm: Normalization mode ('forward', 'backward', or 'ortho').

    Returns:
        The 2D inverse FFT output tensor.
    """
    return _wrap(get_op("Ifft2")()(input._tensor, s=s, axes=dim, norm=norm))


def rfft2(
    input: Tensor,
    s: tuple[int, ...] | None = None,
    dim: tuple[int, ...] = (-2, -1),
    norm: str | None = None,
) -> Tensor:
    """Computes the 2-dimensional discrete Fourier transform of real input.

    Args:
        input: The real-valued input tensor.
        s: Signal size in the transformed dimensions.
        dim: Dimensions to be transformed.
        norm: Normalization mode ('forward', 'backward', or 'ortho').

    Returns:
        The 2D real FFT output tensor.
    """
    return _wrap(get_op("Rfft2")()(input._tensor, s=s, axes=dim, norm=norm))


def irfft2(
    input: Tensor,
    s: tuple[int, ...] | None = None,
    dim: tuple[int, ...] = (-2, -1),
    norm: str | None = None,
) -> Tensor:
    """Computes the inverse of rfft2.

    Args:
        input: The input tensor.
        s: Signal size in the transformed dimensions.
        dim: Dimensions to be transformed.
        norm: Normalization mode ('forward', 'backward', or 'ortho').

    Returns:
        The 2D inverse real FFT output tensor.
    """
    return _wrap(get_op("Irfft2")()(input._tensor, s=s, axes=dim, norm=norm))


def fftn(
    input: Tensor,
    s: tuple[int, ...] | None = None,
    dim: tuple[int, ...] | None = None,
    norm: str | None = None,
) -> Tensor:
    """Computes the N-dimensional discrete Fourier transform of input.

    Args:
        input: The input tensor.
        s: Signal size in the transformed dimensions.
        dim: Dimensions to be transformed.
        norm: Normalization mode ('forward', 'backward', or 'ortho').

    Returns:
        The N-D FFT output tensor.
    """
    return _wrap(get_op("Fftn")()(input._tensor, s=s, axes=dim, norm=norm))


def ifftn(
    input: Tensor,
    s: tuple[int, ...] | None = None,
    dim: tuple[int, ...] | None = None,
    norm: str | None = None,
) -> Tensor:
    """Computes the N-dimensional inverse discrete Fourier transform of input.

    Args:
        input: The input tensor.
        s: Signal size in the transformed dimensions.
        dim: Dimensions to be transformed.
        norm: Normalization mode ('forward', 'backward', or 'ortho').

    Returns:
        The N-D inverse FFT output tensor.
    """
    return _wrap(get_op("Ifftn")()(input._tensor, s=s, axes=dim, norm=norm))


def rfftn(
    input: Tensor,
    s: tuple[int, ...] | None = None,
    dim: tuple[int, ...] | None = None,
    norm: str | None = None,
) -> Tensor:
    """Computes the N-dimensional discrete Fourier transform of real input.

    Args:
        input: The real-valued input tensor.
        s: Signal size in the transformed dimensions.
        dim: Dimensions to be transformed.
        norm: Normalization mode ('forward', 'backward', or 'ortho').

    Returns:
        The N-D real FFT output tensor.
    """
    return _wrap(get_op("Rfftn")()(input._tensor, s=s, axes=dim, norm=norm))


def irfftn(
    input: Tensor,
    s: tuple[int, ...] | None = None,
    dim: tuple[int, ...] | None = None,
    norm: str | None = None,
) -> Tensor:
    """Computes the inverse of rfftn.

    Args:
        input: The input tensor.
        s: Signal size in the transformed dimensions.
        dim: Dimensions to be transformed.
        norm: Normalization mode ('forward', 'backward', or 'ortho').

    Returns:
        The N-D inverse real FFT output tensor.
    """
    return _wrap(get_op("Irfftn")()(input._tensor, s=s, axes=dim, norm=norm))


def fftfreq(n: int, d: float = 1.0) -> Tensor:
    """Computes the discrete Fourier Transform sample frequencies for a signal of size n.

    Args:
        n: The FFT length.
        d: The sampling length scale (spacing between individual samples).

    Returns:
        1D tensor of length n containing the sample frequencies.
    """
    return _wrap(get_op("Fftfreq")()(n, d=d))


def rfftfreq(n: int, d: float = 1.0) -> Tensor:
    """Computes the sample frequencies for rfft with a signal of size n.

    Args:
        n: The real FFT length.
        d: The sampling length scale (spacing between individual samples).

    Returns:
        1D tensor containing the positive sample frequencies.
    """
    return _wrap(get_op("Rfftfreq")()(n, d=d))


def fftshift(input: Tensor, dim: int | tuple[int, ...] | None = None) -> Tensor:
    """Shifts the zero-frequency component to the center of the spectrum.

    Args:
        input: The input tensor.
        dim: The dimensions over which to shift.

    Returns:
        The shifted spectrum tensor.
    """
    return _wrap(get_op("Fftshift")()(input._tensor, axes=dim))


def ifftshift(input: Tensor, dim: int | tuple[int, ...] | None = None) -> Tensor:
    """The inverse of fftshift.

    Args:
        input: The tensor in fftshift order.
        dim: The dimensions over which to shift.

    Returns:
        The unshifted spectrum tensor.
    """
    return _wrap(get_op("Ifftshift")()(input._tensor, axes=dim))
