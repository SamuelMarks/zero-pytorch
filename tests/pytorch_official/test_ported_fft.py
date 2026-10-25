"""Ported Fast Fourier Transform (FFT) functional tests."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
from tests.conftest import assert_allclose
from zero_torch.tensor import Tensor


def test_fft_parity() -> None:
    """Verifies `zero_torch.fft` 1D, 2D, N-D, frequency, and shift operations against analytical DFT values."""
    with EagerMode():
        t = Tensor([1.0, 2.0, 3.0, 4.0])

        fft_1d = torch.fft.fft(t)
        assert_allclose(
            fft_1d.tolist(),
            [10.0 + 0.0j, -2.0 + 2.0j, -2.0 + 0.0j, -2.0 - 2.0j],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.fft.ifft(fft_1d).tolist(),
            [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j],
            rtol=1e-4,
            atol=1e-4,
        )
        rfft_1d = torch.fft.rfft(t)
        assert_allclose(
            rfft_1d.tolist(),
            [10.0 + 0.0j, -2.0 + 2.0j, -2.0 + 0.0j],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.fft.irfft(rfft_1d, n=4).tolist(),
            [1.0, 2.0, 3.0, 4.0],
            rtol=1e-4,
            atol=1e-4,
        )

        t2 = Tensor([[1.0, 2.0], [3.0, 4.0]])
        fft_2d = torch.fft.fft2(t2)
        assert_allclose(
            fft_2d.tolist(),
            [[10.0 + 0.0j, -2.0 + 0.0j], [-4.0 + 0.0j, 0.0 + 0.0j]],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.fft.ifft2(fft_2d).tolist(),
            [[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]],
            rtol=1e-4,
            atol=1e-4,
        )
        rfft_2d = torch.fft.rfft2(t2)
        assert_allclose(
            rfft_2d.tolist(),
            [[10.0 + 0.0j, -2.0 + 0.0j], [-4.0 + 0.0j, 0.0 + 0.0j]],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.fft.irfft2(rfft_2d, s=(2, 2)).tolist(),
            [[1.0, 2.0], [3.0, 4.0]],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.fft.fftn(t2).tolist(),
            fft_2d.tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.fft.ifftn(fft_2d).tolist(),
            [[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.fft.rfftn(t2).tolist(),
            rfft_2d.tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.fft.irfftn(rfft_2d, s=(2, 2)).tolist(),
            [[1.0, 2.0], [3.0, 4.0]],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.fft.fftfreq(4).tolist(),
            [0.0, 0.25, -0.5, -0.25],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.fft.rfftfreq(4).tolist(),
            [0.0, 0.25, 0.5],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.fft.fftshift(t).tolist(),
            [3.0, 4.0, 1.0, 2.0],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.fft.ifftshift(t).tolist(),
            [3.0, 4.0, 1.0, 2.0],
            rtol=1e-4,
            atol=1e-4,
        )
