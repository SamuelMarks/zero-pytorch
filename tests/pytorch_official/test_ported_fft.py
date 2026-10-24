"""Ported Fast Fourier Transform (FFT) parity tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch as real_torch
from ml_switcheroo_compiler.core.config import EagerMode
from ml_switcheroo_compiler.core.errors import UnimplementedMathError

import zero_torch as torch
from zero_torch.tensor import Tensor


def test_fft_parity() -> None:
    """Verifies numerical parity between `zero_torch.fft` and `torch.fft`."""
    with EagerMode():
        arr1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        t = Tensor(arr1d)
        rt = real_torch.tensor(arr1d)

        np.testing.assert_allclose(
            torch.fft.fft(t).tolist(),
            real_torch.fft.fft(rt).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            torch.fft.ifft(t).tolist(),
            real_torch.fft.ifft(rt).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            torch.fft.rfft(t).tolist(),
            real_torch.fft.rfft(rt).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            torch.fft.irfft(t).tolist(),
            real_torch.fft.irfft(rt).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )

        arr2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t2 = Tensor(arr2d)
        rt2 = real_torch.tensor(arr2d)

        np.testing.assert_allclose(
            torch.fft.fft2(t2).tolist(),
            real_torch.fft.fft2(rt2).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            torch.fft.ifft2(t2).tolist(),
            real_torch.fft.ifft2(rt2).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            torch.fft.rfft2(t2).tolist(),
            real_torch.fft.rfft2(rt2).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            torch.fft.irfft2(t2).tolist(),
            real_torch.fft.irfft2(rt2).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            torch.fft.fftn(t2).tolist(),
            real_torch.fft.fftn(rt2).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            torch.fft.ifftn(t2).tolist(),
            real_torch.fft.ifftn(rt2).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            torch.fft.rfftn(t2).tolist(),
            real_torch.fft.rfftn(rt2).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            torch.fft.irfftn(t2).tolist(),
            real_torch.fft.irfftn(rt2).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            torch.fft.fftfreq(4).tolist(),
            real_torch.fft.fftfreq(4).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            torch.fft.rfftfreq(4).tolist(),
            real_torch.fft.rfftfreq(4).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            torch.fft.fftshift(t).tolist(),
            real_torch.fft.fftshift(rt).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            torch.fft.ifftshift(t).tolist(),
            real_torch.fft.ifftshift(rt).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        _ = pytest
        _ = UnimplementedMathError
