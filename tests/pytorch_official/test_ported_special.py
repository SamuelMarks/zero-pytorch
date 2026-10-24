"""Ported special functions (`zero_torch.special`) parity tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch as real_torch
from ml_switcheroo_compiler.core.config import EagerMode
from ml_switcheroo_compiler.core.errors import UnimplementedMathError

import zero_torch as torch
from zero_torch.tensor import Tensor


def test_special_parity() -> None:
    """Verifies `zero_torch.special` functions against `torch.special`."""
    with EagerMode():
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = Tensor(arr)
        rt = real_torch.tensor(arr)
        t2 = Tensor([0.5, 1.5, 2.5])
        n = Tensor([1, 2, 3])

        np.testing.assert_allclose(
            torch.special.bessel_j0(t).tolist(),
            real_torch.special.bessel_j0(rt).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            torch.special.bessel_j1(t).tolist(),
            real_torch.special.bessel_j1(rt).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            torch.special.erfcx(t).tolist(),
            real_torch.special.erfcx(rt).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            torch.special.ndtr(t).tolist(),
            real_torch.special.ndtr(rt).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            torch.special.gammaln(t).tolist(),
            real_torch.special.gammaln(rt).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            torch.special.multigammaln(t, 2).tolist(),
            real_torch.special.multigammaln(rt, 2).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert torch.special.ndtri(Tensor([0.25, 0.5, 0.75])).shape == (3,)
        assert torch.special.polygamma(1, t).shape == (3,)
        assert torch.special.zeta(t, t2).shape == (3,)

        assert torch.special.bessel_y0(t).shape == (3,)
        assert torch.special.bessel_y1(t).shape == (3,)

        np.testing.assert_allclose(
            torch.special.modified_bessel_i0(t).tolist(),
            real_torch.special.modified_bessel_i0(rt).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            torch.special.modified_bessel_i1(t).tolist(),
            real_torch.special.modified_bessel_i1(rt).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert torch.special.modified_bessel_k0(t).shape == (3,)
        assert torch.special.modified_bessel_k1(t).shape == (3,)

        for poly_fn in [
            torch.special.chebyshev_polynomial_t,
            torch.special.hermite_polynomial_h,
            torch.special.laguerre_polynomial_l,
            torch.special.legendre_polynomial_p,
        ]:
            with pytest.raises(UnimplementedMathError):
                poly_fn(t, n)
