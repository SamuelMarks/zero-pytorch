"""Ported special functions (`zero_torch.special`) analytical tests."""

from __future__ import annotations

import math

from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
from tests.conftest import assert_allclose
from zero_torch.tensor import Tensor


def test_special_parity() -> None:
    """Verifies `zero_torch.special` functions against analytical mathematical values."""
    with EagerMode():
        vals = [1.0, 2.0, 3.0]
        t = Tensor(vals)
        t2 = Tensor([0.5, 1.5, 2.5])
        n = Tensor([1, 2, 3])

        assert_allclose(
            torch.special.bessel_j0(t).tolist(),
            [0.76519769, 0.22389078, -0.26005195],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.special.bessel_j1(t).tolist(),
            [0.44005059, 0.57672481, 0.33905896],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.special.erfcx(t).tolist(),
            [math.exp(x * x) * math.erfc(x) for x in vals],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.special.ndtr(t).tolist(),
            [0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) for x in vals],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.special.gammaln(t).tolist(),
            [math.lgamma(x) for x in vals],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.special.multigammaln(t, 2).tolist(),
            torch.mvlgamma(t, 2).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert torch.special.ndtri(Tensor([0.25, 0.5, 0.75])).shape == (3,)
        assert torch.special.polygamma(1, t).shape == (3,)
        assert torch.special.zeta(t, t2).shape == (3,)

        assert torch.special.bessel_y0(t).shape == (3,)
        assert torch.special.bessel_y1(t).shape == (3,)

        assert_allclose(
            torch.special.modified_bessel_i0(t).tolist(),
            [1.26606588, 2.27958530, 4.88079259],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.special.modified_bessel_i1(t).tolist(),
            [0.56515910, 1.59063685, 3.95337022],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.special.modified_bessel_k0(t).tolist(),
            [0.42102444, 0.11389387, 0.03473950],
            rtol=1e-3,
            atol=1e-3,
        )
        assert_allclose(
            torch.special.modified_bessel_k1(t).tolist(),
            [0.60190723, 0.13986588, 0.04015643],
            rtol=1e-3,
            atol=1e-3,
        )
        t_large = Tensor([2.5, 3.5])
        assert_allclose(
            torch.special.modified_bessel_k0(t_large).tolist(),
            [0.06234755, 0.01959890],
            rtol=1e-2,
            atol=1e-2,
        )
        assert_allclose(
            torch.special.modified_bessel_k1(t_large).tolist(),
            [0.07389082, 0.02223940],
            rtol=1e-2,
            atol=1e-2,
        )
        g0 = torch.special.gammaln(Tensor([0.0])).item()
        assert isinstance(g0, float) and math.isinf(g0)

        n_degs = Tensor([0.0, 1.0, 2.0])
        assert_allclose(
            torch.special.chebyshev_polynomial_t(t, n_degs).tolist(),
            [1.0, 2.0, 17.0],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.special.hermite_polynomial_h(t, n_degs).tolist(),
            [1.0, 4.0, 34.0],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.special.laguerre_polynomial_l(t, n_degs).tolist(),
            [1.0, -1.0, -0.5],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            torch.special.legendre_polynomial_p(t, n_degs).tolist(),
            [1.0, 2.0, 13.0],
            rtol=1e-4,
            atol=1e-4,
        )
        for z_poly in [
            torch.special.chebyshev_polynomial_t,
            torch.special.hermite_polynomial_h,
            torch.special.laguerre_polynomial_l,
            torch.special.legendre_polynomial_p,
        ]:
            assert z_poly(t, n).shape == (3,)
