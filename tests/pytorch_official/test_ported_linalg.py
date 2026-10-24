"""Ported linear algebra (`zero_torch.linalg`) parity tests."""

from __future__ import annotations

import numpy as np
import pytest
from ml_switcheroo_compiler.core.config import EagerMode
from ml_switcheroo_compiler.core.errors import UnimplementedMathError

import zero_torch as torch
from zero_torch.tensor import Tensor


def test_linalg_parity() -> None:
    """Verifies `zero_torch.linalg` operations against expected analytical values."""
    with EagerMode():
        t = Tensor([[4.0, 1.0], [1.0, 3.0]])
        t2 = Tensor([[2.0, 0.0], [0.0, 2.0]])
        v3_a = Tensor([1.0, 0.0, 0.0])
        v3_b = Tensor([0.0, 1.0, 0.0])

        assert torch.linalg.cond(t).item() != 0.0
        np.testing.assert_allclose(torch.linalg.det(t).tolist(), 11.0, rtol=1e-4)
        sign, logabsdet = torch.linalg.slogdet(t)
        np.testing.assert_allclose(sign.tolist(), 1.0, rtol=1e-4)
        np.testing.assert_allclose(logabsdet.tolist(), np.log(11.0), rtol=1e-4)
        assert torch.linalg.matrix_rank(t).item() == 2
        assert torch.linalg.matrix_norm(t).item() != 0.0
        assert torch.linalg.vector_norm(t).item() != 0.0

        chol = torch.linalg.cholesky(t)
        np.testing.assert_allclose((chol @ chol.T).tolist(), t.tolist(), rtol=1e-4)
        chol_ex, _ = torch.linalg.cholesky_ex(t)
        np.testing.assert_allclose(chol_ex.tolist(), chol.tolist(), rtol=1e-4)

        eig_vals, eig_vecs = torch.linalg.eig(t)
        assert eig_vals.shape == (2,) and eig_vecs.shape == (2, 2)
        eigh_vals, eigh_vecs = torch.linalg.eigh(t)
        assert eigh_vals.shape == (2,) and eigh_vecs.shape == (2, 2)
        assert torch.linalg.eigvals(t).shape == (2,)
        assert torch.linalg.eigvalsh(t).shape == (2,)

        p_lu, l_lu, u_lu = torch.linalg.lu(t)
        assert p_lu.shape == (2, 2) and l_lu.shape == (2, 2) and u_lu.shape == (2, 2)
        lu_f, pivots = torch.linalg.lu_factor(t)
        assert lu_f.shape == (2, 2) and pivots.shape == (2,)

        q, r = torch.linalg.qr(t)
        np.testing.assert_allclose((q @ r).tolist(), t.tolist(), rtol=1e-4, atol=1e-4)

        u, s, vh = torch.linalg.svd(t)
        assert u.shape == (2, 2) and s.shape == (2,) and vh.shape == (2, 2)
        assert torch.linalg.svdvals(t).shape == (2,)

        inv_t = torch.linalg.inv(t)
        np.testing.assert_allclose(
            (t @ inv_t).tolist(), np.eye(2), rtol=1e-4, atol=1e-4
        )
        inv_ex_t, _ = torch.linalg.inv_ex(t)
        np.testing.assert_allclose(inv_ex_t.tolist(), inv_t.tolist(), rtol=1e-4)

        pinv_t = torch.linalg.pinv(t)
        np.testing.assert_allclose(pinv_t.tolist(), inv_t.tolist(), rtol=1e-4)

        sol = torch.linalg.solve(t, t2)
        np.testing.assert_allclose((t @ sol).tolist(), t2.tolist(), rtol=1e-4)

        assert torch.linalg.tensorinv(Tensor(np.eye(4).reshape(2, 2, 2, 2))).shape == (
            2,
            2,
            2,
            2,
        )
        assert torch.linalg.tensorsolve(
            Tensor(np.eye(4).reshape(2, 2, 2, 2)), t2
        ).shape == (2, 2)
        np.testing.assert_allclose(
            torch.linalg.cross(v3_a, v3_b).tolist(), [0.0, 0.0, 1.0], rtol=1e-4
        )
        assert torch.linalg.matrix_power(t, 2).shape == (2, 2)
        np.testing.assert_allclose(
            torch.linalg.multi_dot([t, t2]).tolist(), (t @ t2).tolist(), rtol=1e-4
        )

        assert torch.linalg.householder_product(t, Tensor([1.0, 1.0])).shape == (2, 2)
        with pytest.raises(UnimplementedMathError):
            torch.linalg.matrix_exp(t)
