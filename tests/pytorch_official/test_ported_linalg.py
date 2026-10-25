"""Ported linear algebra (`zero_torch.linalg`) parity tests."""

from __future__ import annotations

import math

import pytest
from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
from tests.conftest import assert_allclose, eye_nested
from zero_torch.tensor import Tensor


def test_linalg_parity() -> None:
    """Verifies `zero_torch.linalg` operations against expected analytical values."""
    with EagerMode():
        t = Tensor([[4.0, 1.0], [1.0, 3.0]])
        t2 = Tensor([[2.0, 0.0], [0.0, 2.0]])
        v3_a = Tensor([1.0, 0.0, 0.0])
        v3_b = Tensor([0.0, 1.0, 0.0])

        assert torch.linalg.cond(t).item() != 0.0
        assert_allclose(torch.linalg.det(t).tolist(), 11.0, rtol=1e-4)
        sign, logabsdet = torch.linalg.slogdet(t)
        assert_allclose(sign.tolist(), 1.0, rtol=1e-4)
        assert_allclose(logabsdet.tolist(), math.log(11.0), rtol=1e-4)
        assert torch.linalg.matrix_rank(t).item() == 2
        assert torch.linalg.matrix_norm(t).item() != 0.0
        assert torch.linalg.vector_norm(t).item() != 0.0

        chol = torch.linalg.cholesky(t)
        assert_allclose((chol @ chol.T).tolist(), t.tolist(), rtol=1e-4)
        chol_ex, _ = torch.linalg.cholesky_ex(t)
        assert_allclose(chol_ex.tolist(), chol.tolist(), rtol=1e-4)

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
        assert_allclose((q @ r).tolist(), t.tolist(), rtol=1e-4, atol=1e-4)

        u, s, vh = torch.linalg.svd(t)
        assert u.shape == (2, 2) and s.shape == (2,) and vh.shape == (2, 2)
        assert torch.linalg.svdvals(t).shape == (2,)

        inv_t = torch.linalg.inv(t)
        assert_allclose((t @ inv_t).tolist(), eye_nested(2), rtol=1e-4, atol=1e-4)
        inv_ex_t, _ = torch.linalg.inv_ex(t)
        assert_allclose(inv_ex_t.tolist(), inv_t.tolist(), rtol=1e-4)

        pinv_t = torch.linalg.pinv(t)
        assert_allclose(pinv_t.tolist(), inv_t.tolist(), rtol=1e-4)

        sol = torch.linalg.solve(t, t2)
        assert_allclose((t @ sol).tolist(), t2.tolist(), rtol=1e-4)

        eye_4d = torch.eye(4).reshape(2, 2, 2, 2)
        assert torch.linalg.tensorinv(eye_4d).shape == (2, 2, 2, 2)
        assert torch.linalg.tensorsolve(eye_4d, t2).shape == (2, 2)
        assert_allclose(
            torch.linalg.cross(v3_a, v3_b).tolist(), [0.0, 0.0, 1.0], rtol=1e-4
        )
        assert torch.linalg.matrix_power(t, 2).shape == (2, 2)
        assert_allclose(
            torch.linalg.multi_dot([t, t2]).tolist(), (t @ t2).tolist(), rtol=1e-4
        )

        assert torch.linalg.householder_product(t, Tensor([1.0, 1.0])).shape == (2, 2)
        m_exp = torch.linalg.matrix_exp(Tensor([[0.0, 1.0], [-1.0, 0.0]]))
        assert_allclose(
            m_exp.tolist(),
            [[math.cos(1.0), math.sin(1.0)], [-math.sin(1.0), math.cos(1.0)]],
            rtol=1e-4,
            atol=1e-4,
        )
        sol_ex, info_ex = torch.linalg.solve_ex(t, t2)
        assert_allclose((t @ sol_ex).tolist(), t2.tolist(), rtol=1e-4)
        assert info_ex.item() == 0
        lstsq_sol, _, _, _ = torch.linalg.lstsq(t, t2)
        assert_allclose((t @ lstsq_sol).tolist(), t2.tolist(), rtol=1e-4, atol=1e-4)
        m_ords: list[int | float | str] = [
            "fro",
            "nuc",
            1,
            -1,
            2,
            -2,
            float("inf"),
            float("-inf"),
        ]
        for m_ord in m_ords:
            assert torch.linalg.matrix_norm(t, ord=m_ord).shape == ()
            assert torch.linalg.matrix_norm(t, ord=m_ord, keepdim=True).shape == (1, 1)
        with pytest.raises(ValueError):
            torch.linalg.matrix_norm(t, ord="invalid")
        v_ords: list[int | float] = [0, 1, 2, 3.0, float("inf"), float("-inf")]
        for v_ord in v_ords:
            assert torch.linalg.vector_norm(t, ord=v_ord).shape == ()
            assert torch.linalg.vector_norm(t, ord=v_ord, keepdim=True).shape == (1, 1)
            assert torch.linalg.vector_norm(t, ord=v_ord, dim=-1).shape == (2,)
            assert torch.linalg.vector_norm(
                t, ord=v_ord, dim=-1, keepdim=True
            ).shape == (2, 1)
