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
        chol_u = torch.linalg.cholesky(t, upper=True)
        assert_allclose((chol_u.T @ chol_u).tolist(), t.tolist(), rtol=1e-4)
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
        p_nopiv, _, _ = torch.linalg.lu(t, pivot=False)
        assert_allclose(p_nopiv.tolist(), eye_nested(2), rtol=1e-4)
        lu_out = (torch.zeros((2, 2)), torch.zeros((2, 2)), torch.zeros((2, 2)))
        assert torch.linalg.lu(t, out=lu_out) is lu_out
        lu_f, pivots = torch.linalg.lu_factor(t)
        assert lu_f.shape == (2, 2) and pivots.shape == (2,)

        q, r = torch.linalg.qr(t)
        assert_allclose((q @ r).tolist(), t.tolist(), rtol=1e-4, atol=1e-4)
        q_r_mode, r_only = torch.linalg.qr(t, mode="r")
        assert q_r_mode.shape == (0,) and r_only.shape == (2, 2)
        qr_out = (torch.zeros((2, 2)), torch.zeros((2, 2)))
        assert torch.linalg.qr(t, out=qr_out) is qr_out

        u, s, vh = torch.linalg.svd(t)
        assert u.shape == (2, 2) and s.shape == (2,) and vh.shape == (2, 2)
        rect_tall = Tensor([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
        u_red, s_red, vh_red = torch.linalg.svd(rect_tall, full_matrices=False)
        assert u_red.shape == (3, 2) and s_red.shape == (2,) and vh_red.shape == (2, 2)
        rect_wide = Tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        u_w, s_w, vh_w = torch.linalg.svd(rect_wide, full_matrices=False)
        assert u_w.shape == (2, 2) and s_w.shape == (2,) and vh_w.shape == (2, 3)
        assert torch.linalg.svdvals(t).shape == (2,)

        inv_t = torch.linalg.inv(t)
        assert_allclose((t @ inv_t).tolist(), eye_nested(2), rtol=1e-4, atol=1e-4)
        inv_ex_t, _ = torch.linalg.inv_ex(t)
        assert_allclose(inv_ex_t.tolist(), inv_t.tolist(), rtol=1e-4)

        pinv_t = torch.linalg.pinv(t)
        assert_allclose(pinv_t.tolist(), inv_t.tolist(), rtol=1e-4)
        pinv_rcond = torch.linalg.pinv(t, rcond=1e-5, atol=1e-6)
        assert_allclose(pinv_rcond.tolist(), inv_t.tolist(), rtol=1e-4)
        pinv_ten = torch.linalg.pinv(t, rtol=Tensor(1e-5), atol=Tensor(1e-6))
        assert_allclose(pinv_ten.tolist(), inv_t.tolist(), rtol=1e-4)
        pinv_atol_only = torch.linalg.pinv(t, atol=1e-6)
        assert_allclose(pinv_atol_only.tolist(), inv_t.tolist(), rtol=1e-4)

        sol = torch.linalg.solve(t, t2)
        assert_allclose((t @ sol).tolist(), t2.tolist(), rtol=1e-4)
        sol_right = torch.linalg.solve(t, t2, left=False)
        assert_allclose((sol_right @ t).tolist(), t2.tolist(), rtol=1e-4)

        eye_4d = torch.eye(4).reshape(2, 2, 2, 2)
        assert torch.linalg.tensorinv(eye_4d).shape == (2, 2, 2, 2)
        assert torch.linalg.tensorsolve(eye_4d, t2).shape == (2, 2)
        assert_allclose(
            torch.linalg.cross(v3_a, v3_b).tolist(), [0.0, 0.0, 1.0], rtol=1e-4
        )
        v3_col_a = Tensor([[1.0], [0.0], [0.0]])
        v3_col_b = Tensor([[0.0], [1.0], [0.0]])
        assert_allclose(
            torch.linalg.cross(v3_col_a, v3_col_b, dim=0).tolist(),
            [[0.0], [0.0], [1.0]],
            rtol=1e-4,
        )
        assert torch.linalg.matrix_power(t, 2).shape == (2, 2)
        assert_allclose(
            torch.linalg.multi_dot([t, t2]).tolist(), (t @ t2).tolist(), rtol=1e-4
        )

        assert torch.linalg.householder_product(t, Tensor([1.0, 1.0])).shape == (2, 2)
        out_hp = Tensor([[0.0, 0.0], [0.0, 0.0]])
        assert (
            torch.linalg.householder_product(t, Tensor([1.0, 1.0]), out=out_hp)
            is out_hp
        )
        out_mp = Tensor([[0.0, 0.0], [0.0, 0.0]])
        assert torch.linalg.matrix_power(t, 2, out=out_mp) is out_mp
        assert torch.linalg.det(input=t).item() != 0.0
        assert torch.linalg.slogdet(input=t)[0].item() == 1.0
        assert torch.linalg.slogdet(nan=t)[0].item() == 1.0
        assert torch.linalg.norm(t).item() != 0.0
        assert torch.linalg.norm(t, ord="fro").item() != 0.0
        assert torch.linalg.norm(t, dim=(-2, -1)).item() != 0.0
        lu_ex_m, lu_ex_p, lu_ex_info = torch.linalg.lu_factor_ex(t)
        assert lu_ex_m.shape == (2, 2) and lu_ex_info.item() == 0
        assert torch.linalg.lu_solve(lu_ex_m, lu_ex_p, t2).shape == (2, 2)
        assert torch.linalg.lu_solve(
            lu_ex_m, Tensor([0, 1, 0]), t2, left=False, adjoint=True
        ).shape == (2, 2)
        ldl_m, ldl_p = torch.linalg.ldl_factor(t)
        assert ldl_m.shape == (2, 2)
        ldl_ex_m, _ldl_ex_p, ldl_ex_info = torch.linalg.ldl_factor_ex(t)
        assert ldl_ex_m.shape == (2, 2) and ldl_ex_info.item() == 0
        assert torch.linalg.ldl_solve(ldl_m, ldl_p, t2).shape == (2, 2)
        assert torch.linalg.solve_triangular(t, t2, upper=True).shape == (2, 2)
        assert torch.linalg.solve_triangular(
            t, t2, upper=False, left=False, unitriangular=True
        ).shape == (2, 2)
        assert torch.linalg.solve_triangular(
            t, t2, upper=True, unitriangular=True
        ).shape == (2, 2)
        assert_allclose(torch.linalg.diagonal(t).tolist(), [4.0, 3.0], rtol=1e-4)
        assert_allclose(
            torch.linalg.diagonal(t, dim1=1, dim2=0).tolist(), [4.0, 3.0], rtol=1e-4
        )
        assert_allclose(
            torch.linalg.matmul(t, t2).tolist(), (t @ t2).tolist(), rtol=1e-4
        )
        assert_allclose(
            torch.linalg.vecdot(Tensor([1.0, 2.0]), Tensor([3.0, 4.0])).tolist(),
            11.0,
            rtol=1e-4,
        )
        assert torch.linalg.vander(Tensor([1.0, 2.0, 3.0])).shape == (3, 3)
        assert torch.linalg.vander(Tensor([1.0, 2.0, 3.0]), N=2).shape == (3, 2)
        u_pol, p_pol = torch.linalg.polar(t)
        assert_allclose((u_pol @ p_pol).tolist(), t.tolist(), rtol=1e-4, atol=1e-4)
        sqrth_t = torch.linalg.matrix_sqrth(t)
        assert_allclose((sqrth_t @ sqrth_t).tolist(), t.tolist(), rtol=1e-4, atol=1e-4)
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
