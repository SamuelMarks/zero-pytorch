"""Ported autograd functional tests verifying analytical and numerical gradients."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
import zero_torch.nn as nn
import zero_torch.nn.functional as F
from tests.conftest import assert_allclose
from zero_torch.autograd.math_ops import (
    CosBackward,
    ExpBackward,
    LogBackward,
    MeanBackward,
    PermuteBackward,
    ReluBackward,
    ReshapeBackward,
    SigmoidBackward,
    SinBackward,
    SqrtBackward,
    SqueezeBackward,
    SumBackward,
    TanhBackward,
    TransposeBackward,
    UnsqueezeBackward,
)


def test_functional_autograd() -> None:
    """Verifies forward and backward gradients across arithmetic and reduction ops."""
    with EagerMode():
        x = torch.tensor([2.0], requires_grad=True)
        y = torch.tensor([3.0], requires_grad=True)
        z = x * y
        z.backward()

        assert x.grad is not None
        assert y.grad is not None
        assert x.grad.item() == 3.0
        assert y.grad.item() == 2.0

        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = torch.tensor([10.0, 20.0], requires_grad=True)
        out = (a + b).sum()
        out.backward()
        assert a.grad is not None
        assert b.grad is not None
        assert_allclose(a.grad.tolist(), [[1.0, 1.0], [1.0, 1.0]])
        assert_allclose(b.grad.tolist(), [2.0, 2.0])


def test_autograd_unary_reduction_shape_and_modules() -> None:
    """Verifies autograd across unary, activation, reduction, shape ops, detach(), no_grad() decorator, and Linear MLP."""
    with EagerMode():
        # Decorator @torch.no_grad() and @torch.set_grad_enabled(False)
        @torch.no_grad()
        def run_no_grad(t: torch.Tensor) -> torch.Tensor:
            """Multiplies `t` by 2 with gradients disabled.

            Args:
                t: Input tensor.

            Returns:
                Scaled tensor.
            """
            return t * 2.0

        t_req = torch.tensor([3.0], requires_grad=True)
        res_ng = run_no_grad(t_req)
        assert not res_ng.requires_grad
        assert res_ng.grad_fn is None

        # Detach preserves storage and clears requires_grad and grad_fn
        y_active = t_req * 4.0
        y_det = y_active.detach()
        assert not y_det.requires_grad
        assert y_det.grad_fn is None
        assert y_det._tensor is y_active._tensor

        # Unary & activation ops with requires_grad=True and requires_grad=False
        u = torch.tensor([1.0, 4.0], requires_grad=True)
        loss_u = (
            torch.exp(u)
            + torch.log(u)
            + torch.sin(u)
            + torch.cos(u)
            + torch.tanh(u)
            + F.sigmoid(u)
            + F.relu(u)
            + torch.sqrt(u)
        ).mean()
        loss_u.backward()
        assert u.grad is not None and u.grad.shape == (2,)

        # Non-grad branches of Backward classes
        c = torch.tensor([1.0, 2.0], requires_grad=False)
        for cls in (
            ExpBackward,
            LogBackward,
            SinBackward,
            CosBackward,
            TanhBackward,
            SigmoidBackward,
            ReluBackward,
            SqrtBackward,
        ):
            ctx_u = torch.autograd.Context()
            if cls in (ExpBackward, TanhBackward, SigmoidBackward, SqrtBackward):
                ctx_u.save_for_backward(c, c)
            else:
                ctx_u.save_for_backward(c)
            assert cls.backward(ctx_u, torch.ones((2,))) is None

        # MeanBackward and SumBackward with axis & keepdims
        m_in = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        m_axis = m_in.mean(dim=1, keepdim=False).sum()
        m_axis.backward()
        assert m_in.grad is not None
        assert_allclose(m_in.grad.tolist(), [[0.5, 0.5], [0.5, 0.5]])

        m_kd = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        torch.sum(m_kd, axis=1, keepdims=True).sum().backward()
        assert m_kd.grad is not None
        assert_allclose(m_kd.grad.tolist(), [[1.0, 1.0], [1.0, 1.0]])

        ctx_s = torch.autograd.Context()
        ctx_s.save_for_backward(c, None, False)
        assert SumBackward.backward(ctx_s, torch.ones(())) is None
        assert MeanBackward.backward(ctx_s, torch.ones(())) == (None, None, None)

        # Shape ops: reshape, view, transpose, permute, unsqueeze, squeeze, concatenate, stack
        s1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        s_reshaped = s1.view(3, 2).transpose(0, 1)
        s_perm = torch.permute(s_reshaped.unsqueeze(-1), (2, 0, 1)).squeeze(0)
        s_perm.sum().backward()
        assert s1.grad is not None and s1.grad.shape == (2, 3)

        for shape_cls, saved in [
            (ReshapeBackward, (c,)),
            (TransposeBackward, (c, 0, 1)),
            (PermuteBackward, (c, (0,))),
            (UnsqueezeBackward, (c,)),
            (SqueezeBackward, (c,)),
        ]:
            ctx_sh = torch.autograd.Context()
            ctx_sh.save_for_backward(*saved)
            res_b = shape_cls.backward(ctx_sh, torch.ones((2,)))
            assert isinstance(res_b, tuple) and res_b[0] is None

        # Concatenate and stack backward (mixed requires_grad True and False)
        cat_a = torch.tensor([1.0, 2.0], requires_grad=True)
        cat_b = torch.tensor([3.0, 4.0], requires_grad=False)
        cat_out = torch.concatenate([cat_a, cat_b], dim=0)
        cat_out.sum().backward()
        assert cat_a.grad is not None
        assert_allclose(cat_a.grad.tolist(), [1.0, 1.0])

        stk_a = torch.tensor([1.0, 2.0], requires_grad=True)
        stk_b = torch.tensor([3.0, 4.0], requires_grad=False)
        stk_out = torch.stack([stk_a, stk_b], dim=0)
        stk_out.sum().backward()
        assert stk_a.grad is not None
        assert_allclose(stk_a.grad.tolist(), [1.0, 1.0])

        # End-to-end Linear + ReLU + MSELoss backward pass
        lin = nn.Linear(3, 2)
        inp = torch.tensor([[1.0, -1.0, 2.0], [0.5, 1.5, -0.5]], requires_grad=True)
        target = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        pred = F.relu(lin(inp))
        loss = F.mse_loss(pred, target)
        loss.backward()
        assert inp.grad is not None and inp.grad.shape == (2, 3)
        assert lin.weight.grad is not None and lin.weight.grad.shape == (2, 3)
        assert (
            lin.bias is not None
            and lin.bias.grad is not None
            and lin.bias.grad.shape == (2,)
        )
