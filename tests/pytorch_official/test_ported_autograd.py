"""Ported autograd functional tests verifying analytical and numerical gradients."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
import zero_torch.nn as nn
import zero_torch.nn.functional as F
from tests.conftest import assert_allclose
from zero_torch.autograd.math_ops import (
    ClampBackward,
    CosBackward,
    ExpBackward,
    LogBackward,
    LogSoftmaxBackward,
    MaxBackward,
    MeanBackward,
    MinBackward,
    PermuteBackward,
    ReluBackward,
    ReshapeBackward,
    SigmoidBackward,
    SinBackward,
    SliceBackward,
    SoftmaxBackward,
    SplitBackward,
    SqrtBackward,
    SqueezeBackward,
    SumBackward,
    TanhBackward,
    TransposeBackward,
    UnsqueezeBackward,
    WhereBackward,
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
        assert (
            lin.weight is not None
            and lin.weight.grad is not None
            and lin.weight.grad.shape == (2, 3)
        )
        assert (
            lin.bias is not None
            and lin.bias.grad is not None
            and lin.bias.grad.shape == (2,)
        )

        # Test grad, gradcheck, gradgradcheck, and helper utilities in zero_torch.autograd
        gx = torch.tensor([2.0], requires_grad=True)
        gy = torch.tensor([3.0], requires_grad=True)
        gz = gx * gx
        (dgx, dgy_none) = torch.autograd.grad(gz, [gx, gy], allow_unused=True)
        assert dgx is not None and dgx.item() == 4.0 and dgy_none is None
        gz2 = gx * 5.0
        (dgx2, dgy_zero) = torch.autograd.grad(
            [gz2],
            [gx, gy],
            grad_outputs=[torch.tensor([1.0])],
            materialize_grads=True,
        )
        assert dgx2 is not None and dgx2.item() == 5.0
        assert dgy_zero is not None and dgy_zero.item() == 0.0
        gz3 = gx * 2.0
        torch.autograd.backward(gz3, grad_tensors=torch.tensor([1.0]))

        def sq_fn(v: torch.Tensor) -> torch.Tensor:
            """Squares `v`.

            Args:
                v: Input tensor.

            Returns:
                Squared tensor.
            """
            return v * v

        assert torch.autograd.gradcheck(sq_fn, gx)
        assert torch.autograd.gradgradcheck(
            sq_fn, [torch.tensor([2.0], requires_grad=False)]
        )
        assert torch.autograd.cast(torch.Tensor, gx) is gx
        assert torch.autograd.handle_torch_function(sq_fn, [gx], gx).item() == 4.0
        assert not torch.autograd.has_torch_function([gx])
        assert torch.autograd.is_multithreading_enabled()
        assert torch.autograd.is_view_replay_enabled()
        assert torch.autograd.is_tensor_like(gx)
        assert not torch.autograd.is_tensor_like(123)
        assert not torch.autograd.kineto_available()
        assert torch.autograd.overload(sq_fn) is sq_fn
        assert torch.autograd.variable(gx) is gx
        assert torch.autograd.variable().item() == 0.0


def test_autograd_extended_ops_hooks_and_higher_order() -> None:
    """Verifies Where/Clamp/Max/Min/Softmax/LogSoftmax/Split/Slice backward ops, Tensor.register_hook, create_graph=True, and inputs filtering."""
    import pytest

    with EagerMode():
        # Where, Clamp, Max, Min, Softmax, LogSoftmax, Split, Slice
        wx = torch.tensor([1.0, 5.0, -2.0], requires_grad=True)
        wy = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)
        cond = torch.tensor([True, False, True])
        w_out = torch.where(cond, wx, wy)
        c_out = torch.clamp(torch.clamp(w_out, min=0.0), max=15.0)
        c_out.sum().backward()
        assert wx.grad is not None and wx.grad.tolist() == [1.0, 0.0, 0.0]
        assert wy.grad is not None and wy.grad.tolist() == [0.0, 0.0, 0.0]

        mx = torch.tensor([[1.0, 4.0], [3.0, 2.0]], requires_grad=True)
        (torch.max(mx, dim=1) + torch.min(mx, dim=1)).sum().backward()
        assert mx.grad is not None and mx.grad.tolist() == [[1.0, 1.0], [1.0, 1.0]]

        sm_x = torch.tensor([[1.0, 2.0], [0.5, 1.5]], requires_grad=True)
        (
            F.softmax(sm_x, dim=-1) * torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        ).sum().backward()
        assert sm_x.grad is not None and sm_x.grad.shape == (2, 2)

        lsm_x = torch.tensor([[1.0, 2.0], [0.5, 1.5]], requires_grad=True)
        F.nll_loss(F.log_softmax(lsm_x, dim=-1), torch.tensor([0, 1])).backward()
        assert lsm_x.grad is not None and lsm_x.grad.shape == (2, 2)

        sp_x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        p1, p2 = sp_x.chunk(2, dim=0)
        (p1 * 2.0 + p2 * 3.0).sum().backward()
        assert sp_x.grad is not None and sp_x.grad.tolist() == [2.0, 2.0, 3.0, 3.0]

        sl_x = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)
        (sl_x[1] * 4.0 + sl_x[0:2].sum()).backward()
        assert sl_x.grad is not None and sl_x.grad.tolist() == [1.0, 5.0, 0.0]

        sl_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        sl_2d[(0, 1)].backward()
        assert sl_2d.grad is not None

        # Non-grad & edge branches on new Backward classes
        c_nograd = torch.tensor([1.0, 2.0], requires_grad=False)
        assert (
            WhereBackward.apply(True, 1.0, torch.tensor(2.0, requires_grad=True)).item()
            == 1.0
        )
        assert ClampBackward.apply(1.0, None, None).item() == 1.0
        assert MaxBackward.apply(1.0, None, False).item() == 1.0
        assert MinBackward.apply(1.0, None, False).item() == 1.0

        ctx_c = torch.autograd.Context()
        ctx_c.save_for_backward(c_nograd, 0.0, 1.0)
        assert ClampBackward.backward(ctx_c, torch.ones((2,))) == (None, None, None)

        ctx_m = torch.autograd.Context()
        ctx_m.save_for_backward(c_nograd, c_nograd, None, False)
        assert MaxBackward.backward(ctx_m, torch.ones((2,))) == (None, None, None)
        assert MinBackward.backward(ctx_m, torch.ones((2,))) == (None, None, None)

        ctx_sm = torch.autograd.Context()
        ctx_sm.save_for_backward(c_nograd, c_nograd, -1)
        assert SoftmaxBackward.backward(ctx_sm, torch.ones((2,))) == (None, None)
        assert LogSoftmaxBackward.backward(ctx_sm, torch.ones((2,))) == (None, None)

        ctx_sp = torch.autograd.Context()
        ctx_sp.save_for_backward(c_nograd, 0, torch.ones((1,)), torch.ones((1,)))
        assert SplitBackward.backward(ctx_sp, torch.ones((1,))) == (None, None, None)

        ctx_sl = torch.autograd.Context()
        ctx_sl.save_for_backward(c_nograd)
        assert SliceBackward.backward(ctx_sl, torch.ones(())) == (None, None)
        sl_idx_t = torch.tensor([10.0, 20.0], requires_grad=True)
        sl_idx_t[torch.tensor([0])].sum().backward()

        with torch.autograd.enable_grad():
            with torch.autograd.inference_mode(True):
                assert not torch.autograd.is_grad_enabled()

        # Tensor.register_hook, create_graph=True, inputs filtering, allow_unused=False
        with pytest.raises(RuntimeError):
            c_nograd.register_hook(lambda g: g)
        assert c_nograd.register_hook(None) is None

        hx = torch.tensor([3.0], requires_grad=True)
        h_handle = hx.register_hook(lambda g: g * 2.0)
        h_noop = hx.register_hook(lambda _g: None)
        hy = hx * hx
        hy.backward(create_graph=True)
        assert hx.grad is not None and hx.grad.item() == 12.0
        assert h_handle is not None and h_noop is not None

        # Higher-order derivative via create_graph=True in autograd.grad
        x_ho = torch.tensor([2.0], requires_grad=True)
        y_ho = x_ho * x_ho * x_ho
        (dy_dx,) = torch.autograd.grad(y_ho, x_ho, create_graph=True)
        assert dy_dx is not None and dy_dx.item() == 12.0
        (d2y_dx2,) = torch.autograd.grad(dy_dx, x_ho)
        assert d2y_dx2 is not None and d2y_dx2.item() == 12.0

        # Unreachable input raises RuntimeError when allow_unused=False
        unused_t = torch.tensor([1.0], requires_grad=True)
        with pytest.raises(RuntimeError):
            torch.autograd.grad(y_ho, [x_ho, unused_t], allow_unused=False)

        # inputs parameter in backward (Tensor, list, and dict)
        in1 = torch.tensor([2.0], requires_grad=True)
        in2 = torch.tensor([3.0], requires_grad=True)
        in1.backward(inputs=in2)
        assert in1.grad is None
        (in1 * in2).backward(inputs=in1)
        assert in1.grad is not None and in2.grad is None
        (in1 * in2).backward(inputs={"in2": in2})
        assert in2.grad is not None

        with torch.autograd.detect_anomaly():
            with torch.autograd.set_detect_anomaly(True):
                v_t = torch.autograd.variable(
                    [1.0, 2.0], requires_grad=True, dtype="float32"
                )
                assert v_t.requires_grad is True
