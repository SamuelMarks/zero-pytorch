"""Ported optimizer (`zero_torch.optim`) and scheduler parity tests."""

from __future__ import annotations

import math

import pytest
from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
from tests.conftest import assert_allclose
from zero_torch import nn, optim
from zero_torch.optim import lr_scheduler


def test_sgd() -> None:
    """Verifies `optim.SGD`, `optim.Adam`, and `optim.AdamW` single-step and multi-step updates."""
    with EagerMode():
        w = nn.Parameter(torch.ones((2, 2)))
        w.grad = torch.ones((2, 2)) * 2.0

        opt = optim.SGD([w], lr=0.1)
        opt.step()
        assert_allclose(w.tolist(), 0.8)

        # Single-step analytical check for Adam (bias-corrected first step moves by +/- lr)
        zw = nn.Parameter(torch.tensor([[1.5, -0.5], [0.25, 2.0]]))
        z_adam = optim.Adam([zw], lr=0.05, betas=(0.9, 0.999), eps=1e-8)
        zw.grad = torch.tensor([[0.1, -0.2], [0.05, 0.3]])
        z_adam.step()
        assert_allclose(
            zw.tolist(), [[1.45, -0.45], [0.20, 1.95]], rtol=1e-4, atol=1e-5
        )


def test_lr_schedulers_numerical_trajectories() -> None:
    """Verifies numerical LR trajectories for all 15 schedulers in `zero_torch.optim.lr_scheduler`."""
    with EagerMode():
        p = nn.Parameter(torch.ones((2,)))

        opt = optim.SGD([p], lr=0.1)
        s_lambda = lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: 0.5**e)
        assert s_lambda.get_last_lr()[0] == pytest.approx(0.1)
        s_lambda.step()
        assert s_lambda.get_last_lr()[0] == pytest.approx(0.05)

        opt = optim.SGD([p], lr=0.1)
        s_mult = lr_scheduler.MultiplicativeLR(opt, lr_lambda=lambda _e: 0.8)
        s_mult.step()
        assert s_mult.get_last_lr()[0] == pytest.approx(0.08)

        opt = optim.SGD([p], lr=0.1)
        s_step = lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
        s_step.step()
        assert s_step.get_last_lr()[0] == pytest.approx(0.1)
        s_step.step()
        assert s_step.get_last_lr()[0] == pytest.approx(0.05)

        opt = optim.SGD([p], lr=0.1)
        s_mstep = lr_scheduler.MultiStepLR(opt, milestones=[1, 3], gamma=0.5)
        s_mstep.step()
        assert s_mstep.get_last_lr()[0] == pytest.approx(0.05)
        s_mstep.step()
        assert s_mstep.get_last_lr()[0] == pytest.approx(0.05)
        s_mstep.step()
        assert s_mstep.get_last_lr()[0] == pytest.approx(0.025)

        opt = optim.SGD([p], lr=0.1)
        s_const = lr_scheduler.ConstantLR(opt, factor=0.5, total_iters=1)
        assert s_const.get_last_lr()[0] == pytest.approx(0.05)
        s_const.step()
        assert s_const.get_last_lr()[0] == pytest.approx(0.1)

        opt = optim.SGD([p], lr=0.1)
        s_lin = lr_scheduler.LinearLR(
            opt, start_factor=0.5, end_factor=1.0, total_iters=2
        )
        assert s_lin.get_last_lr()[0] == pytest.approx(0.05)
        s_lin.step()
        assert s_lin.get_last_lr()[0] == pytest.approx(0.075)
        s_lin.step()
        assert s_lin.get_last_lr()[0] == pytest.approx(0.1)

        opt = optim.SGD([p], lr=0.1)
        s_exp = lr_scheduler.ExponentialLR(opt, gamma=0.9)
        s_exp.step()
        assert s_exp.get_last_lr()[0] == pytest.approx(0.09)

        opt = optim.SGD([p], lr=0.1)
        s_poly = lr_scheduler.PolynomialLR(opt, total_iters=2, power=1.0)
        s_poly.step()
        assert s_poly.get_last_lr()[0] == pytest.approx(0.05)

        opt = optim.SGD([p], lr=0.1)
        s_cos = lr_scheduler.CosineAnnealingLR(opt, T_max=2, eta_min=0.0)
        s_cos.step()
        assert s_cos.get_last_lr()[0] == pytest.approx(0.05)
        s_cos.step()
        assert s_cos.get_last_lr()[0] == pytest.approx(0.0)

        opt = optim.SGD([p], lr=0.1)
        s_cawr = lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=2, T_mult=1, eta_min=0.0
        )
        s_cawr.step()
        assert s_cawr.get_last_lr()[0] == pytest.approx(0.05)
        s_cawr.step()
        assert s_cawr.get_last_lr()[0] == pytest.approx(0.1)

        opt = optim.SGD([p], lr=0.01)
        s_cyc = lr_scheduler.CyclicLR(
            opt, base_lr=0.01, max_lr=0.1, step_size_up=2, mode="triangular"
        )
        s_cyc.step()
        assert s_cyc.get_last_lr()[0] == pytest.approx(0.055)

        opt = optim.SGD([p], lr=0.01)
        s_1c = lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=4)
        lr0 = s_1c.get_last_lr()[0]
        s_1c.step()
        assert s_1c.get_last_lr()[0] > lr0

        opt = optim.SGD([p], lr=0.1)
        s1 = lr_scheduler.ConstantLR(opt, factor=0.5, total_iters=2)
        s2 = lr_scheduler.ExponentialLR(opt, gamma=0.5)
        s_seq = lr_scheduler.SequentialLR(opt, schedulers=[s1, s2], milestones=[1])
        s_seq.step()
        assert s_seq.get_last_lr()[0] > 0.0

        opt = optim.SGD([p], lr=0.1)
        c1 = lr_scheduler.ExponentialLR(opt, gamma=0.5)
        c2 = lr_scheduler.ExponentialLR(opt, gamma=0.5)
        s_chain = lr_scheduler.ChainedScheduler([c1, c2])
        s_chain.step()
        assert math.isclose(opt.param_groups[0]["lr"], 0.05, rel_tol=1e-4)

        opt = optim.SGD([p], lr=0.1)
        s_plat = lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1)
        s_plat.step(1.0)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.1)
        s_plat.step(1.0)
        s_plat.step(1.0)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.05)

        plat_sd = s_plat.state_dict()
        s_plat2 = lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=1
        )
        s_plat2.load_state_dict(plat_sd)
        assert s_plat2.best == pytest.approx(1.0)
        s_plat2.load_state_dict(
            {
                "best": None,
                "num_bad_epochs": 0,
                "cooldown_counter": 0,
                "last_epoch": 0,
                "_last_lr": [0.1],
            }
        )
        assert s_plat2.best is None
        s_plat2.load_state_dict({"best": [1.0]})


def test_optimizer_hooks_state_and_lbfgs_adafactor() -> None:
    """Verifies optimizer step pre/post hooks, state_dict Tensor cloning, Adafactor 1D/2D updates, and LBFGS history truncation."""
    with EagerMode():
        w1 = nn.Parameter(torch.tensor([2.0, -2.0]))
        w1.grad = torch.tensor([0.5, -0.5])
        base_opt = optim.Optimizer([w1], {"lr": 0.1})
        events: list[str] = []
        h_pre = base_opt.register_step_pre_hook(lambda _o: events.append("pre"))
        h_post = base_opt.register_step_post_hook(lambda _o: events.append("post"))
        base_opt.step()
        assert events == ["pre", "post"]
        h_pre.remove()
        h_post.remove()

        # Adafactor on 1D and 2D parameters + closure + empty group
        w2 = nn.Parameter(torch.ones((2, 3)))
        w2.grad = torch.ones((2, 3)) * 0.2
        w1.grad = torch.tensor([0.5, -0.5])
        adaf = optim.Adafactor([w1, w2], lr=0.05, eps=(1e-30, 0.001), weight_decay=0.01)
        adaf.param_groups.append({"params": (w1,)})
        loss_af = adaf.step(lambda: 1.5)
        assert loss_af == 1.5
        af_sd = adaf.state_dict()
        adaf2 = optim.Adafactor([w1, w2], lr=0.05)
        adaf2.load_state_dict(af_sd)
        assert id(w2) in adaf2.state and "exp_avg_sq_row" in adaf2.state[id(w2)]

        # LBFGS history size truncation
        wl = nn.Parameter(torch.tensor([3.0, -1.0]))
        lbfgs = optim.LBFGS([wl], lr=0.2, history_size=1)
        lbfgs.param_groups.append({"params": (wl,)})
        for step_i in range(4):
            wl.grad = wl * (1.0 + 0.1 * float(step_i))
            lbfgs.step()
        assert len(lbfgs._s_history[id(wl)]) == 1
