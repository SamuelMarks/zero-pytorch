"""Tests for all optimizers in `zero_torch.optim`."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
from tests.conftest import flatten_scalars
from zero_torch import nn, optim
from zero_torch._types import ParamGroupDict


def test_all_optimizers() -> None:
    """Verifies that all optimizers in `zero_torch.optim` strictly decrease a quadratic objective over multiple steps."""
    opts: list[type[optim.Optimizer]] = [
        optim.Adadelta,
        optim.Adafactor,
        optim.Adagrad,
        optim.Adam,
        optim.AdamW,
        optim.Adamax,
        optim.ASGD,
        optim.LBFGS,
        optim.Muon,
        optim.NAdam,
        optim.RAdam,
        optim.RMSprop,
        optim.Rprop,
        optim.SparseAdam,
        optim.SGD,
    ]

    with EagerMode():
        target = torch.tensor([[0.2, -0.3], [0.5, 0.1]])
        for opt_cls in opts:
            w_quad = nn.Parameter(torch.ones((2, 2)))
            q_group: ParamGroupDict = {"params": [w_quad], "lr": 0.05}
            opt_q = opt_cls([q_group])
            init_diff = w_quad - target
            init_loss = float(torch.sum(init_diff * init_diff).item().real)
            for _ in range(4):
                w_quad.grad = w_quad - target
                opt_q.step()
            final_diff = w_quad - target
            final_loss = float(torch.sum(final_diff * final_diff).item().real)
            assert final_loss < init_loss

            w = nn.Parameter(torch.ones((2, 2)))
            w.grad = torch.ones((2, 2)) * 0.1

            group: ParamGroupDict = {
                "params": [w],
                "weight_decay": 0.1,
                "momentum": 0.9,
                "nesterov": True,
                "maximize": True,
                "dampening": 0.1,
            }
            opt = opt_cls([group])

            if opt_cls in (optim.LBFGS, optim.SGD):
                opt.step(lambda: torch.tensor(1.0))
            else:
                opt.step()

            opt.step()

            opt.zero_grad()
            assert w.grad is not None
            opt.zero_grad(set_to_none=True)
            assert w.grad is None

            w2 = nn.Parameter(torch.ones((2, 2)))
            w2.grad = None
            opt2 = opt_cls([w2])
            if opt_cls in (optim.LBFGS, optim.SGD):
                opt2.step(lambda: torch.tensor(1.0))
            else:
                opt2.step()

            if opt_cls is optim.SGD:
                w3 = nn.Parameter(torch.ones((2, 2)))
                w3.grad = torch.ones((2, 2)) * 0.1
                opt3 = optim.SGD([w3], momentum=0.9, nesterov=False)
                opt3.step()
                opt3.step()
                assert all(
                    isinstance(v, (int, float)) and float(v) < 1.0
                    for v in flatten_scalars(w3.tolist())
                )
