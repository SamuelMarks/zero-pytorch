"""Tests for all optimizers in `zero_torch.optim`."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
from zero_torch import nn, optim
from zero_torch._types import ParamGroupDict


def test_all_optimizers() -> None:
    """Verifies that all optimizers in `zero_torch.optim` execute `step()` and `zero_grad()` cleanly."""
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
        for opt_cls in opts:
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
                assert (w3.numpy() < 1.0).all()
