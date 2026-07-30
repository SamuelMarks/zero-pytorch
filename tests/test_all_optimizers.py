from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
from zero_torch import nn, optim

try:
    from ml_switcheroo_compiler.core.errors import (
        ShapeMismatchError,
        UnimplementedMathError,
    )
except ImportError:
    UnimplementedMathError = Exception
    ShapeMismatchError = Exception


def test_all_optimizers():
    opts = [
        optim.Adadelta,
        optim.Adafactor,
        optim.Adagrad,
        optim.Adam,
        optim.AdamW,
        optim.Adamax,
        optim.ASGD,
        optim.LBFGS,
        optim.NAdam,
        optim.RAdam,
        optim.RMSprop,
        optim.Rprop,
        optim.SparseAdam,
        optim.SGD,
        optim.SGD,
    ]

    with EagerMode():
        for Opt in opts:
            w = nn.Parameter(torch.ones((2, 2)))
            w.grad = torch.ones((2, 2)) * 0.1

            # Hit params as list of dicts
            opt = Opt(
                [
                    {
                        "params": [w],
                        "weight_decay": 0.1,
                        "momentum": 0.9,
                        "nesterov": True,
                        "maximize": True,
                        "dampening": 0.1,
                    }
                ]
            )

            if Opt in (optim.LBFGS, optim.SGD):
                opt.step(lambda: torch.tensor(1.0))
            else:
                opt.step()

            opt.step()

            opt.zero_grad()
            opt.zero_grad(set_to_none=True)

            # Hit params as list of tensors
            w2 = nn.Parameter(torch.ones((2, 2)))
            w2.grad = None  # to hit continue
            opt2 = Opt([w2])
            if Opt in (optim.LBFGS, optim.SGD):
                opt2.step(lambda: torch.tensor(1.0))
            else:
                opt2.step()

            # Hit SGD nesterov=False
            if Opt == optim.SGD:
                w3 = nn.Parameter(torch.ones((2, 2)))
                w3.grad = torch.ones((2, 2)) * 0.1
                opt3 = optim.SGD([w3], momentum=0.9, nesterov=False)
                opt3.step()
                opt3.step()
