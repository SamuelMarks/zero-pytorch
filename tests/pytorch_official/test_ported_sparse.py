"""Ported sparse tensor (`zero_torch.sparse`) tests."""

from __future__ import annotations

import ml_switcheroo_compiler
from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
from zero_torch.tracing import _tracer


def test_sparse_parity() -> None:
    """Verifies `sparse_coo_tensor`, `sparse.add`, and `sparse.mm` in eager mode."""
    with EagerMode():
        indices = [[0, 1, 1], [2, 0, 2]]
        values = [3.0, 4.0, 5.0]
        st = torch.sparse_coo_tensor(indices, values, (2, 3))
        assert st.shape == (2, 3)

        st2 = torch.sparse_coo_tensor(indices, values)
        assert st2.shape == (2, 3)

        st3 = torch.sparse_coo_tensor(indices, values, (3, 3, 3))
        assert st3.shape == (3, 3, 3)

        indices_1d = [[0, 2]]
        st_1d = torch.sparse_coo_tensor(indices_1d, [1.0, 2.0], (4,))
        assert st_1d.shape == (4,)

        indices_3d = [[0, 1], [1, 0], [0, 1]]
        st_3d = torch.sparse_coo_tensor(indices_3d, [1.0, 2.0], (2, 2, 2))
        assert st_3d.shape == (2, 2, 2)

        indices_empty: list[list[int]] = [[], []]
        values_empty: list[float] = []
        st4 = torch.sparse_coo_tensor(indices_empty, values_empty)
        assert st4.shape == ()

        res_add = torch.sparse.add(st, st)
        assert res_add.shape == (2, 3)

        sq = torch.sparse_coo_tensor([[0, 1], [0, 1]], [2.0, 3.0], (2, 2))
        res_mm = torch.sparse.mm(sq, sq)
        assert res_mm.shape == (2, 2)


def test_sparse_proxy() -> None:
    """Verifies `sparse_coo_tensor`, `sparse.add`, and `sparse.mm` in graph tracing mode."""
    prev = ml_switcheroo_compiler.core.config.eager_mode
    ml_switcheroo_compiler.core.config.eager_mode = False
    _tracer.start_tracing()
    try:
        indices = [[0, 1, 1], [2, 0, 2]]
        values = [3.0, 4.0, 5.0]
        st = torch.sparse_coo_tensor(indices, values, (2, 3))
        assert st.shape == (2, 3)
        res_add = torch.sparse.add(st, st)
        assert res_add is not None
        res_mm = torch.sparse.mm(st, st)
        assert res_mm is not None
    finally:
        _tracer.stop_tracing()
        ml_switcheroo_compiler.core.config.eager_mode = prev
