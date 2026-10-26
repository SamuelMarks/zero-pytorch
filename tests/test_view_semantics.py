"""Tests for `Tensor` view, reshape, contiguous, and squeeze semantics."""

from __future__ import annotations

import zero_torch as torch
from zero_torch.tracing import _tracer


def test_view() -> None:
    """Verifies `Tensor.view` with variadic args and shape tuple in eager and tracing modes."""
    t = torch.ones((2, 3))
    t2 = t.view(3, 2)
    assert t2.shape == (3, 2)
    t3 = t.view((6,))
    assert t3.shape == (6,)

    _tracer.start_tracing()
    t4 = t.view(3, 2)
    assert t4.shape == (3, 2)
    _tracer.stop_tracing()


def test_reshape() -> None:
    """Verifies `Tensor.reshape` in eager and tracing modes."""
    t = torch.ones((2, 3))
    t2 = t.reshape(3, 2)
    assert t2.shape == (3, 2)


def test_contiguous() -> None:
    """Verifies `Tensor.contiguous` returns a tensor with matching shape."""
    t = torch.ones((2, 3))
    t2 = t.contiguous()
    assert t2.shape == (2, 3)


def test_squeeze() -> None:
    """Verifies `Tensor.squeeze` with and without explicit dimension in eager and tracing modes."""
    t = torch.ones((1, 2, 1, 3))
    t2 = t.squeeze()
    assert t2.shape == (2, 3)

    t3 = t.squeeze(0)
    assert t3.shape == (2, 1, 3)

    _tracer.start_tracing()
    t4 = t.squeeze()
    assert t4.shape == (2, 3)
    t5 = t.squeeze(2)
    assert t5.shape == (1, 2, 3)
    _tracer.stop_tracing()


def test_view_traced_with_id() -> None:
    """Verifies `reshape` and `squeeze` on an intermediate traced tensor."""
    t = torch.ones((2, 3))
    _tracer.start_tracing()
    t2 = t + t
    assert t2.reshape(6).shape == (6,)
    assert t2.squeeze().shape == (2, 3)
    graph = _tracer.stop_tracing()
    assert len(graph.nodes) >= 3


def test_view_inference_mutation_and_tensor_helpers() -> None:
    """Verifies `-1` view inference, view in-place mutation sync, `stride`, `element_size`, `nbytes`, `new_*`, `_tensor_reductions`, and `_movedim_perm`."""
    from ml_switcheroo_compiler.core.config import EagerMode

    from zero_torch._tensor_reductions import _tensor_aminmax, _tensor_std_var
    from zero_torch._tensor_storage import _compute_contiguous_strides
    from zero_torch._tensor_views import _movedim_perm

    with EagerMode():
        base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        v = base.view(3, -1)
        assert v.shape == (3, 2)
        v.fill_(9.0)
        assert base.tolist() == [[9.0, 9.0, 9.0], [9.0, 9.0, 9.0]]

        assert base.stride() == (3, 1)
        assert base.stride(0) == 3
        assert _compute_contiguous_strides(()) == ()
        assert base.element_size() == 4
        assert base.nbytes == 24
        assert torch.zeros((2,), dtype="complex128").element_size() == 16
        assert torch.zeros((2,), dtype="float64").element_size() == 8
        assert torch.zeros((2,), dtype="float16").element_size() == 2
        assert torch.zeros((2,), dtype="int8").element_size() == 1

        ne = base.new_empty((2, 2), requires_grad=True)
        assert ne.shape == (2, 2) and ne.requires_grad is True
        nf = base.new_full((2,), 7.0, requires_grad=True)
        assert nf.tolist() == [7.0, 7.0] and nf.requires_grad is True
        no = base.new_ones((2,), requires_grad=True)
        assert no.tolist() == [1.0, 1.0] and no.requires_grad is True
        nz = base.new_zeros((2,), requires_grad=True)
        assert nz.tolist() == [0.0, 0.0] and nz.requires_grad is True
        nt = base.new_tensor([3.0, 4.0], requires_grad=True)
        assert nt.tolist() == [3.0, 4.0] and nt.requires_grad is True

        # In-place tensor methods (bernoulli_, normal_, random_, uniform_, masked_scatter_, index_*, scatter_*)
        m = torch.zeros((2, 2))
        assert m.bernoulli_(0.5) is m
        assert m.normal_(0.0, 1.0) is m
        assert m.random_() is m
        assert m.uniform_(-1.0, 1.0) is m
        assert (
            m.masked_scatter_(
                torch.tensor([[True, False], [False, True]]), torch.ones((2, 2))
            )
            is m
        )
        assert m.masked_scatter_(None, None) is m
        assert m.index_add_(0, torch.tensor([0, 1]), torch.ones((2, 2)), alpha=2.0) is m
        assert m.index_add_(0, torch.tensor([0]), torch.ones((1, 2))) is m
        assert m.index_copy_(0, torch.tensor([0, 1]), torch.zeros((2, 2))) is m
        assert m.index_fill_(0, torch.tensor([0, 1]), 5.0) is m
        assert m.index_put_((torch.tensor([0, 1]),), torch.ones((2, 2))) is m
        assert m.scatter_(0, torch.tensor([[0, 0], [1, 1]]), torch.zeros((2, 2))) is m
        assert (
            m.scatter_add_(0, torch.tensor([[0, 0], [1, 1]]), torch.ones((2, 2))) is m
        )
        assert m.tolist() == [[1.0, 1.0], [1.0, 1.0]]

        # _tensor_reductions and _movedim_perm
        t_stat = torch.tensor([1.0, 3.0])
        mn, mx = _tensor_aminmax(t_stat)
        assert mn.item() == 1.0 and mx.item() == 3.0
        assert _tensor_std_var(t_stat, take_sqrt=True).item() == 1.0
        assert _tensor_std_var(t_stat, take_sqrt=False).item() == 1.0
        assert _movedim_perm(3, 0, -1) == (1, 2, 0)
        assert _movedim_perm(3, [0, 1], [2, 0]) == (1, 2, 0)


def test_domain_ops_and_aten_packets() -> None:
    """Verifies `_ops_creation`, `_ops_math`, `_ops_reduction`, `_ops_manipulation`, `_ops_blas`, `_ops_runtime`, and `OpOverloadPacket`."""
    import math

    import pytest
    from ml_switcheroo_compiler.core.config import EagerMode
    from ml_switcheroo_compiler.core.dtype import DType

    from zero_torch._ops_creation import _broadcast_shapes_impl, _cartesian_prod_impl
    from zero_torch._ops_manipulation import (
        _atleast_nd_one,
        _block_diag_impl,
        _diag_embed_impl,
        _flip_1d_or_nd,
        _kron_impl,
        _movedim_impl,
        _rot90_impl,
        _unflatten_impl,
    )
    from zero_torch._ops_math import _cdist_impl, _frac_impl, _nan_replace_flat
    from zero_torch._ops_reduction import (
        _bincount_impl,
        _bucketize_impl,
        _cum_extrema_impl,
        _histc_impl,
        _quantile_1d,
        _quantile_impl,
    )
    from zero_torch.ops.aten import OpOverloadPacket, _dispatch_aten

    with EagerMode():
        # _ops_creation
        assert torch.tril_indices(2, 2).tolist() == [[0, 1, 1], [0, 0, 1]]
        assert torch.triu_indices(2, 2).tolist() == [[0, 0, 1], [0, 1, 1]]
        t1d = torch.tensor([1.0, 2.0])
        assert _cartesian_prod_impl([t1d]).tolist() == [1.0, 2.0]
        assert torch.cartesian_prod(t1d, t1d).shape == (4, 2)
        assert torch.combinations(t1d, r=2, with_replacement=True).shape == (3, 2)
        assert torch.combinations(t1d, r=2, with_replacement=False).shape == (1, 2)
        assert _broadcast_shapes_impl() == (1,)
        assert torch.broadcast_shapes((2, 1), (1, 3), 3) == (2, 3)
        with pytest.raises(RuntimeError):
            torch.broadcast_shapes((2, 3), (4, 3))

        # _ops_math
        out_f = torch.zeros((2,))
        assert _frac_impl(torch.tensor([1.5, -2.25]), out=out_f) is out_f
        assert torch.frac(torch.tensor([1.5])).tolist() == [0.5]
        x_cd = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        assert _cdist_impl(x_cd, x_cd, p=2.0).tolist() == [[0.0, 5.0], [5.0, 0.0]]
        assert _cdist_impl(x_cd, x_cd, p=0.0).tolist() == [[0.0, 2.0], [2.0, 0.0]]
        assert _cdist_impl(x_cd, x_cd, p=math.inf).tolist() == [[0.0, 4.0], [4.0, 0.0]]
        assert _cdist_impl(t1d, t1d).tolist() == [1.0, 2.0]
        assert _nan_replace_flat(torch.tensor(math.nan), nan_val=1.0).item() == 1.0
        assert _nan_replace_flat(
            torch.tensor([math.nan, math.inf, -math.inf, 2.0]),
            nan_val=0.0,
            posinf_val=9.0,
            neginf_val=-9.0,
        ).tolist() == [0.0, 9.0, -9.0, 2.0]

        # _ops_reduction
        vmax, imax = torch.cummax(torch.tensor([1.0, 3.0, 2.0]), dim=0)
        assert vmax.tolist() == [1.0, 3.0, 3.0] and imax.tolist() == [0, 1, 1]
        vmin, imin = torch.cummin(torch.tensor([3.0, 1.0, 2.0]), dim=0)
        assert vmin.tolist() == [3.0, 1.0, 1.0] and imin.tolist() == [0, 1, 1]
        assert _cum_extrema_impl(torch.ones((2, 2)), dim=0)[0].shape == (2, 2)
        assert torch.nansum(torch.tensor([1.0, math.nan, 3.0])).item() == 4.0
        assert torch.nanmean(torch.tensor([1.0, math.nan, 3.0])).item() == 2.0
        assert _quantile_1d([], 0.5) == 0.0
        for mode in ("linear", "lower", "higher", "midpoint", "nearest"):
            assert _quantile_1d([1.0, 2.0, 3.0, 4.0], 0.6, mode) >= 1.0
        assert _quantile_impl(
            torch.tensor([1.0, 2.0, 3.0]), q=torch.tensor([0.0, 1.0])
        ).tolist() == [1.0, 3.0]
        assert _bincount_impl(torch.tensor([0, 1, 1, 2])).tolist() == [1, 2, 1]
        assert _bincount_impl(
            torch.tensor([0, 1]), weights=torch.tensor([0.5, 1.5])
        ).tolist() == [0.5, 1.5]
        assert _histc_impl(
            torch.tensor([1.0, 2.0, 3.0]), bins=2, min_val=0.0, max_val=0.0
        ).tolist() == [1.0, 2.0]
        assert _histc_impl(
            torch.tensor([1.0, 5.0]), bins=2, min_val=1.0, max_val=3.0
        ).tolist() == [1.0, 0.0]
        assert _histc_impl(
            torch.tensor([1.0, 1.0]), bins=2, min_val=1.0, max_val=1.0
        ).shape == (2,)
        assert (
            _bucketize_impl(
                torch.tensor(2.0), torch.tensor([1.0, 3.0]), out_int32=True, right=True
            ).item()
            == 1
        )
        assert _bucketize_impl(
            torch.tensor([2.0]), torch.tensor([1.0, 3.0]), out_int32=False, right=False
        ).tolist() == [1]

        # _ops_manipulation
        s0 = torch.tensor(5.0)
        assert _atleast_nd_one(s0, 1).shape == (1,)
        assert _atleast_nd_one(s0, 2).shape == (1, 1)
        assert _atleast_nd_one(s0, 3).shape == (1, 1, 1)
        assert _atleast_nd_one(t1d, 2).shape == (1, 2)
        assert _atleast_nd_one(t1d, 3).shape == (1, 2, 1)
        assert _atleast_nd_one(torch.ones((2, 3)), 3).shape == (2, 3, 1)
        assert _atleast_nd_one(torch.ones((2, 3, 4)), 2).shape == (2, 3, 4)
        assert torch.atleast_1d(s0, t1d)[0].shape == (1,)
        assert torch.atleast_1d(s0)[0].shape == (1,)
        assert torch.atleast_2d(s0, t1d)[0].shape == (1, 1)
        assert torch.atleast_2d(s0)[0].shape == (1, 1)
        assert torch.atleast_3d(s0, t1d)[0].shape == (1, 1, 1)
        assert torch.atleast_3d(s0)[0].shape == (1, 1, 1)
        assert _block_diag_impl([]).shape == (0, 0)
        assert _block_diag_impl([torch.ones((2, 2))]).shape == (2, 2)
        assert torch.block_diag(torch.ones((1, 2)), torch.ones((2, 1))).shape == (3, 3)
        assert _diag_embed_impl(t1d).shape == (2, 2)
        assert _diag_embed_impl(torch.ones((2, 3))).shape == (2, 3, 3)
        assert _kron_impl(t1d, t1d).shape == (4,)
        assert _kron_impl(torch.ones((2, 2)), torch.ones((2, 2))).shape == (4, 4)
        assert _kron_impl(s0, s0).item() == 25.0
        assert _flip_1d_or_nd(s0, [0]).item() == 5.0
        assert torch.fliplr(torch.tensor([[1.0, 2.0], [3.0, 4.0]])).tolist() == [
            [2.0, 1.0],
            [4.0, 3.0],
        ]
        assert torch.flipud(torch.tensor([[1.0, 2.0], [3.0, 4.0]])).tolist() == [
            [3.0, 4.0],
            [1.0, 2.0],
        ]
        assert _rot90_impl(t1d).tolist() == [1.0, 2.0]
        m2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        for k_rot in (0, 1, 2, 3):
            assert _rot90_impl(m2, k=k_rot).shape == (2, 2)
        assert _movedim_impl(t1d, 0, 0).shape == (2,)
        assert _movedim_impl(torch.ones((2, 3, 4)), 0, -1).shape == (3, 4, 2)
        assert torch.movedim(torch.ones((2, 3, 4)), [0, 1], [2, 0]).shape == (3, 4, 2)
        assert _unflatten_impl(s0, 0, (1,)).shape == ()
        assert torch.unflatten(torch.ones((2, 6)), 1, (2, 3)).shape == (2, 2, 3)
        assert torch.unflatten(torch.ones((2, 6)), 1, ()).shape == (2, 6)

        # _ops_blas
        out_mv = torch.zeros((2,))
        assert (
            torch.addmv(
                torch.ones((2,)), torch.eye(2), torch.tensor([2.0, 3.0]), out=out_mv
            )
            is out_mv
        )
        assert torch.addmv(
            torch.ones((2,)), torch.eye(2), torch.tensor([2.0, 3.0])
        ).tolist() == [3.0, 4.0]
        assert out_mv.tolist() == [3.0, 4.0]
        out_r = torch.zeros((2, 2))
        assert (
            torch.addr(
                torch.zeros((2, 2)),
                torch.tensor([1.0, 2.0]),
                torch.tensor([3.0, 4.0]),
                beta=0.0,
                out=out_r,
            )
            is out_r
        )
        assert torch.addr(
            torch.zeros((2, 2)), torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
        ).shape == (2, 2)
        assert out_r.tolist() == [[3.0, 4.0], [6.0, 8.0]]
        b3 = torch.ones((2, 2, 2))
        out_bmm = torch.zeros((2, 2, 2))
        assert torch.baddbmm(b3, b3, b3, beta=0.0, out=out_bmm) is out_bmm
        assert torch.baddbmm(b3, b3, b3).shape == (2, 2, 2)
        out_abmm = torch.zeros((2, 2))
        assert (
            torch.addbmm(torch.zeros((2, 2)), b3, b3, beta=0.0, out=out_abmm)
            is out_abmm
        )
        assert torch.addbmm(torch.zeros((2, 2)), b3, b3).shape == (2, 2)
        out_mm = torch.zeros((2, 2))
        assert (
            torch.addmm(torch.zeros((2, 2)), torch.eye(2), torch.eye(2), out=out_mm)
            is out_mm
        )
        assert torch.addmm(t1d, t1d, t1d).tolist() == [1.0, 2.0]

        # _ops_runtime
        torch.set_default_device("cpu")
        assert torch.get_default_device() == "cpu"
        torch.set_default_dtype(DType.Float64)
        assert torch.get_default_dtype() == DType.Float64
        torch.set_default_dtype("invalid_dt")
        torch.set_default_dtype("float32")
        assert torch.get_default_dtype() == DType.Float32
        for dm in (0, 1, 2):
            torch.set_deterministic_debug_mode(dm)
        assert torch.get_deterministic_debug_mode() == "error"
        torch.set_deterministic_debug_mode("default")
        torch.use_deterministic_algorithms(True)
        assert torch.are_deterministic_algorithms_enabled() is True
        torch.use_deterministic_algorithms(False)
        torch.set_num_threads(2)
        assert torch.get_num_threads() == 2
        torch.set_num_interop_threads(2)
        assert torch.get_num_interop_threads() == 2
        assert torch.is_anomaly_enabled() is False

        # ATen OpOverloadPacket and _dispatch_aten
        pkt = OpOverloadPacket("abs", torch.ops.aten.abs)
        assert pkt(torch.tensor([-2.0])).item() == 2.0
        assert pkt.default(torch.tensor([-3.0])).item() == 3.0
        out_at = torch.zeros((1,))
        assert pkt.out(torch.tensor([-4.0]), out=out_at) is out_at
        assert out_at.item() == 4.0
        assert pkt.out(torch.tensor([-5.0])).item() == 5.0
        assert _dispatch_aten(torch.tensor([-2.0]), _op="abs", out=out_at) is out_at
        assert (
            _dispatch_aten(
                torch.tensor([2.0]), torch.tensor([3.0]), _op="add", out=out_at
            )
            is out_at
        )
        assert _dispatch_aten(torch.tensor([2.0]), 3.0, _op="add").item() == 5.0
