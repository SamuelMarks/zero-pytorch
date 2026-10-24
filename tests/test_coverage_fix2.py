"""Comprehensive unit tests covering 100% of lines and branches across zero_torch."""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Callable, Iterator
from unittest import mock

import pytest
from ml_switcheroo_compiler.core.config import EagerMode, config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import UnimplementedMathError
from ml_switcheroo_compiler.core.tensor import (
    Tensor as SwitcherooTensor,
)
from ml_switcheroo_compiler.core.tensor import (
    TensorConfig,
)

import zero_torch as torch
from zero_torch import nn, optim
from zero_torch._types import OpArg, ParamGroupDict
from zero_torch.autograd import Function
from zero_torch.autograd.engine import _backward
from zero_torch.distributions import Distribution
from zero_torch.nn import (
    functional as F,
)
from zero_torch.nn import (
    functional_dropout as FD,
)
from zero_torch.nn import (
    functional_norm as FN,
)
from zero_torch.nn import (
    functional_pooling as FP,
)
from zero_torch.nn.norm import _apply_channel_affine
from zero_torch.optim import _betas_opt, _float_opt, lr_scheduler
from zero_torch.tensor import (
    Tensor,
    _flatten_nested,
    _to_tensor,
    _wrap,
    _wrap_list,
    _wrap_tuple,
)
from zero_torch.tracing import ProxyTensor, _tracer
from zero_torch.utils import data
from zero_torch.utils.data.dataloader import (
    DataLoader,
    Dataset,
    Sampler,
    default_collate,
)


def test_top_level_ops_and_branches() -> None:
    """Verifies all top-level `zero_torch` math, shape, logic, bitwise, and creation operations."""
    with EagerMode():
        mat = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
        vec = torch.tensor([1.0, 2.0, 3.0])
        vec2 = torch.tensor([2.0, 3.0, 4.0])
        int_vec = torch.tensor([2, 4, 6])
        int_vec2 = torch.tensor([1, 2, 3])
        bool_vec = torch.tensor([True, False, True])
        bool_vec2 = torch.tensor([False, False, True])

        # Unary floating math ops
        for unary_fn in [
            torch.abs,
            torch.acos,
            torch.acosh,
            torch.asin,
            torch.asinh,
            torch.atan,
            torch.atanh,
            torch.cbrt,
            torch.ceil,
            torch.conj,
            torch.cos,
            torch.cosh,
            torch.deg2rad,
            torch.digamma,
            torch.erf,
            torch.erfc,
            torch.erfinv,
            torch.exp,
            torch.exp2,
            torch.expm1,
            torch.fix,
            torch.floor,
            torch.imag,
            torch.isfinite,
            torch.isinf,
            torch.isnan,
            torch.lgamma,
            torch.log,
            torch.log10,
            torch.log1p,
            torch.log2,
            torch.negative,
            torch.positive,
            torch.rad2deg,
            torch.real,
            torch.reciprocal,
            torch.round,
            torch.rsqrt,
            torch.sign,
            torch.sin,
            torch.sinc,
            torch.sinh,
            torch.sqrt,
            torch.square,
            torch.tan,
            torch.tanh,
            torch.trunc,
        ]:
            assert unary_fn(torch.tensor([0.5, 1.5])).shape == (2,)

        # Binary floating math ops
        bin_fns: list[Callable[[OpArg, OpArg], Tensor]] = [
            torch.add,
            torch.atan2,
            torch.copysign,
            torch.divide,
            torch.equal,
            torch.float_power,
            torch.floor_divide,
            torch.fmax,
            torch.fmin,
            torch.fmod,
            torch.greater,
            torch.greater_equal,
            torch.heaviside,
            torch.hypot,
            torch.inner,
            torch.isclose,
            torch.ldexp,
            torch.less,
            torch.less_equal,
            torch.logaddexp,
            torch.logaddexp2,
            torch.maximum,
            torch.minimum,
            torch.mod,
            torch.multiply,
            torch.nextafter,
            torch.not_equal,
            torch.outer,
            torch.power,
            torch.remainder,
            torch.subtract,
            torch.true_divide,
            torch.vdot,
        ]
        for bin_fn in bin_fns:
            assert bin_fn(int_vec, int_vec2).shape is not None

        # Integer / bitwise / logical ops
        for int_bin_fn in [
            torch.bitwise_and,
            torch.bitwise_or,
            torch.bitwise_xor,
            torch.gcd,
            torch.lcm,
            torch.left_shift,
            torch.right_shift,
        ]:
            assert int_bin_fn(int_vec, int_vec2).shape == (3,)
        assert torch.bitwise_not(int_vec).shape == (3,)

        for log_bin_fn in [
            torch.logical_and,
            torch.logical_or,
            torch.logical_xor,
        ]:
            assert log_bin_fn(bool_vec, bool_vec2).shape == (3,)
        assert torch.logical_not(bool_vec).shape == (3,)

        # Reductions
        for red_fn in [
            torch.all,
            torch.any,
            torch.argmax,
            torch.argmin,
            torch.count_nonzero,
            torch.logsumexp,
            torch.max,
            torch.mean,
            torch.min,
            torch.prod,
            torch.std,
            torch.sum,
            torch.variance,
        ]:
            assert red_fn(vec).shape == ()

        assert torch.allclose(vec, vec) is True
        assert torch.dot(vec, vec2).shape == ()
        assert torch.matmul(mat, mat).shape == (2, 2)
        assert torch.tensordot(mat, mat, axes=1).shape == (2, 2)
        assert torch.einsum("ij,jk->ik", mat, mat).shape == (2, 2)
        assert torch.cholesky(mat).shape == (2, 2)
        assert torch.det(mat).shape == ()
        assert torch.diag(mat).shape == (2,)
        assert len(torch.eigh(mat)) == 2
        assert torch.eigvalsh(mat).shape == (2,)
        assert torch.inv(mat).shape == (2, 2)
        assert torch.matrix_power(mat, 2).shape == (2, 2)
        assert torch.pinv(mat).shape == (2, 2)
        assert len(torch.qr(mat)) == 2
        assert len(torch.svd(mat)) == 3
        assert len(torch.frexp(vec)) == 2
        assert len(torch.meshgrid(vec, vec2)) == 2
        assert len(torch.meshgrid(vec, vec2, indexing="xy")) == 2
        assert torch.broadcast_to(vec, (2, 3)).shape == (2, 3)
        assert torch.expand(vec, (2, 3)).shape == (2, 3)
        assert torch.flatten(mat).shape == (4,)
        assert torch.moveaxis(mat, 0, 1).shape == (2, 2)
        assert torch.permute(mat, axes=(1, 0)).shape == (2, 2)
        assert torch.repeat(vec, 2).shape == (6,)
        assert torch.reshape(mat, (4,)).shape == (4,)
        assert torch.roll(vec, 1).shape == (3,)
        assert len(torch.split(torch.ones((4,)), 2)) == 2
        assert torch.squeeze(torch.ones((1, 2)), 0).shape == (2,)
        assert torch.squeeze(torch.ones((1, 2))).shape == (2,)
        assert torch.stack([vec, vec2]).shape == (2, 3)
        assert torch.swapaxes(mat, 0, 1).shape == (2, 2)
        assert torch.take(vec, torch.tensor([0, 2])).shape == (2,)
        assert torch.tile(vec, (2,)).shape == (6,)
        assert torch.tril(mat).shape == (2, 2)
        assert torch.triu(mat).shape == (2, 2)
        assert torch.unsqueeze(vec, 0).shape == (1, 3)
        assert torch.unsqueeze(vec, -1).shape == (3, 1)
        assert len(torch.unstack(mat)) == 2
        assert torch.where(bool_vec, vec, vec2).shape == (3,)
        assert torch.ones((2, 2)).shape == (2, 2)
        assert torch.ones_like(mat).shape == (2, 2)
        assert torch.zeros((2, 2)).shape == (2, 2)
        assert torch.zeros_like(mat).shape == (2, 2)
        assert torch.full((2, 2), 3.0).shape == (2, 2)
        assert torch.full_like(mat, 3.0).shape == (2, 2)
        assert torch.empty((2, 2)).shape == (2, 2)
        assert torch.eye(2).shape == (2, 2)
        assert torch.identity(3).shape == (3, 3)
        assert torch.arange(4).shape == (4,)
        assert torch.linspace(0.0, 1.0, 4).shape == (4,)

        assert torch.bitcast(vec, torch.int32).shape == (3,)
        assert torch.bitcast(vec, dtype=torch.int32).shape == (3,)
        assert torch.cast(vec, torch.float32).shape == (3,)
        assert torch.cast(vec, dtype=torch.float32).shape == (3,)

        assert torch.concatenate(tensors=[vec, vec], dim=0).shape == (6,)
        assert torch.concatenate([vec, vec], 0).shape == (6,)

        idx = torch.tensor([0, 1])
        assert torch.gather(vec, 0, idx).shape == (2,)
        assert torch.gather(vec, dim=0, index=[0, 1]).shape == (2,)
        assert torch.gather(vec, index=idx).shape == (2,)
        assert torch.gather(vec, 0, index=idx).shape == (2,)
        with mock.patch(
            "zero_torch.get_op",
            return_value=lambda: lambda *a, **kw: vec._tensor,
        ):
            assert torch.gather(vec, 0).shape == (3,)

        assert torch.norm(vec, p=2, dim=0, keepdim=True).shape == (1,)
        assert torch.norm(vec, p=3).shape == ()

        assert torch.reshape(mat, newshape=(4,)).shape == (4,)
        assert torch.transpose(mat, 0, 1).shape == (2, 2)
        assert torch.transpose(mat, dim0=0, dim1=1).shape == (2, 2)
        assert torch.transpose(mat, dim0="bad", dim1="bad").shape == (2, 2)
        assert torch.transpose(mat, axes=(1, 0)).shape == (2, 2)

        assert torch.logit(torch.tensor([0.2, 0.8]), eps=1e-4).shape == (2,)
        assert torch.logit(torch.tensor([0.2, 0.8])).shape == (2,)
        assert torch.signbit(torch.tensor([-1.0, 1.0])).shape == (2,)
        assert torch.xlogy(
            torch.tensor([0.0, 2.0]), torch.tensor([1.0, 3.0])
        ).shape == (2,)
        assert torch.mvlgamma(torch.tensor([2.0, 3.0]), p=2).shape == (2,)

        assert torch.nan_to_num(
            torch.tensor([float("nan"), float("inf"), float("-inf")]),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        ).shape == (3,)
        assert torch.nan_to_num(torch.tensor([1.0]), nan=None).shape == (1,)

        assert torch.manual_seed(42) == 42
        assert torch.rand((2, 2)).shape == (2, 2)
        assert torch.rand(2, 2).shape == (2, 2)
        assert torch.randn((2, 2)).shape == (2, 2)
        assert torch.randn(2, 2).shape == (2, 2)
        assert torch.randint(5, (2, 2)).shape == (2, 2)
        assert torch.randint(0, 5).shape == ()
        assert torch.randint(5).shape == ()
        assert torch.randint(0, 5, 3).shape == (3,)
        assert torch.randint(5, None, (2,)).shape == (2,)

        assert torch.clip(vec, min=1.5, max=2.5).shape == (3,)
        assert torch.clip(vec).shape == (3,)
        assert torch.clamp(vec, min=1.5, max=2.5).shape == (3,)
        assert torch.sigmoid(vec).shape == (3,)
        assert torch.softmax(vec, dim=0).shape == (3,)

        for op_fn in (
            torch.binary,
            torch.creation,
            torch.divmod,
            torch.reductions,
            torch.scatter,
            torch.shape,
            torch.unary,
        ):
            with mock.patch("zero_torch._ops") as mock_ops:
                mock_ops.binary.return_value = vec._tensor
                mock_ops.creation.return_value = vec._tensor
                mock_ops.divmod.return_value = vec._tensor
                mock_ops.reductions.return_value = vec._tensor
                mock_ops.scatter.return_value = vec._tensor
                mock_ops.shape.return_value = vec._tensor
                mock_ops.unary.return_value = vec._tensor
                assert op_fn(vec).shape == (3,)

        with mock.patch("zero_torch._dynamic_slice", return_value=vec._tensor):
            assert torch.dynamic_slice(vec).shape == (3,)

        for op_fn in (
            torch.gather_nd,
            torch.scatter_add,
            torch.scatter_nd,
            torch.slice,
            torch.strided_slice,
            torch.update_slice,
        ):
            with mock.patch(
                "zero_torch.get_op",
                return_value=lambda: lambda *a, **kw: vec._tensor,
            ):
                assert op_fn(vec).shape == (3,)


def test_tensor_methods_and_edge_cases() -> None:
    """Verifies `Tensor` methods, dtypes, indexing, tracing, and edge cases."""
    with EagerMode():
        assert _flatten_nested(None) == [0.0]

        t_grad = torch.tensor([1.0, 2.0], requires_grad=True)
        t_grad.retain_grad()
        assert t_grad._retains_grad is True
        assert t_grad.is_leaf is True
        assert t_grad.device == "cpu"
        assert t_grad.dim() == 1
        assert len(t_grad) == 2
        assert len(torch.tensor(5.0)) == 0
        assert torch.tensor(None).numel() == 1

        t_no_grad = torch.tensor([1.0], requires_grad=False)
        with pytest.raises(RuntimeError):
            t_no_grad.retain_grad()
        with pytest.raises(RuntimeError):
            t_no_grad.backward()
        with pytest.raises(RuntimeError):
            t_grad.backward()

        t_none = Tensor(None)
        assert t_none.view(1) is t_none
        assert t_none.reshape(1) is t_none
        assert t_none.detach() is t_none
        assert t_none.tolist() == []
        assert t_none.unsqueeze(0) is t_none
        assert t_none.T is t_none
        with pytest.raises(RuntimeError):
            t_none.numpy()

        t_mat = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert t_mat.reshape([4]).shape == (4,)
        assert t_mat.detach().shape == (2, 2)
        assert t_mat.unsqueeze(0).shape == (1, 2, 2)
        assert t_mat.transpose(0, 1).shape == (2, 2)
        assert len(t_mat.chunk(2, dim=0)) == 2

        t_sp = torch.tensor([1.5, 2.5])
        assert t_sp.digamma().shape == (2,)
        assert t_sp.erfc().shape == (2,)
        assert t_sp.erfinv().shape == (2,)
        assert t_sp.lgamma().shape == (2,)

        t_cmp1 = torch.tensor([1.0, 2.0])
        t_cmp2 = torch.tensor([2.0, 2.0])
        assert (t_cmp1 < t_cmp2).shape == (2,)
        assert (t_cmp1 > t_cmp2).shape == (2,)
        assert (t_cmp1 <= t_cmp2).shape == (2,)
        assert (t_cmp1 >= t_cmp2).shape == (2,)
        assert (t_cmp1 == t_cmp2).shape == (2,)
        assert (t_cmp1 != t_cmp2).shape == (2,)

        assert t_mat[0].shape == (2,)
        assert t_mat[0:1].shape == (1, 2)
        with pytest.raises(IndexError):
            _ = torch.tensor(5.0)[0]
        t_scalar_tolist = torch.tensor([1.0, 2.0])
        with mock.patch.object(Tensor, "tolist", return_value=5.0):
            assert t_scalar_tolist[0].item() == 5.0

        t_np = torch.tensor([1.0, 2.0])
        assert t_np.data is not None
        assert isinstance(_wrap([t_np._tensor]), Tensor)
        assert isinstance(_wrap([]), Tensor)
        assert isinstance(_wrap_list([t_np._tensor]), list)
        assert isinstance(_wrap_list(t_np._tensor), list)
        assert isinstance(_wrap_tuple((t_np._tensor,)), tuple)
        assert isinstance(_wrap_tuple(t_np._tensor), tuple)

        # Dtypes and _to_tensor edge cases
        import numpy as np

        assert Tensor(np.array([1.0], dtype=np.float64)).dtype == DType.Float64
        assert Tensor(np.array([1], dtype=np.int32)).dtype == DType.Int32
        assert Tensor(np.array([1], dtype=np.int64)).dtype == DType.Int64
        assert Tensor(np.array([True], dtype=np.bool_)).dtype == DType.Bool
        assert Tensor(np.array([1], dtype=np.uint8)).dtype == DType.Float32
        assert Tensor([1.0], dtype="float64").dtype == DType.Float64
        assert Tensor([1.0], dtype="invalid_dtype").dtype == DType.Float32
        assert (
            Tensor(ProxyTensor(id="p1", shape=(1,), dtype="invalid")).dtype
            == DType.Float32
        )
        assert Tensor(ProxyTensor(id="p2", shape=(1,), dtype="")).dtype == DType.Float32
        assert Tensor("custom_str").shape == ()
        assert Tensor([1.0], dtype=None).dtype == DType.Float32

        st_scalar = SwitcherooTensor(
            data=3.0, config=TensorConfig(shape=(), dtype=DType.Float32, device="cpu")
        )
        assert Tensor(st_scalar).tolist() == [3.0]
        _tracer.start_tracing()
        try:
            _ = _to_tensor(st_scalar)
            _ = _to_tensor(t_np._tensor)
        finally:
            _tracer.stop_tracing()
        st_none = SwitcherooTensor(
            data=None, config=TensorConfig(shape=(), dtype=DType.Float32, device="cpu")
        )
        assert Tensor(st_none).tolist() == []

        from ml_switcheroo_compiler import ops as _sw_ops

        with mock.patch.object(_sw_ops, "array", side_effect=RuntimeError("fail")):
            t_fallback = Tensor([1.0, 2.0])
            assert t_fallback.shape == (2,)

        # Tracing constant emission in _to_tensor
        prev_eager = config.eager_mode
        config.eager_mode = False
        _tracer.start_tracing()
        try:
            _ = _to_tensor([1.0, 2.0])
        finally:
            _tracer.stop_tracing()
            config.eager_mode = prev_eager


def test_autograd_engine_and_math_ops() -> None:
    """Verifies autograd engine, backward functions, retain_grad, and unbroadcasting."""
    with EagerMode():
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)

        s = a + b
        s.backward(Tensor([1.0, 1.0]), retain_graph=True)
        assert a.grad is not None and b.grad is not None
        s.backward(Tensor([1.0, 1.0]), retain_graph=False)

        a.grad = None
        b.grad = None
        diff = a - b
        diff.backward(Tensor([1.0, 1.0]))
        assert a.grad is not None and b.grad is not None

        a.grad = None
        b.grad = None
        quot = a / b
        quot.backward(Tensor([1.0, 1.0]))
        assert a.grad is not None and b.grad is not None

        a.grad = None
        b.grad = None
        pw = a**b
        pw.backward(Tensor([1.0, 1.0]))
        assert a.grad is not None and b.grad is not None

        a.grad = None
        neg = -a
        neg.backward(Tensor([1.0, 1.0]))
        assert a.grad is not None

        m1 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        m2 = Tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        mm = m1 @ m2
        mm.backward(Tensor([[1.0, 1.0], [1.0, 1.0]]))
        assert m1.grad is not None and m2.grad is not None

        m3 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
        mm_r = m3 @ m2
        mm_r.backward(Tensor([[1.0, 1.0], [1.0, 1.0]]))
        mm_l = m1 @ m3
        mm_l.backward(Tensor([[1.0, 1.0], [1.0, 1.0]]))

        r1 = 2.0 + a
        r2 = 2.0 - a
        r3 = 2.0 * a
        r4 = 2.0 / a
        r5 = a**2.0
        r6 = Tensor([2.0, 3.0], requires_grad=False) ** b
        for r in (r1, r2, r3, r4, r5, r6):
            r.backward(Tensor([1.0, 1.0]))

        # Unreached ancestor node in topo (when an intermediate node returns None for one input branch)
        class DropFirstGradFn(Function):
            """Function that drops the first input gradient."""

            @staticmethod
            def forward(ctx: torch.autograd.Context, x1: Tensor, x2: Tensor) -> Tensor:
                """Adds `x1` and `x2`.

                Args:
                    ctx: Autograd context.
                    x1: First input tensor.
                    x2: Second input tensor.

                Returns:
                    Sum tensor.
                """
                _ = ctx
                return x1 + x2

            @staticmethod
            def backward(
                ctx: torch.autograd.Context, *grad_outputs: Tensor
            ) -> tuple[None, Tensor]:
                """Returns `(None, grad_output)`.

                Args:
                    ctx: Autograd context.
                    *grad_outputs: Upstream gradients.

                Returns:
                    Tuple of `(None, grad_output)`.
                """
                _ = ctx
                return None, grad_outputs[0]

        branch1 = a * 2.0
        branch2 = b * 3.0
        joined = DropFirstGradFn.apply(branch1, branch2)
        joined.backward(Tensor([1.0, 1.0]))
        _backward(Tensor([1.0], requires_grad=False), Tensor([1.0]))

        n1 = Tensor([[1.0, 2.0]], requires_grad=True)
        rm = [[1.0], [2.0]] @ n1
        rm.backward(Tensor([[1.0, 1.0], [1.0, 1.0]]))
        assert n1.grad is not None

        x = Tensor([2.0], requires_grad=True)
        u = x * 2.0
        v1 = u + 1.0
        v2 = u * 3.0
        out = v1 + v2
        out.backward()
        assert x.grad is not None and x.grad.item() == 8.0

        t_leaf = Tensor([3.0], requires_grad=True)
        t_leaf.backward()
        assert t_leaf.grad is not None and t_leaf.grad.item() == 1.0
        torch.autograd.backward(t_leaf)

        with pytest.raises(NotImplementedError):
            Function.forward(torch.autograd.Context(), t_leaf)
        with pytest.raises(NotImplementedError):
            Function.backward(torch.autograd.Context(), t_leaf)

        with torch.set_grad_enabled(False):
            assert torch.autograd.is_grad_enabled() is False
            y_ng = t_leaf + 1.0
            assert y_ng.requires_grad is False

        class MultiOutFn(Function):
            """Custom Function returning multiple outputs."""

            @staticmethod
            def forward(
                ctx: torch.autograd.Context, inp: Tensor, scale: float
            ) -> tuple[Tensor, Tensor, float]:
                """Multiplies and adds `scale` to `inp`.

                Args:
                    ctx: Autograd context.
                    inp: Input tensor.
                    scale: Scalar multiplier.

                Returns:
                    Tuple of two tensors and a float.
                """
                ctx.save_for_backward(inp)
                return inp * scale, inp + scale, 0.0

            @staticmethod
            def backward(
                ctx: torch.autograd.Context, *grad_outputs: Tensor
            ) -> tuple[None, None]:
                """Returns None gradients.

                Args:
                    ctx: Autograd context.
                    *grad_outputs: Upstream gradients.

                Returns:
                    Tuple of `(None, None)`.
                """
                _ = (ctx, grad_outputs)
                return None, None

        r_tuple = MultiOutFn.apply(Tensor([1.0], requires_grad=True), 2.0)
        assert isinstance(r_tuple, tuple)
        assert r_tuple[0].grad_fn is not None
        assert r_tuple[0].grad_fn(Tensor([1.0])) == (None, None)
        _backward(r_tuple[0], Tensor([1.0]))
        _backward(r_tuple[0], Tensor([1.0]))

        leaf = Tensor([2.0, 3.0], requires_grad=True)
        leaf.grad = Tensor([1.0, 1.0])
        _backward(leaf, Tensor([1.0, 1.0]))
        assert leaf.grad is not None

        x_ret = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b_ret = Tensor([10.0, 20.0], requires_grad=True)
        b_col = Tensor([[1.0], [2.0]], requires_grad=True)
        mid = x_ret + b_ret
        mid.retain_grad()
        mid.grad = Tensor([[0.5, 0.5], [0.5, 0.5]])
        mid2 = mid + b_col
        mid2.retain_grad()
        total = mid2.sum()
        total.backward()
        assert mid.grad is not None and mid2.grad is not None
        assert b_ret.grad is not None and b_ret.grad.shape == (2,)
        assert b_col.grad is not None and b_col.grad.shape == (2, 1)
        assert x_ret.sum(dim=0, keepdim=True).shape == (1, 2)

        from zero_torch.autograd.math_ops import SumBackward, _unbroadcast

        assert _unbroadcast(Tensor([[1.0, 2.0]]), (2,)).shape == (2,)
        ctx_sum = Function(
            torch.autograd.Context(), [Tensor([1.0], requires_grad=False)]
        ).ctx
        ctx_sum.save_for_backward(Tensor([1.0], requires_grad=False))
        assert SumBackward.backward(ctx_sum, Tensor(1.0)) is None

        t_scalar = Tensor(5.0)
        assert len(t_scalar.chunk(2)) == 1
        t_zero_dim = Tensor([])
        assert len(t_zero_dim.chunk(2)) == 2
        with pytest.raises(ValueError):
            Tensor([1.0, 2.0]).chunk(0)
        t_odd = Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        odd_chunks = t_odd.chunk(2, dim=-1)
        assert len(odd_chunks) == 2 and odd_chunks[0].shape == (3,)


def test_distributions_and_linalg_branches() -> None:
    """Verifies `Distribution` base properties and `linalg` multi-output branches."""
    with EagerMode():
        d = Distribution(validate_args=True)
        assert d._validate_args is True
        with pytest.raises(NotImplementedError):
            _ = d.mean
        with pytest.raises(NotImplementedError):
            _ = d.variance
        with pytest.raises(NotImplementedError):
            _ = d.stddev

        mat = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
        with mock.patch(
            "zero_torch.linalg.get_op",
            return_value=lambda: lambda *a, **kw: mat._tensor,
        ):
            assert torch.linalg.solve_ex(mat, mat)[0].shape == (2, 2)
        with mock.patch("zero_torch.linalg._ops.lstsq", return_value=mat._tensor):
            assert torch.linalg.lstsq(mat, mat)[0].shape == (2, 2)
        assert torch.slogdet(mat)[0].shape == ()
        assert torch.slogdet(1.0)[0].shape == ()

        bern = torch.distributions.Bernoulli(probs=torch.tensor([0.25, 0.75]))
        assert bern.mean.shape == (2,)
        assert bern.variance.shape == (2,)
        assert bern.entropy().shape == (2,)

        beta = torch.distributions.Beta(
            torch.tensor([2.0, 3.0]), torch.tensor([2.0, 5.0])
        )
        assert beta.mean.shape == (2,)
        assert beta.variance.shape == (2,)

        cat = torch.distributions.Categorical(probs=torch.tensor([0.2, 0.3, 0.5]))
        assert cat.mean.shape == ()
        assert cat.variance.shape == ()
        assert cat.entropy().shape == ()

        gam = torch.distributions.Gamma(
            torch.tensor([2.0, 4.0]), torch.tensor([1.0, 2.0])
        )
        assert gam.mean.shape == (2,)
        assert gam.variance.shape == (2,)

        norm = torch.distributions.Normal(
            torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0])
        )
        assert norm.mean.shape == (2,)
        assert norm.variance.shape == (2,)
        assert norm.entropy().shape == (2,)
        assert norm.cdf(torch.tensor([0.0, 1.0])).shape == (2,)
        assert norm.icdf(torch.tensor([0.5, 0.5])).shape == (2,)

        pois = torch.distributions.Poisson(torch.tensor([3.0, 5.0]))
        assert pois.mean.shape == (2,)
        assert pois.variance.shape == (2,)

        unif = torch.distributions.Uniform(
            torch.tensor([0.0, 1.0]), torch.tensor([2.0, 5.0])
        )
        assert unif.mean.shape == (2,)
        assert unif.variance.shape == (2,)
        assert unif.entropy().shape == (2,)
        assert unif.cdf(torch.tensor([1.0, 3.0])).shape == (2,)
        assert unif.icdf(torch.tensor([0.5, 0.5])).shape == (2,)


def test_nn_modules_and_functional_branches() -> None:
    """Verifies all `zero_torch.nn` modules and `zero_torch.nn.functional` operations."""
    with EagerMode():
        x1d = torch.ones((2, 4, 6))
        x2d = torch.ones((2, 4, 4, 4))
        x3d = torch.ones((1, 4, 3, 3, 3))
        v1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        v2 = torch.tensor([[2.0, 1.0], [0.0, 4.0]])
        y_bin = torch.tensor([1.0, -1.0])

        # Activations
        for act_mod in [
            nn.CELU(),
            nn.ELU(),
            nn.GELU(),
            nn.GLU(dim=-1),
            nn.Hardshrink(),
            nn.Hardsigmoid(),
            nn.Hardswish(),
            nn.Hardtanh(),
            nn.LeakyReLU(),
            nn.LogSigmoid(),
            nn.LogSoftmax(dim=-1),
            nn.Mish(),
            nn.PReLU(),
            nn.RReLU(),
            nn.ReLU(),
            nn.ReLU6(),
            nn.SELU(),
            nn.SiLU(),
            nn.Sigmoid(),
            nn.Softmin(dim=-1),
            nn.Softmax(dim=-1),
            nn.Softplus(),
            nn.Softshrink(),
            nn.Softsign(),
            nn.Tanh(),
            nn.Tanhshrink(),
            nn.Threshold(0.5, 0.0),
        ]:
            assert act_mod(v1).shape == (2, 2)
        assert nn.Softmax2d()(x2d).shape == (2, 4, 4, 4)

        assert F.hard_sigmoid(v1).shape == (2, 2)
        assert F.soft_shrink(v1).shape == (2, 2)
        assert F.softmax2d(x2d).shape == (2, 4, 4, 4)
        assert F.softmin(v1, dim=-1).shape == (2, 2)
        assert F.softplus(1.0).shape == ()
        assert F.tanhshrink(v1).shape == (2, 2)
        assert F.threshold(v1, 0.5, 0.0).shape == (2, 2)
        assert F.layer_norm(v1, (2,)).shape == (2, 2)
        assert F.multi_label_margin_loss(v1, v2).shape == ()
        assert F.multi_label_soft_margin_loss(v1, v2).shape == ()

        # Convolutions, Fold, Unfold
        assert nn.Conv1d(4, 2, kernel_size=2, padding=(1,))(x1d).shape == (2, 2, 7)
        assert nn.Conv1d(4, 2, kernel_size=2, padding="valid", bias=False)(
            x1d
        ).shape == (
            2,
            2,
            5,
        )
        assert nn.Conv2d(4, 2, kernel_size=2, padding=(1, 1))(x2d).shape == (
            2,
            2,
            5,
            5,
        )
        assert nn.Conv2d(4, 2, kernel_size=2, padding="valid", bias=False)(
            x2d
        ).shape == (
            2,
            2,
            3,
            3,
        )
        assert nn.Conv3d(4, 2, kernel_size=2, padding=1)(x3d).shape == (1, 2, 4, 4, 4)
        assert nn.Conv3d(4, 2, kernel_size=2, padding=(1, 1, 1))(x3d).shape == (
            1,
            2,
            4,
            4,
            4,
        )
        assert nn.Conv3d(4, 2, kernel_size=2, padding="valid", bias=False)(
            x3d
        ).shape == (
            1,
            2,
            2,
            2,
            2,
        )
        assert nn.ConvTranspose1d(4, 2, kernel_size=2)(x1d).shape == (2, 2, 7)
        assert nn.ConvTranspose2d(4, 2, kernel_size=2)(x2d).shape == (2, 2, 5, 5)
        assert nn.ConvTranspose3d(4, 2, kernel_size=2)(x3d).shape == (1, 2, 4, 4, 4)

        uf = nn.Unfold(kernel_size=2)
        with mock.patch("zero_torch.nn.conv.unfold", return_value=x2d._tensor):
            assert uf(x2d).shape == (2, 4, 4, 4)
        fd = nn.Fold(output_size=(4, 4), kernel_size=2)
        with mock.patch("zero_torch.nn.conv.fold", return_value=x2d._tensor):
            assert fd(x2d).shape == (2, 4, 4, 4)

        # Distance
        assert nn.CosineSimilarity(dim=-1)(v1, v2).shape == (2,)
        assert nn.PairwiseDistance(p=1.0)(v1, v2).shape == (2,)
        assert nn.PairwiseDistance(p=2.0)(v1, v2).shape == (2,)
        assert nn.PairwiseDistance(p=3.0)(v1, v2).shape == (2,)

        # Dropout & functional_dropout
        for drop_mod in [
            nn.Dropout(0.5),
            nn.Dropout1d(0.5),
            nn.Dropout2d(0.5),
            nn.Dropout3d(0.5),
            nn.AlphaDropout(0.5),
            nn.FeatureAlphaDropout(0.5),
        ]:
            drop_mod.eval()
            assert drop_mod(v1).shape == (2, 2)
        for fd_fn in [
            FD.alpha_dropout,
            FD.dropout,
            FD.dropout1d,
            FD.dropout2d,
            FD.dropout3d,
            FD.feature_alpha_dropout,
        ]:
            assert fd_fn(v1, p=0.5, training=False, inplace=False).shape == (2, 2)
            assert fd_fn(v1, p=0.5, training=True, inplace=False).shape == (2, 2)
            assert fd_fn(v1, p=0.5, training=True, inplace=True).shape == (2, 2)
            assert fd_fn(v1, p=1.0, training=True, inplace=False).shape == (2, 2)

        # Flatten, Identity, Linear, Bilinear, DataParallel, ChannelShuffle
        assert nn.Flatten()(x2d).shape == (128,)
        assert nn.Unflatten(1, (2, 2))(x1d).shape == (2, 2, 2, 6)
        assert nn.Unflatten(-1, 6)(x1d).shape == (2, 4, 6)
        assert nn.Identity()(v1).shape == (2, 2)
        assert nn.Linear(2, 3, bias=False)(v1).shape == (2, 3)
        assert nn.Bilinear(2, 2, 3, bias=True)(v1, v2).shape == (2, 3)
        assert nn.Bilinear(2, 2, 3, bias=False)(v1, v2).shape == (2, 3)
        assert nn.DataParallel(nn.Linear(2, 3))(v1).shape == (2, 3)
        assert nn.ChannelShuffle(2)(x2d).shape == (2, 4, 4, 4)

        # Padding modules
        for pad1 in [
            nn.CircularPad1d(1),
            nn.ConstantPad1d(1, 0.0),
            nn.ReflectionPad1d(1),
            nn.ReplicationPad1d(1),
            nn.ZeroPad1d(1),
        ]:
            assert pad1(x1d).shape == (2, 4, 6)
        for pad2 in [
            nn.CircularPad2d(1),
            nn.ConstantPad2d(1, 0.0),
            nn.ReflectionPad2d(1),
            nn.ReplicationPad2d(1),
            nn.ZeroPad2d(1),
        ]:
            assert pad2(x2d).shape == (2, 4, 4, 4)
        for pad3 in [
            nn.CircularPad3d(1),
            nn.ConstantPad3d(1, 0.0),
            nn.ReflectionPad3d(1),
            nn.ReplicationPad3d(1),
            nn.ZeroPad3d(1),
        ]:
            assert pad3(x3d).shape == (1, 4, 3, 3, 3)

        # Pooling & functional_pooling
        assert FP._format_padding((1, 2), 2) == ((1, 1), (2, 2))
        assert nn.AdaptiveAvgPool2d(2)(x2d).shape is not None
        assert nn.AdaptiveMaxPool2d(2)(x2d).shape is not None
        assert nn.AvgPool1d(2)(x1d).shape is not None
        assert nn.AvgPool2d(2)(x2d).shape is not None
        assert nn.AvgPool3d(2)(x3d).shape is not None
        assert nn.MaxPool1d(2)(x1d).shape is not None
        assert nn.MaxPool2d(2)(x2d).shape is not None
        assert nn.MaxPool3d(2)(x3d).shape is not None
        assert nn.LPPool1d(1.0, 2)(x1d).shape is not None
        assert nn.LPPool1d(2.0, 2, stride=2)(x1d).shape is not None
        assert nn.LPPool2d(1.0, 2)(x2d).shape is not None
        assert nn.LPPool2d(2.0, 2, stride=2)(x2d).shape is not None
        assert nn.LPPool3d(1.0, 2)(x3d).shape is not None
        assert nn.LPPool3d(2.0, 2, stride=2)(x3d).shape is not None
        assert F.avg_pool1d(x1d, 2).shape is not None
        assert F.avg_pool2d(x2d, 2).shape is not None
        assert F.avg_pool3d(x3d, 2).shape is not None
        assert F.max_pool1d(x1d, 2).shape is not None
        assert F.max_pool2d(x2d, 2).shape is not None
        assert F.max_pool3d(x3d, 2).shape is not None

        for unimpl_pool in [
            nn.AdaptiveAvgPool1d(2),
            nn.AdaptiveAvgPool3d(2),
            nn.AdaptiveMaxPool1d(2),
            nn.AdaptiveMaxPool2d(2, return_indices=True),
            nn.AdaptiveMaxPool3d(2),
            nn.FractionalMaxPool2d(2, output_size=1),
            nn.FractionalMaxPool3d(2, output_size=1),
        ]:
            with pytest.raises(UnimplementedMathError):
                unimpl_pool(x2d)
        for unpool_mod in [
            nn.MaxUnpool1d(2),
            nn.MaxUnpool2d(2),
            nn.MaxUnpool3d(2),
        ]:
            with pytest.raises(UnimplementedMathError):
                unpool_mod(x1d, x1d)

        # Losses
        assert nn.loss._Loss()(v1, v2).shape == (2, 2)
        for reduction in ("none", "mean", "sum"):
            assert nn.L1Loss(reduction=reduction)(v1, v2).shape is not None
            assert nn.MSELoss(reduction=reduction)(v1, v2).shape is not None
            assert nn.SmoothL1Loss(reduction=reduction)(v1, v2).shape is not None
            assert nn.HuberLoss(reduction=reduction)(v1, v2).shape is not None
            assert (
                nn.KLDivLoss(reduction=reduction, log_target=False)(
                    v1, torch.ones((2, 2))
                ).shape
                is not None
            )
            assert (
                nn.KLDivLoss(reduction=reduction, log_target=True)(v1, v2).shape
                is not None
            )
            assert (
                nn.BCELoss(reduction=reduction)(
                    torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
                    torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                ).shape
                is not None
            )
            assert nn.BCEWithLogitsLoss(reduction=reduction)(v1, v2).shape is not None
            assert (
                nn.CosineEmbeddingLoss(reduction=reduction)(v1, v2, y_bin).shape
                is not None
            )
            assert nn.HingeEmbeddingLoss(reduction=reduction)(v1, v2).shape is not None
            assert (
                nn.MarginRankingLoss(reduction=reduction)(v1, v2, v1).shape is not None
            )
            assert (
                nn.MultiLabelMarginLoss(reduction=reduction)(v1, v2).shape is not None
            )
            assert (
                nn.MultiLabelSoftMarginLoss(reduction=reduction)(v1, v2).shape
                is not None
            )
            assert (
                nn.MultiMarginLoss(reduction=reduction)(v1, torch.tensor([0, 1])).shape
                is not None
            )
            assert nn.PoissonNLLLoss(reduction=reduction)(v1, v2).shape is not None
            assert nn.SoftMarginLoss(reduction=reduction)(v1, v2).shape is not None
            assert (
                nn.TripletMarginLoss(reduction=reduction)(v1, v2, v1).shape is not None
            )
            assert (
                nn.TripletMarginWithDistanceLoss(reduction=reduction)(v1, v2, v1).shape
                is not None
            )
            assert (
                nn.GaussianNLLLoss(reduction=reduction)(
                    v1, v2, torch.ones((2, 2))
                ).shape
                is not None
            )
            assert (
                nn.CTCLoss(reduction=reduction)(v1, torch.tensor([0, 1])).shape
                is not None
            )
            assert (
                nn.NLLLoss(reduction=reduction)(v1, torch.tensor([0, 1])).shape
                is not None
            )
            assert (
                nn.NLLLoss2d(reduction=reduction)(v1, torch.tensor([0, 1])).shape
                is not None
            )
            assert (
                nn.CrossEntropyLoss(reduction=reduction)(v1, torch.tensor([0, 1])).shape
                is not None
            )

        als = nn.AdaptiveLogSoftmaxWithLoss(4, 10, cutoffs=[2, 5])
        out_als, loss_als = als(torch.ones((2, 4)), torch.tensor([0, 1]))
        assert out_als.shape == (1,) and loss_als.shape == (1,)

        lce = nn.LinearCrossEntropyLoss(2, 3)
        assert lce(v1, torch.tensor([0, 1])).shape == ()

        # nn.init
        assert nn.init.calculate_gain("linear") == 1.0
        assert nn.init.calculate_gain("tanh") == pytest.approx(5.0 / 3.0)
        assert nn.init.calculate_gain("relu") == pytest.approx(2.0**0.5)
        assert nn.init.calculate_gain("leaky_relu", 0.2) > 1.0
        assert nn.init.calculate_gain("selu") == 0.75
        with pytest.raises(ValueError):
            nn.init.calculate_gain("unknown_nonlinearity")

        t_init = torch.ones((3, 3))
        assert nn.init.constant_(t_init, 5.0).shape == (3, 3)
        assert nn.init.constant(t_init, 3.0).shape == (3, 3)
        assert nn.init.constant(tensor=t_init, val=2.0).shape == (3, 3)
        assert nn.init.constant().shape == ()
        assert nn.init.dirac_(torch.ones((2, 2, 3))).shape == (2, 2, 3)
        assert nn.init.dirac_(torch.ones((2, 2, 3, 3))).shape == (2, 2, 3, 3)
        assert nn.init.dirac_(torch.ones((2, 2, 3, 3, 3))).shape == (2, 2, 3, 3, 3)
        assert nn.init.dirac(torch.ones((2, 2, 3))).shape == (2, 2, 3)
        assert nn.init.dirac().shape == ()
        with pytest.raises(ValueError):
            nn.init.dirac_(torch.ones((2, 2)))
        with pytest.raises(ValueError):
            nn.init.dirac_(torch.ones((3, 2, 3)), groups=2)
        assert nn.init.eye_(t_init).shape == (3, 3)
        assert nn.init.eye(t_init).shape == (3, 3)
        assert nn.init.eye().shape == ()
        with pytest.raises(ValueError):
            nn.init.eye_(torch.ones((2, 2, 2)))
        assert nn.init.normal(t_init, 0.0, 1.0).shape == (3, 3)
        assert nn.init.normal().shape == ()
        assert nn.init.uniform(t_init, -1.0, 1.0).shape == (3, 3)
        assert nn.init.uniform().shape == ()
        assert nn.init.ones_(t_init).shape == (3, 3)
        assert nn.init.zeros_(t_init).shape == (3, 3)
        assert nn.init.trunc_normal_(t_init).shape == (3, 3)
        assert nn.init.xavier_uniform(t_init).shape == (3, 3)
        assert nn.init.xavier_uniform().shape == ()
        assert nn.init.xavier_normal_(t_init).shape == (3, 3)
        assert nn.init.xavier_normal(t_init).shape == (3, 3)
        assert nn.init.xavier_normal().shape == ()
        assert nn.init.kaiming_uniform_(t_init, mode="fan_out").shape == (3, 3)
        assert nn.init.kaiming_uniform(t_init).shape == (3, 3)
        assert nn.init.kaiming_uniform().shape == ()
        assert nn.init.kaiming_normal_(t_init, mode="fan_in").shape == (3, 3)
        assert nn.init.kaiming_normal_(t_init, mode="fan_out").shape == (3, 3)
        assert nn.init.kaiming_normal(t_init).shape == (3, 3)
        assert nn.init.kaiming_normal().shape == ()
        assert nn.init.orthogonal_(torch.ones((3, 2))).shape == (3, 2)
        assert nn.init.orthogonal_(torch.ones((2, 3))).shape == (2, 3)
        assert nn.init.orthogonal(t_init).shape == (3, 3)
        assert nn.init.orthogonal().shape == ()
        with pytest.raises(ValueError):
            nn.init.orthogonal_(torch.ones((3,)))
        assert nn.init.sparse_(t_init, sparsity=0.5).shape == (3, 3)
        assert nn.init.sparse(t_init, 0.5).shape == (3, 3)
        assert nn.init.sparse().shape == ()
        with pytest.raises(ValueError):
            nn.init.sparse_(torch.ones((3,)), sparsity=0.5)
        assert nn.init._calculate_fan_in_and_fan_out(torch.ones((2, 3, 4, 4))) == (
            48,
            32,
        )
        with pytest.raises(ValueError):
            nn.init._calculate_fan_in_and_fan_out(torch.ones((3,)))

        # nn.lazy
        for lazy_cls, inp in [
            (nn.LazyBatchNorm1d, x1d),
            (nn.LazyBatchNorm2d, x2d),
            (nn.LazyBatchNorm3d, x3d),
            (nn.LazyConv1d, x1d),
            (nn.LazyConv2d, x2d),
            (nn.LazyConv3d, x3d),
            (nn.LazyInstanceNorm1d, x1d),
            (nn.LazyInstanceNorm2d, x2d),
            (nn.LazyInstanceNorm3d, x3d),
            (nn.LazyLinear, v1),
        ]:
            if "Conv" in lazy_cls.__name__:
                m_lazy = lazy_cls(2, 2)
            elif "Linear" in lazy_cls.__name__:
                m_lazy = lazy_cls(2)
            else:
                m_lazy = lazy_cls()
            _ = m_lazy(inp)
            _ = m_lazy(inp)

        for lazy_t_cls, inp in [
            (nn.LazyConvTranspose1d, x1d),
            (nn.LazyConvTranspose2d, x2d),
            (nn.LazyConvTranspose3d, x3d),
        ]:
            m_lt = lazy_t_cls(2, 2)
            _ = m_lt(inp)
            _ = m_lt(inp)

        # nn.norm & functional_norm
        assert FN._norm_shape_tuple(4) == (4,)
        assert FN._norm_shape_tuple([2, 3]) == (2, 3)
        assert FN._norm_shape_tuple("invalid") == (1,)
        assert FN.normalize(v1, dim=1).shape == (2, 2)
        assert FN.rms_norm(torch.tensor([2.0, 3.0])).shape == (2,)

        assert _apply_channel_affine(
            torch.ones((2,)),
            nn.Parameter(torch.ones((2,))),
            nn.Parameter(torch.zeros((2,))),
        ).shape == (2,)
        assert _apply_channel_affine(torch.ones((2,)), None, None).shape == (2,)
        assert nn.BatchNorm1d(4, affine=False, track_running_stats=False)(
            x1d
        ).shape == (
            2,
            4,
            6,
        )
        assert nn.BatchNorm3d(4)(x3d).shape == (1, 4, 3, 3, 3)
        assert nn.SyncBatchNorm(4)(x2d).shape == (2, 4, 4, 4)
        assert nn.CrossMapLRN2d(2)(x2d).shape == (2, 4, 4, 4)
        assert nn.LocalResponseNorm(2)(x2d).shape == (2, 4, 4, 4)
        assert nn.GroupNorm(2, 4, affine=False)(x2d).shape == (2, 4, 4, 4)
        assert nn.GroupNorm(2, 4, affine=True)(x2d).shape == (2, 4, 4, 4)
        assert nn.InstanceNorm1d(4, affine=True, track_running_stats=True)(
            x1d
        ).shape == (
            2,
            4,
            6,
        )
        assert nn.InstanceNorm2d(4)(x2d).shape == (2, 4, 4, 4)
        assert nn.InstanceNorm3d(4)(x3d).shape == (1, 4, 3, 3, 3)
        assert nn.LayerNorm(2, elementwise_affine=True, bias=False)(v1).shape == (2, 2)
        assert nn.LayerNorm(2, elementwise_affine=False)(v1).shape == (2, 2)
        assert nn.RMSNorm(2, elementwise_affine=True)(v1).shape == (2, 2)
        assert nn.RMSNorm(2, elementwise_affine=False)(v1).shape == (2, 2)

        # nn.module containers & lifecycle
        base_m = nn.Module()
        with pytest.raises(NotImplementedError):
            base_m(v1)
        base_m.add_module("none_mod", None)
        base_m.register_parameter("none_param", None)
        base_m.register_buffer("none_buf", None)
        assert base_m.to("cpu") is base_m

        child = nn.Linear(2, 2)
        child.register_buffer("cb1", torch.ones((2,)))
        parent = nn.Module()
        parent.child = child
        parent.buf = torch.ones((2,))
        parent.register_buffer("b1", torch.ones((2,)))
        assert len(list(parent.buffers(recurse=True))) == 2
        assert len(list(parent.buffers(recurse=False))) == 1
        assert len(list(parent.parameters(recurse=True))) == 2
        assert len(list(parent.parameters(recurse=False))) == 0
        assert len(list(parent.named_children())) == 1
        parent.eval()
        assert child.training is False

        assert len(nn.ModuleList()) == 0
        ml = nn.ModuleList([nn.ReLU(), nn.Identity()])
        assert len(ml) == 2
        assert len(list(ml)) == 2
        assert isinstance(ml[0], nn.ReLU)

        assert len(nn.ParameterList()) == 0
        pl = nn.ParameterList([nn.Parameter(torch.ones((2,)))])
        assert len(pl) == 1
        assert len(list(pl)) == 1

        cont = nn.Container(l1=nn.ReLU())
        assert "l1" in cont._modules

        _ = nn.ModuleDict()
        md1 = nn.ModuleDict({"a": nn.ReLU()})
        md2 = nn.ModuleDict([("a", nn.ReLU())])
        with pytest.raises(NotImplementedError):
            md1()

        _ = nn.ParameterDict()
        pd1 = nn.ParameterDict({"w": nn.Parameter(torch.ones((1,)))})
        pd2 = nn.ParameterDict([("w", nn.Parameter(torch.ones((1,))))])
        with pytest.raises(NotImplementedError):
            pd1()
        _ = (md2, pd2)

        # Parameter tracing
        p_trace = nn.Parameter(torch.ones((2,)))
        assert p_trace.data is not None
        _tracer.start_tracing()
        assert p_trace.data is not None
        _tracer.stop_tracing()

        # nn.pixelshuffle
        ps = nn.PixelShuffle(2)
        assert ps(torch.ones((1, 4, 2, 2))).shape == (1, 1, 4, 4)
        with pytest.raises(ValueError):
            ps(torch.ones((4, 2)))
        with pytest.raises(ValueError):
            ps(torch.ones((1, 3, 2, 2)))

        pus = nn.PixelUnshuffle(2)
        assert pus(torch.ones((1, 1, 4, 4))).shape == (1, 4, 2, 2)
        with pytest.raises(ValueError):
            pus(torch.ones((4, 4)))
        with pytest.raises(ValueError):
            pus(torch.ones((1, 1, 3, 4)))

        # nn.relu & Sequential
        seq1 = nn.Sequential(nn.ReLU(), nn.Identity())
        assert seq1(v1).shape == (2, 2)
        seq2 = nn.Sequential({"relu": nn.ReLU(), "id": nn.Identity()})
        assert seq2(v1).shape == (2, 2)

        # nn.sparse
        emb = nn.Embedding(5, 3, _weight=torch.ones((5, 3)))
        assert emb(torch.tensor([0, 1])).shape == (2, 3)
        assert nn.Embedding(5, 3)(torch.tensor([0, 1])).shape == (2, 3)
        eb_mean = nn.EmbeddingBag(5, 3, mode="mean", _weight=torch.ones((5, 3)))
        eb_sum = nn.EmbeddingBag(5, 3, mode="sum")
        eb_max = nn.EmbeddingBag(5, 3, mode="max")
        assert eb_mean(
            torch.tensor([0, 1]), per_sample_weights=torch.ones((2,))
        ).shape == (1, 3)
        assert eb_sum(torch.tensor([0, 1])).shape == (1, 3)
        assert eb_max(torch.tensor([0, 1])).shape == (1, 3)
        assert eb_mean(torch.tensor([[0, 1], [1, 2]])).shape == (2, 3)
        assert eb_sum(torch.tensor([[0, 1], [1, 2]])).shape == (2, 3)
        assert eb_max(torch.tensor([[0, 1], [1, 2]])).shape == (2, 3)

        # nn.upsampling
        up_bi = nn.Upsample(scale_factor=2.0, mode="bilinear")
        assert up_bi(x2d).shape is not None
        up_near = nn.Upsample(scale_factor=2.0, mode="nearest")
        assert up_near(x2d).shape is not None
        assert nn.UpsamplingBilinear2d(scale_factor=2.0)(x2d).shape is not None
        assert nn.UpsamplingNearest2d(scale_factor=2.0)(x2d).shape is not None

        # Functional branches (eager & non-eager)
        assert F.scaled_dot_product_attention(v1, v1, v1).shape == (2, 2)
        assert F.scaled_dot_product_attention(
            torch.ones((2,)), torch.ones((2,)), torch.ones((2,))
        ).shape == (2,)
        assert (
            F.multilabel_soft_margin_loss(v1, v2, weight=torch.ones((2, 2))).shape == ()
        )
        assert F.gelu(v1, approximate="tanh").shape == (2, 2)
        assert F.glu(torch.ones((3,))).shape == (3,)
        assert F.gaussian_nll_loss(v1, v2, torch.ones((2, 2)), full=True).shape == ()
        assert F.nll_loss(
            v1,
            torch.tensor([0, -100]),
            weight=torch.tensor([1.0, 2.0]),
            reduction="none",
        ).shape == (2,)
        assert F.nll_loss(v1, torch.tensor([0, 1]), reduction="sum").shape == ()
        assert F.nll_loss(torch.ones((2,)), torch.ones((2,))).shape == ()
        assert F.poisson_nll_loss(v1, v2, log_input=False).shape == ()
        assert (
            F.triplet_margin_with_distance_loss(
                v1,
                v2,
                v1,
                distance_function=lambda a, b: torch.sum((a - b) ** 2, axis=-1),
            ).shape
            == ()
        )
        assert F.rrelu(v1, training=True).shape == (2, 2)
        assert F.rrelu(v1, training=False).shape == (2, 2)
        assert F.silu(v1).shape == (2, 2)
        assert F.activations(v1).shape == (2, 2)
        assert F.adaptive_avg_pool2d(x2d, 1).shape == (2, 4, 1, 1)
        assert F.adaptive_avg_pool2d(v1, 1).shape == (2, 2)
        assert F.alpha_dropout(v1).shape == (2, 2)
        assert F.batch_norm(x2d).shape == (2, 4, 4, 4)
        assert F.batch_norm(torch.ones((2,))).shape == (2,)
        assert F.complex(v1, v2).shape == (2, 2)
        assert F.conv_transpose1d(torch.ones((2,)), torch.ones((2,))).shape == (2,)
        assert F.conv_transpose2d(torch.ones((2,)), torch.ones((2,))).shape == (2,)
        assert F.conv_transpose3d(torch.ones((2,)), torch.ones((2,))).shape == (2,)
        assert F.dropout(v1).shape == (2, 2)
        assert F.embedding(torch.tensor([0, 1])).shape == (2,)
        assert F.feature_alpha_dropout(v1).shape == (2, 2)
        with pytest.raises(UnimplementedMathError):
            F.fractional_max_pool2d(x2d, 2)
        assert F.fractional_max_pool2d(v1, 2).shape == (2, 2)
        assert F.group_norm(x2d, 2).shape == (2, 4, 4, 4)
        assert F.group_norm(torch.ones((2,)), 1).shape == (2,)
        assert F.gru_cell(v1).shape == (2, 2)
        assert F.instance_norm(x2d).shape == (2, 4, 4, 4)
        assert F.instance_norm(v1).shape == (2, 2)
        assert F.layer_norm(v1, (2,), torch.ones((2,)), torch.zeros((2,))).shape == (
            2,
            2,
        )
        assert F.lstm_cell(v1).shape == (2, 2)
        assert F.pad(v1).shape == (2, 2)
        assert F.rms_norm(v1).shape == (2, 2)
        assert F.rnn_cell(v1).shape == (2, 2)
        assert F.spatial_dropout(v1).shape == (2, 2)
        assert F.swish(v1).shape == (2, 2)
        assert F.upsample_bilinear(x2d).shape == (2, 4, 4, 4)
        assert F.upsample_nearest(x2d).shape == (2, 4, 4, 4)
        assert F.cross_entropy(v1, v2).shape == ()
        assert (
            F.cross_entropy(v1, torch.tensor([0, 1]), label_smoothing=0.1).shape == ()
        )
        assert (
            F.binary_cross_entropy(
                torch.tensor([0.5, 0.5]),
                torch.tensor([1.0, 0.0]),
                weight=torch.ones((2,)),
            ).shape
            == ()
        )
        assert (
            F.binary_cross_entropy_with_logits(
                torch.tensor([0.5, -0.5]),
                torch.tensor([1.0, 0.0]),
                weight=torch.ones((2,)),
                pos_weight=torch.ones((2,)),
            ).shape
            == ()
        )
        assert F.bilinear(
            v1, v2, torch.ones((3, 2, 2)), bias=torch.ones((3,))
        ).shape == (2, 3)
        assert F.bilinear(v1, v2, torch.ones((3, 2, 2)), bias=None).shape == (2, 3)
        assert F.bilinear(torch.ones(()), torch.ones(()), torch.ones(())).shape == ()

        prev_eager = config.eager_mode
        config.eager_mode = False
        try:
            for fn in [
                F.softmax,
                F.scaled_dot_product_attention,
                F.celu,
                F.multilabel_margin_loss,
                F.elu,
                F.ctc_loss,
                F.multilabel_soft_margin_loss,
                F.gelu,
                F.cosine_embedding_loss,
                F.multi_margin_loss,
                F.glu,
                F.gaussian_nll_loss,
                F.hinge_embedding_loss,
                F.poisson_nll_loss,
                F.huber_loss,
                F.soft_margin_loss,
                F.hardswish,
                F.margin_ranking_loss,
                F.triplet_margin_loss,
                F.leaky_relu,
                F.log_softmax,
                F.mish,
                F.selu,
                F.softplus,
                F.activations,
                F.adaptive_avg_pool2d,
                F.alpha_dropout,
                F.batch_norm,
                F.complex,
                F.conv_transpose1d,
                F.conv_transpose2d,
                F.conv_transpose3d,
                F.dropout,
                F.embedding,
                F.feature_alpha_dropout,
                F.fractional_max_pool2d,
                F.group_norm,
                F.gru_cell,
                F.instance_norm,
                F.layer_norm,
                F.lstm_cell,
                F.pad,
                F.rms_norm,
                F.rnn_cell,
                F.spatial_dropout,
                F.swish,
                F.upsample_bilinear,
                F.upsample_nearest,
            ]:
                with pytest.raises(NotImplementedError):
                    fn(v1)
        finally:
            config.eager_mode = prev_eager


def test_rnn_and_transformer_full_coverage() -> None:
    """Verifies all `RNNCell`, `LSTMCell`, `GRUCell`, `RNN`, `LSTM`, `GRU`, and `Transformer` branches."""
    with EagerMode():
        cell_base = nn.RNNCellBase(2, 3, bias=False, num_chunks=1)
        assert cell_base.bias_ih is None
        cell_base_zero = nn.RNNCellBase(2, 0, bias=False, num_chunks=1)
        assert cell_base_zero.hidden_size == 0
        with pytest.raises(TypeError):
            cell_base(torch.ones((2, 2)))

        rnn_cell_tanh = nn.RNNCell(2, 3, bias=True, nonlinearity="tanh")
        assert rnn_cell_tanh(torch.ones((2,))).shape == (2, 3)
        assert rnn_cell_tanh(torch.ones((4, 2)), torch.ones((4, 3))).shape == (4, 3)

        rnn_cell_relu = nn.RNNCell(2, 3, bias=False, nonlinearity="relu")
        assert rnn_cell_relu(torch.ones((4, 2))).shape == (4, 3)

        lstm_cell = nn.LSTMCell(2, 3, bias=True)
        h1, c1 = lstm_cell(torch.ones((2,)))
        assert h1.shape == (2, 3) and c1.shape == (2, 3)
        h2, c2 = lstm_cell(torch.ones((4, 2)), (torch.ones((4, 3)), torch.ones((4, 3))))
        assert h2.shape == (4, 3) and c2.shape == (4, 3)
        lstm_cell_nb = nn.LSTMCell(2, 3, bias=False)
        h3, _ = lstm_cell_nb(torch.ones((4, 2)), torch.ones((4, 3)))
        assert h3.shape == (4, 3)

        gru_cell = nn.GRUCell(2, 3, bias=True)
        assert gru_cell(torch.ones((2,))).shape == (2, 3)
        assert gru_cell(torch.ones((4, 2)), torch.ones((4, 3))).shape == (4, 3)
        gru_cell_nb = nn.GRUCell(2, 3, bias=False)
        assert gru_cell_nb(torch.ones((4, 2))).shape == (4, 3)

        x_seq = torch.ones((3, 2, 4))
        x_bf = torch.ones((2, 3, 4))
        x_unbatched = torch.ones((3, 4))

        rnn_zero = nn.RNN(4, 0, num_layers=1)
        assert rnn_zero.hidden_size == 0

        rnn1 = nn.RNN(
            4,
            3,
            num_layers=2,
            nonlinearity="relu",
            bias=True,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )
        out_r1, h_r1 = rnn1(x_bf)
        assert (
            isinstance(h_r1, Tensor)
            and out_r1.shape == (2, 3, 6)
            and h_r1.shape == (4, 2, 3)
        )

        rnn2 = nn.RNN(
            4, 3, num_layers=1, nonlinearity="tanh", bias=False, bidirectional=False
        )
        out_r2, h_r2 = rnn2(x_unbatched, torch.ones((1, 3)))
        assert (
            isinstance(h_r2, Tensor) and out_r2.shape == (3, 3) and h_r2.shape == (1, 3)
        )

        gru = nn.GRU(4, 3, num_layers=1, bias=True, bidirectional=False)
        out_g, h_g = gru(x_seq, (torch.ones((1, 2, 3)), torch.ones((1, 2, 3))))
        assert (
            isinstance(h_g, Tensor)
            and out_g.shape == (3, 2, 3)
            and h_g.shape == (1, 2, 3)
        )

        lstm = nn.LSTM(4, 3, num_layers=1, bias=True, bidirectional=False)
        out_l, hc_l = lstm(x_seq)
        assert out_l.shape == (3, 2, 3) and isinstance(hc_l, tuple)
        out_l2, _ = lstm(x_seq, hc_l)
        assert out_l2.shape == (3, 2, 3)

        # Transformer & MultiheadAttention
        mha_same = nn.MultiheadAttention(
            4, 2, dropout=0.1, bias=True, add_bias_kv=True, batch_first=False
        )
        q_2d = torch.ones((3, 4))
        attn_2d, _ = mha_same(q_2d, q_2d, q_2d, need_weights=False)
        assert attn_2d.shape == (3, 4)
        attn_2d_w, w_2d = mha_same(q_2d, q_2d, q_2d, need_weights=True)
        assert attn_2d_w.shape == (3, 4) and w_2d is not None

        q_3d = torch.ones((2, 3, 4))
        attn_out, attn_w = mha_same(
            q_3d, q_3d, q_3d, need_weights=True, average_attn_weights=True
        )
        assert attn_out.shape == (2, 3, 4) and attn_w is not None

        attn_out2, attn_w2 = mha_same(
            q_3d,
            q_3d,
            q_3d,
            attn_mask=torch.zeros((2, 2)),
            need_weights=True,
            average_attn_weights=False,
        )
        assert attn_out2.shape == (2, 3, 4) and attn_w2 is not None

        mha_diff = nn.MultiheadAttention(
            4, 2, bias=False, kdim=3, vdim=3, batch_first=True
        )
        q_bf = torch.ones((3, 2, 4))
        kv_bf = torch.ones((3, 2, 3))
        attn_bf, _ = mha_diff(q_bf, kv_bf, kv_bf, need_weights=False)
        assert attn_bf.shape == (3, 2, 4)

        mha_diff_bias = nn.MultiheadAttention(
            4, 2, bias=True, kdim=3, vdim=3, batch_first=True
        )
        attn_bfb, _ = mha_diff_bias(q_bf, kv_bf, kv_bf, need_weights=False)
        assert attn_bfb.shape == (3, 2, 4)
        mha_same_nobias = nn.MultiheadAttention(4, 2, bias=False)
        assert mha_same_nobias(q_3d, q_3d, q_3d)[0].shape == (2, 3, 4)

        enc_layer_post = nn.TransformerEncoderLayer(
            4, 2, dim_feedforward=8, norm_first=False, activation="relu"
        )
        assert enc_layer_post(torch.ones((3, 2, 4))).shape == (3, 2, 4)

        dec_layer_post = nn.TransformerDecoderLayer(
            4, 2, dim_feedforward=8, norm_first=False, activation="relu"
        )
        assert dec_layer_post(torch.ones((3, 2, 4)), torch.ones((3, 2, 4))).shape == (
            3,
            2,
            4,
        )

        enc_layer_norm_first = nn.TransformerEncoderLayer(
            4, 2, dim_feedforward=8, norm_first=True, activation="gelu"
        )
        assert enc_layer_norm_first(torch.ones((3, 2, 4))).shape == (3, 2, 4)

        dec_layer_norm_first = nn.TransformerDecoderLayer(
            4, 2, dim_feedforward=8, norm_first=True, activation="gelu"
        )
        assert dec_layer_norm_first(
            torch.ones((3, 2, 4)), torch.ones((3, 2, 4))
        ).shape == (3, 2, 4)

        assert nn.TransformerEncoder(None, 1)(torch.ones((3, 2, 4))).shape == (
            3,
            2,
            4,
        )
        assert nn.TransformerDecoder(None, 1)(torch.ones((3, 2, 4))).shape == (
            3,
            2,
            4,
        )
        custom_enc = nn.TransformerEncoder(
            enc_layer_norm_first, 1, norm=nn.LayerNorm(4)
        )
        custom_dec = nn.TransformerDecoder(
            dec_layer_norm_first, 1, norm=nn.LayerNorm(4)
        )
        tf_custom = nn.Transformer(
            d_model=4, nhead=2, custom_encoder=custom_enc, custom_decoder=custom_dec
        )
        tf_custom.register_parameter("one_d", nn.Parameter(torch.ones((4,))))
        tf_custom.reset_parameters()
        assert tf_custom(torch.ones((3, 2, 4)), torch.ones((3, 2, 4))).shape == (
            3,
            2,
            4,
        )
        tf_default = nn.Transformer(
            d_model=4, nhead=2, num_encoder_layers=1, num_decoder_layers=1
        )
        assert tf_default(torch.ones((3, 2, 4))).shape == (3, 2, 4)


def test_optim_and_lr_schedulers_full_coverage() -> None:
    """Verifies all optimizers and learning rate schedulers in `zero_torch.optim`."""
    with EagerMode():
        p = nn.Parameter(torch.tensor([2.0, 3.0]))
        p.grad = torch.tensor([0.5, 0.5])
        p_no_grad = nn.Parameter(torch.tensor([1.0]))

        sgd_group: ParamGroupDict = {"params": [p, p_no_grad], "lr": 0.1}
        sgd = optim.SGD([sgd_group], momentum=0.9, maximize=True)
        sgd.param_groups.append({"params": 123})
        loss_val = sgd.step(closure=lambda: 1.5)
        assert loss_val == 1.5
        sgd.step()
        sgd.zero_grad(set_to_none=True)
        assert p.grad is None
        p.grad = torch.tensor([0.5, 0.5])
        sgd.zero_grad(set_to_none=False)
        assert p.grad is not None

        muon = optim.Muon([p], lr=0.01, weight_decay=0.1)
        muon.param_groups.append({"params": 123})
        assert muon.step() is None
        assert muon.step(closure=lambda: 2.0) == 2.0

        assert _float_opt(torch.tensor(0.05), 0.1) == pytest.approx(0.05)
        assert _float_opt("invalid", 0.1) == 0.1
        assert _betas_opt("invalid", (0.9, 0.99)) == (0.9, 0.99)

        for opt_cls, kw in [
            (optim.Adam, {"amsgrad": True, "weight_decay": 0.01}),
            (optim.Adam, {"amsgrad": False, "decoupled_weight_decay": True}),
            (optim.AdamW, {"weight_decay": 0.01}),
            (optim.Adamax, {"weight_decay": 0.01}),
            (optim.Adagrad, {"weight_decay": 0.01, "lr_decay": 0.01}),
            (optim.Adadelta, {"weight_decay": 0.01}),
            (optim.RMSprop, {"weight_decay": 0.01, "momentum": 0.9, "centered": True}),
            (optim.RMSprop, {"weight_decay": 0.0, "momentum": 0.0, "centered": False}),
            (optim.NAdam, {}),
            (optim.RAdam, {}),
            (optim.SparseAdam, {}),
        ]:
            w_opt = nn.Parameter(torch.tensor([2.0, -2.0]))
            w_opt.grad = torch.tensor([0.5, -0.5])
            inst = opt_cls([w_opt, p_no_grad], **kw)
            inst.param_groups.append({"params": 123})
            assert inst.step(closure=lambda: 1.0) == 1.0
            inst.step()

        # LRScheduler base & subclasses
        opt = optim.SGD([p], lr=0.1)
        base_sched = lr_scheduler.LRScheduler(opt)
        assert base_sched.get_last_lr() == [0.1]
        base_sched.step(epoch=3)
        assert base_sched.last_epoch == 3
        st = base_sched.state_dict()
        base_sched.load_state_dict(st)
        base_sched.load_state_dict({"last_epoch": [1.0]})

        s_lambda1 = lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: 0.9**e)
        s_lambda1.step()
        s_lambda2 = lr_scheduler.LambdaLR(opt, lr_lambda=[lambda e: 0.9**e])
        s_lambda2.step()

        s_mult1 = lr_scheduler.MultiplicativeLR(opt, lr_lambda=lambda e: 0.95)
        s_mult1.step()
        s_mult2 = lr_scheduler.MultiplicativeLR(opt, lr_lambda=[lambda e: 0.95])
        s_mult2.step()

        s_step = lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
        s_step.step()

        s_mstep = lr_scheduler.MultiStepLR(opt, milestones=[1, 3], gamma=0.5)
        s_mstep.step()
        s_mstep.step()

        s_const = lr_scheduler.ConstantLR(opt, factor=0.5, total_iters=1)
        s_const.step()
        s_const.step()

        s_lin = lr_scheduler.LinearLR(
            opt, start_factor=0.5, end_factor=1.0, total_iters=1
        )
        s_lin.step()
        s_lin.step()

        s_exp = lr_scheduler.ExponentialLR(opt, gamma=0.9)
        s_exp.step()

        s_poly = lr_scheduler.PolynomialLR(opt, total_iters=1, power=2.0)
        s_poly.step()
        s_poly.step()

        s_cos = lr_scheduler.CosineAnnealingLR(opt, T_max=4, eta_min=0.01)
        s_cos.step()

        s_cos_wr = lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=2, T_mult=2)
        s_cos_wr.step()

        s_cyc1 = lr_scheduler.CyclicLR(
            opt,
            base_lr=0.01,
            max_lr=0.1,
            step_size_up=1,
            step_size_down=2,
            mode="triangular",
        )
        s_cyc1.step()
        s_cyc1.step()
        s_cyc1.step()
        s_cyc2 = lr_scheduler.CyclicLR(
            opt, base_lr=[0.01], max_lr=[0.1], step_size_up=1, mode="triangular2"
        )
        s_cyc2.step()
        s_cyc3 = lr_scheduler.CyclicLR(
            opt, base_lr=0.01, max_lr=0.1, step_size_up=1, mode="exp_range"
        )
        s_cyc3.step()
        s_cyc4 = lr_scheduler.CyclicLR(
            opt, base_lr=0.01, max_lr=0.1, step_size_up=1, scale_fn=lambda c: 0.5
        )
        s_cyc4.step()

        s_1c_cos = lr_scheduler.OneCycleLR(
            opt, max_lr=0.1, total_steps=2, pct_start=0.5, anneal_strategy="cos"
        )
        s_1c_cos.step()
        s_1c_cos.step()
        s_1c_lin = lr_scheduler.OneCycleLR(
            opt,
            max_lr=[0.1],
            epochs=1,
            steps_per_epoch=2,
            pct_start=0.5,
            anneal_strategy="linear",
        )
        s_1c_lin.step()
        s_1c_lin.step()
        s_1c_def = lr_scheduler.OneCycleLR(opt, max_lr=0.1)
        s_1c_def.step()

        s_chain = lr_scheduler.ChainedScheduler([s_step, s_exp])
        s_chain.step()

        s_seq = lr_scheduler.SequentialLR(
            opt, schedulers=[s_step, s_exp], milestones=[1]
        )
        s_seq.step()
        s_seq.step()

        plat_min = lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=0, cooldown=1, threshold_mode="rel"
        )
        plat_min.step(1.0)
        plat_min.step(2.0)
        plat_min.step(3.0)
        assert len(plat_min.get_last_lr()) == 1

        plat_max = lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", patience=0, threshold_mode="abs", min_lr=[0.0], eps=100.0
        )
        plat_max.step(1.0)
        plat_max.step(2.0)
        plat_max.step(0.5)


def test_utils_data_and_dataloader_full_coverage() -> None:
    """Verifies all `zero_torch.utils.data` datasets, samplers, datapipes, and DataLoader branches."""
    with EagerMode():
        ds = data.TensorDataset(torch.tensor([10.0, 20.0, 30.0]))
        assert len(ds) == 3
        assert ds[1][0].item() == 20.0

        s_base: Sampler[int] = Sampler(ds)
        assert len(s_base) == 3
        with pytest.raises(NotImplementedError):
            len(Sampler())

        rs_rep = data.RandomSampler(ds, replacement=True, num_samples=4)
        assert len(list(rs_rep)) == 4 and len(rs_rep) == 4

        ss = data.SequentialSampler(ds)
        assert len(ss) == 3

        with pytest.raises(ValueError):
            data.StackDataset()
        st_tuple = data.StackDataset(ds, ds)
        assert len(st_tuple) == 3 and len(st_tuple[0]) == 2
        st_dict = data.StackDataset(datasets={"a": ds, "b": ds})
        assert len(st_dict) == 3 and isinstance(st_dict[0], dict)

        sub = data.Subset(ds, [0, 2])
        assert len(sub) == 2 and sub[1][0].item() == 30.0

        srs = data.SubsetRandomSampler([0, 1, 2])
        assert len(list(srs)) == 3 and len(srs) == 3

        dl_custom_bs = DataLoader(ds, batch_size=None, batch_sampler=[[0, 1]])
        assert len(list(dl_custom_bs)) == 1

        class EmptyIterDS(data.IterableDataset[int]):
            """IterableDataset yielding 1, 2."""

            def __iter__(self) -> Iterator[int]:
                """Returns an iterator over `[1, 2]`.

                Returns:
                    Iterator yielding 1 and 2.
                """
                return iter([1, 2])

        dl_iter = DataLoader(EmptyIterDS(), batch_size=None)
        assert list(dl_iter) == [1, 2]

        class NonIterDS(Dataset[tuple[Tensor, ...]]):
            """Unimplemented map-style Dataset."""

        dl_non_iter = DataLoader(NonIterDS(), batch_size=None)
        dl_non_iter.sampler = None
        assert list(dl_non_iter) == []

        assert default_collate([b"a", b"b"]) == [b"a", b"b"]

        Point = namedtuple("Point", ["x", "y"])
        collated_pt = default_collate([Point(1, 2), Point(3, 4)])
        assert isinstance(collated_pt, Point)

        chain_ds = data.ChainDataset([EmptyIterDS(), ds])
        assert len([x for x in chain_ds]) == 5
        with pytest.raises(NotImplementedError):
            len(chain_ds)
        chain_map = data.ChainDataset([ds, ds])
        assert len(chain_map) == 6

        concat_ds: data.ConcatDataset[tuple[Tensor, ...]] = data.ConcatDataset(
            [ds, ds, NonIterDS()]
        )
        assert len(concat_ds) == 6
        assert concat_ds[0][0].item() == 10.0
        assert concat_ds[4][0].item() == 20.0
        assert concat_ds[-1][0].item() == 30.0
        with pytest.raises(ValueError):
            _ = concat_ds[-10]
        with pytest.raises(IndexError):
            _ = concat_ds[10]
        concat_custom = data.ConcatDataset([ds], cumulative_sizes=[3])
        assert len(concat_custom) == 3

        df_pipe = data.DFIterDataPipe(1, a=2)
        assert df_pipe.args == (1,)
        iter_pipe = data.IterDataPipe()
        assert iter_pipe.functions == {}
        map_pipe = data.MapDataPipe()
        assert map_pipe.functions == {}
        data.functional_datapipe("test_pipe")
        data.guaranteed_datapipes_determinism(True)
        data.non_deterministic(
            data.IterDataPipe,
            deterministic_fn=lambda: False,
            arg=data.IterDataPipe,
        )
        data.runtime_validation_disabled(False)

        chunk = data.DataChunk([10, 20, 30])
        assert list(chunk) == [10, 20, 30]
        assert len(chunk) == 3
        assert chunk[1] == 20
        assert list(chunk.raw_iterator()) == [10, 20, 30]

        dist_s1 = data.DistributedSampler(
            ds, num_replicas=2, rank=0, shuffle=True, drop_last=False
        )
        dist_s1.set_epoch(2)
        assert len(list(dist_s1)) == 2 and len(dist_s1) == 2
        dist_s2 = data.DistributedSampler(
            ds, num_replicas=2, rank=1, shuffle=False, drop_last=True
        )
        assert len(list(dist_s2)) == 1 and len(dist_s2) == 1

        wrs_rep = data.WeightedRandomSampler(
            [0.1, 0.9], num_samples=4, replacement=True
        )
        assert len(list(wrs_rep)) == 4 and len(wrs_rep) == 4
        wrs_norep = data.WeightedRandomSampler(
            torch.tensor([0.2, 0.3, 0.5]), num_samples=2, replacement=False
        )
        assert len(list(wrs_norep)) == 2
        wrs_scalar = data.WeightedRandomSampler(
            torch.tensor(1.0), num_samples=1, replacement=True
        )
        assert len(list(wrs_scalar)) == 1
