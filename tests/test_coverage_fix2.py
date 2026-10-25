"""Comprehensive unit tests covering 100% of lines and branches across zero_torch."""

from __future__ import annotations

import math
from collections import namedtuple
from collections.abc import Callable, Iterator

import pytest
from ml_switcheroo_compiler.core.config import EagerMode, config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import (
    Tensor as SwitcherooTensor,
)
from ml_switcheroo_compiler.core.tensor import (
    TensorConfig,
)

import zero_torch as torch
from tests.conftest import assert_allclose, eye_nested, flatten_scalars
from zero_torch import nn, optim
from zero_torch._types import NestedList, ParamGroupDict, Scalar, TensorLike
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
from zero_torch.optim import _scalar_to_float, lr_scheduler
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


class _MockArrayData:
    """Pure-Python mock array implementing `_ArrayDataProtocol` for arbitrary dtype strings."""

    def __init__(self, data: NestedList | Scalar, dtype_str: str) -> None:
        """Initializes `_MockArrayData`.

        Args:
            data: Underlying nested list or scalar payload.
            dtype_str: Data type string returned by `.dtype`.
        """
        self._data = data
        self._dtype_str = dtype_str

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the 1D shape tuple `(1,)`.

        Returns:
            Shape tuple `(1,)`.
        """
        return (1,)

    @property
    def dtype(self) -> str:
        """Returns the configured dtype string.

        Returns:
            Dtype string.
        """
        return self._dtype_str

    def tolist(self) -> NestedList | Scalar:
        """Returns the stored Python list or scalar payload.

        Returns:
            Stored payload.
        """
        return self._data


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

        # Unary floating math ops with analytical expected values
        u_vals = [0.5, 0.75]
        u_in = torch.tensor(u_vals)
        unary_expected: list[tuple[Callable[[TensorLike], Tensor], NestedList]] = [
            (torch.abs, [abs(x) for x in u_vals]),
            (torch.acos, [math.acos(x) for x in u_vals]),
            (torch.asin, [math.asin(x) for x in u_vals]),
            (torch.asinh, [math.asinh(x) for x in u_vals]),
            (torch.atan, [math.atan(x) for x in u_vals]),
            (torch.atanh, [math.atanh(x) for x in u_vals]),
            (torch.ceil, [float(math.ceil(x)) for x in u_vals]),
            (torch.conj, [x for x in u_vals]),
            (torch.cos, [math.cos(x) for x in u_vals]),
            (torch.cosh, [math.cosh(x) for x in u_vals]),
            (torch.deg2rad, [math.radians(x) for x in u_vals]),
            (torch.digamma, [-1.96351003, -1.08586085]),
            (torch.erf, [math.erf(x) for x in u_vals]),
            (torch.erfc, [math.erfc(x) for x in u_vals]),
            (torch.erfinv, [0.47693628, 0.81341985]),
            (torch.exp, [math.exp(x) for x in u_vals]),
            (torch.exp2, [2.0**x for x in u_vals]),
            (torch.expm1, [math.expm1(x) for x in u_vals]),
            (torch.fix, [float(math.trunc(x)) for x in u_vals]),
            (torch.floor, [float(math.floor(x)) for x in u_vals]),
            (torch.isfinite, [True, True]),
            (torch.isinf, [False, False]),
            (torch.isnan, [False, False]),
            (torch.lgamma, [math.lgamma(x) for x in u_vals]),
            (torch.log, [math.log(x) for x in u_vals]),
            (torch.log10, [math.log10(x) for x in u_vals]),
            (torch.log1p, [math.log1p(x) for x in u_vals]),
            (torch.log2, [math.log2(x) for x in u_vals]),
            (torch.negative, [-x for x in u_vals]),
            (torch.positive, [x for x in u_vals]),
            (torch.rad2deg, [math.degrees(x) for x in u_vals]),
            (torch.real, [x for x in u_vals]),
            (torch.reciprocal, [1.0 / x for x in u_vals]),
            (torch.round, [float(round(x)) for x in u_vals]),
            (torch.rsqrt, [1.0 / math.sqrt(x) for x in u_vals]),
            (torch.sign, [1.0, 1.0]),
            (torch.sin, [math.sin(x) for x in u_vals]),
            (
                torch.sinc,
                [math.sin(math.pi * x) / (math.pi * x) for x in u_vals],
            ),
            (torch.sinh, [math.sinh(x) for x in u_vals]),
            (torch.sqrt, [math.sqrt(x) for x in u_vals]),
            (torch.square, [x * x for x in u_vals]),
            (torch.tan, [math.tan(x) for x in u_vals]),
            (torch.tanh, [math.tanh(x) for x in u_vals]),
            (torch.trunc, [float(math.trunc(x)) for x in u_vals]),
        ]
        for unary_fn, exp_u in unary_expected:
            assert_allclose(
                unary_fn(u_in).tolist(),
                exp_u,
                rtol=1e-4,
                atol=1e-4,
            )
        assert_allclose(
            torch.acosh(torch.tensor([1.5, 2.5])).tolist(),
            [math.acosh(1.5), math.acosh(2.5)],
            rtol=1e-4,
        )
        assert_allclose(
            torch.cbrt(torch.tensor([8.0, 27.0])).tolist(),
            [2.0, 3.0],
            rtol=1e-4,
        )
        assert_allclose(torch.imag(u_in).tolist(), [0.0, 0.0])

        # Binary floating math ops with analytical assertions
        v1_vals = [1.0, 2.0, 3.0]
        v2_vals = [2.0, 3.0, 4.0]
        iv2_vals = [1, 2, 3]
        bin_expected: list[
            tuple[Callable[[TensorLike, TensorLike], Tensor], NestedList | Scalar]
        ] = [
            (torch.add, [a + b for a, b in zip(v1_vals, v2_vals)]),
            (torch.atan2, [math.atan2(a, b) for a, b in zip(v1_vals, v2_vals)]),
            (torch.copysign, [math.copysign(a, b) for a, b in zip(v1_vals, v2_vals)]),
            (torch.divide, [a / b for a, b in zip(v1_vals, v2_vals)]),
            (torch.equal, [False, False, False]),
            (torch.float_power, [a**b for a, b in zip(v1_vals, v2_vals)]),
            (torch.floor_divide, [a // b for a, b in zip(v1_vals, v2_vals)]),
            (torch.fmax, [max(a, b) for a, b in zip(v1_vals, v2_vals)]),
            (torch.fmin, [min(a, b) for a, b in zip(v1_vals, v2_vals)]),
            (torch.fmod, [math.fmod(a, b) for a, b in zip(v1_vals, v2_vals)]),
            (torch.greater, [False, False, False]),
            (torch.greater_equal, [False, False, False]),
            (torch.hypot, [math.hypot(a, b) for a, b in zip(v1_vals, v2_vals)]),
            (torch.inner, sum(a * b for a, b in zip(v1_vals, v2_vals))),
            (torch.less, [True, True, True]),
            (torch.less_equal, [True, True, True]),
            (
                torch.logaddexp,
                [math.log(math.exp(a) + math.exp(b)) for a, b in zip(v1_vals, v2_vals)],
            ),
            (
                torch.logaddexp2,
                [math.log2(2.0**a + 2.0**b) for a, b in zip(v1_vals, v2_vals)],
            ),
            (torch.maximum, [max(a, b) for a, b in zip(v1_vals, v2_vals)]),
            (torch.minimum, [min(a, b) for a, b in zip(v1_vals, v2_vals)]),
            (torch.multiply, [a * b for a, b in zip(v1_vals, v2_vals)]),
            (torch.nextafter, [math.nextafter(a, b) for a, b in zip(v1_vals, v2_vals)]),
            (torch.not_equal, [True, True, True]),
            (torch.outer, [[a * b for b in v2_vals] for a in v1_vals]),
            (torch.power, [a**b for a, b in zip(v1_vals, v2_vals)]),
            (torch.remainder, [a % b for a, b in zip(v1_vals, v2_vals)]),
            (torch.subtract, [a - b for a, b in zip(v1_vals, v2_vals)]),
            (torch.true_divide, [a / b for a, b in zip(v1_vals, v2_vals)]),
            (torch.vdot, sum(a * b for a, b in zip(v1_vals, v2_vals))),
        ]
        for bin_fn, exp_b in bin_expected:
            assert_allclose(
                bin_fn(vec, vec2).tolist(),
                exp_b,
                rtol=1e-4,
                atol=1e-4,
            )
        assert_allclose(
            torch.ldexp(vec, int_vec2).tolist(),
            [math.ldexp(a, b) for a, b in zip(v1_vals, iv2_vals)],
            rtol=1e-4,
        )
        assert_allclose(
            torch.heaviside(
                torch.tensor([-1.0, 0.0, 2.0]), torch.tensor([0.5, 0.5, 0.5])
            ).tolist(),
            [0.0, 0.5, 1.0],
        )
        assert_allclose(torch.isclose(vec, vec).tolist(), [True, True, True])
        assert_allclose(torch.mod(int_vec, int_vec2).tolist(), [0, 0, 0])

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
        assert torch.dot(vec, vec2).item() == pytest.approx(20.0)
        assert torch.matmul(mat, mat).shape == (2, 2)
        assert torch.tensordot(mat, mat, axes=1).shape == (2, 2)
        assert torch.einsum("ij,jk->ik", mat, mat).shape == (2, 2)
        assert torch.cholesky(mat).shape == (2, 2)
        assert torch.det(mat).item() == pytest.approx(5.0)
        assert torch.diag(mat).tolist() == [2.0, 3.0]
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
        assert torch.stack([vec, vec2], 1).shape == (3, 2)
        assert torch.swapaxes(mat, 0, 1).shape == (2, 2)
        assert torch.take(vec, torch.tensor([0, 2])).tolist() == [1.0, 3.0]
        assert torch.tile(vec, (2,)).shape == (6,)
        assert torch.tril(mat).shape == (2, 2)
        assert torch.triu(mat).shape == (2, 2)
        assert torch.unsqueeze(vec, 0).shape == (1, 3)
        assert torch.unsqueeze(vec, -1).shape == (3, 1)
        assert len(torch.unstack(mat)) == 2
        assert torch.where(bool_vec, vec, vec2).tolist() == [1.0, 3.0, 3.0]
        assert torch.ones((2, 2)).shape == (2, 2)
        assert torch.ones_like(mat).shape == (2, 2)
        assert torch.zeros((2, 2)).shape == (2, 2)
        assert torch.zeros_like(mat).shape == (2, 2)
        assert torch.full((2, 2), 3.0).shape == (2, 2)
        assert torch.full_like(mat, 3.0).shape == (2, 2)
        assert torch.empty((2, 2)).shape == (2, 2)
        assert torch.eye(2).tolist() == [[1.0, 0.0], [0.0, 1.0]]
        assert torch.identity(3).shape == (3, 3)
        assert torch.arange(4).tolist() == [0, 1, 2, 3]
        assert torch.linspace(0.0, 1.0, 4).shape == (4,)

        assert torch.bitcast(vec, torch.int32).shape == (3,)
        assert torch.bitcast(vec, dtype=torch.int32).shape == (3,)
        assert torch.cast(vec, torch.float32).shape == (3,)
        assert torch.cast(vec, dtype=torch.float32).shape == (3,)

        assert torch.concatenate(tensors=[vec, vec], dim=0).shape == (6,)
        assert torch.concatenate([vec, vec], 0).shape == (6,)

        idx = torch.tensor([0, 1])
        assert torch.gather(vec, 0, idx).tolist() == [1.0, 2.0]
        assert torch.gather(vec, dim=0, index=[0, 1]).tolist() == [1.0, 2.0]
        assert torch.gather(vec, index=idx).tolist() == [1.0, 2.0]
        assert torch.gather(vec, 0, index=idx).tolist() == [1.0, 2.0]

        assert torch.norm(vec, p=2, dim=0, keepdim=True).shape == (1,)
        assert torch.norm(vec, p=3).shape == ()

        assert torch.reshape(mat, newshape=(4,)).shape == (4,)
        assert torch.transpose(mat, 0, 1).shape == (2, 2)
        assert torch.transpose(mat, dim0=0, dim1=1).shape == (2, 2)
        assert torch.transpose(mat, dim0="bad", dim1="bad").shape == (2, 2)
        assert torch.transpose(mat, axes=(1, 0)).shape == (2, 2)

        assert torch.logit(torch.tensor([0.2, 0.8]), eps=1e-4).shape == (2,)
        assert torch.logit(torch.tensor([0.2, 0.8])).shape == (2,)
        assert torch.signbit(torch.tensor([-1.0, 1.0])).tolist() == [True, False]
        assert torch.xlogy(
            torch.tensor([0.0, 2.0]), torch.tensor([1.0, 3.0])
        ).shape == (2,)
        assert torch.mvlgamma(torch.tensor([2.0, 3.0]), p=2).shape == (2,)

        assert torch.nan_to_num(
            torch.tensor([float("nan"), float("inf"), float("-inf")]),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        ).tolist() == [0.0, 10.0, -10.0]
        assert torch.nan_to_num(torch.tensor([1.0]), nan=None).tolist() == [1.0]

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

        assert torch.clip(vec, min=1.5, max=2.5).tolist() == [1.5, 2.0, 2.5]
        assert torch.clip(vec).tolist() == [1.0, 2.0, 3.0]
        assert torch.clamp(vec, min=1.5, max=2.5).tolist() == [1.5, 2.0, 2.5]
        assert torch.sigmoid(vec).shape == (3,)
        assert torch.softmax(vec, dim=0).shape == (3,)

    # Real TracingMode execution for low-level indexing/slicing ops
    _tracer.start_tracing()
    try:
        vt = torch.ones((4,))
        it = torch.tensor([0, 1])
        assert isinstance(torch.gather(vt, 0), Tensor)
        assert torch.dynamic_slice(vt, (0,), (2,)).shape == (2,)
        assert isinstance(torch.gather_nd(vt, it), Tensor)
        assert isinstance(torch.scatter_add(vt, it, it), Tensor)
        assert isinstance(torch.scatter_nd(it, it, (4,)), Tensor)
        assert isinstance(torch.slice(vt, (0,), (2,)), Tensor)
        assert isinstance(torch.strided_slice(vt, (0,), (2,), (1,)), Tensor)
        assert isinstance(torch.update_slice(vt, it, (0,)), Tensor)
    finally:
        _tracer.stop_tracing()


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
        assert torch.tensor(None).numel() == 0
        assert Tensor(None).numel() == 0
        assert torch.tensor(5.0).numel() == 1
        assert torch.zeros((0, 3)).numel() == 0

        t_no_grad = torch.tensor([1.0], requires_grad=False)
        with pytest.raises(RuntimeError):
            t_no_grad.retain_grad()
        with pytest.raises(RuntimeError):
            t_no_grad.backward()
        with pytest.raises(RuntimeError):
            t_grad.backward()

        t_none = Tensor(None)
        assert t_none.shape == (0,)
        assert t_none.view(0).shape == (0,)
        assert t_none.reshape(0).shape == (0,)
        assert t_none.detach().shape == (0,)
        assert t_none.tolist() == []
        assert t_none.unsqueeze(0).shape == (1, 0)
        assert t_none.T.shape == (0,)
        assert t_none.numpy().shape == (0,)

        t_mat = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert t_mat.reshape([4]).shape == (4,)
        assert t_mat.detach().shape == (2, 2)
        assert t_mat.unsqueeze(0).shape == (1, 2, 2)
        assert t_mat.transpose(0, 1).shape == (2, 2)
        assert len(t_mat.chunk(2, dim=0)) == 2

        t_sp = torch.tensor([0.5, 0.75])
        assert t_sp.digamma().shape == (2,)
        assert t_sp.erfc().shape == (2,)
        assert t_sp.erfinv().shape == (2,)
        assert t_sp.lgamma().shape == (2,)

        t_cmp1 = torch.tensor([1.0, 2.0])
        t_cmp2 = torch.tensor([2.0, 2.0])
        assert (t_cmp1 < t_cmp2).tolist() == [True, False]
        assert (t_cmp1 > t_cmp2).tolist() == [False, False]
        assert (t_cmp1 <= t_cmp2).tolist() == [True, True]
        assert (t_cmp1 >= t_cmp2).tolist() == [False, True]
        assert (t_cmp1 == t_cmp2).tolist() == [False, True]
        assert (t_cmp1 != t_cmp2).tolist() == [True, False]
        none_val: None = None
        assert not bool(t_cmp1 == none_val)
        assert bool(t_cmp1 != none_val)
        assert not bool(t_cmp1 == "abc")
        assert bool(t_cmp1 != "abc")
        assert bool(t_cmp1) is True
        assert bool(t_none) is False

        assert t_mat[0].tolist() == [1.0, 2.0]
        assert t_mat[0:1].tolist() == [[1.0, 2.0]]
        assert t_mat[(0, 1)].item() == 2.0
        assert t_mat[torch.tensor([0], dtype=torch.int32)].shape == (1, 2)
        assert t_mat[(torch.tensor([0], dtype=torch.int32), 1)].shape == (1,)
        t_copy_dst = torch.zeros((2, 2))
        t_copy_dst.copy_(1.0)
        assert t_copy_dst.tolist() == [[1.0, 1.0], [1.0, 1.0]]
        t_copy_dst.copy_(t_mat)
        assert t_copy_dst.tolist() == [[1.0, 2.0], [3.0, 4.0]]
        with pytest.raises(IndexError):
            _ = torch.tensor(5.0)[0]
        st_list_data = SwitcherooTensor(
            data=[10.0, 20.0],
            config=TensorConfig(shape=(2,), dtype=DType.Float32, device="cpu"),
        )
        t_list_data = Tensor(st_list_data)
        assert t_list_data[0].item() == 10.0
        assert t_list_data[(0,)] is t_list_data

        t_np = torch.tensor([1.0, 2.0])
        assert t_np.data is not None
        assert isinstance(_wrap([t_np._tensor]), Tensor)
        assert isinstance(_wrap([]), Tensor)
        assert isinstance(_wrap_list([t_np._tensor]), list)
        assert isinstance(_wrap_list(t_np._tensor), list)
        assert isinstance(_wrap_tuple((t_np._tensor,)), tuple)
        assert isinstance(_wrap_tuple(t_np._tensor), tuple)

        # Dtypes and _to_tensor edge cases
        assert Tensor(_MockArrayData([1.0], "float64")).dtype == DType.Float64
        assert Tensor(_MockArrayData([1.0], "float16")).dtype == DType.float16
        assert Tensor([1.0], dtype="bfloat16").dtype == DType.bfloat16
        assert Tensor(_MockArrayData([1], "int32")).dtype == DType.Int32
        assert Tensor(_MockArrayData([1], "int64")).dtype == DType.Int64
        assert Tensor(_MockArrayData([1], "int16")).dtype == DType.int16
        assert Tensor(_MockArrayData([1], "int8")).dtype == DType.int8
        assert Tensor(_MockArrayData([True], "bool")).dtype == DType.Bool
        assert Tensor(_MockArrayData([1], "uint8")).dtype == DType.uint8
        assert Tensor(_MockArrayData([1], "uint16")).dtype == DType.uint16
        assert Tensor(_MockArrayData([1], "uint32")).dtype == DType.uint32
        assert Tensor(_MockArrayData([1], "uint64")).dtype == DType.uint64
        assert Tensor(_MockArrayData([0.0], "S1")).dtype == DType.Float32
        assert (
            Tensor(_MockArrayData([1.0 + 0.0j], "complex64")).dtype == DType.complex64
        )
        assert (
            Tensor(_MockArrayData([1.0 + 0.0j], "complex128")).dtype == DType.complex128
        )
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

        # Ragged nested list triggers _ops.array fallback naturally without mock
        t_fallback = Tensor([[1.0, 2.0], [3.0]])
        assert t_fallback.shape == (2, 2)

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
        sol_ex, info_ex = torch.linalg.solve_ex(mat, mat)
        assert_allclose(sol_ex.tolist(), [[1.0, 0.0], [0.0, 1.0]], atol=1e-4)
        assert info_ex.item() == 0
        lstsq_sol, _, _, _ = torch.linalg.lstsq(mat, mat)
        assert_allclose(lstsq_sol.tolist(), [[1.0, 0.0], [0.0, 1.0]], atol=1e-4)
        assert torch.slogdet(mat)[0].item() == pytest.approx(1.0)
        assert torch.slogdet(1.0)[0].item() == pytest.approx(1.0)

        bern = torch.distributions.Bernoulli(probs=torch.tensor([0.25, 0.75]))
        assert_allclose(bern.mean.tolist(), [0.25, 0.75])
        assert_allclose(bern.variance.tolist(), [0.1875, 0.1875])
        assert bern.entropy().shape == (2,)
        with pytest.raises(NotImplementedError):
            bern.rsample()

        beta = torch.distributions.Beta(
            torch.tensor([2.0, 3.0]), torch.tensor([2.0, 5.0])
        )
        assert_allclose(beta.mean.tolist(), [0.5, 0.375])
        assert beta.variance.shape == (2,)

        cat = torch.distributions.Categorical(probs=torch.tensor([0.2, 0.3, 0.5]))
        assert cat.mean.shape == ()
        assert cat.variance.shape == ()
        assert cat.entropy().shape == ()

        gam = torch.distributions.Gamma(
            torch.tensor([2.0, 4.0]), torch.tensor([1.0, 2.0])
        )
        assert_allclose(gam.mean.tolist(), [2.0, 2.0])
        assert_allclose(gam.variance.tolist(), [2.0, 1.0])

        norm = torch.distributions.Normal(
            torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0])
        )
        assert_allclose(norm.mean.tolist(), [0.0, 1.0])
        assert_allclose(norm.variance.tolist(), [1.0, 4.0])
        assert norm.entropy().shape == (2,)
        assert_allclose(
            norm.cdf(torch.tensor([0.0, 1.0])).tolist(), [0.5, 0.5], atol=1e-4
        )
        assert_allclose(
            norm.icdf(torch.tensor([0.5, 0.5])).tolist(), [0.0, 1.0], atol=1e-4
        )

        pois = torch.distributions.Poisson(torch.tensor([0.0, 3.0, 35.0]))
        assert_allclose(pois.mean.tolist(), [0.0, 3.0, 35.0])
        assert_allclose(pois.variance.tolist(), [0.0, 3.0, 35.0])
        assert pois.sample().shape == (3,)

        unif = torch.distributions.Uniform(
            torch.tensor([0.0, 1.0]), torch.tensor([2.0, 5.0])
        )
        assert_allclose(unif.mean.tolist(), [1.0, 3.0])
        assert unif.variance.shape == (2,)
        assert unif.entropy().shape == (2,)
        assert_allclose(unif.cdf(torch.tensor([1.0, 3.0])).tolist(), [0.5, 0.5])
        assert_allclose(unif.icdf(torch.tensor([0.5, 0.5])).tolist(), [1.0, 3.0])


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
        assert nn.GLU(dim=-1)(v1).shape == (2, 1)
        assert nn.Softmax2d()(x2d).shape == (2, 4, 4, 4)

        assert F.hard_sigmoid(v1).shape == (2, 2)
        assert F.soft_shrink(v1).shape == (2, 2)
        assert F.softmax2d(x2d).shape == (2, 4, 4, 4)
        assert F.softmin(v1, dim=-1).shape == (2, 2)
        assert F.softplus(1.0).shape == ()
        assert F.tanhshrink(v1).shape == (2, 2)
        assert F.threshold(v1, 0.5, 0.0).shape == (2, 2)
        assert F.layer_norm(v1, (2,)).shape == (2, 2)
        assert (
            F.multi_label_margin_loss(v1, torch.tensor([[0, -1], [1, -1]])).shape == ()
        )
        assert F.multi_label_soft_margin_loss(v1, v2).shape == ()

        # Convolutions, Fold, Unfold
        assert nn.Conv1d(4, 2, kernel_size=2, padding=(1,))(x1d).shape == (2, 2, 7)
        assert nn.Conv1d(4, 2, kernel_size=3, padding="same")(x1d).shape == (
            2,
            4 if False else 2,
            6,
        )
        assert nn.Conv1d(4, 2, kernel_size=2, padding="valid", bias=False)(
            x1d
        ).shape == (2, 2, 5)
        assert nn.Conv2d(4, 2, kernel_size=2, padding=(1, 1))(x2d).shape == (
            2,
            2,
            5,
            5,
        )
        assert nn.Conv2d(4, 2, kernel_size=2, padding="valid", bias=False)(
            x2d
        ).shape == (2, 2, 3, 3)
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
        ).shape == (1, 2, 2, 2, 2)
        assert nn.ConvTranspose1d(4, 2, kernel_size=2)(x1d).shape == (2, 2, 7)
        assert nn.ConvTranspose1d(4, 2, kernel_size=2, bias=False)(x1d).shape == (
            2,
            2,
            7,
        )
        assert nn.ConvTranspose2d(4, 2, kernel_size=2, padding=1)(x2d).shape == (
            2,
            2,
            3,
            3,
        )
        assert nn.ConvTranspose2d(4, 2, kernel_size=2)(x2d).shape == (2, 2, 5, 5)
        assert nn.ConvTranspose3d(4, 2, kernel_size=2)(x3d).shape == (1, 2, 4, 4, 4)

        # Unfold and Fold execution against analytical values
        uf = nn.Unfold(kernel_size=2, padding=1, stride=2)
        uf_out = uf(x2d)
        assert uf_out.shape == (2, 16, 9)

        fd = nn.Fold(output_size=(4, 4), kernel_size=2, padding=1, stride=2)
        fd_out = fd(uf_out)
        assert fd_out.shape == (2, 4, 4, 4)
        assert_allclose(fd_out.tolist(), 1.0, rtol=1e-4)

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
            assert fd_fn(
                torch.ones((2, 2)), p=0.5, training=True, inplace=True
            ).shape == (2, 2)
            assert fd_fn(v1, p=1.0, training=True, inplace=False).shape == (2, 2)
            assert fd_fn(torch.ones((2,)), p=0.5, training=True).shape == (2,)
            assert fd_fn(x1d, p=0.5, training=True).shape == (2, 4, 6)
            assert fd_fn(
                torch.ones((2,)), p=1.0, training=True, inplace=True
            ).shape == (2,)

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

        # Padding modules (1D, 2D, 3D with constant, reflect, replicate, circular)
        for pad1 in [
            nn.CircularPad1d(1),
            nn.ConstantPad1d((1, 1), 2.0),
            nn.ReflectionPad1d(1),
            nn.ReplicationPad1d(1),
            nn.ZeroPad1d(1),
        ]:
            assert pad1(x1d).shape == (2, 4, 8)
        for pad2 in [
            nn.CircularPad2d(1),
            nn.ConstantPad2d(1, 0.0),
            nn.ReflectionPad2d(1),
            nn.ReplicationPad2d(1),
            nn.ZeroPad2d(1),
        ]:
            assert pad2(x2d).shape == (2, 4, 6, 6)
        for pad3 in [
            nn.CircularPad3d(1),
            nn.ConstantPad3d(1, 0.0),
            nn.ReflectionPad3d(1),
            nn.ReplicationPad3d(1),
            nn.ZeroPad3d(1),
        ]:
            assert pad3(x3d).shape == (1, 4, 5, 5, 5)
        assert F.pad(torch.ones((1, 1, 1)), (1, 1), mode="reflect").shape == (1, 1, 3)
        assert F.pad(torch.ones((1, 2)), (0, 0), mode="constant").shape == (1, 2)
        assert F.pad(torch.ones((1, 2)), 1, mode="constant", value=3.0).shape == (1, 4)
        assert F.pad(torch.ones((1, 2)), "invalid").shape == (1, 2)
        with pytest.raises(ValueError):
            F.pad(x1d, (1, 1), mode="invalid_mode")

        # Pooling & functional_pooling with exact output shapes and values
        assert FP._format_padding((1, 2), 2) == ((1, 1), (2, 2))
        assert FP._format_padding(1, 2) == ((1, 1), (1, 1))
        assert FP._resolve_opt_size(None, 2, (4, 4)) == (4, 4)
        assert FP._resolve_opt_size((None, 2), 2, (4, 4)) == (4, 2)

        assert nn.AdaptiveAvgPool1d(3)(x1d).shape == (2, 4, 3)
        assert nn.AdaptiveAvgPool1d(6)(x1d).shape == (2, 4, 6)
        assert nn.AdaptiveAvgPool2d(2)(x2d).shape == (2, 4, 2, 2)
        assert_allclose(nn.AdaptiveAvgPool2d(2)(x2d).tolist(), 1.0)
        assert nn.AdaptiveAvgPool3d(2)(x3d).shape == (1, 4, 2, 2, 2)

        am1_no_idx = nn.AdaptiveMaxPool1d(3)(x1d)
        assert isinstance(am1_no_idx, Tensor) and am1_no_idx.shape == (2, 4, 3)
        am1_res = FP.adaptive_max_pool1d(x1d, 3, return_indices=True)
        assert isinstance(am1_res, tuple) and am1_res[0].shape == (2, 4, 3)
        am2_no_idx = nn.AdaptiveMaxPool2d(2)(x2d)
        assert isinstance(am2_no_idx, Tensor) and am2_no_idx.shape == (2, 4, 2, 2)
        am2_res = nn.AdaptiveMaxPool2d(2, return_indices=True)(x2d)
        assert isinstance(am2_res, tuple) and am2_res[0].shape == (2, 4, 2, 2)
        am3_no_idx = nn.AdaptiveMaxPool3d(2)(x3d)
        assert isinstance(am3_no_idx, Tensor) and am3_no_idx.shape == (1, 4, 2, 2, 2)
        am3_res = FP.adaptive_max_pool3d(x3d, 2, return_indices=True)
        assert isinstance(am3_res, tuple) and am3_res[0].shape == (1, 4, 2, 2, 2)

        assert nn.AvgPool1d(2)(x1d).shape == (2, 4, 3)
        assert nn.AvgPool1d(2, ceil_mode=True, count_include_pad=False)(
            torch.ones((2, 4, 5))
        ).shape == (2, 4, 3)
        assert FP.avg_pool1d(x1d, 2, stride=2, padding=1, ceil_mode=True).shape == (
            2,
            4,
            4,
        )
        assert FP.avg_pool2d(x2d, 2, divisor_override=2).shape == (2, 4, 2, 2)
        assert_allclose(FP.avg_pool2d(x2d, 2, divisor_override=2).tolist(), 2.0)
        assert nn.AvgPool2d(2)(x2d).shape == (2, 4, 2, 2)
        assert nn.AvgPool3d(2)(x3d).shape == (1, 4, 1, 1, 1)

        mp1_res = FP.max_pool1d(x1d, 2, return_indices=True)
        assert isinstance(mp1_res, tuple)
        mp1_out, mp1_idx = mp1_res
        assert mp1_out.shape == (2, 4, 3) and mp1_idx.shape == (2, 4, 3)
        assert nn.MaxUnpool1d(2)(mp1_out, mp1_idx).shape == (2, 4, 6)
        assert nn.MaxUnpool1d(2)(mp1_out, mp1_idx, output_size=(6,)).shape == (2, 4, 6)
        assert nn.MaxUnpool1d(2)(mp1_out, mp1_idx, output_size=(2, 4, 6)).shape == (
            2,
            4,
            6,
        )
        with pytest.raises(ValueError):
            nn.MaxUnpool1d(2)(mp1_out, mp1_idx, output_size=(1, 2, 3, 4))

        mp2_res = FP.max_pool2d(
            torch.arange(0, 16, dtype=torch.float32).reshape(1, 1, 4, 4),
            2,
            return_indices=True,
        )
        assert isinstance(mp2_res, tuple)
        mp2_out, mp2_idx = mp2_res
        assert mp2_out.tolist() == [[[[5.0, 7.0], [13.0, 15.0]]]]
        assert mp2_idx.tolist() == [[[[5, 7], [13, 15]]]]
        unp2 = nn.MaxUnpool2d(2)(mp2_out, mp2_idx)
        assert unp2.shape == (1, 1, 4, 4) and unp2[0, 0, 1, 1].item() == 5.0

        mp3_res = FP.max_pool3d(x3d, 2, return_indices=True)
        assert isinstance(mp3_res, tuple)
        mp3_out, mp3_idx = mp3_res
        assert mp3_out.shape == (1, 4, 1, 1, 1)
        assert nn.MaxUnpool3d(2)(mp3_out, mp3_idx).shape == (1, 4, 2, 2, 2)

        assert isinstance(nn.MaxPool1d(2)(x1d), Tensor)
        assert isinstance(nn.MaxPool2d(2)(x2d), Tensor)
        assert isinstance(nn.MaxPool3d(2)(x3d), Tensor)

        fmp2_no_idx = nn.FractionalMaxPool2d(2, output_size=2)(x2d)
        assert isinstance(fmp2_no_idx, Tensor) and fmp2_no_idx.shape == (2, 4, 2, 2)
        fmp2_r = nn.FractionalMaxPool2d(
            2,
            output_ratio=0.5,
            return_indices=True,
            _random_samples=torch.full((2, 4, 2), 0.5),
        )(x2d)
        assert isinstance(fmp2_r, tuple) and fmp2_r[0].shape == (2, 4, 2, 2)
        fmp3_no_idx = nn.FractionalMaxPool3d(2, output_size=2)(x3d)
        assert isinstance(fmp3_no_idx, Tensor) and fmp3_no_idx.shape == (1, 4, 2, 2, 2)
        fmp3_short = FP.fractional_max_pool3d(torch.ones((2,)), 2, output_size=1)
        assert isinstance(fmp3_short, Tensor) and fmp3_short.shape == (2,)
        with pytest.raises(ValueError):
            FP._fractional_max_pool_nd(x2d, 2, None, None, False, None, 2)

        assert nn.LPPool1d(1.0, 2)(x1d).shape == (2, 4, 3)
        assert nn.LPPool1d(2.0, 2, stride=2)(x1d).shape == (2, 4, 3)
        assert nn.LPPool2d(1.0, 2)(x2d).shape == (2, 4, 2, 2)
        assert nn.LPPool2d(2.0, 2, stride=2)(x2d).shape == (2, 4, 2, 2)
        assert nn.LPPool3d(1.0, 2)(x3d).shape == (1, 4, 1, 1, 1)
        assert nn.LPPool3d(2.0, 2, stride=2)(x3d).shape == (1, 4, 1, 1, 1)
        assert F.avg_pool1d(x1d, 2).shape == (2, 4, 3)
        assert F.avg_pool2d(x2d, 2).shape == (2, 4, 2, 2)
        assert F.avg_pool3d(x3d, 2).shape == (1, 4, 1, 1, 1)
        assert isinstance(F.max_pool1d(x1d, 2), Tensor)
        assert isinstance(F.max_pool2d(x2d, 2), Tensor)
        assert isinstance(F.max_pool3d(x3d, 2), Tensor)

        # Losses with exact value and reduction verification
        assert nn.loss._Loss().reduction == "mean"
        assert_allclose(
            nn.L1Loss(reduction="none")(v1, v2).tolist(), [[1.0, 1.0], [3.0, 0.0]]
        )
        assert nn.L1Loss(reduction="mean")(v1, v2).item() == pytest.approx(1.25)
        assert nn.L1Loss(reduction="sum")(v1, v2).item() == pytest.approx(5.0)
        assert_allclose(
            nn.MSELoss(reduction="none")(v1, v2).tolist(), [[1.0, 1.0], [9.0, 0.0]]
        )
        assert nn.MSELoss(reduction="mean")(v1, v2).item() == pytest.approx(2.75)
        assert nn.MSELoss(reduction="sum")(v1, v2).item() == pytest.approx(11.0)

        for reduction in ("none", "mean", "sum"):
            expected_1d = (2,) if reduction == "none" else ()
            expected_2d = (2, 2) if reduction == "none" else ()
            assert nn.SmoothL1Loss(reduction=reduction)(v1, v2).shape == expected_2d
            assert nn.HuberLoss(reduction=reduction)(v1, v2).shape == expected_2d
            assert (
                nn.KLDivLoss(reduction=reduction, log_target=False)(
                    v1, torch.ones((2, 2))
                ).shape
                == expected_2d
            )
            assert (
                nn.KLDivLoss(reduction=reduction, log_target=True)(v1, v2).shape
                == expected_2d
            )
            assert (
                nn.BCELoss(reduction=reduction)(
                    torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
                    torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                ).shape
                == expected_2d
            )
            assert (
                nn.BCEWithLogitsLoss(reduction=reduction)(v1, v2).shape == expected_2d
            )
            assert (
                nn.CosineEmbeddingLoss(reduction=reduction)(v1, v2, y_bin).shape
                == expected_1d
            )
            assert (
                nn.HingeEmbeddingLoss(reduction=reduction)(v1, v2).shape == expected_2d
            )
            assert (
                nn.MarginRankingLoss(reduction=reduction)(v1, v2, v1).shape
                == expected_2d
            )
            assert (
                nn.MultiLabelMarginLoss(reduction=reduction)(
                    v1, torch.tensor([[0, -1], [1, 0]])
                ).shape
                == expected_1d
            )
            assert (
                nn.MultiLabelSoftMarginLoss(reduction=reduction)(v1, v2).shape
                == expected_1d
            )
            assert (
                nn.MultiMarginLoss(reduction=reduction)(v1, torch.tensor([0, 1])).shape
                == expected_1d
            )
            assert (
                nn.PoissonNLLLoss(reduction=reduction, full=True)(v1, v2).shape
                == expected_2d
            )
            assert nn.SoftMarginLoss(reduction=reduction)(v1, v2).shape == expected_2d
            assert (
                nn.TripletMarginLoss(reduction=reduction, swap=True)(v1, v2, v1).shape
                == expected_1d
            )
            assert (
                nn.TripletMarginWithDistanceLoss(reduction=reduction, swap=True)(
                    v1, v2, v1
                ).shape
                == expected_1d
            )
            assert (
                nn.GaussianNLLLoss(reduction=reduction)(
                    v1, v2, torch.ones((2, 2))
                ).shape
                == expected_2d
            )
            assert nn.CTCLoss(reduction=reduction, zero_infinity=True)(
                torch.zeros((3, 2, 3)) - 1.0,
                torch.tensor([[1, 2], [2, 1]]),
                torch.tensor([3, 3]),
                torch.tensor([2, 2]),
            ).shape == (expected_1d if reduction == "none" else ())
            assert (
                nn.NLLLoss(reduction=reduction)(v1, torch.tensor([0, 1])).shape
                == expected_1d
            )
            assert (
                nn.NLLLoss2d(reduction=reduction)(v1, torch.tensor([0, 1])).shape
                == expected_1d
            )
            assert (
                nn.CrossEntropyLoss(reduction=reduction)(v1, torch.tensor([0, 1])).shape
                == expected_1d
            )
        assert nn.KLDivLoss(reduction="batchmean")(v1, torch.ones((2, 2))).shape == ()
        assert (
            F.ctc_loss(
                torch.zeros((3, 3)) - 1.0,
                torch.tensor([1, 2]),
                [3],
                [2],
                reduction="sum",
            ).shape
            == ()
        )
        assert (
            F.nll_loss(
                torch.ones((2, 3, 2)), torch.zeros((2, 2), dtype=DType.Int64)
            ).shape
            == ()
        )
        assert F.nll_loss(torch.tensor(1.0), torch.tensor(0)).shape == ()
        assert (
            F.cross_entropy(
                torch.ones((2, 3, 2)),
                torch.ones((2, 3, 2)) / 3.0,
                weight=torch.ones((3,)),
                label_smoothing=0.1,
            ).shape
            == ()
        )
        assert F.cross_entropy(
            torch.ones((2, 3, 2)),
            torch.zeros((2, 2), dtype=DType.Int64),
            weight=torch.ones((3,)),
            label_smoothing=0.1,
            reduction="none",
        ).shape == (2, 2)
        assert (
            F.cross_entropy(
                v1,
                torch.tensor([0, 1]),
                weight=torch.ones((2,)),
                label_smoothing=0.1,
                reduction="sum",
            ).shape
            == ()
        )

        als = nn.AdaptiveLogSoftmaxWithLoss(4, 10, cutoffs=[2, 5])
        out_als, loss_als = als(torch.ones((2, 4)), torch.tensor([0, 1]))
        assert out_als.shape == (2,) and loss_als.shape == ()
        loss_als_val = loss_als.item()
        assert isinstance(loss_als_val, float) and loss_als_val > 0.0

        lce = nn.LinearCrossEntropyLoss(2, 3)
        assert lce(v1, torch.tensor([0, 1])).shape == ()

        # nn.init with statistical and structural property assertions
        assert nn.init.calculate_gain("linear") == 1.0
        assert nn.init.calculate_gain("tanh") == pytest.approx(5.0 / 3.0)
        assert nn.init.calculate_gain("relu") == pytest.approx(2.0**0.5)
        assert nn.init.calculate_gain("leaky_relu", 0.2) > 1.0
        assert nn.init.calculate_gain("selu") == 0.75
        with pytest.raises(ValueError):
            nn.init.calculate_gain("unknown_nonlinearity")

        t_init = torch.ones((3, 3))
        nn.init.constant_(t_init, 5.0)
        assert_allclose(t_init.tolist(), 5.0)
        assert nn.init.constant(t_init, 3.0).shape == (3, 3)
        assert nn.init.constant(tensor=t_init, val=2.0).shape == (3, 3)
        assert nn.init.constant().shape == ()
        d_1x1 = nn.init.dirac_(torch.zeros((2, 2, 1)))
        assert_allclose(d_1x1.squeeze(-1).tolist(), [[1.0, 0.0], [0.0, 1.0]])
        assert nn.init.dirac_(torch.ones((2, 2, 3, 3))).shape == (2, 2, 3, 3)
        assert nn.init.dirac_(torch.ones((2, 2, 3, 3, 3))).shape == (2, 2, 3, 3, 3)
        assert nn.init.dirac(torch.ones((2, 2, 3))).shape == (2, 2, 3)
        assert nn.init.dirac().shape == ()
        with pytest.raises(ValueError):
            nn.init.dirac_(torch.ones((2, 2)))
        with pytest.raises(ValueError):
            nn.init.dirac_(torch.ones((3, 2, 3)), groups=2)
        nn.init.eye_(t_init)
        assert_allclose(t_init.tolist(), eye_nested(3))
        assert nn.init.eye(t_init).shape == (3, 3)
        assert nn.init.eye().shape == ()
        with pytest.raises(ValueError):
            nn.init.eye_(torch.ones((2, 2, 2)))
        assert nn.init.normal(t_init, 0.0, 1.0).shape == (3, 3)
        assert nn.init.normal().shape == ()
        assert nn.init.uniform(t_init, -1.0, 1.0).shape == (3, 3)
        assert nn.init.uniform().shape == ()
        nn.init.ones_(t_init)
        assert_allclose(t_init.tolist(), 1.0)
        nn.init.zeros_(t_init)
        assert_allclose(t_init.tolist(), 0.0)
        tn_out = nn.init.trunc_normal_(torch.zeros((4, 4)), a=-0.5, b=0.5)
        tn_vals = flatten_scalars(tn_out.tolist())
        assert all(
            isinstance(v, (int, float)) and -0.5 <= float(v) <= 0.5 for v in tn_vals
        )
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
        q_tall = nn.init.orthogonal_(torch.ones((3, 2)))
        assert_allclose((q_tall.T @ q_tall).tolist(), eye_nested(2), atol=1e-4)
        q_wide = nn.init.orthogonal_(torch.ones((2, 3)))
        assert_allclose((q_wide @ q_wide.T).tolist(), eye_nested(2), atol=1e-4)
        q_sq = nn.init.orthogonal(t_init)
        assert_allclose((q_sq.T @ q_sq).tolist(), eye_nested(3), atol=1e-4)
        assert nn.init.orthogonal().shape == ()
        with pytest.raises(ValueError):
            nn.init.orthogonal_(torch.ones((3,)))
        sp_mat = nn.init.sparse_(torch.ones((4, 3)), sparsity=0.5)
        sp_rows = sp_mat.tolist()
        assert isinstance(sp_rows, list)
        col_zeros = [
            sum(1 for row in sp_rows if isinstance(row, list) and row[col] == 0.0)
            for col in range(3)
        ]
        assert col_zeros == [2, 2, 2]
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
        assert FN.batch_norm(torch.tensor([1.0, 3.0])).shape == (2,)
        assert FN.batch_norm([1.0, 3.0]).shape == (2,)
        assert FN.layer_norm([1.0, 3.0], (2,)).shape == (2,)
        assert FN.rms_norm([1.0, 3.0], (2,)).shape == (2,)
        assert FN.instance_norm([1.0, 3.0]).shape == (2,)
        assert FN.instance_norm(
            v1, None, None, torch.ones((2,)), torch.zeros((2,))
        ).shape == (2, 2)
        assert FN.instance_norm(v1).shape == (2, 2)

        bn_eval = nn.BatchNorm2d(4)
        _ = bn_eval(x2d)
        bn_eval.eval()
        assert bn_eval(x2d).shape == (2, 4, 4, 4)

        in_eval = nn.InstanceNorm1d(4, affine=True, track_running_stats=True)
        _ = in_eval(x1d)
        in_eval.eval()
        assert in_eval(x1d).shape == (2, 4, 6)

        assert _apply_channel_affine(
            torch.ones((2,)),
            nn.Parameter(torch.ones((2,))),
            nn.Parameter(torch.zeros((2,))),
        ).shape == (2,)
        assert _apply_channel_affine(torch.ones((2,)), None, None).shape == (2,)
        assert nn.BatchNorm1d(4, affine=False, track_running_stats=False)(
            x1d
        ).shape == (2, 4, 6)
        assert nn.BatchNorm3d(4)(x3d).shape == (1, 4, 3, 3, 3)
        assert nn.SyncBatchNorm(4)(x2d).shape == (2, 4, 4, 4)
        assert nn.CrossMapLRN2d(2)(x2d).shape == (2, 4, 4, 4)
        assert nn.LocalResponseNorm(2)(x2d).shape == (2, 4, 4, 4)
        assert nn.GroupNorm(2, 4, affine=False)(x2d).shape == (2, 4, 4, 4)
        assert nn.GroupNorm(2, 4, affine=True)(x2d).shape == (2, 4, 4, 4)
        assert F.group_norm(x2d, 2, torch.ones((4,)), torch.zeros((4,))).shape == (
            2,
            4,
            4,
            4,
        )
        assert nn.InstanceNorm1d(4, affine=True, track_running_stats=True)(
            x1d
        ).shape == (2, 4, 6)
        assert nn.InstanceNorm2d(4)(x2d).shape == (2, 4, 4, 4)
        assert nn.InstanceNorm3d(4)(x3d).shape == (1, 4, 3, 3, 3)
        assert nn.LayerNorm(2, elementwise_affine=True, bias=False)(v1).shape == (2, 2)
        assert nn.LayerNorm(2, elementwise_affine=False)(v1).shape == (2, 2)
        assert nn.RMSNorm(2, elementwise_affine=True)(v1).shape == (2, 2)
        assert nn.RMSNorm(2, elementwise_affine=False)(v1).shape == (2, 2)

        # nn.module containers & lifecycle
        base_m = nn.Module()
        base_m.add_module("none_mod", None)
        base_m.register_parameter("none_param", None)
        base_m.register_buffer("none_buf", None)
        assert base_m.to("cpu") is base_m

        child = nn.Linear(2, 2)
        child.register_buffer("cb1", torch.ones((2,)))
        parent = nn.Module()
        parent.add_module("child", child)
        parent.register_buffer("b1", torch.ones((2,)))
        parent.register_parameter("w_temp", nn.Parameter(torch.ones((2,))))
        parent.register_parameter("w_temp", None)
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
        assert F.embedding(
            torch.tensor([0, 1]),
            torch.ones((5, 3)) * 2.0,
            padding_idx=0,
            max_norm=1.0,
        ).shape == (2, 3)
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

        # nn.upsampling & F.interpolate
        up_bi = nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False)
        assert up_bi(x2d).shape == (2, 4, 8, 8)
        up_near = nn.Upsample(size=(6, 6), mode="nearest")
        assert up_near(x2d).shape == (2, 4, 6, 6)
        assert nn.UpsamplingBilinear2d(size=6)(x2d).shape == (2, 4, 6, 6)
        assert nn.UpsamplingNearest2d(scale_factor=(2.0, 2.0))(x2d).shape == (
            2,
            4,
            8,
            8,
        )
        assert F.interpolate(
            torch.ones((2, 4, 1, 1)), size=(2, 2), mode="bilinear"
        ).shape == (2, 4, 2, 2)
        assert F.interpolate(x2d, size=(4, 4), mode="nearest").shape == (2, 4, 4, 4)
        assert F.interpolate(v1, size=4).shape == (2, 2)

        # Functional branches (eager & tracing)
        assert F.scaled_dot_product_attention(v1, v1, v1).shape == (2, 2)
        assert F.scaled_dot_product_attention(
            v1, v1, v1, attn_mask=torch.zeros((2, 2))
        ).shape == (2, 2)
        assert F.scaled_dot_product_attention(
            torch.ones((2,)), torch.ones((2,)), torch.ones((2,))
        ).shape == (2,)
        assert (
            F.multilabel_soft_margin_loss(v1, v2, weight=torch.ones((2, 2))).shape == ()
        )
        assert F.gelu(v1, approximate="tanh").shape == (2, 2)
        assert F.glu(torch.ones((3,))).shape == (3,)
        assert F.glu(torch.ones(())).shape == ()
        assert F.gaussian_nll_loss(v1, v2, torch.ones((2, 2)), full=True).shape == ()
        assert F.nll_loss(
            v1,
            torch.tensor([0, -100]),
            weight=torch.tensor([1.0, 2.0]),
            reduction="none",
        ).shape == (2,)
        assert F.nll_loss(v1, torch.tensor([0, 1]), reduction="sum").shape == ()
        assert F.nll_loss(torch.ones((2,)), torch.tensor(0)).shape == ()
        assert (
            F.nll_loss(torch.ones((2,)), torch.tensor(0), weight=torch.ones((2,))).shape
            == ()
        )
        assert F.triplet_margin_loss(v1, v2, v1, swap=False).shape == ()
        assert F.poisson_nll_loss(v1, v2, log_input=False).shape == ()
        assert (
            F.triplet_margin_with_distance_loss(
                v1,
                v2,
                v1,
                swap=True,
                distance_function=lambda a, b: torch.sum((a - b) ** 2, axis=-1),
            ).shape
            == ()
        )
        assert F.rrelu(v1, training=True).shape == (2, 2)
        assert F.rrelu(v1, training=False).shape == (2, 2)
        assert F.prelu(x2d, torch.ones((4,))).shape == (2, 4, 4, 4)
        assert F.silu(v1).shape == (2, 2)
        assert F.activations(v1).shape == (2, 2)
        assert F.adaptive_avg_pool2d(x2d, 1).shape == (2, 4, 1, 1)
        assert F.adaptive_avg_pool2d(torch.ones((2,)), 1).shape == (2,)
        assert F.alpha_dropout(v1).shape == (2, 2)
        assert F.batch_norm(x2d).shape == (2, 4, 4, 4)
        assert F.batch_norm(torch.ones((2,))).shape == (2,)
        assert F.complex(v1, v2).shape == (2, 2)
        assert F.conv_transpose1d(torch.ones((2,))).shape == (2,)
        assert F.conv_transpose1d(torch.ones((2,)), torch.ones((2,))).shape == (2,)
        assert F.conv_transpose2d(torch.ones((2,)), torch.ones((2,))).shape == (2,)
        assert F.conv_transpose3d(torch.ones((2,)), torch.ones((2,))).shape == (2,)
        assert F.dropout(v1).shape == (2, 2)
        assert F.dropout1d(x1d).shape == (2, 4, 6)
        assert F.dropout2d(x2d).shape == (2, 4, 4, 4)
        assert F.dropout3d(x3d).shape == (1, 4, 3, 3, 3)
        assert F.embedding(torch.tensor([0, 1])).shape == (2,)
        assert F.embedding(
            torch.tensor([0, 1]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True),
            max_norm=1.0,
        ).shape == (2, 2)
        assert F.pad(x1d, (1, 0), mode="constant").shape == (2, 4, 7)
        assert F.pad(x1d, (0, 1), mode="constant").shape == (2, 4, 7)
        v_req = torch.tensor([1.0, 2.0], requires_grad=True)
        assert F.relu(v_req).requires_grad is True
        assert F.sigmoid(v_req).requires_grad is True
        assert F.tanh(v_req).requires_grad is True
        assert (
            F.triplet_margin_with_distance_loss(
                v1,
                v2,
                v1,
                swap=False,
                distance_function=lambda a, b: torch.sum((a - b) ** 2, axis=-1),
            ).shape
            == ()
        )
        assert F.triplet_margin_with_distance_loss(v1, v2, v1, swap=False).shape == ()
        assert F.feature_alpha_dropout(v1).shape == (2, 2)
        fmp_v1 = F.fractional_max_pool2d(v1, 2)
        assert isinstance(fmp_v1, Tensor) and fmp_v1.shape == (2, 2)
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
        assert F.spatial_dropout(x1d).shape == (2, 4, 6)
        assert F.spatial_dropout(x2d).shape == (2, 4, 4, 4)
        assert F.spatial_dropout(x3d).shape == (1, 4, 3, 3, 3)
        assert F.swish(v1).shape == (2, 2)
        assert F.upsample_bilinear(x2d).shape == (2, 4, 4, 4)
        assert F.upsample_nearest(x2d).shape == (2, 4, 4, 4)
        assert F.cross_entropy(v1, v2).shape == ()
        assert (
            F.cross_entropy(v1, v2, weight=torch.ones((2,)), label_smoothing=0.1).shape
            == ()
        )
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

    # Real TracingMode execution for functional and pooling ops
    prev_eager = config.eager_mode
    config.eager_mode = False
    _tracer.start_tracing()
    try:
        tv = torch.ones((2, 2))
        tx2d = torch.ones((2, 4, 4, 4))
        w2d = torch.ones((4, 2, 2, 2))
        assert F.softmax(tv).shape == (2, 2)
        assert F.celu(tv).shape == (2, 2)
        assert F.elu(tv).shape == (2, 2)
        assert F.gelu(tv).shape == (2, 2)
        assert F.hardswish(tv).shape == (2, 2)
        assert F.leaky_relu(tv).shape == (2, 2)
        assert F.log_softmax(tv).shape == (2, 2)
        assert F.mish(tv).shape == (2, 2)
        assert F.selu(tv).shape == (2, 2)
        assert F.softplus(tv).shape == (2, 2)
        assert F.swish(tv).shape == (2, 2)
        assert F.adaptive_avg_pool2d(tx2d, 2).shape == (2, 4, 2, 2)
        assert isinstance(FP.adaptive_max_pool2d(tx2d, 2, return_indices=False), Tensor)
        assert isinstance(FP.adaptive_max_pool2d(tx2d, 2, return_indices=True), tuple)
        assert FP.avg_pool2d(tx2d, 2).shape == (2, 4, 2, 2)
        assert isinstance(FP.fractional_max_pool2d(tx2d, 2, output_size=2), Tensor)
        assert isinstance(
            FP.fractional_max_pool2d(tx2d, 2, output_size=2, return_indices=True), tuple
        )
        assert FP.max_unpool2d(tx2d, tx2d, 2).shape == (2, 4, 8, 8)
        assert FP.unfold(tx2d, 2).shape == (2, 16, 9)
        assert FP.fold(torch.ones((2, 16, 9)), (4, 4), 2).shape == (2, 4, 4, 4)
        assert F.conv_transpose2d(tx2d, w2d).shape == (2, 2, 5, 5)
        assert F.embedding(
            torch.zeros((2, 3), dtype=DType.Int64), torch.ones((5, 4))
        ).shape == (2, 3, 4)
        assert F.pad(tx2d, (1, 1, 1, 1)).shape == (2, 4, 6, 6)
        assert F.interpolate(tx2d, size=(6, 6)).shape == (2, 4, 6, 6)
        assert F.multilabel_margin_loss(tv, tv).shape is not None
        assert F.ctc_loss(tv).shape is not None
        assert F.multi_margin_loss(tv).shape is not None
    finally:
        _tracer.stop_tracing()
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
        with pytest.raises(ValueError):
            rnn_cell_tanh(torch.ones((4, 2)), (torch.ones((4, 3)), torch.ones((4, 3))))

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
        with pytest.raises(ValueError):
            gru_cell(torch.ones((4, 2)), (torch.ones((4, 3)), torch.ones((4, 3))))
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
        out_g, h_g = gru(x_seq, torch.ones((1, 2, 3)))
        assert (
            isinstance(h_g, Tensor)
            and out_g.shape == (3, 2, 3)
            and h_g.shape == (1, 2, 3)
        )
        with pytest.raises(ValueError):
            gru(x_seq, (torch.ones((1, 2, 3)), torch.ones((1, 2, 3))))

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
            q_3d,
            q_3d,
            q_3d,
            need_weights=True,
            average_attn_weights=True,
            is_causal=True,
            key_padding_mask=torch.tensor(
                [[False, True], [False, False], [False, True]], dtype="bool"
            ),
            attn_mask=torch.tensor([[False, True], [False, False]], dtype="bool"),
        )
        assert attn_out.shape == (2, 3, 4) and attn_w is not None

        attn_out2, attn_w2 = mha_same(
            q_3d,
            q_3d,
            q_3d,
            attn_mask=torch.zeros((2, 2)),
            key_padding_mask=torch.zeros((3, 2)),
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

        sgd_group: ParamGroupDict = {
            "params": [p, p_no_grad],
            "lr": 0.1,
            "initial_lr": 0.1,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "momentum": 0.9,
            "dampening": 0.0,
            "nesterov": False,
            "rho": 0.9,
            "lr_decay": 0.0,
            "initial_accumulator_value": 0.0,
            "lambd": 1e-4,
            "alpha": 0.75,
            "t0": 1e6,
            "max_iter": 20,
            "max_eval": None,
            "tolerance_grad": 1e-7,
            "tolerance_change": 1e-9,
            "history_size": 100,
            "line_search_fn": None,
            "momentum_decay": 0.004,
            "decoupled_weight_decay": False,
            "centered": False,
            "etas": (0.5, 1.2),
            "step_sizes": (1e-6, 50.0),
            "ns_coefficients": (3.4445, -4.775, 2.0315),
            "ns_steps": 5,
            "adjust_lr_fn": None,
            "maximize": True,
            "beta2_decay": -0.8,
            "d": 1.0,
        }
        sgd = optim.SGD([sgd_group], momentum=0.9, maximize=True)
        sgd.param_groups.append({})
        loss_val = sgd.step(closure=lambda: 1.5)
        assert loss_val == 1.5
        sgd.step()
        sgd.zero_grad(set_to_none=True)
        assert p.grad is None
        p.grad = torch.tensor([0.5, 0.5])
        sgd.zero_grad(set_to_none=False)
        assert p.grad is not None

        muon = optim.Muon([p], lr=0.01, weight_decay=0.1)
        muon.param_groups.append({})
        assert muon.step() is None
        assert muon.step(closure=lambda: 2.0) == 2.0
        p2d_tall = nn.Parameter(torch.ones((3, 2)))
        p2d_tall.grad = torch.ones((3, 2)) * 0.1
        p2d_wide = nn.Parameter(torch.ones((2, 3)))
        p2d_wide.grad = torch.ones((2, 3)) * 0.1
        muon2d = optim.Muon([p2d_tall, p2d_wide, p_no_grad], lr=0.01, nesterov=False)
        muon2d.step()
        muon2d.add_param_group({"params": [p], "lr": 0.02})
        muon2d_sd = muon2d.state_dict()
        muon2d.load_state_dict(muon2d_sd)
        muon2d.load_state_dict({})

        asgd_p = nn.Parameter(torch.tensor([2.0, -2.0]))
        asgd_p.grad = torch.tensor([0.5, -0.5])
        asgd_opt = optim.ASGD([asgd_p, p_no_grad], lr=0.01)
        asgd_opt.param_groups.append({})
        assert asgd_opt.step() is None
        assert asgd_opt.step(closure=lambda: 1.0) == 1.0

        base_opt = optim.Optimizer([p, p_no_grad], {"lr": 0.01})
        base_opt.param_groups.append({})
        assert base_opt.step(closure=lambda: 1.0) == 1.0

        rprop_p = nn.Parameter(torch.tensor([2.0, -2.0]))
        rprop_p.grad = torch.tensor([0.5, -0.5])
        rprop_opt = optim.Rprop([rprop_p, p_no_grad], lr=0.01)
        rprop_opt.param_groups.append({})
        assert rprop_opt.step() is None
        rprop_p.grad = torch.tensor([-0.5, 0.5])
        assert rprop_opt.step(closure=lambda: 1.0) == 1.0

        assert _scalar_to_float(torch.tensor(0.05)) == pytest.approx(0.05)
        assert _scalar_to_float(0.1) == pytest.approx(0.1)

        for opt_cls in (
            optim.Adam,
            optim.AdamW,
            optim.Adamax,
            optim.Adagrad,
            optim.Adadelta,
            optim.RMSprop,
            optim.NAdam,
            optim.RAdam,
            optim.SparseAdam,
        ):
            w_opt = nn.Parameter(torch.tensor([2.0, -2.0]))
            w_opt.grad = torch.tensor([0.5, -0.5])
            inst = opt_cls([w_opt, p_no_grad])
            inst.param_groups.append({})
            assert inst.step(closure=lambda: 1.0) == 1.0
            inst.step()

        w_ams = nn.Parameter(torch.tensor([2.0, -2.0]))
        w_ams.grad = torch.tensor([0.5, -0.5])
        opt_ams = optim.Adam([w_ams], amsgrad=True, weight_decay=0.01)
        opt_ams.step()
        opt_ams.step()

        w_dec = nn.Parameter(torch.tensor([2.0, -2.0]))
        w_dec.grad = torch.tensor([0.5, -0.5])
        opt_dec = optim.Adam([w_dec], decoupled_weight_decay=True, weight_decay=0.01)
        opt_dec.step()

        w_rms = nn.Parameter(torch.tensor([2.0, -2.0]))
        w_rms.grad = torch.tensor([0.5, -0.5])
        opt_rms = optim.RMSprop([w_rms], weight_decay=0.01, momentum=0.9, centered=True)
        opt_rms.step()
        opt_rms.step()

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

        s_mult1 = lr_scheduler.MultiplicativeLR(opt, lr_lambda=lambda _e: 0.95)
        s_mult1.step()
        s_mult2 = lr_scheduler.MultiplicativeLR(opt, lr_lambda=[lambda _e: 0.95])
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
            opt, base_lr=0.01, max_lr=0.1, step_size_up=1, scale_fn=lambda _c: 0.5
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
        ds_multi = data.TensorDataset(torch.ones((2,)), torch.zeros((2,)))
        assert len(ds_multi) == 2
        with pytest.raises(ValueError):
            data.TensorDataset(torch.ones((2,)), torch.ones((3,)))

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
        _ = Callable
