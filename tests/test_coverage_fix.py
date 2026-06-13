from zero_torch.tracing import ProxyTensor, _tracer
import zero_torch
from zero_torch import Tensor


def test_tracer_tape():
    _graph = _tracer.start_tracing()
    assert _tracer.is_tracing

    pt1 = ProxyTensor("id1", (2, 2))
    pt2 = ProxyTensor("id2", (2, 2))

    # Run math ops
    _pt_add = pt1 + pt2
    _pt_radd = 1 + pt1
    _pt_sub = pt1 - pt2
    _pt_rsub = 1 - pt1
    _pt_mul = pt1 * pt2
    _pt_rmul = 1 * pt1
    _pt_div = pt1 / pt2
    _pt_rdiv = 1 / pt1
    _pt_pow = pt1**2
    _pt_fdiv = pt1 // 2
    _pt_rfdiv = 2 // pt1
    _pt_mod = pt1 % 2
    _pt_rmod = 2 % pt1
    _pt_and = pt1 & pt2
    _pt_rand = 1 & pt1
    _pt_or = pt1 | pt2
    _pt_ror = 1 | pt1
    _pt_xor = pt1 ^ pt2
    _pt_rxor = 1 ^ pt1
    _pt_lshift = pt1 << 1
    _pt_rlshift = 1 << pt1
    _pt_rshift = pt1 >> 1
    _pt_rrshift = 1 >> pt1

    _pt_neg = -pt1
    _pt_pos = +pt1
    _pt_abs = abs(pt1)
    _pt_inv = ~pt1

    _pt_slice = pt1[0:1]
    _pt_matmul = pt1 @ pt2
    pt1.__pos__()
    pt1.__abs__()
    pt1.__invert__()
    pt1.__getitem__(0)
    pt1.__neg__()
    pt1.__lshift__(1)
    pt1.__rlshift__(1)
    pt1.__rshift__(1)
    pt1.__rrshift__(1)
    pt1.__floordiv__(1)
    pt1.__rfloordiv__(1)
    pt1.__mod__(1)
    pt1.__rmod__(1)
    pt1.__and__(1)
    pt1.__rand__(1)
    pt1.__or__(1)
    pt1.__ror__(1)
    pt1.__xor__(1)
    pt1.__rxor__(1)
    pt1.__pow__(1)

    _tracer.stop_tracing()
    assert not _tracer.is_tracing


def test_rand_tracing():
    _tracer.start_tracing()
    _t_rand = zero_torch.rand(2, 2)
    _t_randn = zero_torch.randn(2, 2)
    _t_randint = zero_torch.randint(0, 10, (2, 2))
    _tracer.stop_tracing()


def test_new_apis_tracing():
    _tracer.start_tracing()
    t1 = Tensor([0.5, 0.5])
    zero_torch.logit(t1)
    zero_torch.mvlgamma(t1, 2)
    zero_torch.nan_to_num(t1, nan=0.0, posinf=1.0, neginf=-1.0)
    zero_torch.signbit(t1)
    zero_torch.true_divide(t1, t1)
    zero_torch.xlogy(t1, t1)
    _tracer.stop_tracing()


def test_manual_seed():
    zero_torch.manual_seed(42)


def test_detach_none():
    t = Tensor(None)
    t.detach()
