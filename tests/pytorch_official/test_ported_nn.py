"""Ported neural network (`zero_torch.nn`) module and functional tests."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
import zero_torch.nn.functional as F
from tests.conftest import assert_allclose
from zero_torch import nn
from zero_torch.tensor import Tensor


def test_module_lifecycle() -> None:
    """Verifies `nn.Module` parameter/buffer registration, `state_dict`, and `forward`."""

    class MyModule(nn.Module):
        """Test module with a Linear layer and registered running_mean buffer."""

        def __init__(self) -> None:
            """Initializes MyModule."""
            super().__init__()
            self.linear = nn.Linear(3, 4)
            self.add_module("linear", self.linear)
            self.running_mean: Tensor | None = torch.zeros(4)
            self.register_buffer("running_mean", self.running_mean)

        def forward(self, x: Tensor) -> Tensor:
            """Applies the linear transformation and adds `running_mean`.

            Args:
                x: Input tensor of shape `(*, 3)`.

            Returns:
                Output tensor of shape `(*, 4)`.
            """
            assert self.running_mean is not None
            return self.linear(x) + self.running_mean

    with EagerMode():
        m = MyModule()

        sd = m.state_dict()
        assert "linear.weight" in sd
        assert "linear.bias" in sd
        assert "running_mean" in sd
        assert sd["linear.weight"].shape == (4, 3)
        assert sd["running_mean"].shape == (4,)

        params = list(m.parameters())
        assert len(params) == 2

        buffers = list(m.buffers())
        assert len(buffers) == 1

        x = torch.ones((2, 3))
        out = m(x)
        assert out.shape == (2, 4)


def test_functional_linear() -> None:
    """Verifies `F.linear` matrix multiplication and bias addition."""
    with EagerMode():
        x = torch.ones((2, 3))
        w = torch.ones((4, 3)) * 2.0
        b = torch.ones((4,))

        out = F.linear(x, w, b)
        assert out.shape == (2, 4)
        assert_allclose(out.tolist(), 7.0)


def test_conv2d() -> None:
    """Verifies `nn.Conv2d` forward pass with deterministic weights."""
    with EagerMode():
        x = torch.ones((2, 3, 5, 5))
        conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)

        conv.weight = nn.Parameter(torch.ones((4, 3, 3, 3)))
        conv.bias = nn.Parameter(torch.zeros(4))

        out = conv(x)
        assert out.shape == (2, 4, 5, 5)


def test_batchnorm2d() -> None:
    """Verifies `nn.BatchNorm2d` affine parameter registration, running stats update, and eval mode."""
    with EagerMode():
        bn = nn.BatchNorm2d(3)
        assert bn.weight is not None and bn.weight.shape == (3,)
        assert bn.bias is not None and bn.bias.shape == (3,)
        assert bn.running_mean is not None and bn.running_mean.shape == (3,)
        assert bn.running_var is not None and bn.running_var.shape == (3,)
        x = torch.ones((2, 3, 4, 4)) * 5.0
        out = bn(x)
        assert out.shape == (2, 3, 4, 4)
        assert_allclose(bn.running_mean.tolist(), [0.5, 0.5, 0.5], rtol=1e-4)
        bn.eval()
        out_eval = bn(x)
        assert out_eval.shape == (2, 3, 4, 4)

        lazy_lin = nn.LazyLinear(4)
        assert lazy_lin(torch.ones((2, 3))).shape == (2, 4)
        sd = lazy_lin.state_dict()
        assert "weight" in sd and "bias" in sd
        assert sd["weight"].shape == (4, 3)


def test_losses() -> None:
    """Verifies `nn.MSELoss` and `nn.CrossEntropyLoss` non-zero loss evaluation."""
    with EagerMode():
        pred = torch.tensor([[0.5, -0.5], [1.0, 2.0]])
        target_mse = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
        mse = nn.MSELoss()
        loss_mse = mse(pred, target_mse)
        assert loss_mse.shape == ()
        assert_allclose(loss_mse.tolist(), 0.125, rtol=1e-4)

        ce = nn.CrossEntropyLoss()
        loss_ce = ce(pred, torch.tensor([0, 1]))
        assert loss_ce.shape == ()
        assert loss_ce.item() != 0.0


def test_module_serialization_hooks_and_traversal() -> None:
    """Verifies `load_state_dict`, `named_parameters`, `named_buffers`, `named_modules`, `apply`, `requires_grad_`, `zero_grad`, dtype/device conversions, and hooks."""
    import pytest

    class SubBlock(nn.Module):
        """Child block with shared/persistent buffers and parameters."""

        def __init__(self) -> None:
            """Initializes SubBlock."""
            super().__init__()
            self.w = nn.Parameter(torch.ones((2, 2)))
            self.register_parameter("w", self.w)
            self.register_buffer("persist_buf", torch.ones((2,)), persistent=True)
            self.register_buffer("temp_buf", torch.zeros((2,)), persistent=False)

        def forward(self, x: Tensor) -> Tensor:
            """Multiplies input by `self.w`.

            Args:
                x: Input tensor.

            Returns:
                Scaled tensor.
            """
            return x * self.w

    class ParentNet(nn.Module):
        """Parent network holding `SubBlock` and shared references."""

        def __init__(self) -> None:
            """Initializes ParentNet."""
            super().__init__()
            self.sub1 = SubBlock()
            self.add_module("sub1", self.sub1)
            self.add_module("sub2", self.sub1)
            self.register_parameter("root_p", nn.Parameter(torch.ones((2,))))
            self.register_parameter("dup_p", self.sub1.w)
            self.register_buffer("root_b", torch.ones((2,)))
            self.register_buffer("dup_b", self.sub1._buffers["persist_buf"])

        def forward(self, x: Tensor) -> Tensor:
            """Forwards `x` through `sub1`.

            Args:
                x: Input tensor.

            Returns:
                Output tensor.
            """
            return self.sub1(x)

    with EagerMode():
        net = ParentNet()
        # Re-registering buffer as persistent after non-persistent and clearing None
        net.sub1.register_buffer("temp_buf", torch.zeros((2,)), persistent=True)
        net.sub1.register_buffer("temp_buf", torch.zeros((2,)), persistent=False)
        net.sub1.register_buffer("gone_buf", torch.zeros((1,)), persistent=False)
        net.sub1.register_buffer("gone_buf", None)

        # state_dict excludes non-persistent buffers (`temp_buf`)
        sd = net.state_dict()
        assert "sub1.temp_buf" not in sd
        assert "sub1.persist_buf" in sd

        # load_state_dict with exact match (in-place copy and assign=True)
        new_sd = {k: torch.full(v.shape, 3.0) for k, v in sd.items()}
        incompat = net.load_state_dict(new_sd, strict=True, assign=False)
        assert incompat.missing_keys == []
        assert incompat.unexpected_keys == []
        assert_allclose(net.sub1.w.tolist(), [[3.0, 3.0], [3.0, 3.0]])

        incompat_assign = net.load_state_dict(new_sd, strict=True, assign=True)
        assert incompat_assign.missing_keys == []

        # load_state_dict with missing and unexpected keys
        partial_sd = {"root_p": torch.ones((2,)), "extra_key": torch.ones((1,))}
        with pytest.raises(RuntimeError):
            net.load_state_dict(partial_sd, strict=True)
        incompat_relaxed = net.load_state_dict(partial_sd, strict=False)
        assert "sub1.w" in incompat_relaxed.missing_keys
        assert incompat_relaxed.unexpected_keys == ["extra_key"]

        # named_parameters / named_buffers / named_modules with remove_duplicate True & False
        np_dedup = list(net.named_parameters(remove_duplicate=True))
        np_all = list(net.named_parameters(remove_duplicate=False))
        assert len(np_all) > len(np_dedup)
        assert len(list(net.parameters(recurse=False))) == 2

        nb_dedup = list(net.named_buffers(remove_duplicate=True))
        nb_all = list(net.named_buffers(remove_duplicate=False))
        assert len(nb_all) > len(nb_dedup)
        assert len(list(net.buffers(recurse=False))) == 2

        nm_dedup = list(net.named_modules(remove_duplicate=True))
        nm_all = list(net.named_modules(remove_duplicate=False))
        assert len(nm_all) > len(nm_dedup)
        assert len(list(net.modules())) == len(nm_dedup)
        assert len(list(net.children())) == 2
        assert len(list(net.named_children())) == 2

        # apply, requires_grad_, zero_grad
        visited: list[nn.Module] = []
        assert net.apply(visited.append) is net
        assert len(visited) == 3

        assert net.requires_grad_(False) is net
        assert all(not p.requires_grad for p in net.parameters())
        net.requires_grad_(True)
        for p in net.parameters():
            p.grad = torch.ones_like(p)
        net.zero_grad(set_to_none=False)
        assert all(
            p.grad is not None and p.grad.tolist() == torch.zeros_like(p).tolist()
            for p in net.parameters()
        )
        net.zero_grad(set_to_none=True)
        assert all(p.grad is None for p in net.parameters())
        net.zero_grad(set_to_none=True)

        # dtype & device conversions
        assert net.cpu() is net
        assert net.cuda(0) is net
        assert net.double() is net
        assert "float64" in str(net.sub1.w.dtype)
        assert net.half() is net
        assert "float16" in str(net.sub1.w.dtype)
        assert net.bfloat16() is net
        assert net.float() is net
        assert "float32" in str(net.sub1.w.dtype)
        assert net.type("float64") is net
        assert net.type("invalid_dtype") is net
        assert net.to("cpu", "float32") is net
        assert net.to(dtype="float32") is net
        assert net._convert_tensors(None) is net

        # Forward pre-hooks, forward hooks, backward hooks, and RemovableHandle context manager
        call_order: list[str] = []

        def pre1(_m: nn.Module, inp: tuple[Tensor, ...]) -> Tensor:
            """Doubles the input tensor.

            Args:
                _m: Module instance.
                inp: Input arguments tuple.

            Returns:
                Doubled tensor.
            """
            call_order.append("pre1")
            return inp[0] * 2.0

        def pre0(_m: nn.Module, inp: tuple[Tensor, ...]) -> tuple[Tensor]:
            """Adds 1 to the input tensor.

            Args:
                _m: Module instance.
                inp: Input arguments tuple.

            Returns:
                Single-element tuple with modified tensor.
            """
            call_order.append("pre0")
            return (inp[0] + 1.0,)

        def pre_noop(_m: nn.Module, _inp: tuple[Tensor, ...]) -> None:
            """No-op pre-hook returning None.

            Args:
                _m: Module instance.
                _inp: Input arguments tuple.
            """
            call_order.append("pre_noop")

        def fwd1(_m: nn.Module, _inp: tuple[Tensor, ...], out: Tensor) -> Tensor:
            """Adds 10 to output.

            Args:
                _m: Module instance.
                _inp: Input tuple.
                out: Output tensor.

            Returns:
                Shifted output tensor.
            """
            call_order.append("fwd1")
            return out + 10.0

        def fwd0(_m: nn.Module, _inp: tuple[Tensor, ...], _out: Tensor) -> None:
            """No-op forward hook returning None.

            Args:
                _m: Module instance.
                _inp: Input tuple.
                _out: Output tensor.
            """
            call_order.append("fwd0")

        def bwd1(
            _m: nn.Module,
            grad_in: tuple[Tensor | None, ...],
            _grad_out: tuple[Tensor | None, ...],
        ) -> tuple[Tensor | None, ...]:
            """Identity backward hook.

            Args:
                _m: Module instance.
                grad_in: Input gradients.
                _grad_out: Output gradients.

            Returns:
                Input gradients unchanged.
            """
            call_order.append("bwd1")
            return grad_in

        h_pre1 = net.sub1.register_forward_pre_hook(pre1)
        h_pre0 = net.sub1.register_forward_pre_hook(pre0, prepend=True)
        with net.sub1.register_forward_pre_hook(pre_noop):
            h_fwd1 = net.sub1.register_forward_hook(fwd1)
            h_fwd0 = net.sub1.register_forward_hook(fwd0, prepend=True)
            h_bwd1 = net.sub1.register_full_backward_hook(bwd1)
            h_bwd0 = net.sub1.register_full_backward_hook(bwd1, prepend=True)
            res = net.sub1(torch.ones((2, 2)))
            assert call_order == ["pre0", "pre1", "pre_noop", "fwd0", "fwd1"]
            assert_allclose(res.tolist(), [[22.0, 22.0], [22.0, 22.0]])
            h_bwd0.remove()
            h_bwd1.remove()
        h_pre0.remove()
        h_pre1.remove()
        h_fwd0.remove()
        h_fwd1.remove()


def test_module_containers_complete() -> None:
    """Verifies indexing, slicing, mutation, and dictionary methods on `ModuleList`, `ParameterList`, `ModuleDict`, `ParameterDict`, and `Sequential`."""
    with EagerMode():
        m1 = nn.ReLU()
        m2 = nn.Sigmoid()
        m3 = nn.Tanh()

        # ModuleList
        ml = nn.ModuleList([m1, m2])
        ml.insert(1, m3)
        assert len(ml) == 3
        assert ml[1] is m3
        ml[1] = m2
        assert ml[1] is m2
        sliced_ml = ml[0:2]
        assert isinstance(sliced_ml, nn.ModuleList) and len(sliced_ml) == 2
        popped = ml.pop()
        assert popped is m2 and len(ml) == 2
        ml += [m3]
        assert len(ml) == 3
        del ml[0]
        assert len(ml) == 2

        # ParameterList
        p1 = nn.Parameter(torch.ones((2,)))
        p2 = nn.Parameter(torch.zeros((2,)))
        p3 = nn.Parameter(torch.full((2,), 5.0))
        pl = nn.ParameterList([p1, p2])
        assert len(pl) == 2
        assert pl[0] is p1
        pl[1] = p3
        assert pl[1] is p3
        pl_slice = pl[0:1]
        assert isinstance(pl_slice, nn.ParameterList) and len(pl_slice) == 1
        pl += [p2]
        assert len(pl) == 3
        assert list(pl) == [p1, p3, p2]

        # ModuleDict
        md = nn.ModuleDict({"a": m1, "b": m2})
        md["c"] = m3
        assert "c" in md and len(md) == 3
        assert md["a"] is m1
        assert list(md) == ["a", "b", "c"]
        assert list(md.keys()) == ["a", "b", "c"]
        assert len(list(md.values())) == 3
        assert len(list(md.items())) == 3
        assert md.pop("b") is m2
        del md["c"]
        assert len(md) == 1
        md.clear()
        assert len(md) == 0

        # ParameterDict
        pd = nn.ParameterDict({"p1": p1})
        pd["p2"] = p2
        assert "p2" in pd and len(pd) == 2
        assert pd["p1"] is p1
        assert list(pd) == ["p1", "p2"]
        assert list(pd.keys()) == ["p1", "p2"]
        assert len(list(pd.values())) == 2
        assert len(list(pd.items())) == 2
        assert pd.pop("p1") is p1
        del pd["p2"]
        assert len(pd) == 0
        pd["p3"] = p3
        pd.clear()
        assert len(pd) == 0

        # Sequential
        seq = nn.Sequential(m1, m2)
        seq.append(m3)
        assert len(seq) == 3
        assert list(seq) == [m1, m2, m3]
        assert seq[0] is m1
        seq[0] = m3
        assert seq[0] is m3
        seq_slice = seq[0:2]
        assert isinstance(seq_slice, nn.Sequential) and len(seq_slice) == 2
        del seq[1:3]
        assert len(seq) == 1
        del seq[0]
        assert len(seq) == 0


def test_nn_layers_padding_mode_output_size_proj_and_adaptive_softmax() -> None:
    """Verifies `padding_mode` in Conv, `output_size` in ConvTranspose, `proj_size` & `dropout` in LSTM, `padding_idx` & `offsets` in EmbeddingBag, `AdaptiveLogSoftmaxWithLoss`, and `MultiheadAttention` `add_bias_kv`/`add_zero_attn`/callable activations."""
    import pytest

    with EagerMode():
        # Linear & Bilinear with 0 features
        assert nn.Linear(0, 2).weight is not None
        assert nn.Bilinear(0, 2, 2).weight is not None

        # Conv padding_mode != 'zeros'
        c_ref = nn.Conv1d(2, 2, kernel_size=3, padding=1, padding_mode="reflect")
        assert c_ref(torch.ones((1, 2, 4))).shape == (1, 2, 4)

        # ConvTranspose output_size valid and invalid
        ct1 = nn.ConvTranspose1d(2, 2, kernel_size=2, stride=2, output_padding=(0,))
        assert ct1(torch.ones((1, 2, 3)), output_size=(6,)).shape == (1, 2, 6)
        with pytest.raises(ValueError):
            ct1(torch.ones((1, 2, 3)), output_size=(10,))

        ct2 = nn.ConvTranspose2d(
            2, 2, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1)
        )
        assert ct2(torch.ones((1, 2, 2, 2)), output_size=(4, 4)).shape == (1, 2, 4, 4)

        # LSTM with proj_size > 0 and stacked dropout > 0.0
        lstm_proj = nn.LSTM(4, 6, num_layers=2, dropout=0.2, proj_size=3)
        out_p, state_p = lstm_proj(torch.ones((5, 2, 4)))
        assert isinstance(state_p, tuple)
        hn_p, cn_p = state_p
        assert out_p.shape == (5, 2, 3)
        assert hn_p.shape == (2, 2, 3)
        assert cn_p.shape == (2, 2, 6)

        # Embedding & EmbeddingBag with padding_idx, offsets, and include_last_offset
        emb = nn.Embedding(5, 3, padding_idx=0)
        assert emb.weight is not None and emb.weight[0].tolist() == [0.0, 0.0, 0.0]

        eb = nn.EmbeddingBag(
            5, 3, mode="sum", padding_idx=-1, include_last_offset=False
        )
        eb_out = eb(torch.tensor([0, 1, 2, 3]), offsets=torch.tensor([0, 2, 2]))
        assert eb_out.shape == (3, 3)
        assert eb_out[1].tolist() == [0.0, 0.0, 0.0]

        eb_last = nn.EmbeddingBag(5, 3, mode="max", include_last_offset=True)
        assert eb_last(
            torch.tensor([0, 1, 2]), offsets=torch.tensor([0, 2, 3])
        ).shape == (2, 3)

        # AdaptiveLogSoftmaxWithLoss log_prob, predict, custom head & tail
        als = nn.AdaptiveLogSoftmaxWithLoss(4, 6, cutoffs=[2, 4])
        x_als = torch.ones((3, 4))
        assert als.log_prob(x_als).shape == (3, 6)
        assert als.predict(x_als).shape == (3,)
        out_als, loss_als = als(x_als, torch.tensor([0, 1, 2]))
        assert out_als.shape == (3,) and loss_als.shape == ()

        als_custom = nn.AdaptiveLogSoftmaxWithLoss(
            4,
            6,
            cutoffs=[2, 4],
            head=nn.Identity(),
            tail=nn.ModuleList([nn.Identity()]),
        )
        assert als_custom(torch.ones((2, 4)), torch.tensor([0, 1]))[0].shape == (2,)

        # MultiheadAttention with add_bias_kv and add_zero_attn, and callable activation in Transformer layers
        mha = nn.MultiheadAttention(4, 2, add_bias_kv=True, add_zero_attn=True)
        q_in = torch.ones((3, 2, 4))
        attn_out, attn_w = mha(q_in, q_in, q_in, average_attn_weights=False)
        assert (
            attn_out.shape == (3, 2, 4)
            and attn_w is not None
            and attn_w.shape == (2, 2, 3, 5)
        )

        mha_plain = nn.MultiheadAttention(4, 2)
        mha_plain(
            q_in,
            q_in,
            q_in,
            attn_mask=torch.zeros((3, 3)),
            key_padding_mask=torch.zeros((2, 3)),
        )

        enc_l = nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, activation=F.relu)
        assert enc_l(q_in).shape == (3, 2, 4)
        dec_l = nn.TransformerDecoderLayer(4, 2, dim_feedforward=8, activation=F.relu)
        assert dec_l(q_in, q_in).shape == (3, 2, 4)
