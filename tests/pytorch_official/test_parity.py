"""Numerical parity tests against reference PyTorch."""

from __future__ import annotations

import torch as real_torch
from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch
from tests.conftest import assert_allclose
from zero_torch import nn, optim
from zero_torch.tensor import Tensor


def test_core_math_parity() -> None:
    """Verifies numerical parity between `zero_torch` and `torch` across core math operations."""
    with EagerMode():
        arr_a = [[0.25, 0.5], [0.75, 1.25]]
        arr_b = [[1.0, 2.0], [3.0, 4.0]]
        z_a = zero_torch.tensor(arr_a)
        z_b = zero_torch.tensor(arr_b)
        t_a = real_torch.tensor(arr_a)
        t_b = real_torch.tensor(arr_b)

        assert_allclose(
            zero_torch.add(z_a, z_b).tolist(),
            real_torch.add(t_a, t_b).tolist(),
            rtol=1e-5,
        )
        assert_allclose(
            zero_torch.subtract(z_a, z_b).tolist(),
            real_torch.sub(t_a, t_b).tolist(),
            rtol=1e-5,
        )
        assert_allclose(
            zero_torch.multiply(z_a, z_b).tolist(),
            real_torch.mul(t_a, t_b).tolist(),
            rtol=1e-5,
        )
        assert_allclose(
            zero_torch.divide(z_a, z_b).tolist(),
            real_torch.div(t_a, t_b).tolist(),
            rtol=1e-5,
        )
        assert_allclose(
            zero_torch.matmul(z_a, z_b).tolist(),
            real_torch.matmul(t_a, t_b).tolist(),
            rtol=1e-5,
        )
        assert_allclose(
            zero_torch.exp(z_a).tolist(), real_torch.exp(t_a).tolist(), rtol=1e-5
        )
        assert_allclose(
            zero_torch.log(z_a).tolist(), real_torch.log(t_a).tolist(), rtol=1e-5
        )
        assert_allclose(
            zero_torch.sin(z_a).tolist(), real_torch.sin(t_a).tolist(), rtol=1e-5
        )
        assert_allclose(
            zero_torch.cos(z_a).tolist(), real_torch.cos(t_a).tolist(), rtol=1e-5
        )
        assert_allclose(
            zero_torch.tanh(z_a).tolist(), real_torch.tanh(t_a).tolist(), rtol=1e-5
        )
        assert_allclose(
            zero_torch.sigmoid(z_a).tolist(),
            real_torch.sigmoid(t_a).tolist(),
            rtol=1e-5,
        )
        assert_allclose(
            zero_torch.softmax(z_a, dim=-1).tolist(),
            real_torch.softmax(t_a, dim=-1).tolist(),
            rtol=1e-5,
        )


def test_fft_parity_with_torch() -> None:
    """Verifies numerical parity between `zero_torch.fft` and `torch.fft`."""
    with EagerMode():
        arr1d = [1.0, 2.0, 3.0, 4.0]
        t = Tensor(arr1d)
        rt = real_torch.tensor(arr1d)

        assert_allclose(
            zero_torch.fft.fft(t).tolist(),
            real_torch.fft.fft(rt).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.fft.ifft(t).tolist(),
            real_torch.fft.ifft(rt).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.fft.rfft(t).tolist(),
            real_torch.fft.rfft(rt).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.fft.irfft(t).tolist(),
            real_torch.fft.irfft(rt).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )

        arr2d = [[1.0, 2.0], [3.0, 4.0]]
        t2 = Tensor(arr2d)
        rt2 = real_torch.tensor(arr2d)

        assert_allclose(
            zero_torch.fft.fft2(t2).tolist(),
            real_torch.fft.fft2(rt2).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.fft.ifft2(t2).tolist(),
            real_torch.fft.ifft2(rt2).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.fft.rfft2(t2).tolist(),
            real_torch.fft.rfft2(rt2).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.fft.irfft2(t2).tolist(),
            real_torch.fft.irfft2(rt2).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.fft.fftn(t2).tolist(),
            real_torch.fft.fftn(rt2).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.fft.ifftn(t2).tolist(),
            real_torch.fft.ifftn(rt2).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.fft.rfftn(t2).tolist(),
            real_torch.fft.rfftn(rt2).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.fft.irfftn(t2).tolist(),
            real_torch.fft.irfftn(rt2).resolve_conj().tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.fft.fftfreq(4).tolist(),
            real_torch.fft.fftfreq(4).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.fft.rfftfreq(4).tolist(),
            real_torch.fft.rfftfreq(4).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.fft.fftshift(t).tolist(),
            real_torch.fft.fftshift(rt).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.fft.ifftshift(t).tolist(),
            real_torch.fft.ifftshift(rt).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )


def test_optim_adam_parity_with_torch() -> None:
    """Verifies multi-step `optim.Adam` parity against `torch.optim.Adam`."""
    with EagerMode():
        zw = nn.Parameter(zero_torch.tensor([[1.5, -0.5], [0.25, 2.0]]))
        rw = real_torch.nn.Parameter(real_torch.tensor([[1.5, -0.5], [0.25, 2.0]]))
        z_adam = optim.Adam([zw], lr=0.05, betas=(0.9, 0.999), eps=1e-8)
        r_adam = real_torch.optim.Adam([rw], lr=0.05, betas=(0.9, 0.999), eps=1e-8)
        for step_i in range(1, 5):
            g_arr = [[0.1 * step_i, -0.2 * step_i], [0.05 * step_i, 0.3 * step_i]]
            zw.grad = zero_torch.tensor(g_arr)
            rw.grad = real_torch.tensor(g_arr)
            z_adam.step()
            r_adam.step()
            assert_allclose(zw.tolist(), rw.detach().tolist(), rtol=1e-4, atol=1e-5)


def test_special_parity_with_torch() -> None:
    """Verifies `zero_torch.special` functions against `torch.special`."""
    with EagerMode():
        arr = [1.0, 2.0, 3.0]
        t = Tensor(arr)
        rt = real_torch.tensor(arr)

        assert_allclose(
            zero_torch.special.bessel_j0(t).tolist(),
            real_torch.special.bessel_j0(rt).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.special.bessel_j1(t).tolist(),
            real_torch.special.bessel_j1(rt).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.special.erfcx(t).tolist(),
            real_torch.special.erfcx(rt).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.special.ndtr(t).tolist(),
            real_torch.special.ndtr(rt).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.special.gammaln(t).tolist(),
            real_torch.special.gammaln(rt).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.special.multigammaln(t, 2).tolist(),
            real_torch.special.multigammaln(rt, 2).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.special.modified_bessel_i0(t).tolist(),
            real_torch.special.modified_bessel_i0(rt).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.special.modified_bessel_i1(t).tolist(),
            real_torch.special.modified_bessel_i1(rt).tolist(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_allclose(
            zero_torch.special.modified_bessel_k0(t).tolist(),
            real_torch.special.modified_bessel_k0(rt).tolist(),
            rtol=1e-3,
            atol=1e-3,
        )
        assert_allclose(
            zero_torch.special.modified_bessel_k1(t).tolist(),
            real_torch.special.modified_bessel_k1(rt).tolist(),
            rtol=1e-3,
            atol=1e-3,
        )

        n_degs = Tensor([0.0, 1.0, 2.0])
        rn_degs = real_torch.tensor([0, 1, 2])
        for z_poly, r_poly in [
            (
                zero_torch.special.chebyshev_polynomial_t,
                real_torch.special.chebyshev_polynomial_t,
            ),
            (
                zero_torch.special.hermite_polynomial_h,
                real_torch.special.hermite_polynomial_h,
            ),
            (
                zero_torch.special.laguerre_polynomial_l,
                real_torch.special.laguerre_polynomial_l,
            ),
            (
                zero_torch.special.legendre_polynomial_p,
                real_torch.special.legendre_polynomial_p,
            ),
        ]:
            assert_allclose(
                z_poly(t, n_degs).tolist(),
                r_poly(rt, rn_degs).tolist(),
                rtol=1e-4,
                atol=1e-4,
            )


def test_unfold_fold_parity_with_torch() -> None:
    """Verifies `nn.Unfold` and `nn.Fold` against `torch.nn.Unfold` and `torch.nn.Fold`."""
    with EagerMode():
        x2d = zero_torch.ones((2, 4, 4, 4))
        rx2d = real_torch.ones((2, 4, 4, 4))
        uf = nn.Unfold(kernel_size=2, padding=1, stride=2)
        uf_out = uf(x2d)
        r_uf_out = real_torch.nn.Unfold(kernel_size=2, padding=1, stride=2)(rx2d)
        assert uf_out.shape == tuple(r_uf_out.shape)
        assert_allclose(uf_out.tolist(), r_uf_out.tolist(), rtol=1e-4)

        fd = nn.Fold(output_size=(4, 4), kernel_size=2, padding=1, stride=2)
        fd_out = fd(uf_out)
        r_fd_out = real_torch.nn.Fold(
            output_size=(4, 4), kernel_size=2, padding=1, stride=2
        )(r_uf_out)
        assert fd_out.shape == tuple(r_fd_out.shape)
        assert_allclose(fd_out.tolist(), r_fd_out.tolist(), rtol=1e-4)


def test_training_step_parity_with_torch() -> None:
    """Verifies end-to-end forward + MSELoss + backward + optimizer.step() parity across SGD, Adam, AdamW, and RMSprop."""
    with EagerMode():
        w_init = [[0.5, -0.25], [1.0, 0.75]]
        b_init = [0.1, -0.2]
        x_val = [[1.0, 2.0], [0.5, -1.0]]
        y_val = [[1.5, 0.5], [0.0, 1.0]]

        for opt_name in ("SGD", "Adam", "AdamW", "RMSprop"):
            z_lin = nn.Linear(2, 2)
            assert z_lin.weight is not None and z_lin.bias is not None
            z_lin.weight.copy_(zero_torch.tensor(w_init))
            z_lin.bias.copy_(zero_torch.tensor(b_init))

            r_lin = real_torch.nn.Linear(2, 2)
            with real_torch.no_grad():
                r_lin.weight.copy_(real_torch.tensor(w_init))
                r_lin.bias.copy_(real_torch.tensor(b_init))

            if opt_name == "SGD":
                z_opt: optim.Optimizer = optim.SGD(
                    z_lin.parameters(), lr=0.05, momentum=0.9
                )
                r_opt: real_torch.optim.Optimizer = real_torch.optim.SGD(
                    r_lin.parameters(), lr=0.05, momentum=0.9
                )
            elif opt_name == "Adam":
                z_opt = optim.Adam(z_lin.parameters(), lr=0.05)
                r_opt = real_torch.optim.Adam(r_lin.parameters(), lr=0.05)
            elif opt_name == "AdamW":
                z_opt = optim.AdamW(z_lin.parameters(), lr=0.05, weight_decay=0.01)
                r_opt = real_torch.optim.AdamW(
                    r_lin.parameters(), lr=0.05, weight_decay=0.01
                )
            else:
                z_opt = optim.RMSprop(z_lin.parameters(), lr=0.01)
                r_opt = real_torch.optim.RMSprop(r_lin.parameters(), lr=0.01)

            z_loss_fn = nn.MSELoss()
            r_loss_fn = real_torch.nn.MSELoss()

            z_opt.zero_grad()
            r_opt.zero_grad()

            z_pred = z_lin(zero_torch.tensor(x_val))
            r_pred = r_lin(real_torch.tensor(x_val))
            assert_allclose(z_pred.tolist(), r_pred.detach().tolist(), rtol=1e-5)

            z_loss = z_loss_fn(z_pred, zero_torch.tensor(y_val))
            r_loss = r_loss_fn(r_pred, real_torch.tensor(y_val))
            assert_allclose(z_loss.tolist(), r_loss.detach().tolist(), rtol=1e-5)

            z_loss.backward()
            r_loss.backward()
            z_opt.step()
            r_opt.step()

            assert_allclose(
                z_lin.weight.tolist(),
                r_lin.weight.detach().tolist(),
                rtol=1e-4,
                atol=1e-5,
            )
            assert_allclose(
                z_lin.bias.tolist(),
                r_lin.bias.detach().tolist(),
                rtol=1e-4,
                atol=1e-5,
            )
