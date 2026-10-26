"""Ported probability distributions test suite."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
from tests.conftest import assert_allclose
from zero_torch.distributions import (
    Bernoulli,
    Beta,
    Categorical,
    Distribution,
    ExponentialFamily,
    Gamma,
    Normal,
    Poisson,
    Uniform,
)


def test_distributions() -> None:
    """Verifies sampling, `log_prob`, `mean`, `variance`, `entropy`, `cdf`, and `icdf` across distributions."""
    with EagerMode():
        dist = Distribution()
        assert dist.batch_shape == ()
        assert dist.event_shape == ()
        with pytest.raises(NotImplementedError):
            dist.rsample()
        with pytest.raises(NotImplementedError):
            dist.sample()

        class RSampleDist(Distribution):
            """Distribution subclass with `has_rsample = True`."""

            has_rsample: bool = True

            def rsample(
                self, sample_shape: Sequence[int] | None = None
            ) -> torch.Tensor:
                """Returns a scalar zero tensor.

                Args:
                    sample_shape: Optional sample shape.

                Returns:
                    Zero tensor.
                """
                return torch.zeros(sample_shape if sample_shape is not None else ())

        assert RSampleDist().sample().item() == 0.0
        with pytest.raises(NotImplementedError):
            dist.log_prob(torch.tensor(0.0))
        with pytest.raises(NotImplementedError):
            dist.expand((2,))
        with pytest.raises(NotImplementedError):
            dist.cdf(torch.tensor(0.0))
        with pytest.raises(NotImplementedError):
            dist.icdf(torch.tensor(0.0))
        with pytest.raises(NotImplementedError):
            dist.enumerate_support()
        with pytest.raises(NotImplementedError):
            dist.entropy()

        ef = ExponentialFamily()
        with pytest.raises(NotImplementedError):
            assert ef._natural_params is not None
        with pytest.raises(NotImplementedError):
            ef._log_normalizer()
        with pytest.raises(NotImplementedError):
            assert ef._mean_carrier_measure != 0.0
        with pytest.raises(NotImplementedError):
            ef.entropy()

        # Normal
        n = Normal(torch.tensor(0.0), torch.tensor(1.0))
        assert n.batch_shape == ()
        assert n.mean.item() == 0.0
        assert n.variance.item() == 1.0
        assert n.stddev.item() == 1.0
        assert n.cdf(torch.tensor(0.0)).item() == pytest.approx(0.5, abs=1e-4)
        assert n.icdf(torch.tensor(0.5)).item() == pytest.approx(0.0, abs=1e-4)
        s = n.sample((2, 3))
        assert s.shape == (2, 3)
        assert n.rsample().shape == ()
        lp = n.log_prob(torch.tensor(0.0))
        assert lp.shape == ()

        # Uniform
        u = Uniform(torch.tensor(0.0), torch.tensor(2.0))
        assert u.mean.item() == pytest.approx(1.0)
        assert u.variance.item() == pytest.approx(4.0 / 12.0)
        assert u.cdf(torch.tensor(1.0)).item() == pytest.approx(0.5)
        assert u.icdf(torch.tensor(0.5)).item() == pytest.approx(1.0)
        assert u.rsample().shape == ()
        s_u = u.sample((2, 3))
        assert s_u.shape == (2, 3)
        lp_u = u.log_prob(torch.tensor(0.5))
        assert lp_u.item() == pytest.approx(-math.log(2.0), abs=1e-4)

        # Bernoulli
        b = Bernoulli(probs=torch.tensor([0.2, 0.8]))
        assert_allclose(b.mean.tolist(), [0.2, 0.8], rtol=1e-4)
        assert_allclose(b.variance.tolist(), [0.16, 0.16], rtol=1e-4)
        s_b = b.sample((2,))
        assert s_b.shape == (2, 2)
        lp_b = b.log_prob(torch.tensor([1.0, 0.0]))
        assert lp_b.shape == (2,)
        assert_allclose(
            lp_b.tolist(), [math.log(0.2), math.log(0.2)], rtol=1e-3, atol=1e-3
        )

        b2 = Bernoulli(logits=torch.tensor([-1.0, 1.0]))
        assert b2.sample().shape == (2,)

        with pytest.raises(ValueError):
            Bernoulli()

        # Categorical
        c = Categorical(probs=torch.tensor([0.2, 0.3, 0.5]))
        assert c.mean.item() == pytest.approx(1.3, abs=1e-4)
        s_c = c.sample((2,))
        assert s_c.shape == (2,)
        lp_c = c.log_prob(torch.tensor([0, 1]))
        assert lp_c.shape == (2,)

        c2 = Categorical(logits=torch.tensor([-1.0, 0.0, 1.0]))
        assert c2.sample().shape == ()

        with pytest.raises(ValueError):
            Categorical(probs=torch.tensor(1.0))

        with pytest.raises(ValueError):
            Categorical(logits=torch.tensor(1.0))

        with pytest.raises(ValueError):
            Categorical()

        # Poisson
        p = Poisson(torch.tensor([1.0, 5.0]))
        assert_allclose(p.mean.tolist(), [1.0, 5.0])
        assert_allclose(p.variance.tolist(), [1.0, 5.0])
        assert p.sample().shape == (2,)
        s_p = p.sample((2,))
        assert s_p.shape == (2, 2)
        lp_p = p.log_prob(torch.tensor([1.0, 2.0]))
        assert lp_p.shape == (2,)

        # Gamma
        g = Gamma(torch.tensor([2.0, 3.0]), torch.tensor([1.0, 1.0]))
        assert_allclose(g.mean.tolist(), [2.0, 3.0])
        assert_allclose(g.variance.tolist(), [2.0, 3.0])
        assert g.rsample().shape == (2,)
        s_g = g.sample((2,))
        assert s_g.shape == (2, 2)
        lp_g = g.log_prob(torch.tensor([1.5, 2.5]))
        assert lp_g.shape == (2,)

        # Beta
        beta = Beta(torch.tensor([2.0, 2.0]), torch.tensor([2.0, 2.0]))
        assert_allclose(beta.mean.tolist(), [0.5, 0.5])
        assert beta.rsample().shape == (2,)
        s_beta = beta.sample((2,))
        assert s_beta.shape == (2, 2)
        lp_beta = beta.log_prob(torch.tensor([0.5, 0.5]))
        assert lp_beta.shape == (2,)


def test_distributions_utils() -> None:
    """Verifies `_broadcast_shape` shape broadcasting rules and error handling."""
    from zero_torch.distributions.utils import _broadcast_shape

    assert _broadcast_shape((2, 3), (1, 3)) == (2, 3)
    assert _broadcast_shape((2, 1), (1, 3)) == (2, 3)
    assert _broadcast_shape((), (2, 3)) == (2, 3)
    assert _broadcast_shape((2, 3), ()) == (2, 3)

    with pytest.raises(RuntimeError):
        _broadcast_shape((2, 3), (3, 3))


def test_distributions_expand_support_and_validation() -> None:
    """Verifies `expand`, `enumerate_support`, `_natural_params`, `_log_normalizer`, `entropy`, and `validate_args`."""
    with EagerMode():
        # Normal
        n = Normal(torch.tensor(0.0), torch.tensor(1.0), validate_args=True)
        n_exp = n.expand((2, 3))
        assert n_exp.batch_shape == (2, 3)
        np0, np1 = n._natural_params
        assert np0.item() == pytest.approx(0.0)
        assert np1.item() == pytest.approx(-0.5)
        assert n._log_normalizer(np0, np1).item() == pytest.approx(
            0.5 * math.log(2.0 * math.pi), abs=1e-4
        )
        with pytest.raises(ValueError):
            Normal(torch.tensor(0.0), torch.tensor(0.0), validate_args=True)

        # Uniform
        u = Uniform(torch.tensor(0.0), torch.tensor(2.0), validate_args=True)
        assert u.expand((3,)).batch_shape == (3,)
        u_vec = Uniform(
            torch.tensor([0.0, 1.0]), torch.tensor([2.0, 3.0]), validate_args=True
        )
        assert u_vec.batch_shape == (2,)
        with pytest.raises(ValueError):
            Uniform(torch.tensor(2.0), torch.tensor(1.0), validate_args=True)

        # Bernoulli
        b = Bernoulli(probs=torch.tensor([0.2, 0.8]), validate_args=True)
        assert b.expand((3, 2)).batch_shape == (3, 2)
        sup_b = b.enumerate_support(expand=True)
        assert sup_b.shape == (2, 2)
        sup_b_no = b.enumerate_support(expand=False)
        assert sup_b_no.shape == (2, 1)
        (b_np,) = b._natural_params
        assert b._log_normalizer(b_np).shape == (2,)
        with pytest.raises(ValueError):
            Bernoulli(probs=torch.tensor([-0.1, 0.5]), validate_args=True)

        # Categorical
        c = Categorical(probs=torch.tensor([[0.2, 0.3, 0.5]]), validate_args=True)
        assert c.expand((4, 1)).batch_shape == (4, 1)
        sup_c = c.enumerate_support(expand=True)
        assert sup_c.shape == (3, 1)
        sup_c_no = c.enumerate_support(expand=False)
        assert sup_c_no.shape == (3, 1)
        c_scalar = Categorical(probs=torch.tensor([0.2, 0.8]))
        assert c_scalar.enumerate_support(expand=True).shape == (2,)
        with pytest.raises(ValueError):
            Categorical(probs=torch.tensor([-0.1, 1.1]), validate_args=True)
        with pytest.raises(ValueError):
            Categorical(probs=torch.tensor([0.0, 0.0]), validate_args=True)

        # Beta
        beta = Beta(torch.tensor(2.0), torch.tensor(3.0), validate_args=True)
        assert beta.expand((2,)).batch_shape == (2,)
        b_c1, b_c0 = beta._natural_params
        assert beta._log_normalizer(b_c1, b_c0).shape == ()
        assert beta.entropy().shape == ()
        with pytest.raises(ValueError):
            Beta(torch.tensor(0.0), torch.tensor(1.0), validate_args=True)

        # Gamma
        g = Gamma(torch.tensor(2.0), torch.tensor(1.0), validate_args=True)
        assert g.expand((2,)).batch_shape == (2,)
        g_np0, g_np1 = g._natural_params
        assert g._log_normalizer(g_np0, g_np1).shape == ()
        assert g.entropy().shape == ()
        with pytest.raises(ValueError):
            Gamma(torch.tensor(-1.0), torch.tensor(1.0), validate_args=True)

        # Poisson
        p = Poisson(torch.tensor(2.5), validate_args=True)
        assert p.expand((4,)).batch_shape == (4,)
        (p_np,) = p._natural_params
        assert p._log_normalizer(p_np).item() == pytest.approx(2.5, abs=1e-4)
        with pytest.raises(ValueError):
            Poisson(torch.tensor(-1.0), validate_args=True)
