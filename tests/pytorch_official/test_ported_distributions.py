"""Ported probability distributions test suite."""

from __future__ import annotations

import numpy as np
import pytest
from ml_switcheroo_compiler.core.config import EagerMode

import zero_torch as torch
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
                self, sample_shape: tuple[int, ...] | None = None
            ) -> torch.Tensor:
                """Returns a scalar zero tensor.

                Args:
                    sample_shape: Optional sample shape.

                Returns:
                    Zero tensor.
                """
                _ = sample_shape
                return torch.tensor(0.0)

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
            _ = ef._natural_params
        with pytest.raises(NotImplementedError):
            ef._log_normalizer()
        with pytest.raises(NotImplementedError):
            _ = ef._mean_carrier_measure
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
        assert lp_u.item() == pytest.approx(-np.log(2.0), abs=1e-4)

        # Bernoulli
        b = Bernoulli(probs=torch.tensor([0.2, 0.8]))
        np.testing.assert_allclose(b.mean.tolist(), [0.2, 0.8], rtol=1e-4)
        np.testing.assert_allclose(b.variance.tolist(), [0.16, 0.16], rtol=1e-4)
        s_b = b.sample((2,))
        assert s_b.shape == (2, 2)
        lp_b = b.log_prob(torch.tensor([1.0, 0.0]))
        assert lp_b.shape == (2,)
        np.testing.assert_allclose(
            lp_b.tolist(), [np.log(0.2), np.log(0.2)], rtol=1e-3, atol=1e-3
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
        np.testing.assert_allclose(p.mean.tolist(), [1.0, 5.0])
        np.testing.assert_allclose(p.variance.tolist(), [1.0, 5.0])
        assert p.sample().shape == (2,)
        s_p = p.sample((2,))
        assert s_p.shape == (2, 2)
        lp_p = p.log_prob(torch.tensor([1.0, 2.0]))
        assert lp_p.shape == (2,)

        # Gamma
        g = Gamma(torch.tensor([2.0, 3.0]), torch.tensor([1.0, 1.0]))
        np.testing.assert_allclose(g.mean.tolist(), [2.0, 3.0])
        np.testing.assert_allclose(g.variance.tolist(), [2.0, 3.0])
        assert g.rsample().shape == (2,)
        s_g = g.sample((2,))
        assert s_g.shape == (2, 2)
        lp_g = g.log_prob(torch.tensor([1.5, 2.5]))
        assert lp_g.shape == (2,)

        # Beta
        beta = Beta(torch.tensor([2.0, 2.0]), torch.tensor([2.0, 2.0]))
        np.testing.assert_allclose(beta.mean.tolist(), [0.5, 0.5])
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
