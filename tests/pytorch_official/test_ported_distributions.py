try:
    from ml_switcheroo_compiler.core.errors import (
        ShapeMismatchError,
        UnimplementedMathError,
    )
except ImportError:
    UnimplementedMathError = Exception
    ShapeMismatchError = Exception
from unittest.mock import patch

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


def test_distributions():
    with EagerMode():
        # Base classes
        dist = Distribution()
        assert dist.batch_shape == ()
        assert dist.event_shape == ()
        dist.has_rsample = True
        dist.rsample = lambda sample_shape=None: torch.tensor(0.0)
        assert dist.sample().item() == 0.0
        del dist.rsample
        with pytest.raises(NotImplementedError):
            dist.rsample()
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
        s = n.sample((2, 3))
        assert s.shape == (2, 3)
        assert n.rsample().shape == ()
        lp = n.log_prob(torch.tensor(0.0))
        assert lp.shape == ()

        # Uniform
        u = Uniform(torch.tensor(0.0), torch.tensor(1.0))
        assert u.rsample() is not None
        s = u.sample((2, 3))
        assert s.shape == (2, 3)
        lp = u.log_prob(torch.tensor(0.5))
        assert lp.shape == ()

        # Bernoulli
        b = Bernoulli(probs=torch.tensor([0.2, 0.8]))
        s = b.sample((2,))
        assert s.shape == (2, 2)
        lp = b.log_prob(torch.tensor([1.0, 0.0]))
        assert lp.shape == (2,)

        b2 = Bernoulli(logits=torch.tensor([-1.0, 1.0]))
        assert b2.sample().shape == (2,)

        with pytest.raises(ValueError):
            Bernoulli()

        # Categorical
        c = Categorical(probs=torch.tensor([0.2, 0.3, 0.5]))
        s = c.sample((2,))
        assert s.shape == (2,)
        lp = c.log_prob(torch.tensor([0, 1]))
        assert lp.shape == (2,)

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
        assert p.sample() is not None
        s = p.sample((2,))
        assert s.shape == (2, 2)
        with patch("zero_torch.special.gammaln", return_value=torch.tensor([0.0, 0.0])):
            lp = p.log_prob(torch.tensor([1.0, 2.0]))
            assert lp.shape == (2,)

        # Gamma
        g = Gamma(torch.tensor([2.0, 3.0]), torch.tensor([1.0, 1.0]))
        assert g.rsample() is not None
        s = g.sample((2,))
        assert s.shape == (2, 2)
        with patch("zero_torch.special.gammaln", return_value=torch.tensor([0.0, 0.0])):
            lp = g.log_prob(torch.tensor([1.5, 2.5]))
            assert lp.shape == (2,)

        # Beta
        beta = Beta(torch.tensor([2.0, 2.0]), torch.tensor([2.0, 2.0]))
        assert beta.rsample() is not None
        s = beta.sample((2,))
        assert s.shape == (2, 2)
        with patch("zero_torch.special.gammaln", return_value=torch.tensor([0.0, 0.0])):
            lp = beta.log_prob(torch.tensor([0.5, 0.5]))
            assert lp.shape == (2,)


def test_distributions_utils():
    from zero_torch.distributions.utils import _broadcast_shape

    assert _broadcast_shape((2, 3), (1, 3)) == (2, 3)
    assert _broadcast_shape((2, 1), (1, 3)) == (2, 3)
    assert _broadcast_shape((), (2, 3)) == (2, 3)
    assert _broadcast_shape((2, 3), ()) == (2, 3)

    with pytest.raises(RuntimeError):
        _broadcast_shape((2, 3), (3, 3))
