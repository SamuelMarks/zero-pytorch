import zero_torch as torch

from .exp_family import ExponentialFamily
from .utils import _broadcast_shape


class Beta(ExponentialFamily):
    """Beta distribution."""

    has_rsample = True

    def __init__(self, concentration1, concentration0, validate_args=None):
        """Initializes the distribution."""
        self.concentration1 = concentration1
        self.concentration0 = concentration0
        batch_shape = _broadcast_shape(concentration1.shape, concentration0.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    def rsample(self, sample_shape=None):
        """Generates a sample_shape shaped reparameterized sample."""
        if sample_shape is None:
            sample_shape = ()
        # Use Gamma approximations to sample Beta: X = G1 / (G1 + G0)
        from .gamma import Gamma

        g1 = Gamma(self.concentration1, torch.ones_like(self.concentration1)).rsample(
            sample_shape
        )
        g0 = Gamma(self.concentration0, torch.ones_like(self.concentration0)).rsample(
            sample_shape
        )
        return g1 / (g1 + g0)

    def log_prob(self, value):
        """Returns the log of the probability density function evaluated at `value`."""
        # log(value^(c1-1) * (1-value)^(c0-1) / B(c1, c0))
        # B(c1, c0) = Gamma(c1) * Gamma(c0) / Gamma(c1 + c0)
        # log B(c1, c0) = gammaln(c1) + gammaln(c0) - gammaln(c1 + c0)
        log_B = (
            torch.special.gammaln(self.concentration1)
            + torch.special.gammaln(self.concentration0)
            - torch.special.gammaln(self.concentration1 + self.concentration0)
        )
        return (
            (self.concentration1 - 1) * torch.log(value)
            + (self.concentration0 - 1) * torch.log(1 - value)
            - log_B
        )
