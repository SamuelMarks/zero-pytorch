import zero_torch as torch

from .exp_family import ExponentialFamily
from .utils import _broadcast_shape


class Gamma(ExponentialFamily):
    """Gamma distribution."""

    has_rsample = True

    def __init__(self, concentration, rate, validate_args=None):
        """Initializes the distribution."""
        self.concentration = concentration
        self.rate = rate
        batch_shape = _broadcast_shape(concentration.shape, rate.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    def rsample(self, sample_shape=None):
        """Generates a sample_shape shaped reparameterized sample."""
        if sample_shape is None:
            sample_shape = ()
        shape = tuple(sample_shape) + self.batch_shape
        # Very simple normal approximation for Gamma (Wilson-Hilferty) for eager testing
        # X ~ Gamma(k, theta). k=concentration, theta=1/rate
        # X \approx k * theta * (1 - 1/(9k) + N(0,1) / sqrt(9k))^3
        n = torch.randn(*shape)
        k = self.concentration
        theta = 1.0 / self.rate
        res = k * theta * (1.0 - 1.0 / (9.0 * k) + n / torch.sqrt(9.0 * k)) ** 3
        return torch.maximum(res, torch.tensor(1e-6))

    def log_prob(self, value):
        """Returns the log of the probability density function evaluated at `value`."""
        # log(rate^conc / Gamma(conc) * value^(conc-1) * e^(-rate*value))
        return (
            self.concentration * torch.log(self.rate)
            + (self.concentration - 1) * torch.log(value)
            - self.rate * value
            - torch.special.gammaln(self.concentration)
        )
