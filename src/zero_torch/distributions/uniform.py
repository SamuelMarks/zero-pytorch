import zero_torch as torch

from .distribution import Distribution
from .utils import _broadcast_shape


class Uniform(Distribution):
    """Uniform distribution."""

    has_rsample = True

    def __init__(self, low, high, validate_args=None):
        """Initializes the distribution."""
        self.low = low
        self.high = high
        batch_shape = _broadcast_shape(low.shape, high.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    def rsample(self, sample_shape=None):
        """Generates a sample_shape shaped reparameterized sample."""
        if sample_shape is None:
            sample_shape = ()
        shape = tuple(sample_shape) + self.batch_shape
        rand_tensor = torch.rand(*shape)
        return self.low + rand_tensor * (self.high - self.low)

    def log_prob(self, value):
        """Returns the log of the probability density function evaluated at `value`."""
        prob = torch.where(
            torch.logical_and(value >= self.low, value < self.high),
            torch.ones_like(value) / (self.high - self.low),
            torch.zeros_like(value),
        )
        return torch.log(prob)
