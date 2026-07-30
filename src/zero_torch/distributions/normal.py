import math

import zero_torch as torch

from .exp_family import ExponentialFamily
from .utils import _broadcast_shape


class Normal(ExponentialFamily):
    """Normal distribution."""

    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        """Initializes the distribution."""
        self.loc = loc
        self.scale = scale
        batch_shape = _broadcast_shape(loc.shape, scale.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    def rsample(self, sample_shape=None):
        """Generates a sample_shape shaped reparameterized sample."""
        if sample_shape is None:
            sample_shape = ()
        shape = tuple(sample_shape) + self.batch_shape
        return torch.randn(*shape) * self.scale + self.loc

    def log_prob(self, value):
        """Returns the log of the probability density function evaluated at `value`."""
        var = self.scale**2
        log_scale = torch.log(self.scale)
        return (
            -((value - self.loc) ** 2) / (2 * var)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )
