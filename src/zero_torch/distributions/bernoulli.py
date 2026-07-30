import zero_torch as torch

from .exp_family import ExponentialFamily


class Bernoulli(ExponentialFamily):
    """Bernoulli distribution."""

    has_rsample = True

    def __init__(self, probs=None, logits=None, validate_args=None):
        """Initializes the distribution."""
        if probs is None and logits is None:
            raise ValueError("Either probs or logits must be specified")
        if probs is not None:
            self.probs = probs
            self.logits = torch.log(probs / (1 - probs))
            batch_shape = probs.shape
        else:
            self.logits = logits
            self.probs = torch.nn.functional.sigmoid(logits)
            batch_shape = logits.shape
        super().__init__(batch_shape, validate_args=validate_args)

    def rsample(self, sample_shape=None):
        """Generates a sample_shape shaped reparameterized sample."""
        if sample_shape is None:
            sample_shape = ()
        shape = tuple(sample_shape) + self.batch_shape
        return torch.cast(torch.rand(*shape) < self.probs, torch.float32)

    def sample(self, sample_shape=None):
        """Generates a sample_shape shaped sample."""
        return self.rsample(sample_shape).detach()

    def log_prob(self, value):
        """Returns the log of the probability density function evaluated at `value`."""
        logits = torch.broadcast_to(self.logits, value.shape)
        # Using the standard numerically stable binary cross entropy logic:
        # log_prob = value * logits - log(1 + exp(logits))
        return value * logits - torch.nn.functional.softplus(logits)
