import zero_torch as torch

from .distribution import Distribution


class Categorical(Distribution):
    """Categorical distribution."""

    has_rsample = False

    def __init__(self, probs=None, logits=None, validate_args=None):
        """Initializes the distribution."""
        if probs is None and logits is None:
            raise ValueError("Either probs or logits must be specified")

        if probs is not None:
            if len(probs.shape) < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            self.probs = probs / torch.sum(probs, axis=-1, keepdims=True)
            self.logits = torch.log(self.probs)
        else:
            if len(logits.shape) < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            self.logits = logits - torch.logsumexp(logits, axis=-1, keepdims=True)
            self.probs = torch.exp(self.logits)

        self._param = self.probs if probs is not None else self.logits
        batch_shape = self._param.shape[:-1]
        super().__init__(batch_shape, validate_args=validate_args)

    def sample(self, sample_shape=None):
        """Generates a sample_shape shaped sample."""
        if sample_shape is None:
            sample_shape = ()
        shape = tuple(sample_shape) + self.batch_shape
        # This is a bit tricky without a dedicated categorical sampler.
        # But ml-switcheroo-compiler has stateless_categorical if we want, or we can just use Gumbel-max trick
        # Gumbel-max: argmax(logits - log(-log(U)))
        u = torch.rand(*(shape + (self.logits.shape[-1],)))
        gumbel = -torch.log(-torch.log(u))
        return torch.argmax(self.logits + gumbel, axis=-1)

    def log_prob(self, value):
        """Returns the log of the probability density function evaluated at `value`."""
        # Gather the log prob for the given class index
        # PyTorch equivalent: self.logits.gather(-1, value.unsqueeze(-1)).squeeze(-1)
        value = torch.cast(value, torch.int32)
        logits = torch.broadcast_to(self.logits, value.shape + (self.logits.shape[-1],))
        return torch.gather(logits, -1, value.unsqueeze(-1)).squeeze(-1)
