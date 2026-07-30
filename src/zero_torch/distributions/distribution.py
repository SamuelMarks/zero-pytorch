import zero_torch as torch


class Distribution:
    """Base class for probability distributions."""

    has_rsample = False
    has_enumerate_support = False
    _validate_args = False

    def __init__(self, batch_shape=None, event_shape=None, validate_args=None):
        """Initializes the distribution."""
        if batch_shape is None:
            batch_shape = ()
        if event_shape is None:
            event_shape = ()
        self._batch_shape = tuple(batch_shape)
        self._event_shape = tuple(event_shape)

    @property
    def batch_shape(self):
        """Returns the shape over which parameters are batched."""
        return self._batch_shape

    @property
    def event_shape(self):
        """Returns the shape of a single sample (without batching)."""
        return self._event_shape

    def expand(self, batch_shape, _instance=None):
        """Returns a new distribution instance with batch dimensions expanded."""
        raise NotImplementedError

    def sample(self, sample_shape=None):
        """Generates a sample_shape shaped sample."""
        if sample_shape is None:
            sample_shape = ()
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=None):
        """Generates a sample_shape shaped reparameterized sample."""
        raise NotImplementedError

    def log_prob(self, value):
        """Returns the log of the probability density/mass function."""
        raise NotImplementedError

    def cdf(self, value):
        """Returns the cumulative density/mass function."""
        raise NotImplementedError

    def icdf(self, value):
        """Returns the inverse cumulative density/mass function."""
        raise NotImplementedError

    def enumerate_support(self, expand=True):
        """Returns tensor containing all values supported by a discrete distribution."""
        raise NotImplementedError

    def entropy(self):
        """Returns entropy of distribution, batched over batch_shape."""
        raise NotImplementedError
