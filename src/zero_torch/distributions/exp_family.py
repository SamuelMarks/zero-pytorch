from .distribution import Distribution


class ExponentialFamily(Distribution):
    """ExponentialFamily distribution."""

    @property
    def _natural_params(self):
        """Returns the natural parameters."""
        raise NotImplementedError

    def _log_normalizer(self, *natural_params):
        """Returns the log normalizer."""
        raise NotImplementedError

    @property
    def _mean_carrier_measure(self):
        """Returns the mean carrier measure."""
        raise NotImplementedError

    def entropy(self):
        """Method to compute the entropy for ExponentialFamily distributions."""
        raise NotImplementedError
