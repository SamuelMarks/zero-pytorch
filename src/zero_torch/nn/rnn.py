"""RNN modules."""

from .module import Module


class RNNBase(Module):
    """Base class for RNNs."""

    def __init__(
        self,
        mode: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.proj_size = proj_size

    def forward(self, input, hx=None):
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class RNN(RNNBase):
    """Applies a multi-layer Elman RNN."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__(
            "RNN",
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
        )
        self.nonlinearity = nonlinearity


class LSTM(RNNBase):
    """Applies a multi-layer LSTM."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
    ) -> None:
        super().__init__(
            "LSTM",
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            proj_size,
        )


class GRU(RNNBase):
    """Applies a multi-layer GRU."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__(
            "GRU",
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
        )


class RNNCellBase(Module):
    """Base class for RNN cells."""

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool, num_chunks: int
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_chunks = num_chunks

    def forward(self, input, hx=None):
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class RNNCell(RNNCellBase):
    """An Elman RNN cell."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
    ) -> None:
        super().__init__(input_size, hidden_size, bias, 1)
        self.nonlinearity = nonlinearity


class LSTMCell(RNNCellBase):
    """A long short-term memory (LSTM) cell."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__(input_size, hidden_size, bias, 4)


class GRUCell(RNNCellBase):
    """A gated recurrent unit (GRU) cell."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__(input_size, hidden_size, bias, 3)
