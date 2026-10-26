"""RNN modules."""

from __future__ import annotations

import math

import zero_torch
from zero_torch._types import DeviceLike, DTypeLike
from zero_torch.nn.init import uniform_
from zero_torch.tensor import Tensor

from .module import Module, Parameter


class RNNCellBase(Module):
    """Base class for recurrent neural network cells (`RNNCell`, `LSTMCell`, `GRUCell`)."""

    __constants__: list[str] = ["input_size", "hidden_size", "bias"]
    weight_ih: Parameter | None = None
    weight_hh: Parameter | None = None
    num_chunks: int = 1

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        num_chunks: int,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes RNNCellBase.

        Args:
            input_size: The number of expected features in the input `x`.
            hidden_size: The number of features in the hidden state `h`.
            bias: If False, then the layer does not use bias weights `b_ih` and `b_hh`.
            num_chunks: Number of gate chunks (`1` for RNN, `3` for GRU, `4` for LSTM).
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.bias_flag: bool = bias
        self.num_chunks: int = num_chunks

        self.weight_ih: Parameter = Parameter(
            zero_torch.empty((num_chunks * hidden_size, input_size))
        )
        self.weight_hh: Parameter = Parameter(
            zero_torch.empty((num_chunks * hidden_size, hidden_size))
        )
        self.register_parameter("weight_ih", self.weight_ih)
        self.register_parameter("weight_hh", self.weight_hh)

        self.bias_ih: Parameter | None
        self.bias_hh: Parameter | None
        if bias:
            self.bias_ih = Parameter(zero_torch.empty((num_chunks * hidden_size,)))
            self.bias_hh = Parameter(zero_torch.empty((num_chunks * hidden_size,)))
        else:
            self.bias_ih = None
            self.bias_hh = None
        self.register_parameter("bias_ih", self.bias_ih)
        self.register_parameter("bias_hh", self.bias_hh)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets all learnable weights and biases using uniform initialization."""
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0.0
        for weight in self.parameters():
            uniform_(weight, -stdv, stdv)

    def forward(
        self,
        input: Tensor,
        hx: Tensor | tuple[Tensor, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Raises TypeError because `RNNCellBase` is an abstract base class.

        Args:
            input: Input tensor.
            hx: Initial hidden state.

        Returns:
            Never returns.
        """
        _ = (input, hx)
        raise TypeError("RNNCellBase is an abstract class.")


class RNNCell(RNNCellBase):
    """An Elman RNN cell with tanh or ReLU non-linearity."""

    __constants__: list[str] = ["input_size", "hidden_size", "bias", "nonlinearity"]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes RNNCell.

        Args:
            input_size: The number of expected features in the input `x`.
            hidden_size: The number of features in the hidden state `h`.
            bias: If False, then the layer does not use bias weights.
            nonlinearity: The non-linearity to use (`'tanh'` or `'relu'`).
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        super().__init__(input_size, hidden_size, bias, 1, device=device, dtype=dtype)
        self.nonlinearity: str = nonlinearity

    def forward(
        self,
        input: Tensor,
        hx: Tensor | tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Computes a single Elman RNN cell step.

        Args:
            input: Input tensor of shape `(batch, input_size)` or `(input_size,)`.
            hx: Hidden state tensor of shape `(batch, hidden_size)` or `(hidden_size,)`.

        Returns:
            Next hidden state tensor `h'`.
        """
        if isinstance(hx, tuple):
            raise ValueError("Expected a Tensor as hidden state, got a tuple")
        if hx is None:
            h_prev = zero_torch.zeros(
                (input.shape[0], self.hidden_size), dtype=input.dtype
            )
        else:
            h_prev = hx

        assert self.weight_ih is not None and self.weight_hh is not None
        igates = zero_torch.matmul(input, self.weight_ih.transpose(-2, -1))
        hgates = zero_torch.matmul(h_prev, self.weight_hh.transpose(-2, -1))
        if self.bias_ih is not None and self.bias_hh is not None:
            igates = igates + self.bias_ih
            hgates = hgates + self.bias_hh
        res = igates + hgates
        if self.nonlinearity == "relu":
            return zero_torch.nn.functional.relu(res)
        return zero_torch.tanh(res)


class LSTMCell(RNNCellBase):
    """A long short-term memory (LSTM) cell."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes LSTMCell.

        Args:
            input_size: The number of expected features in the input `x`.
            hidden_size: The number of features in the hidden state `h`.
            bias: If False, then the layer does not use bias weights.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        super().__init__(input_size, hidden_size, bias, 4, device=device, dtype=dtype)

    def forward(
        self,
        input: Tensor,
        hx: Tensor | tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Computes a single LSTM cell step.

        Args:
            input: Input tensor of shape `(batch, input_size)`.
            hx: Optional tuple `(h_0, c_0)` of hidden and cell states.

        Returns:
            Tuple `(h_1, c_1)` of next hidden and cell state tensors.
        """
        if hx is None or not isinstance(hx, tuple):
            hx_0 = zero_torch.zeros(
                (input.shape[0], self.hidden_size), dtype=input.dtype
            )
            cx_0 = zero_torch.zeros(
                (input.shape[0], self.hidden_size), dtype=input.dtype
            )
        else:
            hx_0, cx_0 = hx

        assert self.weight_ih is not None and self.weight_hh is not None
        gi = zero_torch.matmul(input, self.weight_ih.transpose(-2, -1))
        gh = zero_torch.matmul(hx_0, self.weight_hh.transpose(-2, -1))
        if self.bias_ih is not None and self.bias_hh is not None:
            gi = gi + self.bias_ih
            gh = gh + self.bias_hh
        gates = gi + gh
        chunks = gates.chunk(4, dim=-1)
        i = zero_torch.sigmoid(chunks[0])
        f = zero_torch.sigmoid(chunks[1])
        g = zero_torch.tanh(chunks[2])
        o = zero_torch.sigmoid(chunks[3])
        cx = f * cx_0 + i * g
        hy = o * zero_torch.tanh(cx)
        return hy, cx


class GRUCell(RNNCellBase):
    """A gated recurrent unit (GRU) cell."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes GRUCell.

        Args:
            input_size: The number of expected features in the input `x`.
            hidden_size: The number of features in the hidden state `h`.
            bias: If False, then the layer does not use bias weights.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        super().__init__(input_size, hidden_size, bias, 3, device=device, dtype=dtype)

    def forward(
        self,
        input: Tensor,
        hx: Tensor | tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Computes a single GRU cell step.

        Args:
            input: Input tensor of shape `(batch, input_size)`.
            hx: Optional hidden state tensor `h_0`.

        Returns:
            Next hidden state tensor `h_1`.
        """
        if isinstance(hx, tuple):
            raise ValueError("Expected a Tensor as hidden state, got a tuple")
        if hx is None:
            h_prev = zero_torch.zeros(
                (input.shape[0], self.hidden_size), dtype=input.dtype
            )
        else:
            h_prev = hx

        assert self.weight_ih is not None and self.weight_hh is not None
        gi = zero_torch.matmul(input, self.weight_ih.transpose(-2, -1))
        gh = zero_torch.matmul(h_prev, self.weight_hh.transpose(-2, -1))
        if self.bias_ih is not None and self.bias_hh is not None:
            gi = gi + self.bias_ih
            gh = gh + self.bias_hh
        i_chunks = gi.chunk(3, dim=-1)
        h_chunks = gh.chunk(3, dim=-1)
        resetgate = zero_torch.sigmoid(i_chunks[0] + h_chunks[0])
        inputgate = zero_torch.sigmoid(i_chunks[1] + h_chunks[1])
        newgate = zero_torch.tanh(i_chunks[2] + resetgate * h_chunks[2])
        return newgate + inputgate * (h_prev - newgate)


class RNNBase(Module):
    """Base class for multi-layer recurrent neural networks (`RNN`, `LSTM`, `GRU`)."""

    __constants__: list[str] = [
        "mode",
        "input_size",
        "hidden_size",
        "num_layers",
        "bias",
        "batch_first",
        "dropout",
        "bidirectional",
        "proj_size",
    ]
    __jit_unused_properties__: list[str] = ["all_weights"]
    mode: str = "RNN_TANH"
    proj_size: int = 0

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
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes RNNBase.

        Args:
            mode: RNN cell mode (`'RNN_TANH'`, `'RNN_RELU'`, `'LSTM'`, or `'GRU'`).
            input_size: The number of expected features in the input `x`.
            hidden_size: The number of features in the hidden state `h`.
            num_layers: Number of recurrent layers.
            bias: If False, then the layer does not use bias weights.
            batch_first: If True, input and output tensors are provided as `(batch, seq, feature)`.
            dropout: If non-zero, introduces a Dropout layer on the outputs of each RNN layer.
            bidirectional: If True, becomes a bidirectional RNN.
            proj_size: If `> 0`, will use LSTM with projections of corresponding size.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.mode: str = mode
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.bias_flag: bool = bias
        self.batch_first: bool = batch_first
        self.dropout: float = float(dropout)
        self.bidirectional: bool = bidirectional
        self.proj_size: int = proj_size

        num_directions = 2 if bidirectional else 1
        if mode == "LSTM":
            num_chunks = 4
        elif mode == "GRU":
            num_chunks = 3
        else:
            num_chunks = 1

        self._all_weights: list[
            tuple[
                Parameter,
                Parameter,
                Parameter | None,
                Parameter | None,
                Parameter | None,
            ]
        ] = []
        real_hidden_size = proj_size if proj_size > 0 else hidden_size
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = (
                    input_size if layer == 0 else real_hidden_size * num_directions
                )
                w_ih = Parameter(
                    zero_torch.empty((num_chunks * hidden_size, layer_input_size))
                )
                w_hh = Parameter(
                    zero_torch.empty((num_chunks * hidden_size, real_hidden_size))
                )
                suffix = "_reverse" if direction == 1 else ""
                self.register_parameter(f"weight_ih_l{layer}{suffix}", w_ih)
                self.register_parameter(f"weight_hh_l{layer}{suffix}", w_hh)

                b_ih: Parameter | None
                b_hh: Parameter | None
                if bias:
                    b_ih = Parameter(zero_torch.empty((num_chunks * hidden_size,)))
                    b_hh = Parameter(zero_torch.empty((num_chunks * hidden_size,)))
                else:
                    b_ih = None
                    b_hh = None
                self.register_parameter(f"bias_ih_l{layer}{suffix}", b_ih)
                self.register_parameter(f"bias_hh_l{layer}{suffix}", b_hh)

                w_hr: Parameter | None = None
                if proj_size > 0:
                    w_hr = Parameter(zero_torch.empty((proj_size, hidden_size)))
                    self.register_parameter(f"weight_hr_l{layer}{suffix}", w_hr)
                self._all_weights.append((w_ih, w_hh, b_ih, b_hh, w_hr))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets all learnable weights and biases using uniform initialization."""
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0.0
        for weight in self.parameters():
            uniform_(weight, -stdv, stdv)

    def _step_gate(
        self,
        x_t: Tensor,
        h_prev: Tensor,
        c_prev: Tensor,
        weights: tuple[
            Parameter,
            Parameter,
            Parameter | None,
            Parameter | None,
            Parameter | None,
        ],
    ) -> tuple[Tensor, Tensor]:
        """Executes a single time-step update for one layer/direction.

        Args:
            x_t: Input slice at time `t` of shape `(batch, layer_input_size)`.
            h_prev: Previous hidden state of shape `(batch, real_hidden_size)`.
            c_prev: Previous cell state of shape `(batch, hidden_size)`.
            weights: Tuple `(w_ih, w_hh, b_ih, b_hh, w_hr)` for the current layer/direction.

        Returns:
            Tuple `(h_next, c_next)` of updated hidden and cell state tensors.
        """
        w_ih, w_hh, b_ih, b_hh, w_hr = weights
        gi = zero_torch.matmul(x_t, w_ih.transpose(-2, -1))
        gh = zero_torch.matmul(h_prev, w_hh.transpose(-2, -1))
        if b_ih is not None and b_hh is not None:
            gi = gi + b_ih
            gh = gh + b_hh

        if self.mode == "RNN_RELU":
            return zero_torch.nn.functional.relu(gi + gh), c_prev
        if self.mode == "GRU":
            i_chunks = gi.chunk(3, dim=-1)
            h_chunks = gh.chunk(3, dim=-1)
            resetgate = zero_torch.sigmoid(i_chunks[0] + h_chunks[0])
            inputgate = zero_torch.sigmoid(i_chunks[1] + h_chunks[1])
            newgate = zero_torch.tanh(i_chunks[2] + resetgate * h_chunks[2])
            return newgate + inputgate * (h_prev - newgate), c_prev
        if self.mode == "LSTM":
            gates = gi + gh
            chunks = gates.chunk(4, dim=-1)
            i = zero_torch.sigmoid(chunks[0])
            f = zero_torch.sigmoid(chunks[1])
            g = zero_torch.tanh(chunks[2])
            o = zero_torch.sigmoid(chunks[3])
            cx = f * c_prev + i * g
            hy = o * zero_torch.tanh(cx)
            if w_hr is not None:
                hy = zero_torch.matmul(hy, w_hr.transpose(-2, -1))
            return hy, cx
        return zero_torch.tanh(gi + gh), c_prev

    def forward(
        self,
        input: Tensor,
        hx: Tensor | tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor]]:
        """Applies the multi-layer recurrent network to `input`.

        Args:
            input: Input sequence tensor of shape `(L, N, H_in)`, `(N, L, H_in)` if `batch_first=True`, or `(L, H_in)`.
            hx: Optional initial hidden state (or `(h_0, c_0)` tuple for LSTM).

        Returns:
            Tuple `(output, h_n)` (or `(output, (h_n, c_n))` for LSTM).
        """
        is_batched = input.dim() == 3
        seq_input = (
            input.transpose(0, 1) if (is_batched and self.batch_first) else input
        )
        if not is_batched:
            seq_input = seq_input.unsqueeze(1)

        seq_len = seq_input.shape[0]
        batch_size = seq_input.shape[1]
        num_directions = 2 if self.bidirectional else 1
        real_num_layers = self.num_layers * num_directions
        real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size

        h_0: Tensor
        c_0: Tensor
        if hx is None:
            h_0 = zero_torch.zeros(
                (real_num_layers, batch_size, real_hidden_size),
                dtype=seq_input.dtype,
            )
            c_0 = zero_torch.zeros(
                (real_num_layers, batch_size, self.hidden_size),
                dtype=seq_input.dtype,
            )
        elif isinstance(hx, tuple):
            if self.mode != "LSTM":
                raise ValueError(
                    f"{self.mode} expected a Tensor as hidden state, got a tuple"
                )
            h_0, c_0 = hx
        else:
            h_0 = hx
            c_0 = zero_torch.zeros(
                (real_num_layers, batch_size, self.hidden_size),
                dtype=seq_input.dtype,
            )

        output = seq_input
        next_hx: list[Tensor] = []
        next_cx: list[Tensor] = []

        for layer in range(self.num_layers):
            f_idx = layer * num_directions
            h_f = h_0[f_idx]
            c_f = c_0[f_idx]
            weights_f = self._all_weights[f_idx]

            f_outs: list[Tensor] = []
            for t in range(seq_len):
                h_f, c_f = self._step_gate(output[t], h_f, c_f, weights_f)
                f_outs.append(h_f)

            next_hx.append(h_f)
            next_cx.append(c_f)

            if self.bidirectional:
                b_idx = layer * num_directions + 1
                h_b = h_0[b_idx]
                c_b = c_0[b_idx]
                weights_b = self._all_weights[b_idx]
                b_outs: list[Tensor] = []
                for t in range(seq_len - 1, -1, -1):
                    h_b, c_b = self._step_gate(output[t], h_b, c_b, weights_b)
                    b_outs.append(h_b)
                b_outs.reverse()
                layer_output = [
                    zero_torch.concatenate([f, b], dim=-1)
                    for f, b in zip(f_outs, b_outs)
                ]
                next_hx.append(h_b)
                next_cx.append(c_b)
            else:
                layer_output = f_outs

            output = zero_torch.stack(layer_output, dim=0)
            if self.dropout > 0.0 and self.training and layer < self.num_layers - 1:
                output = zero_torch.nn.functional.dropout(
                    output, p=self.dropout, training=self.training
                )

        if is_batched and self.batch_first:
            output = output.transpose(0, 1)

        if not is_batched:
            output = output.squeeze(1)
            next_hx = [h.squeeze(0) for h in next_hx]
            next_cx = [c.squeeze(0) for c in next_cx]

        h_n = zero_torch.stack(next_hx, dim=0)
        if self.mode == "LSTM":
            c_n = zero_torch.stack(next_cx, dim=0)
            return output, (h_n, c_n)
        return output, h_n


class RNN(RNNBase):
    """Applies a multi-layer Elman RNN with `tanh` or `relu` non-linearity to an input sequence."""

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
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes RNN.

        Args:
            input_size: The number of expected features in the input `x`.
            hidden_size: The number of features in the hidden state `h`.
            num_layers: Number of recurrent layers.
            nonlinearity: The non-linearity to use (`'tanh'` or `'relu'`).
            bias: If False, then the layer does not use bias weights.
            batch_first: If True, input and output tensors are provided as `(batch, seq, feature)`.
            dropout: If non-zero, introduces a Dropout layer on the outputs of each RNN layer.
            bidirectional: If True, becomes a bidirectional RNN.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        self.nonlinearity: str = nonlinearity
        mode = "RNN_TANH" if nonlinearity == "tanh" else "RNN_RELU"
        super().__init__(
            mode,
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            device=device,
            dtype=dtype,
        )


class LSTM(RNNBase):
    """Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence."""

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
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes LSTM.

        Args:
            input_size: The number of expected features in the input `x`.
            hidden_size: The number of features in the hidden state `h`.
            num_layers: Number of recurrent layers.
            bias: If False, then the layer does not use bias weights.
            batch_first: If True, input and output tensors are provided as `(batch, seq, feature)`.
            dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer.
            bidirectional: If True, becomes a bidirectional LSTM.
            proj_size: If `> 0`, uses LSTM with projections of corresponding size.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
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
            device=device,
            dtype=dtype,
        )


class GRU(RNNBase):
    """Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes GRU.

        Args:
            input_size: The number of expected features in the input `x`.
            hidden_size: The number of features in the hidden state `h`.
            num_layers: Number of recurrent layers.
            bias: If False, then the layer does not use bias weights.
            batch_first: If True, input and output tensors are provided as `(batch, seq, feature)`.
            dropout: If non-zero, introduces a Dropout layer on the outputs of each GRU layer.
            bidirectional: If True, becomes a bidirectional GRU.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        super().__init__(
            "GRU",
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            device=device,
            dtype=dtype,
        )
