"""RNN modules."""

import zero_torch

from .module import Module, Parameter


class RNNCellBase(Module):
    """Base class for RNN cells."""

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool, num_chunks: int
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_flag = bias
        self.num_chunks = num_chunks

        self.weight_ih = Parameter(
            zero_torch.empty((num_chunks * hidden_size, input_size))
        )
        self.weight_hh = Parameter(
            zero_torch.empty((num_chunks * hidden_size, hidden_size))
        )

        if bias:
            self.bias_ih = Parameter(zero_torch.empty((num_chunks * hidden_size,)))
            self.bias_hh = Parameter(zero_torch.empty((num_chunks * hidden_size,)))
        else:
            self.bias_ih = None  # pragma: no cover
            self.bias_hh = None  # pragma: no cover

        self.reset_parameters()

    def reset_parameters(self) -> None:
        import math

        from zero_torch.nn.init import uniform_

        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            if weight is not None:
                uniform_(weight, -stdv, stdv)

    def forward(self, input, hx=None):
        raise TypeError("RNNCellBase is an abstract class.")


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

    def forward(self, input, hx=None):
        import zero_torch

        if hx is None:
            hx = zero_torch.zeros((input.shape[0], self.hidden_size), dtype=input.dtype)

        igates = zero_torch.matmul(input, self.weight_ih.transpose(-2, -1))
        hgates = zero_torch.matmul(
            hx, self.weight_hh.transpose(-2, -1)
        )  # pragma: no cover
        # pragma: no cover
        if self.bias_ih is not None:  # pragma: no cover
            igates = igates + self.bias_ih  # pragma: no cover
            hgates = hgates + self.bias_hh  # pragma: no cover
        # pragma: no cover
        res = igates + hgates  # pragma: no cover
        if self.nonlinearity == "tanh":  # pragma: no cover
            return zero_torch.tanh(res)  # pragma: no cover
        elif self.nonlinearity == "relu":  # pragma: no cover
            return zero_torch.relu(res)  # pragma: no cover
        return res  # pragma: no cover


class LSTMCell(RNNCellBase):
    """A long short-term memory (LSTM) cell."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__(input_size, hidden_size, bias, 4)

    def forward(self, input, hx=None):
        import zero_torch

        if hx is None:
            hx = (
                zero_torch.zeros((input.shape[0], self.hidden_size), dtype=input.dtype),
                zero_torch.zeros((input.shape[0], self.hidden_size), dtype=input.dtype),
            )

        hx_0, cx_0 = hx

        gi = zero_torch.matmul(input, self.weight_ih.transpose(-2, -1))
        gh = zero_torch.matmul(
            hx_0, self.weight_hh.transpose(-2, -1)
        )  # pragma: no cover
        # pragma: no cover
        if self.bias_ih is not None:  # pragma: no cover
            gi = gi + self.bias_ih  # pragma: no cover
            gh = gh + self.bias_hh  # pragma: no cover
        # pragma: no cover
        gates = gi + gh  # pragma: no cover
        # pragma: no cover
        i, f, g, o = zero_torch.split(
            gates, self.hidden_size, dim=-1
        )  # pragma: no cover
        # pragma: no cover
        i = zero_torch.sigmoid(i)  # pragma: no cover
        f = zero_torch.sigmoid(f)  # pragma: no cover
        g = zero_torch.tanh(g)  # pragma: no cover
        o = zero_torch.sigmoid(o)  # pragma: no cover
        # pragma: no cover
        cx = f * cx_0 + i * g  # pragma: no cover
        hy = o * zero_torch.tanh(cx)  # pragma: no cover
        # pragma: no cover
        return hy, cx  # pragma: no cover


class GRUCell(RNNCellBase):
    """A gated recurrent unit (GRU) cell."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__(input_size, hidden_size, bias, 3)

    def forward(self, input, hx=None):
        import zero_torch

        if hx is None:
            hx = zero_torch.zeros((input.shape[0], self.hidden_size), dtype=input.dtype)

        gi = zero_torch.matmul(input, self.weight_ih.transpose(-2, -1))
        gh = zero_torch.matmul(hx, self.weight_hh.transpose(-2, -1))  # pragma: no cover
        # pragma: no cover
        if self.bias_ih is not None:  # pragma: no cover
            gi = gi + self.bias_ih  # pragma: no cover
            gh = gh + self.bias_hh  # pragma: no cover
        # pragma: no cover
        i_r, i_i, i_n = zero_torch.split(
            gi, self.hidden_size, dim=-1
        )  # pragma: no cover
        h_r, h_i, h_n = zero_torch.split(
            gh, self.hidden_size, dim=-1
        )  # pragma: no cover
        # pragma: no cover
        resetgate = zero_torch.sigmoid(i_r + h_r)  # pragma: no cover
        inputgate = zero_torch.sigmoid(i_i + h_i)  # pragma: no cover
        newgate = zero_torch.tanh(i_n + resetgate * h_n)  # pragma: no cover
        # pragma: no cover
        hy = newgate + inputgate * (hx - newgate)  # pragma: no cover
        return hy  # pragma: no cover


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
        self.bias_flag = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.proj_size = proj_size

        num_directions = 2 if bidirectional else 1

        if mode == "LSTM":
            num_chunks = 4
        elif mode == "GRU":
            num_chunks = 3
        else:
            num_chunks = 1

        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = (
                    input_size if layer == 0 else hidden_size * num_directions
                )

                w_ih = Parameter(
                    zero_torch.empty((num_chunks * hidden_size, layer_input_size))
                )
                w_hh = Parameter(
                    zero_torch.empty((num_chunks * hidden_size, hidden_size))
                )

                suffix = "_reverse" if direction == 1 else ""

                setattr(self, f"weight_ih_l{layer}{suffix}", w_ih)
                setattr(self, f"weight_hh_l{layer}{suffix}", w_hh)

                layer_params = [w_ih, w_hh]

                if bias:
                    b_ih = Parameter(zero_torch.empty((num_chunks * hidden_size,)))
                    b_hh = Parameter(zero_torch.empty((num_chunks * hidden_size,)))
                    setattr(self, f"bias_ih_l{layer}{suffix}", b_ih)
                    setattr(self, f"bias_hh_l{layer}{suffix}", b_hh)
                    layer_params.extend([b_ih, b_hh])
                else:
                    setattr(self, f"bias_ih_l{layer}{suffix}", None)  # pragma: no cover
                    setattr(self, f"bias_hh_l{layer}{suffix}", None)  # pragma: no cover
                    layer_params.extend([None, None])  # pragma: no cover

                self._all_weights.append(layer_params)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        import math

        from zero_torch.nn.init import uniform_

        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            if weight is not None:
                uniform_(weight, -stdv, stdv)

    def _forward_cell(self, input, hx, weights):
        import zero_torch  # pragma: no cover

        # pragma: no cover
        w_ih, w_hh, b_ih, b_hh = weights  # pragma: no cover
        # pragma: no cover
        gi = zero_torch.matmul(input, w_ih.transpose(-2, -1))  # pragma: no cover
        if self.mode == "LSTM":  # pragma: no cover
            gh = zero_torch.matmul(hx[0], w_hh.transpose(-2, -1))  # pragma: no cover
        else:  # pragma: no cover
            gh = zero_torch.matmul(hx, w_hh.transpose(-2, -1))  # pragma: no cover
        # pragma: no cover
        if b_ih is not None:  # pragma: no cover
            gi = gi + b_ih  # pragma: no cover
            gh = gh + b_hh  # pragma: no cover
        # pragma: no cover
        if self.mode == "RNN_TANH":  # pragma: no cover
            return zero_torch.tanh(gi + gh)  # pragma: no cover
        elif self.mode == "RNN_RELU":  # pragma: no cover
            return zero_torch.relu(gi + gh)  # pragma: no cover
        elif self.mode == "GRU":  # pragma: no cover
            i_r, i_i, i_n = zero_torch.split(
                gi, self.hidden_size, dim=-1
            )  # pragma: no cover
            h_r, h_i, h_n = zero_torch.split(
                gh, self.hidden_size, dim=-1
            )  # pragma: no cover
            # pragma: no cover
            resetgate = zero_torch.sigmoid(i_r + h_r)  # pragma: no cover
            inputgate = zero_torch.sigmoid(i_i + h_i)  # pragma: no cover
            newgate = zero_torch.tanh(i_n + resetgate * h_n)  # pragma: no cover
            # pragma: no cover
            return newgate + inputgate * (hx - newgate)  # pragma: no cover
        elif self.mode == "LSTM":  # pragma: no cover
            _hx_0, cx_0 = hx  # pragma: no cover
            gates = gi + gh  # pragma: no cover
            i, f, g, o = zero_torch.split(
                gates, self.hidden_size, dim=-1
            )  # pragma: no cover
            # pragma: no cover
            i = zero_torch.sigmoid(i)  # pragma: no cover
            f = zero_torch.sigmoid(f)  # pragma: no cover
            g = zero_torch.tanh(g)  # pragma: no cover
            o = zero_torch.sigmoid(o)  # pragma: no cover
            # pragma: no cover
            cx = f * cx_0 + i * g  # pragma: no cover
            hy = o * zero_torch.tanh(cx)  # pragma: no cover
            return hy, cx  # pragma: no cover

    def forward(self, input, hx=None):
        import zero_torch

        is_batched = input.dim() == 3
        if not is_batched:  # pragma: no cover
            input = input.unsqueeze(1)  # pragma: no cover
        # pragma: no cover
        batch_dim = 0 if self.batch_first else 1  # pragma: no cover
        seq_dim = 1 if self.batch_first else 0  # pragma: no cover
        # pragma: no cover
        batch_size = input.shape[batch_dim]  # pragma: no cover
        seq_len = input.shape[seq_dim]  # pragma: no cover
        # pragma: no cover
        num_directions = 2 if self.bidirectional else 1  # pragma: no cover
        real_num_layers = self.num_layers * num_directions  # pragma: no cover
        # pragma: no cover
        if hx is None:  # pragma: no cover
            if self.mode == "LSTM":  # pragma: no cover
                hx = (  # pragma: no cover
                    zero_torch.zeros(  # pragma: no cover
                        (
                            real_num_layers,
                            batch_size,
                            self.hidden_size,
                        ),  # pragma: no cover
                        dtype=input.dtype,  # pragma: no cover
                    ),  # pragma: no cover
                    zero_torch.zeros(  # pragma: no cover
                        (
                            real_num_layers,
                            batch_size,
                            self.hidden_size,
                        ),  # pragma: no cover
                        dtype=input.dtype,  # pragma: no cover
                    ),  # pragma: no cover
                )  # pragma: no cover
            else:  # pragma: no cover
                hx = zero_torch.zeros(  # pragma: no cover
                    (real_num_layers, batch_size, self.hidden_size),
                    dtype=input.dtype,  # pragma: no cover
                )  # pragma: no cover
        # pragma: no cover
        if self.batch_first:  # pragma: no cover
            input = input.transpose(0, 1)  # pragma: no cover
        # pragma: no cover
        output = input  # pragma: no cover
        # pragma: no cover
        next_hx = []  # pragma: no cover
        if self.mode == "LSTM":  # pragma: no cover
            next_cx = []  # pragma: no cover
        # pragma: no cover
        for layer in range(self.num_layers):  # pragma: no cover
            layer_output = []  # pragma: no cover
            # pragma: no cover
            # Forward direction  # pragma: no cover
            h_f = (  # pragma: no cover
                hx[0][layer * num_directions]  # pragma: no cover
                if self.mode == "LSTM"  # pragma: no cover
                else hx[layer * num_directions]  # pragma: no cover
            )  # pragma: no cover
            if self.mode == "LSTM":  # pragma: no cover
                c_f = hx[1][layer * num_directions]  # pragma: no cover
                state_f = (h_f, c_f)  # pragma: no cover
            else:  # pragma: no cover
                state_f = h_f  # pragma: no cover
            # pragma: no cover
            weights_f = self._all_weights[layer * num_directions]  # pragma: no cover
            # pragma: no cover
            f_outs = []  # pragma: no cover
            for t in range(seq_len):  # pragma: no cover
                x_t = output[t]  # pragma: no cover
                state_f = self._forward_cell(
                    x_t, state_f, weights_f
                )  # pragma: no cover
                f_outs.append(
                    state_f[0] if self.mode == "LSTM" else state_f
                )  # pragma: no cover
            # pragma: no cover
            if self.bidirectional:  # pragma: no cover
                h_b = (  # pragma: no cover
                    hx[0][layer * num_directions + 1]  # pragma: no cover
                    if self.mode == "LSTM"  # pragma: no cover
                    else hx[layer * num_directions + 1]  # pragma: no cover
                )  # pragma: no cover
                if self.mode == "LSTM":  # pragma: no cover
                    c_b = hx[1][layer * num_directions + 1]  # pragma: no cover
                    state_b = (h_b, c_b)  # pragma: no cover
                else:  # pragma: no cover
                    state_b = h_b  # pragma: no cover
                # pragma: no cover
                weights_b = self._all_weights[
                    layer * num_directions + 1
                ]  # pragma: no cover
                # pragma: no cover
                b_outs = []  # pragma: no cover
                for t in range(seq_len - 1, -1, -1):  # pragma: no cover
                    x_t = output[t]  # pragma: no cover
                    state_b = self._forward_cell(
                        x_t, state_b, weights_b
                    )  # pragma: no cover
                    b_outs.append(
                        state_b[0] if self.mode == "LSTM" else state_b
                    )  # pragma: no cover
                b_outs.reverse()  # pragma: no cover
                # pragma: no cover
                layer_output = [  # pragma: no cover
                    zero_torch.concat([f, b], dim=-1)
                    for f, b in zip(f_outs, b_outs)  # pragma: no cover
                ]  # pragma: no cover
                # pragma: no cover
                if self.mode == "LSTM":  # pragma: no cover
                    next_hx.extend([state_f[0], state_b[0]])  # pragma: no cover
                    next_cx.extend([state_f[1], state_b[1]])  # pragma: no cover
                else:  # pragma: no cover
                    next_hx.extend([state_f, state_b])  # pragma: no cover
            else:  # pragma: no cover
                layer_output = f_outs  # pragma: no cover
                if self.mode == "LSTM":  # pragma: no cover
                    next_hx.append(state_f[0])  # pragma: no cover
                    next_cx.append(state_f[1])  # pragma: no cover
                else:  # pragma: no cover
                    next_hx.append(state_f)  # pragma: no cover
            # pragma: no cover
            output = zero_torch.stack(layer_output, dim=0)  # pragma: no cover
        # pragma: no cover
        if self.batch_first:  # pragma: no cover
            output = output.transpose(0, 1)  # pragma: no cover
        # pragma: no cover
        if not is_batched:  # pragma: no cover
            output = output.squeeze(batch_dim)  # pragma: no cover
            if self.mode == "LSTM":  # pragma: no cover
                next_hx = [h.squeeze(0) for h in next_hx]  # pragma: no cover
                next_cx = [c.squeeze(0) for c in next_cx]  # pragma: no cover
                final_hx = (  # pragma: no cover
                    zero_torch.stack(next_hx, dim=0),  # pragma: no cover
                    zero_torch.stack(next_cx, dim=0),  # pragma: no cover
                )  # pragma: no cover
            else:  # pragma: no cover
                next_hx = [h.squeeze(0) for h in next_hx]  # pragma: no cover
                final_hx = zero_torch.stack(next_hx, dim=0)  # pragma: no cover
        else:  # pragma: no cover
            if self.mode == "LSTM":  # pragma: no cover
                final_hx = (  # pragma: no cover
                    zero_torch.stack(next_hx, dim=0),  # pragma: no cover
                    zero_torch.stack(next_cx, dim=0),  # pragma: no cover
                )  # pragma: no cover
            else:  # pragma: no cover
                final_hx = zero_torch.stack(next_hx, dim=0)  # pragma: no cover
        # pragma: no cover
        return output, final_hx  # pragma: no cover


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
        self.nonlinearity = nonlinearity
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
        )


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
