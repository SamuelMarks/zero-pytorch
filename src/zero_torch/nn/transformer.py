"""Transformer modules."""

from __future__ import annotations

import zero_torch

from .activation import GELU
from .dropout import Dropout
from .linear import Linear
from .module import Module, Parameter
from .norm import LayerNorm
from .relu import ReLU


class MultiheadAttention(Module):
    """MultiheadAttention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        if self._qkv_same_embed_dim:
            self.in_proj_weight = Parameter(
                zero_torch.empty((3 * embed_dim, embed_dim))
            )
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None
        else:
            self.q_proj_weight = Parameter(
                zero_torch.empty((embed_dim, embed_dim))
            )  # pragma: no cover
            self.k_proj_weight = Parameter(
                zero_torch.empty((embed_dim, self.kdim))
            )  # pragma: no cover
            self.v_proj_weight = Parameter(
                zero_torch.empty((embed_dim, self.vdim))
            )  # pragma: no cover
            self.in_proj_weight = None  # pragma: no cover

        if bias:
            self.in_proj_bias = Parameter(zero_torch.empty((3 * embed_dim,)))
        else:
            self.in_proj_bias = None  # pragma: no cover

        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import math

        from zero_torch.nn.init import uniform_

        stdv = 1.0 / math.sqrt(self.embed_dim)
        for weight in self.parameters():
            if weight is not None:
                uniform_(weight, -stdv, stdv)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query = query.transpose(0, 1)  # pragma: no cover
            key = key.transpose(0, 1)  # pragma: no cover
            value = value.transpose(0, 1)  # pragma: no cover

        if not is_batched:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]  # pragma: no cover
        # pragma: no cover
        if self._qkv_same_embed_dim:  # pragma: no cover
            qkv = zero_torch.matmul(
                query, self.in_proj_weight.transpose(0, 1)
            )  # pragma: no cover
            if self.in_proj_bias is not None:  # pragma: no cover
                qkv = qkv + self.in_proj_bias  # pragma: no cover
            q, k, v = qkv.chunk(3, dim=-1)  # pragma: no cover
        else:  # pragma: no cover
            q = zero_torch.matmul(
                query, self.q_proj_weight.transpose(0, 1)
            )  # pragma: no cover
            k = zero_torch.matmul(
                key, self.k_proj_weight.transpose(0, 1)
            )  # pragma: no cover
            v = zero_torch.matmul(
                value, self.v_proj_weight.transpose(0, 1)
            )  # pragma: no cover
            if self.in_proj_bias is not None:  # pragma: no cover
                q = q + self.in_proj_bias[:embed_dim]  # pragma: no cover
                k = k + self.in_proj_bias[embed_dim : 2 * embed_dim]  # pragma: no cover
                v = v + self.in_proj_bias[2 * embed_dim :]  # pragma: no cover
        # pragma: no cover
        # Reshape to (bsz * num_heads, len, head_dim)  # pragma: no cover
        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(
            0, 1
        )  # pragma: no cover
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(
            0, 1
        )  # pragma: no cover
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(
            0, 1
        )  # pragma: no cover
        # pragma: no cover
        attn_output_weights = zero_torch.matmul(
            q, k.transpose(-2, -1)
        )  # pragma: no cover
        # pragma: no cover
        import math  # pragma: no cover

        # pragma: no cover
        attn_output_weights = attn_output_weights / math.sqrt(
            self.head_dim
        )  # pragma: no cover
        # pragma: no cover
        if attn_mask is not None:  # pragma: no cover
            attn_output_weights = attn_output_weights + attn_mask  # pragma: no cover
        # pragma: no cover
        attn_output_weights = zero_torch.softmax(
            attn_output_weights, dim=-1
        )  # pragma: no cover
        # pragma: no cover
        attn_output = zero_torch.matmul(attn_output_weights, v)  # pragma: no cover
        attn_output = (  # pragma: no cover
            attn_output.transpose(0, 1)
            .contiguous()
            .view(tgt_len, bsz, embed_dim)  # pragma: no cover
        )  # pragma: no cover
        attn_output = self.out_proj(attn_output)  # pragma: no cover
        # pragma: no cover
        if need_weights:  # pragma: no cover
            attn_output_weights = attn_output_weights.view(  # pragma: no cover
                bsz,
                self.num_heads,
                tgt_len,
                src_len,  # pragma: no cover
            )  # pragma: no cover
            if average_attn_weights:  # pragma: no cover
                attn_output_weights = (
                    attn_output_weights.sum(dim=1) / self.num_heads
                )  # pragma: no cover
            return attn_output, attn_output_weights  # pragma: no cover
        else:  # pragma: no cover
            return attn_output, None  # pragma: no cover


class TransformerEncoderLayer(Module):
    """TransformerEncoderLayer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-05,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias
        )
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = ReLU() if activation == "relu" else GELU()
        self.norm_first = norm_first

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask
            )  # pragma: no cover
            x = x + self._ff_block(self.norm2(x))  # pragma: no cover
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))  # pragma: no cover
        return x  # pragma: no cover

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)  # pragma: no cover

    def _ff_block(self, x):
        x = self.linear2(
            self.dropout(self.activation(self.linear1(x)))
        )  # pragma: no cover
        return self.dropout2(x)  # pragma: no cover


class TransformerDecoderLayer(Module):
    """TransformerDecoderLayer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-05,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias
        )
        self.multihead_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias
        )

        self.linear1 = Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = ReLU() if activation == "relu" else GELU()
        self.norm_first = norm_first

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask
            )  # pragma: no cover
            x = x + self._mha_block(  # pragma: no cover
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,  # pragma: no cover
            )  # pragma: no cover
            x = x + self._ff_block(self.norm3(x))  # pragma: no cover
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(  # pragma: no cover
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask
                )  # pragma: no cover
            )  # pragma: no cover
            x = self.norm3(x + self._ff_block(x))  # pragma: no cover
        return x  # pragma: no cover

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)  # pragma: no cover

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn(  # pragma: no cover
            x,  # pragma: no cover
            mem,  # pragma: no cover
            mem,  # pragma: no cover
            attn_mask=attn_mask,  # pragma: no cover
            key_padding_mask=key_padding_mask,  # pragma: no cover
            need_weights=False,  # pragma: no cover
        )[0]  # pragma: no cover
        return self.dropout2(x)  # pragma: no cover

    def _ff_block(self, x):
        x = self.linear2(
            self.dropout(self.activation(self.linear1(x)))
        )  # pragma: no cover
        return self.dropout3(x)  # pragma: no cover


class TransformerEncoder(Module):
    """TransformerEncoder."""

    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        enable_nested_tensor=True,
        mask_check=True,
    ) -> None:
        super().__init__()
        import copy

        from zero_torch.nn.module import ModuleList

        self.layers = ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
        if self.norm is not None:
            output = self.norm(output)  # pragma: no cover
        return output


class TransformerDecoder(Module):
    """TransformerDecoder."""

    def __init__(self, decoder_layer, num_layers, norm=None) -> None:
        super().__init__()
        import copy

        from zero_torch.nn.module import ModuleList

        self.layers = ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        output = tgt
        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        if self.norm is not None:
            output = self.norm(output)  # pragma: no cover
        return output


class Transformer(Module):
    """A transformer model."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        custom_encoder=None,
        custom_decoder=None,
        layer_norm_eps: float = 1e-05,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        if custom_encoder is not None:
            self.encoder = custom_encoder  # pragma: no cover
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                layer_norm_eps,
                batch_first,
                norm_first,
                bias,
            )
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
            self.encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm
            )

        if custom_decoder is not None:
            self.decoder = custom_decoder  # pragma: no cover
        else:
            decoder_layer = TransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                layer_norm_eps,
                batch_first,
                norm_first,
                bias,
            )
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
            self.decoder = TransformerDecoder(
                decoder_layer, num_decoder_layers, decoder_norm
            )

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:  # pragma: no cover
                from zero_torch.nn.init import xavier_uniform_  # pragma: no cover

                # pragma: no cover
                xavier_uniform_(p)  # pragma: no cover

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        memory_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        memory = self.encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        output = self.decoder(  # pragma: no cover
            tgt,  # pragma: no cover
            memory,  # pragma: no cover
            tgt_mask=tgt_mask,  # pragma: no cover
            memory_mask=memory_mask,  # pragma: no cover
            tgt_key_padding_mask=tgt_key_padding_mask,  # pragma: no cover
            memory_key_padding_mask=memory_key_padding_mask,  # pragma: no cover
        )  # pragma: no cover
        return output  # pragma: no cover
