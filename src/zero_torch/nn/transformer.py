"""Transformer modules."""

from __future__ import annotations

import copy
import math

import zero_torch
from zero_torch._types import DeviceLike, DTypeLike
from zero_torch.nn.init import uniform_, xavier_uniform_
from zero_torch.tensor import Tensor

from .activation import GELU
from .dropout import Dropout
from .linear import Linear
from .module import Module, ModuleList, Parameter
from .norm import LayerNorm
from .relu import ReLU


class MultiheadAttention(Module):
    """Allows the model to jointly attend to information from different representation subspaces."""

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
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes MultiheadAttention.

        Args:
            embed_dim: Total dimension of the model.
            num_heads: Number of parallel attention heads.
            dropout: Dropout probability on `attn_output_weights`.
            bias: If specified, adds bias to input / output projection layers.
            add_bias_kv: If specified, adds bias to the key and value sequences at `dim=0`.
            add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at `dim=1`.
            kdim: Total number of features for keys.
            vdim: Total number of features for values.
            batch_first: If True, input and output tensors are provided as `(batch, seq, feature)`.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (add_bias_kv, add_zero_attn, device, dtype)
        super().__init__()
        self.embed_dim: int = embed_dim
        self.kdim: int = kdim if kdim is not None else embed_dim
        self.vdim: int = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim: bool = (
            self.kdim == embed_dim and self.vdim == embed_dim
        )

        self.num_heads: int = num_heads
        self.dropout: float = dropout
        self.batch_first: bool = batch_first
        self.head_dim: int = embed_dim // num_heads

        self.in_proj_weight: Parameter | None
        self.q_proj_weight: Parameter | None
        self.k_proj_weight: Parameter | None
        self.v_proj_weight: Parameter | None
        if self._qkv_same_embed_dim:
            self.in_proj_weight = Parameter(
                zero_torch.empty((3 * embed_dim, embed_dim))
            )
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None
        else:
            self.q_proj_weight = Parameter(zero_torch.empty((embed_dim, embed_dim)))
            self.k_proj_weight = Parameter(zero_torch.empty((embed_dim, self.kdim)))
            self.v_proj_weight = Parameter(zero_torch.empty((embed_dim, self.vdim)))
            self.in_proj_weight = None

        self.in_proj_bias: Parameter | None
        if bias:
            self.in_proj_bias = Parameter(zero_torch.empty((3 * embed_dim,)))
        else:
            self.in_proj_bias = None

        self.out_proj: Linear = Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets all learnable parameters using uniform initialization."""
        stdv = 1.0 / math.sqrt(self.embed_dim)
        for weight in self.parameters():
            uniform_(weight, -stdv, stdv)

    def __call__(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Invokes `forward` on MultiheadAttention.

        Args:
            query: Query embeddings.
            key: Key embeddings.
            value: Value embeddings.
            key_padding_mask: Mask for keys per batch.
            need_weights: If True, returns `attn_output_weights`.
            attn_mask: 2D or 3D mask preventing attention to certain positions.
            average_attn_weights: If True, returns attention weights averaged across heads.
            is_causal: If True, applies a causal mask.

        Returns:
            Tuple `(attn_output, attn_output_weights)`.
        """
        return self.forward(
            query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Computes multi-head attention.

        Args:
            query: Query embeddings of shape `(L, E_q)`, `(L, N, E_q)`, or `(N, L, E_q)`.
            key: Key embeddings of shape `(S, E_k)`, `(S, N, E_k)`, or `(N, S, E_k)`.
            value: Value embeddings of shape `(S, E_v)`, `(S, N, E_v)`, or `(N, S, E_v)`.
            key_padding_mask: Optional mask for padding keys.
            need_weights: If True, returns `attn_output_weights` in addition to `attn_outputs`.
            attn_mask: Optional 2D or 3D attention mask.
            average_attn_weights: If True, returns attention weights averaged across heads.
            is_causal: If True, applies a causal mask.

        Returns:
            Tuple `(attn_output, attn_output_weights)`.
        """
        _ = (key_padding_mask, is_causal)
        k_in = key if key is not None else query
        v_in = value if value is not None else query

        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query = query.transpose(0, 1)
            k_in = k_in.transpose(0, 1)
            v_in = v_in.transpose(0, 1)

        if not is_batched:
            query = query.unsqueeze(1)
            k_in = k_in.unsqueeze(1)
            v_in = v_in.unsqueeze(1)

        tgt_len, bsz, embed_dim = query.shape
        src_len = k_in.shape[0]

        if self._qkv_same_embed_dim:
            assert self.in_proj_weight is not None
            qkv = zero_torch.matmul(query, self.in_proj_weight.transpose(0, 1))
            if self.in_proj_bias is not None:
                qkv = qkv + self.in_proj_bias
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            assert self.q_proj_weight is not None
            assert self.k_proj_weight is not None
            assert self.v_proj_weight is not None
            q = zero_torch.matmul(query, self.q_proj_weight.transpose(0, 1))
            k = zero_torch.matmul(k_in, self.k_proj_weight.transpose(0, 1))
            v = zero_torch.matmul(v_in, self.v_proj_weight.transpose(0, 1))
            if self.in_proj_bias is not None:
                b_q, b_k, b_v = self.in_proj_bias.chunk(3, dim=-1)
                q = q + b_q
                k = k + b_k
                v = v + b_v

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_output_weights = zero_torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )

        if attn_mask is not None:
            attn_output_weights = attn_output_weights + attn_mask

        attn_output_weights = zero_torch.nn.functional.softmax(
            attn_output_weights, dim=-1
        )
        attn_output = zero_torch.matmul(attn_output_weights, v)
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        )
        attn_output = self.out_proj(attn_output)

        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(0, 1)
        elif not is_batched:
            attn_output = attn_output.squeeze(1)

        if need_weights:
            weights_4d = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                weights_out = weights_4d.sum(dim=1) / float(self.num_heads)
            else:
                weights_out = weights_4d
            if not is_batched:
                weights_out = weights_out.squeeze(0)
            return attn_output, weights_out
        return attn_output, None


class TransformerEncoderLayer(Module):
    """TransformerEncoderLayer is made up of self-attention and feedforward network."""

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
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes TransformerEncoderLayer.

        Args:
            d_model: The number of expected features in the input.
            nhead: The number of heads in the multiheadattention models.
            dim_feedforward: The dimension of the feedforward network model.
            dropout: The dropout value.
            activation: The activation function of the intermediate layer (`'relu'` or `'gelu'`).
            layer_norm_eps: The eps value in layer normalization components.
            batch_first: If True, input and output tensors are provided as `(batch, seq, feature)`.
            norm_first: If True, layer norm is done prior to attention and feedforward operations.
            bias: If False, Linear and LayerNorm layers will not learn an additive bias.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.self_attn: MultiheadAttention = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias
        )
        self.linear1: Linear = Linear(d_model, dim_feedforward, bias=bias)
        self.dropout: Dropout = Dropout(dropout)
        self.linear2: Linear = Linear(dim_feedforward, d_model, bias=bias)

        self.norm1: LayerNorm = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2: LayerNorm = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1: Dropout = Dropout(dropout)
        self.dropout2: Dropout = Dropout(dropout)

        self.activation: Module = ReLU() if activation == "relu" else GELU()
        self.norm_first: bool = norm_first

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Passes the input through the encoder layer.

        Args:
            src: The sequence to the encoder layer.
            src_mask: The mask for the `src` sequence.
            src_key_padding_mask: The mask for the `src` keys per batch.
            is_causal: If specified, applies a causal mask as `src_mask`.

        Returns:
            Encoded sequence tensor.
        """
        _ = is_causal
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
    ) -> Tensor:
        """Executes the self-attention sub-block.

        Args:
            x: Input tensor.
            attn_mask: Optional attention mask.
            key_padding_mask: Optional key padding mask.

        Returns:
            Self-attention output tensor after dropout.
        """
        attn_out, _ = self.self_attn.forward(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.dropout1(attn_out)

    def _ff_block(self, x: Tensor) -> Tensor:
        """Executes the position-wise feed-forward sub-block.

        Args:
            x: Input tensor.

        Returns:
            Feed-forward output tensor after dropout.
        """
        out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(out)


class TransformerDecoderLayer(Module):
    """TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network."""

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
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes TransformerDecoderLayer.

        Args:
            d_model: The number of expected features in the input.
            nhead: The number of heads in the multiheadattention models.
            dim_feedforward: The dimension of the feedforward network model.
            dropout: The dropout value.
            activation: The activation function of the intermediate layer (`'relu'` or `'gelu'`).
            layer_norm_eps: The eps value in layer normalization components.
            batch_first: If True, input and output tensors are provided as `(batch, seq, feature)`.
            norm_first: If True, layer norm is done prior to self attention, multihead attention and feedforward operations.
            bias: If False, Linear and LayerNorm layers will not learn an additive bias.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.self_attn: MultiheadAttention = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias
        )
        self.multihead_attn: MultiheadAttention = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias
        )

        self.linear1: Linear = Linear(d_model, dim_feedforward, bias=bias)
        self.dropout: Dropout = Dropout(dropout)
        self.linear2: Linear = Linear(dim_feedforward, d_model, bias=bias)

        self.norm1: LayerNorm = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2: LayerNorm = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3: LayerNorm = LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1: Dropout = Dropout(dropout)
        self.dropout2: Dropout = Dropout(dropout)
        self.dropout3: Dropout = Dropout(dropout)

        self.activation: Module = ReLU() if activation == "relu" else GELU()
        self.norm_first: bool = norm_first

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        """Passes the inputs (and masks) through the decoder layer.

        Args:
            tgt: The sequence to the decoder layer.
            memory: The sequence from the last layer of the encoder.
            tgt_mask: The mask for the `tgt` sequence.
            memory_mask: The mask for the `memory` sequence.
            tgt_key_padding_mask: The mask for the `tgt` keys per batch.
            memory_key_padding_mask: The mask for the `memory` keys per batch.
            tgt_is_causal: If specified, applies a causal mask as `tgt_mask`.
            memory_is_causal: If specified, applies a causal mask as `memory_mask`.

        Returns:
            Decoded sequence tensor.
        """
        _ = (tgt_is_causal, memory_is_causal)
        mem = memory if memory is not None else tgt
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(
                self.norm2(x), mem, memory_mask, memory_key_padding_mask
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(
                x + self._mha_block(x, mem, memory_mask, memory_key_padding_mask)
            )
            x = self.norm3(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
    ) -> Tensor:
        """Executes the self-attention sub-block.

        Args:
            x: Input tensor.
            attn_mask: Optional attention mask.
            key_padding_mask: Optional key padding mask.

        Returns:
            Self-attention output tensor after dropout.
        """
        attn_out, _ = self.self_attn.forward(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.dropout1(attn_out)

    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
    ) -> Tensor:
        """Executes the cross multi-head attention sub-block.

        Args:
            x: Target query tensor.
            mem: Encoder memory tensor.
            attn_mask: Optional cross-attention mask.
            key_padding_mask: Optional memory key padding mask.

        Returns:
            Cross-attention output tensor after dropout.
        """
        attn_out, _ = self.multihead_attn.forward(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.dropout2(attn_out)

    def _ff_block(self, x: Tensor) -> Tensor:
        """Executes the position-wise feed-forward sub-block.

        Args:
            x: Input tensor.

        Returns:
            Feed-forward output tensor after dropout.
        """
        out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(out)


class TransformerEncoder(Module):
    """TransformerEncoder is a stack of N encoder layers."""

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer | None,
        num_layers: int,
        norm: Module | None = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True,
    ) -> None:
        """Initializes TransformerEncoder.

        Args:
            encoder_layer: An instance of the `TransformerEncoderLayer` class.
            num_layers: The number of sub-encoder-layers in the encoder.
            norm: Optional layer normalization component.
            enable_nested_tensor: If True, input will automatically convert to nested tensor.
            mask_check: If True, validates mask properties.
        """
        _ = (enable_nested_tensor, mask_check)
        super().__init__()
        self.layers: ModuleList = ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
            if encoder_layer is not None
            else []
        )
        self.num_layers: int = num_layers
        self.norm: Module | None = norm

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool | None = None,
    ) -> Tensor:
        """Passes the input through the encoder layers in turn.

        Args:
            src: The sequence to the encoder.
            mask: The mask for the `src` sequence.
            src_key_padding_mask: The mask for the `src` keys per batch.
            is_causal: If specified, applies a causal mask as `mask`.

        Returns:
            Encoded sequence tensor.
        """
        _ = is_causal
        output = src
        for mod in self.layers:
            output = mod(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(Module):
    """TransformerDecoder is a stack of N decoder layers."""

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer | None,
        num_layers: int,
        norm: Module | None = None,
    ) -> None:
        """Initializes TransformerDecoder.

        Args:
            decoder_layer: An instance of the `TransformerDecoderLayer` class.
            num_layers: The number of sub-decoder-layers in the decoder.
            norm: Optional layer normalization component.
        """
        super().__init__()
        self.layers: ModuleList = ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
            if decoder_layer is not None
            else []
        )
        self.num_layers: int = num_layers
        self.norm: Module | None = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool | None = None,
        memory_is_causal: bool = False,
    ) -> Tensor:
        """Passes the inputs (and masks) through the decoder layer in turn.

        Args:
            tgt: The sequence to the decoder.
            memory: The sequence from the last layer of the encoder.
            tgt_mask: The mask for the `tgt` sequence.
            memory_mask: The mask for the `memory` sequence.
            tgt_key_padding_mask: The mask for the `tgt` keys per batch.
            memory_key_padding_mask: The mask for the `memory` keys per batch.
            tgt_is_causal: If specified, applies a causal mask as `tgt_mask`.
            memory_is_causal: If specified, applies a causal mask as `memory_mask`.

        Returns:
            Decoded sequence tensor.
        """
        _ = (tgt_is_causal, memory_is_causal)
        mem = memory if memory is not None else tgt
        output = tgt
        for mod in self.layers:
            output = mod(
                output,
                mem,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class Transformer(Module):
    """A standard encoder-decoder Transformer model."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        custom_encoder: Module | None = None,
        custom_decoder: Module | None = None,
        layer_norm_eps: float = 1e-05,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initializes Transformer.

        Args:
            d_model: The number of expected features in the encoder/decoder inputs.
            nhead: The number of heads in the multiheadattention models.
            num_encoder_layers: The number of sub-encoder-layers in the encoder.
            num_decoder_layers: The number of sub-decoder-layers in the decoder.
            dim_feedforward: The dimension of the feedforward network model.
            dropout: The dropout value.
            activation: The activation function of encoder/decoder intermediate layer (`'relu'` or `'gelu'`).
            custom_encoder: Optional custom encoder module.
            custom_decoder: Optional custom decoder module.
            layer_norm_eps: The eps value in layer normalization components.
            batch_first: If True, input and output tensors are provided as `(batch, seq, feature)`.
            norm_first: If True, encoder and decoder layers will perform LayerNorms before attention/feedforward.
            bias: If False, Linear and LayerNorm layers will not learn an additive bias.
            device: Desired device of parameters.
            dtype: Desired data type of parameters.
        """
        _ = (device, dtype)
        super().__init__()
        self.d_model: int = d_model
        self.nhead: int = nhead

        self.encoder: Module
        if custom_encoder is not None:
            self.encoder = custom_encoder
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

        self.decoder: Module
        if custom_decoder is not None:
            self.decoder = custom_decoder
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

    def reset_parameters(self) -> None:
        """Initiates parameters in the transformer model using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor | None = None,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        src_is_causal: bool | None = None,
        tgt_is_causal: bool | None = None,
        memory_is_causal: bool = False,
    ) -> Tensor:
        """Take in and process masked source/target sequences.

        Args:
            src: The sequence to the encoder.
            tgt: The sequence to the decoder.
            src_mask: The additive mask for the `src` sequence.
            tgt_mask: The additive mask for the `tgt` sequence.
            memory_mask: The additive mask for the encoder output.
            src_key_padding_mask: The Tensor mask for `src` keys per batch.
            tgt_key_padding_mask: The Tensor mask for `tgt` keys per batch.
            memory_key_padding_mask: The Tensor mask for `memory` keys per batch.
            src_is_causal: If specified, applies a causal mask as `src_mask`.
            tgt_is_causal: If specified, applies a causal mask as `tgt_mask`.
            memory_is_causal: If specified, applies a causal mask as `memory_mask`.

        Returns:
            Transformer output tensor.
        """
        _ = (src_is_causal, tgt_is_causal, memory_is_causal)
        target_seq = tgt if tgt is not None else src
        memory = self.encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        return self.decoder(
            target_seq,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
