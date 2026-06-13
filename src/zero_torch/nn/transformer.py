"""Transformer modules."""

from .module import Module


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

    def forward(self, src, tgt):
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


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

    def forward(self, src):
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class TransformerDecoder(Module):
    """TransformerDecoder."""

    def __init__(self, decoder_layer, num_layers, norm=None) -> None:
        super().__init__()

    def forward(self, tgt, memory):
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


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

    def forward(self, src):
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


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

    def forward(self, tgt, memory):
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError


class MultiheadAttention(Module):
    """MultiheadAttention."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
    ) -> None:
        super().__init__()

    def forward(self, query, key, value):
        import ml_switcheroo_compiler.core.errors

        raise ml_switcheroo_compiler.core.errors.UnimplementedMathError
