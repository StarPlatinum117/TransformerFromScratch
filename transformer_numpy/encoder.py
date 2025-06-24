import numpy as np
from typing import Optional

from attention import MultiHeadAttention
from layers import FeedForward, LayerNorm, PositionalEncoding


class TransformerEncoder:
    def __init__(
            self, *,
            vocab_size: int,
            num_layers:int,
            num_heads: int,
            d_model: int,
            d_ff: int,
            max_len: Optional[int] = 5000):
        self.d_model = d_model
        self.token_embedding = np.random.randn(vocab_size, d_model).astype(np.float32)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.encoders = [EncoderBlock(num_heads, d_model, d_ff) for _ in range(num_layers)]

    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """Runs the Transformer encoder stack over input tokens.
        Args:
            x: (batch_size, src_seq_len) token indices for encoder input, dtype=int.
            mask: (1, 1, src_seq_len, src_seq_len) mask for encoder self-attention.
                  True for masked positions.
        Returns:
            output: (batch_size, src_seq_len, d_model) encoder stack embeddings, dtype=float32.
        """
        embedded = self.token_embedding[x] * np.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        output = self.pos_encoding(embedded)
        for encoder in self.encoders:
            output = encoder(output, mask)
        return output


class EncoderBlock:
    def __init__(self, num_heads: int, d_model: int, d_ff: int):
        # Instantiate attention sub-block.
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        # Instantiate layernorms.
        self.layernorm1 = LayerNorm(d_model)
        self.layernorm2 = LayerNorm(d_model)
        # Instantiate feed-forward layer.
        self.ff = FeedForward(d_model, d_ff)

    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Applies the operations corresponding to the Transformer Encoder block:
        - Multi-head attention.
        - Residual addition and normalization.
        - Feed-forward layer.
        - Residual addition and normalization.
        Args:
            x:    (batch_size, src_seq_len, d_model) input embeddings, dtype=float32.
            mask: (1, 1, src_seq_len, src_seq_len) True for masked positions.
        Returns:
            output: (batch_size, src_seq_len, d_model) encoded embeddings, dtype=float32.
        """
        # Apply multi-head attention.
        output = self.multihead_attention.self_attention(input_embeddings=x, mask=mask)
        # Add residual and normalize.
        output = self.layernorm1(output + x)
        # Apply feed-forward layer.
        hidden = self.ff(output)
        # Add residual and normalize.
        output = self.layernorm2(hidden + output)
        return output