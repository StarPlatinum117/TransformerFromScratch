import numpy as np
from typing import Optional


class FeedForward:
    def __init__(self, d_model: int, d_ff: int):
        self.W1 = np.random.randn(d_model, d_ff).astype(np.float32)
        self.b1 = np.zeros(d_ff, dtype=np.float32)
        self.W2 = np.random.randn(d_ff, d_model).astype(np.float32)
        self.b2 = np.zeros(d_model, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Applies a two-layer MLP transformation to x with ReLU activation.
        The seq_len of x may differ based on encoder/decoder.
        Args:
            x: (batch_size, seq_len, d_model) input embeddings, dtype=float32.
        Returns:
            output: (batch_size, seq_len, d_model) output embeddings, dtype=float32.
        """
        hidden = x @ self.W1 + self.b1  # (batch_size, seq_len, d_model) x (d_model, d_ff) + (d_ff,) = (bs, sl, dff)
        hidden = np.maximum(0, hidden)  # ReLU activation
        output = hidden @ self.W2 + self.b2  # (bs, sl, dff) x (dff, dm) + (dm,) = (bs, sl, dm)
        return output


class LayerNorm:
    def __init__(self, d_model: int, eps: Optional[float] = 1e-5):
        self.eps = eps
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Normalizes input x across feature dimensions with learnable scale gamma and shift beta.
        The seq_len of x may differ based on encoder/decoder.
        Args:
            x: (batch_size, seq_len, d_model) input embeddings, dtype=float32.
        Returns:
            x_norm: (batch_size, seq_len, d_model) normalized, scaled and shifted embeddings.
        """
        mean = np.mean(x, axis=-1, keepdims=True)  # (bs, sl, 1)
        var = np.var(x, axis=-1, keepdims=True)  # (bs, sl, 1)
        x_norm = (self.gamma * (x - mean) / np.sqrt(var + self.eps)) + self.beta
        return x_norm


class PositionalEncoding:
    def __init__(self, d_model: int, max_len: int = 5000):
        assert d_model > 0, f"Model dimensionality must be positive! {d_model=}"
        assert d_model % 2 == 0, f"Model dimensionality must be even! {d_model=}"
        assert max_len > 0, f"Max length for encodings must be positive! {max_len=}"
        self.max_len = max_len
        self.d_model = d_model
        # Create position and feature index arrays.
        pos = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
        i = np.arange(d_model // 2)[np.newaxis, :]  # (1, d_model)
        # Compute denominators based on index.
        denom = 10_000 ** (2 * i / d_model)  # (1, d_model)
        # Create encodings and apply sin and cos functions based on index.
        encodings = np.zeros((max_len, d_model), dtype=np.float32)
        encodings[:, 0::2] = np.sin(pos / denom)  # (ml,1) x (1,dm) = (ml, dm)
        encodings[:, 1::2] = np.cos(pos / denom)
        # Add batch dimension.
        self.encodings = encodings[np.newaxis, :, :]  # (1, seq_len, d_model)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Adds precomputed sinusoidal positional encodings to the input x.
        The seq_len of x may differ based on encoder/decoder.
        Args:
            x: (batch_size, seq_len, d_model) input embeddings, dtype=float32.
        Returns:
            (batch_size, seq_len, d_model) embeddings with positional encodings.
        Raises:
            ValueError: if seq_len > max_len.
        """
        # Trim the encodings to match the sequence length, which should be less than max_len.
        seq_len = x.shape[1]
        assert seq_len <= self.max_len, f"Sequence length {seq_len} is bigger than max length {self.max_len}!"
        return x + self.encodings[:, :seq_len, :]
