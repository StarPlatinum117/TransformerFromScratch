import numpy as np
from typing import Optional


class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})!"
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads

        # Weight matrices for Q, K, V projections.
        self.W_q = np.random.randn(d_model, d_model).astype(np.float32)
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32)
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32)
        # Final linear projection weight matrix.
        self.Wo = np.random.randn(d_model, d_model).astype(np.float32)

    def forward(
            self, *,
            x_query: np.ndarray,
            x_key: np.ndarray,
            x_value: np.ndarray,
            mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Implements multi-head attention.
        The seq_len of x may differ based on encoder/decoder.
        Args:
            x_query:    (batch_size, seq_len, d_model) input to be projected into Q, dtype=float32
            x_key:      (batch_size, seq_len, d_model) input to be projected into K, dtype=float32
            x_value:    (batch_size, seq_len, d_model) input to be projected into V, dtype=float32
            mask:       (batch_size, num_heads, seq_len, seq_len) True for masked positions.
        Returns:
            output: (batch_size, seq_len, d_model) unified representation of all attention heads, dtype=float32.
        """
        batch_size = x_query.shape[0]
        q_len = x_query.shape[1]
        k_len = x_key.shape[1]
        v_len = x_value.shape[1]
        # Linearly project x.
        Q = x_query @ self.W_q  # (batch_size, seq_len, d_model) x (d_model, d_model) = (bs, sl, dm)
        K = x_key @ self.W_k
        V = x_value @ self.W_v
        # Reshape QKV into heads -> (batch_size, seq_len, num_heads, dk).
        # Tranpose to (batch_size, num_heads, seq_len, dk) for correct broadcasting during attention computation.
        Q = Q.reshape((batch_size, q_len, self.num_heads, self.dk)).transpose((0, 2, 1, 3))
        K = K.reshape((batch_size, k_len, self.num_heads, self.dk)).transpose((0, 2, 1, 3))
        V = V.reshape((batch_size, v_len, self.num_heads, self.dk)).transpose((0, 2, 1, 3))
        # Compute multi-head attention. Output: (bs, nh, sl, dk). Weights: (nh, sl, sl).
        output, attention_weights = scaled_dot_product_attention(queries=Q, keys=K, values=V, mask=mask)
        # Transpose and reshape back to (bs, sl, dm).
        output = output.transpose((0, 2, 1, 3)).reshape((batch_size, q_len, self.d_model))
        # Final linear projection.
        output = output @ self.Wo  # (bs, sl, dm) x (dm, dm) = (bs, sl, dm)
        return output

    def self_attention(self, *, input_embeddings: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        return self.forward(x_query=input_embeddings, x_key=input_embeddings, x_value=input_embeddings, mask=mask)

    def cross_attention(
            self, *,
            x_query: np.ndarray , x_key_value: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        return self.forward(x_query=x_query, x_key=x_key_value, x_value=x_key_value, mask=mask)


def scaled_dot_product_attention(
        *,
        queries: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        mask: Optional = None) -> tuple[np.ndarray, np.ndarray]:
    """Computes attention scores given Q, K, V matrices.
    The seq_len of QKV matrices may differ based on encoder/decoder.
    Args:
        queries: (batch_size, num_heads, seq_len, dk) queries matrix Q, dytpe=float32.
        keys: (batch_size, num_heads, seq_len, dk) keys matrix K, dytpe=float32.
        values: (batch_size, num_heads, seq_len, dk) values matrix V, dytpe=float32.
        mask: (1, 1, seq_len, seq_len) True for masked positions.
    Returns:
        output: (batch_size, num_heads, seq_len, dk) self-attention values, dtype=float32.
        attention_weights: (batch_size, num_heads, seq_len, seq_len) self-attention weights.
    """
    Q, K, V = queries, keys, values
    # Get the dimension of key vectors.
    dk = K.shape[-1]
    # Compute the attention scores using Q and V. Transpose to accommodate batch dimension.
    scores = Q @ K.transpose((0, 1, 3, 2))  # (bs,nh,sl,dk) x (bs,nh,dk,sl) = (bs,nh,sl,sl)
    # Scale the scores with dimension of key vectors.
    scores /= np.sqrt(dk)
    # Apply mask, if needed.
    if mask is not None:
        fully_masked_queries = mask.all(axis=-1)
        if np.any(fully_masked_queries):
            print("Warning: fully masked queries detected at indices:", np.where(fully_masked_queries))
            raise Exception
        scores = np.where(mask, -np.inf, scores)
    # Subtract the max before taking softmax (for numerical stability). Softmax is shift invariant.
    scores -= np.max(scores, axis=-1, keepdims=True)
    # Apply softmax.
    exp_scores = np.exp(scores)
    attention_weights = (
            exp_scores / (1e-9 + np.sum(exp_scores, axis=-1, keepdims=True))  # (bs,nh,sl,sl)x(bs,nh,sl,1)=(bs,nh,sl,sl)
    )
    # Multiply values by scores.
    output = attention_weights @ V  # (bs,nh,sl,sl) x (bs,nh,sl,dk) = (bs,nh,sl,dk)

    return output, attention_weights
