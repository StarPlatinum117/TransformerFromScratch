import logging
import numpy as np
from typing import Iterator
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
        Notes:
            The matrices QKV, the attention weights and output are saved for the backward pass.
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

        # Compute multi-head attention. Output: (bs, nh, sl, dk). Weights: (bs, nh, sl, sl).
        attn_output, attn_weights = self.scaled_dot_product_attention(queries=Q, keys=K, values=V, mask=mask)
        # Transpose and reshape output back to (bs, sl, dm).
        attn_output = attn_output.transpose((0, 2, 1, 3)).reshape((batch_size, q_len, self.d_model))
        # Final linear projection.
        output = attn_output @ self.Wo  # (bs, sl, dm) x (dm, dm) = (bs, sl, dm)
        # Save everything necessary for the backward pass
        self.x_query, self.x_key, self.x_value = x_query, x_key, x_value
        self.Q, self.K, self.V = Q, K, V
        self.q_len, self.k_len, self.v_len = q_len, k_len, v_len
        self.attn_output, self.attn_weights = attn_output, attn_weights

        return output

    def self_attention(self, *, input_embeddings: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        return self.forward(x_query=input_embeddings, x_key=input_embeddings, x_value=input_embeddings, mask=mask)

    def cross_attention(
            self, *,
            x_query: np.ndarray, x_key_value: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        return self.forward(x_query=x_query, x_key=x_key_value, x_value=x_key_value, mask=mask)

    def backward(self, dout: np.ndarray, self_attention: bool) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Backpropagates the gradient through the Multi-Head Attention layer.

        This method computes the gradients w.r.t. input x_query, x_key, x_value, and the weight matrices.
        Unless specified otherwise, all gradients are of dtype=float32.

        Args:
            dout: (batch_size, tgt_seq_len, d_model) gradient of the loss w.r.t. the output of the MHA layer.
            self_attention: whether the MHA layer is used for self-attention (True) or cross-attention (False).

        Returns:
            - If self_attention is True:
                grad_input: (batch_size, seq_len, d_model)
            - If self_attention is False (e.g. during cross-attention):
                grad_input_query: (batch_size, seq_len, d_model)
                grad_input_key_value: (batch_size, seq_len, d_model)

        Notes:
            This method also computes and stores the following gradients for the optimizer step:
                - grad_Wo: ∂L/∂Wo, shape (d_model, d_model)
        """
        batch_size, seq_len, d_model = dout.shape
        # Backpropagate through the final linear projection. Flatten batch and sequence dimensions for broadcasting.
        dout_flat = dout.reshape((batch_size * seq_len, d_model))  # (flat, dm)
        attn_output_flat = self.attn_output.reshape((batch_size * seq_len, d_model))  # (flat, dm)
        self.grad_Wo = attn_output_flat.T @ dout_flat  # (dm, dm) saved for the optimizer step
        grad_attn_output = dout @ self.Wo.T            # (bs, sl, dm)
        # Backpropagate through multi-head attention.
        grad_attn_output = grad_attn_output.reshape(
            (batch_size, self.q_len, self.num_heads, self.dk)
        ).transpose((0, 2, 1, 3))  # (bs, nh, sl_q, dk)
        grad_Q, grad_K, grad_V = self.scaled_dot_product_attention_backward(
            queries=self.Q,
            keys=self.K,
            values=self.V,
            attn_weights=self.attn_weights,
            dout=grad_attn_output,
        )  # each of shape (bs, nh, sl, dk)
        # Project Q, K, V gradients to input space.
        grad_input_Q = self._project_qkv_backward(dout=grad_Q, input_x=self.x_query, W=self.W_q, grad_W_attr="grad_W_q")
        grad_input_K = self._project_qkv_backward(dout=grad_K, input_x=self.x_key, W=self.W_k, grad_W_attr="grad_W_k")
        grad_input_V = self._project_qkv_backward(dout=grad_V, input_x=self.x_value, W=self.W_v, grad_W_attr="grad_W_v")

        if self_attention:
            return grad_input_Q + grad_input_K + grad_input_V
        else:
            return grad_input_Q, grad_input_K + grad_input_V

    def _project_qkv_backward(
            self, *,
            dout: np.ndarray,
            input_x: np.ndarray,
            W: np.ndarray,
            grad_W_attr: str,
    ) -> np.ndarray:
        """Backpropagates through a Q/K/V projection linear layer and computes the gradient w.r.t. the input.
    Args:
        dout:    (batch_size, num_heads, seq_len, dk) gradient of the loss w.r.t. Q, K, or V after
                 the attention computation.
        input_x: (batch_size, seq_len, d_model) the input used to compute Q, K, or V during the forward pass.
        W:       (d_model, d_model) the projection weight matrix used for Q, K, or V during the forward pass.
        grad_W_attr: name of attribute to store grad_W (e.g. 'grad_W_q').

    Returns:
        grad_input: (batch_size, seq_len, d_model) gradient of the loss w.r.t. the input that was used to compute Q/K/V.

    Notes:
        Also computes and stores:
            - grad_W_q / grad_W_k / grad_W_v: ∂L/∂W_{q,k,v}, shape (d_model, d_model)
    """
        bs, sl, dm = input_x.shape
        _, nh, _, dk = dout.shape
        # Flatten for projection matrix gradient.
        grad_output_flat = dout.transpose((0, 2, 1, 3)).reshape(bs * sl, nh * dk)
        input_flat = input_x.reshape(bs * sl, dm)
        # Compute gradient w.r.t. W.
        grad_W = input_flat.T @ grad_output_flat  # (dm, dm)
        # Save it for the optimizer step.
        setattr(self, grad_W_attr, grad_W)
        # Backprop into input
        grad_input = grad_output_flat @ W.T  # (flat, dm)
        grad_input = grad_input.reshape((bs, sl, dm))
        return grad_input

    def get_parameters_and_gradients(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Returns learnable parameters and their gradients for the optimizer step."""
        return iter([
            (self.W_q, self.grad_W_q),
            (self.W_k, self.grad_W_k),
            (self.W_v, self.grad_W_v),
            (self.Wo, self.grad_Wo),
        ])

    @staticmethod
    def scaled_dot_product_attention(
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
                logging.debug(f"Warning: fully masked queries detected at indices: {np.where(fully_masked_queries)}")
            scores = np.where(mask, -np.inf, scores)
        # Subtract the max before taking softmax (for numerical stability). Softmax is shift invariant.
        scores -= np.max(scores, axis=-1, keepdims=True)
        # Apply softmax.
        exp_scores = np.exp(scores)
        attention_weights = (
                exp_scores / (1e-9 + np.sum(exp_scores, axis=-1, keepdims=True))
        )  # (bs,nh,sl,sl)x(bs,nh,sl,1)=(bs,nh,sl,sl)
        # Multiply values by scores.
        output = attention_weights @ V  # (bs,nh,sl,sl) x (bs,nh,sl,dk) = (bs,nh,sl,dk)

        return output, attention_weights

    @staticmethod
    def scaled_dot_product_attention_backward(
            *,
            queries: np.ndarray,
            keys: np.ndarray,
            values: np.ndarray,
            attn_weights: np.ndarray,
            dout: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backpropagates the gradient through the scaled dot-product attention.
        All input matrices are computed during the forward pass and have dtype=float32.
        Args:
            queries: (bs, nh, sl_q, dk) queries matrix Q.
            keys:    (bs, nh, sl_k, dk) keys matrix K.
            values:  (bs, nh, sl_v, dk) values matrix V.
            attn_weights: (bs, nh, sl_q, sl_k) self-attention weights.
            dout:    (bs, nh, sl_q, dk) gradient of the loss w.r.t. the output of the scaled dot-product attention.

        Returns:
            grad_Q: (bs, nh, sl_q, dk)
            grad_K: (bs, nh, sl_k, dk)
            grad_V: (bs, nh, sl_v, dk)
        """
        Q, K, V = queries, keys, values
        dk = Q.shape[-1]
        scale = 1. / np.sqrt(dk)
        # Backpropagate through final score-value multiplication.
        grad_attn_weights = dout @ V.transpose((0, 1, 3, 2))  # (bs,nh,sl_q,dk) x (bs,nh,dk,sl_v) = (bs,nh,sl_q,sl_v)
        grad_V = attn_weights.transpose((0, 1, 3, 2)) @ dout  # (bs,nh,sl_k, sl_q) x (bs,nh,sl_q,dk) = (bs,nh,sl_k,dk)
        # Backpropagate through the softmax + scaling.
        weighted_grad_sum = np.sum(grad_attn_weights * attn_weights, axis=-1, keepdims=True)  # (bs,nh,sl_q,1)
        grad_scores = attn_weights * (grad_attn_weights - weighted_grad_sum)  # (bs,nh,sl_q,sl_k)
        # Backpropagate through the scores' computation.
        grad_Q = grad_scores @ keys * scale  # (bs, nh, sl_q, dk)
        grad_K = grad_scores.transpose(0, 1, 3, 2) @ queries * scale  # (bs,nh,sl_k,dk)

        return grad_Q, grad_K, grad_V
