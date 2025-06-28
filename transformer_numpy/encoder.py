import numpy as np
from typing import Iterator
from typing import Optional

from attention import MultiHeadAttention
from layers import FeedForward, LayerNorm, PositionalEncoding


class TransformerEncoder:
    def __init__(
            self, *,
            vocab_size: int,
            num_layers: int,
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
        # Cache input x for backward pass.
        self.input_x = x
        # Convert token indices to embeddings.
        embedded = self.token_embedding[x] * np.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        # Apply positional encodings.
        output = self.pos_encoding(embedded)
        for encoder in self.encoders:
            output = encoder(output, mask)
        return output

    def backward(self, dout: np.ndarray) -> None:
        """Backpropagates the gradient through the Encoder stack.
        Args:
            dout: (batch_size, src_seq_len, d_model) gradient of the loss w.r.t. the output of the encoder stack,
                  dtype=float32.
        """
        batch_size, seq_len, _ = dout.shape
        # Backpropagate through encoder blocks.
        grad_input = dout
        for encoder in reversed(self.encoders):
            grad_input = encoder.backward(grad_input)
        # No backpropagation needed for positional encoding since it is fixed (sinusoidal), not learnable.
        # Backpropagate through token embeddings.
        self.grad_token_embeddings = np.zeros_like(self.token_embedding, dtype=np.float32)
        for b in range(batch_size):
            for t in range(seq_len):
                idx = self.input_x[b, t]  # token index for batch b, position t
                self.grad_token_embeddings[idx] += grad_input[b, t]  # accumulate gradient for this token index

    def get_parameters_and_gradients(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Returns the parameters and gradients of the Transformer Encoder for optimization.
        Returns:
            Iterator of tuples (parameter, gradient) for each learnable parameter in the Transformer encoder.
        Notes:
            Parameters and gradients of positional encodings are not yielded. They are fixed in this implementation.
        """
        # Yield parameters of each encoder block.
        for encoder in self.encoders:
            yield from encoder.get_parameters_and_gradients()
        # Yield token embeddings and their gradients.
        yield self.token_embedding, self.grad_token_embeddings


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
        attn_output = self.multihead_attention.self_attention(input_embeddings=x, mask=mask)
        # Add residual and normalize.
        res1 = attn_output + x
        out1 = self.layernorm1(res1)
        # Apply feed-forward layer.
        hidden = self.ff(out1)
        # Add residual and normalize.
        res2 = hidden + out1
        output = self.layernorm2(res2)

        # Store everything needed for the backward pass.
        self.input_x, self.res1, self.res2 = x, res1, res2
        self.attn_output, self.hidden = attn_output, hidden

        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backpropagates the gradient through the Encoder block.
        Unless otherwise specified, all the tensors have dtype = float32.
        Args:
            dout: (batch_size, src_seq_len, d_model) gradient of the loss w.r.t. the output of the Encoder block.
        Returns:
            grad_input: (batch_size, src_seq_len, d_model) gradient of the loss w.r.t. the input of the Encoder block.
        Notes:
            The gradient variable names follow the forward pass:
                - attn_output                        → grad_attn_output
                - res1 = attn_output + x             → grad_res1
                - out1 = layernorm1(res1)            → grad_out1
                - hidden = ff(out1)                  → grad_hidden
                - res2 = hidden + out1               → grad_res2
        """
        # Backpropagate through LayerNorm2 and residual addition res2 = hidden + out1.
        grad_res2 = self.layernorm2.backward(dout)
        grad_hidden = grad_res2
        grad_out1 = grad_res2
        # Backpropagate through Feed-Forward layer.
        grad_ff_out1 = self.ff.backward(grad_hidden)
        grad_out1 += grad_ff_out1
        # Backpropagate through LayerNorm1 and residual addition res1 = attn_output + x.
        grad_res1 = self.layernorm1.backward(grad_out1)
        grad_self_attn = grad_res1
        grad_input_direct = grad_res1
        # Backpropagate through Multi-Head Attention.
        grad_input_attn = self.multihead_attention.backward(grad_self_attn, self_attention=True)
        # Combine residual paths into the final input gradient.
        grad_input = grad_input_direct + grad_input_attn
        return grad_input

    def get_parameters_and_gradients(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Returns the parameters and gradients of the Encoder block for optimization.
        Returns:
            Iterator of tuples (parameter, gradient) for each learnable parameter in the Encoder block.
        """
        # Yield parameters and gradients of multi-head attention.
        yield from self.multihead_attention.get_parameters_and_gradients()
        # Yield parameters and gradients of feed-forward layer.
        yield from self.ff.get_parameters_and_gradients()
        # Yield parameters and gradients of layer norms.
        for layer in [self.layernorm1, self.layernorm2]:
            yield layer.get_parameters_and_gradients()
