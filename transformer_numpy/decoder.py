import numpy as np
from typing import Optional

from attention import MultiHeadAttention
from layers import FeedForward, LayerNorm, PositionalEncoding
from utils import numerically_stable_softmax


class TransformerDecoder:
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
        self.decoders = [DecoderBlock(num_heads, d_model, d_ff) for _ in range(num_layers)]
        self.Wo = np.random.randn(d_model, vocab_size).astype(np.float32)
        self.bo = np.zeros(vocab_size, dtype=np.float32)

    @staticmethod
    def _generate_self_attention_mask(seq_len: int) -> np.ndarray:
        """Creates upper-triangular mask to avoid attending to future tokens.
        Args:
            seq_len: target sequence length.
        Returns:
            mask: (1, 1, seq_len, seq_len) True for masked positions. Includes batch and head dimensions.
        """
        mask = np.ones((seq_len, seq_len))
        mask = np.triu(mask).astype(bool)
        # Add batch and head dimensions.
        mask = mask[np.newaxis, np.newaxis, :, :]  # (1, 1, seq_len, seq_len)
        return mask

    def __call__(
            self, *,
            x: np.ndarray,
            encoder_output: np.ndarray,
            src_mask: Optional[np.ndarray],
            from_logits: Optional[bool] = False) -> np.ndarray:
        """Runs the Transformer decoder stack over input tokens.
        Args:
            x: (batch_size, tgt_seq_len) token indices for decoder input, dtype=int.
            encoder_output: (batch_size, src_seq_len, d_model) encoder stack output for cross-attention, dtype=float32.
            src_mask: (1, 1, tgt_seq_len, src_seq_len) optional mask for cross-attention.
                      True for masked positions.
            from_logits: if True, returns raw logits. Otherwise, returns softmax probabilities.

        Returns:
            (batch_size, tgt_seq_len, vocab_size) logits or probabilities.
        """
        seq_len = x.shape[1]
        embedded = self.token_embedding[x] * np.sqrt(self.d_model)
        output = self.pos_encoding(embedded)
        tgt_mask = self._generate_self_attention_mask(seq_len)
        for decoder in self.decoders:
            output = decoder(x=output, encoder_output=encoder_output, tgt_mask=tgt_mask, src_mask=src_mask)
        # Cache output for the backward step.
        self.output = output
        # Apply final linear layer and possibly softmax.
        logits = output @ self.Wo + self.bo  # (bs, sl, dm) x (dm, vocab_size) + (vocab_size,) = (bs, sl, vs)
        if from_logits:
            return logits
        probs = numerically_stable_softmax(logits)
        return probs

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backpropagates the gradient through the final linear projection layer and the decoder stack.
        Args:
            dout: (batch_size, tgt_seq_len, vocab_size) gradient of loss w.r.t. logits, dtype=float32
        Returns:
            grad_encoder_output: (batch_size, tgt_seq_len, d_model) gradient to pass into the
                                 encoder stack, dtype=float32.
        Notes:
            The gradients grad_Wo, grad_bo, and grad_output come from standard matrix calculus:
                - logits = output @ Wo + bo.
                - ∂L/∂Wo = output.T @ ∂L/∂logits.
                - ∂L/∂bo = sum over batch and sequence positions of ∂L/∂logits.
                - ∂L/∂output = ∂L/∂logits @ Wo.T.

            For more details on these derivations, see:
                - Stanford CS231n notes on backprop through fully connected layers
                  https://cs231n.github.io/optimization-2/#fc
                - "Matrix Calculus for Deep Learning" by Terence Parr and Jeremy Howard
                  https://explained.ai/matrix-calculus/index.html

            This method also stores the following gradients for the optimizer step:
                - grad_Wo: shape (d_model, vocab_size)
                - grad_bo: shape (vocab_size,)
        """
        # Backpropagate through final linear projection layer. First, flatten batch and seq_len dimensions.
        batch_size, seq_len, vocab_size = dout.shape
        dout_flat = dout.reshape(batch_size * seq_len, vocab_size)  # (bs*sl, vs) = (flat, vs)
        output_flat = self.output.reshape(batch_size * seq_len, self.d_model)  # (bs*sl, dm) = (flat, dm)
        # Compute gradient w.r.t. Wo.
        self.grad_Wo = output_flat.T @ dout_flat  # (dm, flat) x (flat, vs) = (dm, vs)
        # Compute gradient w.r.t. bo.
        self.grad_bo = np.sum(dout, axis=(0, 1))  # (vocab_size,)
        # Compute gradient w.r.t. decoder stack's output before projection. This is sent into the decoder stack.
        grad_output = dout @ self.Wo.T  # (bs, sl, vs) x (vs, dm) = (bs, sl, dm)

        # Backpropagate through decoder blocks.
        # Accumulate gradient to encoder output.
        grad_encoder_output = np.zeros((batch_size, seq_len, self.d_model))  # this is sent into the encoder stack.
        for decoder in reversed(self.decoders):
            grad_output, grad_enc = decoder.backward(grad_output)
            grad_encoder_output += grad_enc  # (bs, sl, dm)
        # No backpropagation needed for positional encoding since it is fixed (sinusoidal), not learnable.

        # Return grad_encoder_output to pass into encoder stack.
        return grad_encoder_output


class DecoderBlock:
    def __init__(self, num_heads: int, d_model, d_ff: int):
        # Instantiate masked attention sub-block.
        self.masked_multihead_attention = MultiHeadAttention(d_model, num_heads)
        # Instantiate cross-attention sub-block.
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        # Instantiate layernorms.
        self.layernorm1 = LayerNorm(d_model)
        self.layernorm2 = LayerNorm(d_model)
        self.layernorm3 = LayerNorm(d_model)
        # Instantiate feed-forward layer.
        self.ff = FeedForward(d_model, d_ff)

    def __call__(
            self, *,
            x: np.ndarray,
            encoder_output: np.ndarray,
            tgt_mask: np.ndarray,
            src_mask: Optional[np.ndarray]) -> np.ndarray:
        """Applies the operations corresponding to the Transformer Decoder block:
        - Multi-head attention with mask for future tokens.
        - Residual addition and normalization.
        - Cross attention with decoder query and encoder output keys and values.
        - Add & normalize, feed-forward layer, add & normalize.
        Args:
            x: (batch_size, tgt_seq_len, d_model) decoder input embeddings, dtype=float32.
            encoder_output: (bs, src_seq_len, dm), encoder stack output to use for cross-attention, dtype=float32.
            tgt_mask: (bs, num_heads, tgt_sl, tgt_sl) mask for self-attention future tokens. True for masked positions.
            src_mask: (bs, num_heads, tgt_sl, src_sl) mask for cross-attention. True for masked positions.
        Returns:
            output: (bs, tgt_sl, dm) decoded embeddings, dtype=float32.
        """
        # Apply masked multi-head attention. Attending to future tokens is not allowed.
        self_attn_output = self.masked_multihead_attention.self_attention(input_embeddings=x, mask=tgt_mask)
        # Add residual and normalize.
        res1 = self_attn_output + x
        out1 = self.layernorm1(res1)
        # Apply cross-attention on encoder output.
        cross_attn = self.cross_attention.cross_attention(x_query=out1, x_key_value=encoder_output, mask=src_mask)
        # Add residual and normalize.
        res2 = cross_attn + res1
        out2 = self.layernorm2(res2)
        # Apply feed-forward layer.
        hidden = self.ff(out2)
        # Add residual and normalize.
        res3 = hidden + res2
        output = self.layernorm3(res3)
        return output

    def backward(self, dout: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Backpropagates the gradient through the Transformer Decoder block.

        Args:
            dout: (batch_size, tgt_seq_len, d_model) gradient of loss w.r.t. decoder stack's output, dtype=float32.

        Returns:
            grad_input: (batch_size, tgt_seq_len, d_model) gradient of loss w.r.t. decoder block input x, dtype=float32.
            grad_encoder_output: (batch_size, src_seq_len, d_model) gradient of loss w.r.t. encoder output (used as
                                 key and value in cross-attention), dtype=float32.

        Notes:
            The gradient variable names follow the forward pass:
                - res1 = self_attn_output + x        → grad_res1
                - out1 = layernorm1(res1)            → grad_out1
                - res2 = cross_attn + res1           → grad_res2
                - out2 = layernorm2(res2)            → grad_out2
                - res3 = hidden + res2               → grad_res3
                - output = layernorm3(res3)          → dout
        """
        # Backpropagate through LayerNorm3 and residual res3 = hidden + res2.
        grad_res3 = self.layernorm3.backward(dout)  # (bs, sl, dm)
        grad_hidden = grad_res3  # for feed-forward
        grad_out2 = grad_res3    # for residual path toward layernorm2
        # Backpropagate through Feed-Forward layer.
        grad_ff_out2 = self.ff.backward(grad_hidden)  # (bs, sl, dm)
        grad_out2 += grad_ff_out2  # add residual gradient
        # Backpropagate through LayerNorm2 and residual res 2 = cross_attn + res1.
        grad_res2 = self.layernorm2.backward(grad_out2)  # (bs, sl, dm)
        grad_cross_attn = grad_res2  # for cross-attention
        grad_out1 = grad_res2        # for residual path toward layernorm1
        # Backpropagate through Cross-Attention layer. Both grads have shape (bs, sl, dm).
        grad_out1_attn, grad_encoder_output = self.cross_attention.backward(grad_cross_attn, self_attention=False)
        grad_out1 += grad_out1_attn  # add residual gradient
        # Backpropagate through LayerNorm1 and residual res1 = self_attn_output + x.
        grad_res1 = self.layernorm1.backward(grad_out1)  # (bs, sl, dm)
        grad_self_attn = grad_res1     # toward self-attention
        grad_input_direct = grad_res1  # residual path to x
        # Backpropagate through Masked Self-Attention layer.
        grad_input_attn = self.masked_multihead_attention.backward(grad_self_attn, self_attention=True)  # (bs, sl, dm)
        # Combine residual paths into final input gradient.
        grad_input = grad_input_direct + grad_input_attn
        return grad_input, grad_encoder_output

