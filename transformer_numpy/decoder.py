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
        """Computes the backward flow through the Decoder
        Args:
            dout: (batch_size, tgt_seq_len, vocab_size) gradient of loss w.r.t. logits, dype=float32.
        Returns:
            grad_encoder_output: add text, dtype=float32.
        """
        # Backpropagate through final linear projection layer: logits = output @ Wo + bo.
        # Compute gradient w.r.t. Wo: dL/dWo = dL/dlogits x dlogits/dWo = dout @ output (bs,tgt_sl,vs)x(bs,tgt_sl,dm)
        # Reshape for matrix multiplication.
        batch_size, seq_len, vocab_size = dout.shape
        dout_flat = dout.reshape(batch_size * seq_len, vocab_size)  # (bs*sl, vs)
        output_flat = self.output.reshape(batch_size * seq_len, self.d_model)  # (bs*sl, dm)
        grad_Wo = dout.T @ self.output  # (vs, bs*sl) @ (bs*sl, dm) = (vs, dm)
        grad_bo = ...
        grad_output = ...  # decoder's output before projection.
        # Backpropagate through all decoder blocks.
        for decoder in reversed(self.decoders):
            decoder.backward(grad_Wo, grad_bo)
        # Backpropagate through input embeddings.
        

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
        output = self.masked_multihead_attention.self_attention(input_embeddings=x, mask=tgt_mask)
        # Add residual and normalize.
        output = self.layernorm1(output + x)
        # Apply cross-attention on encoder output.
        encoder_attn = self.cross_attention.cross_attention(x_query=x, x_key_value=encoder_output, mask=src_mask)
        # Add residual and normalize.
        output = self.layernorm2(encoder_attn + output)
        # Apply feed-forward layer.
        hidden = self.ff(output)
        # Add residual and normalize.
        output = self.layernorm3(hidden + output)
        return output
