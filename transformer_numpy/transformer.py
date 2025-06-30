import numpy as np
from typing import Iterator
from typing import Optional

from decoder import TransformerDecoder
from encoder import TransformerEncoder


class Transformer:
    def __init__(
            self, *,
            src_vocab_size: int,
            tgt_vocab_size: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            num_heads: int,
            d_model: int,
            d_ff: int,
            max_len: Optional[int] = 5000):
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            d_model=d_model,
            d_ff=d_ff,
            max_len=max_len,
        )
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            d_model=d_model,
            d_ff=d_ff,
            max_len=max_len,
        )

    def __call__(
            self, *,
            src: np.ndarray,
            tgt: np.ndarray,
            src_mask: Optional[np.ndarray],
            from_logits: Optional[bool] = False) -> np.ndarray:
        """Runs the Transformer architecture over src and tgt tokens.
        Args:
            src: (batch_size, src_seq_len) token indices for encoder input, dtype=int32.
            tgt: (batch_size, tgt_seq_len) token indices for decoder input, dtype=int32.
            src_mask: (1, 1, src_seq_len, src_seq_len) optional mask for encoder self-attention and
                      decoder cross-attention. True for masked positions.
            from_logits: if True, returns raw logits. Otherwise, returns softmax probabilities.
        Returns:
            decoder_output: (batch_size, tgt_seq_len, vocab_size) logits or probabilities, dtype=float32.
        """
        # Encode source input.
        encoder_output = self.encoder(src, src_mask)
        # Decode with encoder output as context.
        decoder_output = self.decoder(x=tgt, encoder_output=encoder_output, src_mask=src_mask, from_logits=from_logits)
        return decoder_output

    def backward(self, *, dout: np.ndarray) -> None:
        """Computes the backward flow through the Transformer.
        Args:
            dout: (batch_size, tgt_seq_len, vocab_size) gradient of loss w.r.t. logits, dype=float32.
        """
        # Backpropagate through decoder, starting from final linear projection
        grad_decoder = self.decoder.backward(dout=dout)
        # Then through encoder.
        self.encoder.backward(dout=grad_decoder)

    def get_parameters_and_gradients(self) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
        """Returns the parameters and gradients of the Transformer model for optimization.
        Returns:
            generator yielding tuples of (parameter, gradient) for each parameter in the model.
        """
        yield from self.encoder.get_parameters_and_gradients()
        yield from self.decoder.get_parameters_and_gradients()
