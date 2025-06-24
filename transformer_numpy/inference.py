import numpy as np
from typing import Optional

from transformer import Transformer


def generate(
        model: Transformer,
        src: np.ndarray,
        start_token_id: int,
        end_token_id: int,
        max_len: Optional[int] = 50,
) -> list[int]:
    """Greedily generates tokens autoregressively.
    Args:
        model: Transformer instance.
        src: (1, src_seq_len) array of input token IDs.
        start_token_id: ID to start generation with.
        end_token_id: ID to end generation with.
        max_len: maximum number of tokens to generate.
    Returns:
        List of generated token IDs (excluding start token).
    """
    tgt = [start_token_id]
    # Start the generation process.
    for _ in range(max_len):
        # Add batch dimension.
        tgt_arr = np.array(tgt, dtype=np.int32)[np.newaxis, :]  # (1, curr_len)
        # Run the Transformer model.
        decoder_output = model(src=src, tgt=tgt_arr, src_mask=None, from_logits=False)
        # Get the probabilities for the next token by taking only element in batch (0) and last generation (-1).
        next_token_probs = decoder_output[0, -1]  # (vocab_size,)
        # Get the token with the highest probability. Append it to the current target.
        next_token = int(np.argmax(next_token_probs))
        tgt.append(next_token)
        # Conclude if end token is predicted.
        if next_token == end_token_id:
            break
    # Return predicted tokens, except start token.
    return tgt[1:]


vocab_size = 50
src = np.array([[5, 10, 20]], dtype=np.int32)  # (1, src_seq_len)
model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    num_encoder_layers=2,
    num_decoder_layers=2,
    num_heads=2,
    d_model=32,
    d_ff=64,
)

# Generate
output_ids = generate(model, src=src, start_token_id=1, end_token_id=2, max_len=10)
print("Generated token IDs:", output_ids)

