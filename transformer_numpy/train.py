import logging
import numpy as np

from utils import calculate_cross_entropy_loss
from tokenizer import get_identity_dataset
from transformer import Transformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# ========================== Training settings. =======================================================================
np.random.seed(1)
vocab_size = 50
model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    num_encoder_layers=2,
    num_decoder_layers=2,
    num_heads=2,
    d_model=32,
    d_ff=64,
)
n_epochs = 1000
learning_rate = 0.001
from_logits = True
data, tokenizer = get_identity_dataset(trivial=False)  # trivial dataset for testing
# =====================================================================================================================
losses = []
# Training loop.
for epoch in range(n_epochs):
    epoch_loss = 0.
    for (src, tgt) in data:
        # Prepare input and output.
        # Each position in decoder output is the expected prediction of the corresponding position in decoder input.
        tgt_input = tgt[:-1]
        tgt_output = tgt[1:]
        # Transform to arrays to match expected shape and dtype.
        src_arr = np.array(src, dtype=int)[np.newaxis, :]  # dtype=int for token_embeddings matrix. (1, src_seq_len)
        tgt_arr = np.array(tgt_input, dtype=int)[np.newaxis, :]  # same. (1, tgt_seq_len)
        expected_output = np.array(tgt_output, dtype=int)[np.newaxis, :]  # (1, tgt_seq_len)
        # Forward pass.
        predictions = model(
            src=src_arr, tgt=tgt_arr, src_mask=None, from_logits=from_logits
        )  # (1, tgt_seq_len, vocab_size)
        # Loss computation.
        loss, grad_logits = calculate_cross_entropy_loss(
            y_pred=predictions, y_true=expected_output, from_logits=from_logits
        )
        # Accumulate batch losses.
        epoch_loss += loss
        # Zero gradients
        if epoch != 0:
            for _, param, grad in model.get_parameters_and_gradients():
                grad[...] = 0  # in-place zeroing
        # Back-propagation.
        model.backward(dout=grad_logits)
        # Optimizer step.
        for name, param, grad in model.get_parameters_and_gradients():
            param -= learning_rate * grad
    # Intermediate output logging for sanity check.
    if epoch % 10 == 0:
        decoded = np.argmax(predictions[0], axis=-1)
        logging.info(f"Input: {src}")
        logging.info(f"Prediction: {decoded.tolist()}")
        logging.info(f"Target: {tgt_output}")
    # Log epoch metrics.
    logging.info(f"Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss: .4f}")
