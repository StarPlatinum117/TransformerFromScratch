import numpy as np
from typing import Optional


def numerically_stable_softmax(x: np.ndarray) -> np.ndarray:
    # Subtract max. Softmax is shift-invariant.
    x -= np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def get_predicted_probability_for_gt_label(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Extracts the predicted probability for GT labels y_true from y_pred.

    For each element in the sequence, it grabs the probability predicted for its corresponding GT label.
    Conceptually, the indexing operation can be thought as:

    y_pred[batch num, pos in seq, GT label for this position of this batch]

    Which means "given a batch number and a position in the sequence, retrieve the predicted probability
    for the GT label". Numpy's advanced indexing is used to accomplish it.

    Args:
        y_pred: (batch_size, tgt_seq_len, vocab_size) predicted probability distribution for each position.
        y_true: (batch_size, tgt_seq_len) GT label for each position.
    Returns:
        (batch_size, tgt_seq_len) GT label predicted probability for each position.
    """
    batch_size, seq_len = y_true.shape
    batch_indices = np.arange(batch_size)[:, None]  # (batch_size, 1)
    seq_indices = np.arange(seq_len)[None, :]  # (1, tgt_seq_len)
    y_pred_for_gt_label = y_pred[batch_indices, seq_indices, y_true]  # (batch_size, tgt_seq_len)
    return y_pred_for_gt_label


def calculate_cross_entropy_loss(
        *, y_pred: np.ndarray, y_true: np.ndarray,
        from_logits: Optional[bool] = False, eps: Optional[float] = 1e-15) -> tuple[float, np.ndarray]:
    """
        Cross-entropy loss for integer class labels.

        Args:
            y_pred: (batch_size, tgt_seq_len, vocab_size) predicted logits or probabilities for each token position
            in the output sequence, dtype=float32.
            y_true: (batch_size, tgt_seq_len) ground-truth labels, dtype=int.
            from_logits: True if predictions are in the form of logits instead of probabilities.
            eps: small constant to avoid log(0).
        Returns:
            cse_loss: scalar cross-entropy loss averaged over all tokens and batches.
            grad_logits: (batch_size, tgt_seq_len, vocab_size) gradient of loss w.r.t. logits.
    """
    if from_logits:
        probs = numerically_stable_softmax(y_pred)
    else:
        probs = y_pred
    # Clip probs to avoid log(0).
    probs = np.clip(probs, eps, 1 - eps)  # (batch_size, seq_len, vocab_size)

    # Compute cross-entropy loss from predicted probs for GT labels.
    gt_probs = get_predicted_probability_for_gt_label(y_pred=probs, y_true=y_true)
    cse_loss = -np.log(gt_probs)  # (batch_size, tgt_seq_len)
    cse_loss = float(np.mean(cse_loss))

    # Compute gradients w.r.t. logits. The result of ∂L/∂logits is well known (see comment below).
    batch_size, seq_len = y_true.shape
    grad_logits = probs.copy()
    batch_indices = np.arange(batch_size)[:, None]  # (batch_size, 1)
    seq_indices = np.arange(seq_len)  # (seq_len,)
    # ∂L/∂logits = Subtract 1 from the predicted probability of the correct class.
    grad_logits[batch_indices, seq_indices, y_true] -= 1  # (batch_size, seq_len, vocab_size)
    grad_logits /= (batch_size * seq_len)  # normalize over all tokens

    return cse_loss, grad_logits
