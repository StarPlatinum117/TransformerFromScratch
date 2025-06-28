import numpy as np
from typing import Iterator
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
        self.input_x = x  # save x (and intermediate steps) for the backward pass
        self.hidden = x @ self.W1 + self.b1  # (bs, sl, dm) x (dm, dff) + (dff,) = (bs, sl, dff)
        self.activation = np.maximum(0, self.hidden)  # ReLU activation
        self.output = self.activation @ self.W2 + self.b2  # (bs, sl, dff) x (dff, dm) + (dm,) = (bs, sl, dm)
        return self.output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backpropagates the gradient through the Feed-Forward network.
        Args:
            dout: (batch_size, seq_len, d_model) gradient of the loss w.r.t. the output of the FFN, dtype=float32.

        Returns:
            grad_input: (batch_size, seq_len, d_model) gradient of the loss w.r.t. the FFN input (used in
                        residual connection), dtype=float32

        Notes:
            This method also computes and stores the following gradients for the optimizer step:
                - grad_Wn: ∂L/∂Wn, shape (d_ff, d_model)
                - grad_bn: ∂L/∂bn, shape (d_model,)
            Where n = 1, 2 corresponds to the linear layers in the FFN.
        """
        batch_size, seq_len, d_model = dout.shape
        _, _, d_ff = self.hidden.shape
        # Backpropagate through second linear layer. Flatten batch and sequence dimensions for correct broadcasting.
        dout_flat = dout.reshape((batch_size * seq_len, d_model))                # (flat, dm)
        activation_flat = self.activation.reshape((batch_size * seq_len, d_ff))  # (flat, dff)
        self.grad_W2 = activation_flat.T @ dout_flat                             # (dff, dm)
        self.grad_b2 = np.sum(dout, axis=(0, 1))  # (dm,)
        grad_activation = dout @ self.W2.T   # (bs, sl, dff)  gradient w.r.t. ReLU activation output
        # Backpropagate through ReLU.
        grad_hidden = grad_activation * (self.hidden > 0)  # (bs, sl, dff)
        # Backpropagate through the first linear layer. Flatten dimensions again.
        grad_hidden_flat = grad_hidden.reshape((batch_size * seq_len, d_ff))
        input_x_flat = self.input_x.reshape((batch_size * seq_len, d_model))
        self.grad_W1 = input_x_flat.T @ grad_hidden_flat   # (dm, dff)
        self.grad_b1 = np.sum(grad_hidden, axis=(0, 1))  # (dff,)
        # Gradient w.r.t. FFN input.
        grad_input = grad_hidden @ self.W1.T  # (bs, sl, dm)
        return grad_input

    def get_parameters_and_gradients(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Returns learnable parameters and their gradients for the optimizer step."""
        return iter([
            (self.W1, self.grad_W1),
            (self.b1, self.grad_b1),
            (self.W2, self.grad_W2),
            (self.b2, self.grad_b2),
        ])


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
        self.mean = np.mean(x, axis=-1, keepdims=True)  # (bs, sl, 1)
        self.var = np.var(x, axis=-1, keepdims=True)  # (bs, sl, 1)
        self.std = np.sqrt(self.var + self.eps)
        self.x_hat = (x - self.mean) / self.std
        self.x_norm = self.gamma * self.x_hat + self.beta
        return self.x_norm

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backpropagates the gradient through the Layer Normalization layer.

        Args:
            dout: (batch_size, tgt_seq_len, d_model) gradient of the loss w.r.t. LayerNorm output.

        Returns:
            grad_input: (batch_size, tgt_seq_len, d_model) gradient of the loss w.r.t input x.

        Notes:
            This method also computes and stores the following gradients for the optimizer step:
                - grad_gamma: ∂L/∂γ, shape (1, 1, d_model)
                - grad_beta: ∂L/∂β, shape (1, 1, d_model)
        """
        batch_size, seq_len, d_model = dout.shape
        self.grad_gamma = np.sum(dout * self.x_hat, axis=(0, 1), keepdims=True)  # (1, 1, dm)
        self.grad_beta = np.sum(dout, axis=(0, 1), keepdims=True)  # (1, 1, dm)
        # Backpropagate into x.
        dx_hat = dout * self.gamma  # (bs, sl, dm)
        sum_dxhat = np.sum(dx_hat, axis=-1, keepdims=True)  # (bs, sl, 1)
        sum_dxhat_xhat = np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)  # (bs, sl, 1)

        grad_input = (
                (dx_hat - sum_dxhat / d_model - self.x_hat * sum_dxhat_xhat / d_model)
                / self.std
        )
        return grad_input

    def get_parameters_and_gradients(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Returns learnable parameters and their gradients for the optimizer step."""
        return iter([
            (self.gamma, self.grad_gamma),
            (self.beta, self.grad_beta),
        ])


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
