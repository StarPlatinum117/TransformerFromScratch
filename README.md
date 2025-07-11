Transformer from Scratch
This project implements a mini Transformer architecture from scratch using:
- NumPy only — no deep learning frameworks involved.
It includes both the forward and backward passes, and is capable of learning simple sequence-to-sequence tasks using teacher forcing and cross-entropy loss.

🧠 Project Goals

Understand and implement the Transformer architecture at a low level.

Manually code backpropagation for all layers (e.g., Multi-Head Attention, LayerNorm, FeedForward).

Train on a toy dataset and validate learning.

Build intuition for how gradients propagate in deep sequence models.

📜 Dataset

Handcrafted toy dataset with identity mappings like:

Input:  "how are you"

Target: "how are you"

Each sample is encoded with a custom tokenizer, with support for special tokens like <s>, </s>, and <pad>.

🏗️ Components Implemented

- Token embedding layer
- Positional encoding
- Multi-head self-attention
- Feed-forward network
- Layer normalization
- Transformer Encoder/Decoder
- Full backpropagation
- Gradient descent optimizer

🔁 Training Loop

The training loop includes:
- Forward pass
- Cross-entropy loss computation
- Manual gradient computation (backward pass)
- Parameter update via SGD

Logging is enabled to track loss and predictions at intervals.

🧪 Results

The model learns identity mappings and eventually achieves near-perfect predictions on all training samples, confirming the correctness of the implementation.

Example log:

Epoch 990/1000 - Loss:  0.3199

Input:      [1, 8, 6, 9, 10, 11, 2]

Prediction: [8, 6, 9, 10, 11, 2]

Target:     [8, 6, 9, 10, 11, 2]

🚀 Usage

To run training:

python train.py

Make sure all modules (e.g., model.py, layers.py, tokenizer.py) are in the same directory or properly imported.

📂 Structure

transformer_numpy

├── attention.py

├── decoder.py

├── encoder.py

├── inference.py

├── init.py

├── layers.py

├── tokenizer.py

├── train.py

├── transformer.py

├── utils.py

└── README.md

📚 Learnings

How attention mechanisms distribute focus across tokens.

How to derive gradients for complex architectures by hand.

Deep understanding of Transformer internals and training dynamics.

**Useful links:**
- https://jalammar.github.io/illustrated-transformer/
