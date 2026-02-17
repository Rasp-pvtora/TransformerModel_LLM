# TransformerModel_LLM

A clean, educational PyTorch implementation of an encoder–decoder Transformer (small LLM-style model).

---

## 🔧 What this project is

**TransformerModel_LLM** provides a readable-from-scratch implementation of the core Transformer building blocks in `TransformerModel_LLM.py`:

- `InputEmbeddings`, `PositionalEncoding` — token & positional embeddings
- `MultiHeadAttention` — scaled dot-product multi-head attention
- `FeedForwardSubLayer` — position-wise feed-forward network
- `EncoderLayer` / `TransformerEncoder` — encoder stack
- `DecoderLayer` / `TransformerDecoder` — decoder stack with cross-attention
- `Transformer` — full encoder-decoder model that returns `log`-probabilities

This repository is intended for learning, experiments and small-scale prototyping.

---

## ✅ Features

- Pure PyTorch implementation (no torch.nn.Transformer usage)
- Positional encodings (sin/cos)
- Layer normalization, dropout, and residual connections
- Decoder outputs `log_softmax` (suitable for `nn.NLLLoss`)

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch (install with `pip install torch`)

---

## 🚀 Quick start (example)

```python
from TransformerModel_LLM import Transformer
import torch
import torch.nn as nn

# model hyperparameters
vocab_size = 10000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 128
dropout = 0.1

model = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# dummy input tensors (int token ids)
batch_size = 2
src = torch.randint(0, vocab_size, (batch_size, 20))   # (batch, src_seq_len)
tgt = torch.randint(0, vocab_size, (batch_size, 10))   # (batch, tgt_seq_len)

# forward pass — returns log-probabilities (batch, tgt_seq_len, vocab_size)
log_probs = model(src, tgt, src_mask=None, tgt_mask=None, cross_mask=None)

# loss (decoder returns log_softmax -> use NLLLoss)
loss_fn = nn.NLLLoss()
loss = loss_fn(log_probs.view(-1, vocab_size), tgt.view(-1))
loss.backward()
```

Notes:
- `src` and `tgt` are LongTensors of token IDs with shape `(batch, seq_len)`.
- Masks are optional; if used they should be broadcastable to attention scores (commonly `(batch, 1, seq_q, seq_k)`).

---

## 🔍 API (high level)

- `Transformer(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)` — full model
- `TransformerEncoder`, `TransformerDecoder` — encoder / decoder stacks
- `Transformer.forward(x, tgt, src_mask, tgt_mask, cross_mask)` — returns decoder log-probs

---

## Examples & tests

- Run the runnable training example: `python examples/train.py` (toy data, quick demo).
- Run the unit / smoke tests (requires `pytest`): `python -m pytest tests -q`.

## 💡 Tips & extensions

- Swap `nn.NLLLoss` for `nn.CrossEntropyLoss` by removing the `log_softmax` in the decoder output.
- Add tokenization, padding and mask generation for real datasets.
- Implement learning-rate schedules (warmup + Adam) for training stability.

---

## 📚 Reference

- "Attention Is All You Need" — Vaswani et al., 2017

---

## 🧾 License

This project contains example code for learning and experimentation.

---

