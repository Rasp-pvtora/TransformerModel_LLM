#!/usr/bin/env python3
"""
examples/train.py

Tiny, runnable training example for the `Transformer` in `TransformerModel_LLM.py`.
- Trains on synthetic data (toy copy task).
- Decoder returns `log_softmax` so we use `nn.NLLLoss`.

Run:
    python examples/train.py

This script is intentionally small and dependency-free (only requires PyTorch).
"""

import argparse
import random
import torch
import torch.nn as nn
from TransformerModel_LLM import Transformer


def subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    """Create a causal (lower-triangular) attention mask.

    Mask values are boolean: True = keep, False = mask out.
    Returned shape is (1, 1, size, size) so it broadcasts to (batch, heads, q_len, k_len).
    """
    return torch.tril(torch.ones((size, size), dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)


def generate_batch(batch_size, src_len, tgt_len, vocab_size, device):
    # simple random integer sequences (0..vocab_size-1)
    src = torch.randint(0, vocab_size, (batch_size, src_len), dtype=torch.long, device=device)
    # target input is the sequence we ask decoder to predict (toy task)
    tgt_input = torch.randint(0, vocab_size, (batch_size, tgt_len), dtype=torch.long, device=device)
    # target labels (same shape) used with NLLLoss
    tgt_output = tgt_input.clone()
    return src, tgt_input, tgt_output


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model = Transformer(
        args.vocab_size,
        args.d_model,
        args.num_heads,
        args.num_layers,
        args.d_ff,
        args.max_seq_length,
        args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()

    print(f"Training on device: {device} — vocab={args.vocab_size}, d_model={args.d_model}")

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for step in range(args.steps_per_epoch):
            src, tgt_input, tgt_output = generate_batch(args.batch_size, args.src_len, args.tgt_len, args.vocab_size, device)

            src_mask = None
            tgt_mask = subsequent_mask(args.tgt_len, device)
            cross_mask = None

            log_probs = model(src, tgt_input, src_mask, tgt_mask, cross_mask)

            loss = criterion(log_probs.view(-1, args.vocab_size), tgt_output.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / args.steps_per_epoch
        print(f"Epoch {epoch:2d} — avg loss: {avg_loss:.4f}")

    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")


def cli_args():
    p = argparse.ArgumentParser(description="Tiny training demo for TransformerModel_LLM")
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--max-seq-length", type=int, dest="max_seq_length", default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, dest="batch_size", default=16)
    p.add_argument("--src-len", type=int, dest="src_len", default=20)
    p.add_argument("--tgt-len", type=int, dest="tgt_len", default=20)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--steps-per-epoch", type=int, dest="steps_per_epoch", default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-path", type=str, default="")
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    return p.parse_args()


if __name__ == "__main__":
    args = cli_args()
    train(args)
