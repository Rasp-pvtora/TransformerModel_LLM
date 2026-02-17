import torch
import torch.nn as nn
from TransformerModel_LLM import Transformer


def test_transformer_forward_shape_and_logprob():
    """Smoke test: forward pass shape and log-probability property (log_softmax sums to 1).
    """
    device = torch.device("cpu")

    vocab_size = 20
    d_model = 16
    num_heads = 4
    num_layers = 1
    d_ff = 32
    max_seq_length = 10

    model = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.0).to(device)

    batch_size = 2
    src_len = 6
    tgt_len = 5

    src = torch.randint(0, vocab_size, (batch_size, src_len), dtype=torch.long, device=device)
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len), dtype=torch.long, device=device)

    # forward
    out = model(src, tgt, src_mask=None, tgt_mask=None, cross_mask=None)

    # shape
    assert out.shape == (batch_size, tgt_len, vocab_size)

    # log-softmax property: logsumexp over vocab should be ~ 0
    lse = torch.logsumexp(out, dim=-1)
    assert torch.allclose(lse, torch.zeros_like(lse), atol=1e-5)


def test_backward_loss_and_gradients():
    device = torch.device("cpu")

    vocab_size = 15
    d_model = 12
    num_heads = 3
    num_layers = 1
    d_ff = 32
    max_seq_length = 8

    model = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.0).to(device)
    model.train()

    batch_size = 2
    src_len = 5
    tgt_len = 4

    src = torch.randint(0, vocab_size, (batch_size, src_len), dtype=torch.long, device=device)
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len), dtype=torch.long, device=device)

    out = model(src, tgt, src_mask=None, tgt_mask=None, cross_mask=None)

    loss_fn = nn.NLLLoss()
    loss = loss_fn(out.view(-1, vocab_size), tgt.view(-1))

    # should backprop without error and produce gradients
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)
