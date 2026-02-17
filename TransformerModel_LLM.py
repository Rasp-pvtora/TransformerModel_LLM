import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ------------------------------------------------------
#  EMBEDDINGS
# ------------------------------------------------------

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# ------------------------------------------------------
#  POSITIONAL ENCODING
# ------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ------------------------------------------------------
#  MULTI-HEAD ATTENTION
# ------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model, bias=False)
        self.key_linear = nn.Linear(d_model, d_model, bias=False)
        self.value_linear = nn.Linear(d_model, d_model, bias=False)
        self.output_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        seq_length = x.size(1)
        x = x.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def compute_attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, value)

    def combine_heads(self, x, batch_size):
        seq_length = x.size(2)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.reshape(batch_size, seq_length, self.d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.split_heads(self.query_linear(query), batch_size)
        key = self.split_heads(self.key_linear(key), batch_size)
        value = self.split_heads(self.value_linear(value), batch_size)

        attention = self.compute_attention(query, key, value, mask)
        output = self.combine_heads(attention, batch_size)
        return self.output_linear(output)


# ------------------------------------------------------
#  FEED FORWARD SUBLAYER
# ------------------------------------------------------

class FeedForwardSubLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# ------------------------------------------------------
#  ENCODER LAYER
# ------------------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff_sublayer = FeedForwardSubLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        attn_output = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.ff_sublayer(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# ------------------------------------------------------
#  ENCODER STACK
# ------------------------------------------------------

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length):
        super().__init__()
        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, src_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x


# ------------------------------------------------------
#  DECODER LAYER WITH CROSS-ATTENTION
# ------------------------------------------------------

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff_sublayer = FeedForwardSubLayer(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, tgt_mask, cross_mask):
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        cross_attn_output = self.cross_attn(x, y, y, cross_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        ff_output = self.ff_sublayer(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


# ------------------------------------------------------
#  DECODER STACK
# ------------------------------------------------------

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length):
        super().__init__()
        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output, tgt_mask, cross_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, cross_mask)

        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


# ------------------------------------------------------
#  FULL TRANSFORMER (ENCODER + DECODER)
# ------------------------------------------------------

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, num_layers,
                                          num_heads, d_ff, dropout, max_seq_length)
        self.decoder = TransformerDecoder(vocab_size, d_model, num_layers,
                                          num_heads, d_ff, dropout, max_seq_length)

    def forward(self, x, tgt, src_mask, tgt_mask, cross_mask):
        encoder_output = self.encoder(x, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, cross_mask)
        return decoder_output