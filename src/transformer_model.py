import math
import torch
import torch.nn as nn

'''
Reference paper:

Attention Is All You Need
https://arxiv.org/abs/1706.03762
'''


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # multiply by embeddings by dimension, refer to paper 3.4
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbeddings(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        # dropout percentage to counter overfitting
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        # Apply sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)  # Tensor of shape (1, seq_len, d_model

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps  # epsilon to compensate for small variance
        self.alpha = nn.Parameter(torch.ones[1])  # Multiplied
        self.bias = nn.Parameter(torch.ones[1])  # Added

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
