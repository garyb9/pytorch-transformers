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


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_model, d_ff)  # W2 and B2

    def forward(self, x: torch.Tensor):
        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_ff) --> (Batch, Seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model, h: int, dropout: float) -> None:
        super().__init__()
        assert d_model % h == 0, "d_model is not divisible by heads"
        self.d_model = d_model
        self.h = h

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)  # Wq - query weight matrix
        self.w_k = nn.Linear(d_model, d_model)  # Wk - keys weight matrix
        self.w_v = nn.Linear(d_model, d_model)  # Wv - value weight matrix

        self.w_o = nn.Linear(d_model, d_model)  # Wo - output weight matrix
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (Batch, h, seq_len, d_k) --> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # essentially minus infinity
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(
            dim=-1)  # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, h, d_k) --> (Batch, h, Seq_len, d_k)
        query = query.view(
            query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(
            key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(
            value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout)

        # (Batch, h, Seq_len, d_k) --> (Batch, Seq_len, h, d_k) --> (Batch, Seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(
            x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)
