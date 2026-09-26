from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .config import ModelConfig


class InputEmbedding(nn.Embedding):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    pe: Tensor

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * torch.exp(div_term))
        pe[:, 1::2] = torch.cos(position * torch.exp(div_term))
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention_scores: Tensor | None = None

    @staticmethod
    def attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None,
        dropout: nn.Dropout,
    ) -> tuple[Tensor, Tensor]:
        scores = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(~mask.to(torch.bool), -1e9)
        weights = scores.softmax(dim=-1)
        if dropout is not None:
            weights = dropout(weights)
        return weights @ value, weights

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        query = self._split_heads(self.w_q(q))
        key = self._split_heads(self.w_k(k))
        value = self._split_heads(self.w_v(v))
        out, self.attention_scores = self.attention(query, key, value, mask, self.dropout)
        batch, heads, seq_len, d_k = out.shape
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, heads * d_k)
        return self.w_o(out)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        residual_mode: str = "post",
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.residual_mode = residual_mode
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def _residual(self, x: Tensor, sublayer, norm: nn.LayerNorm) -> Tensor:
        if self.residual_mode == "pre":
            return x + self.dropout(sublayer(norm(x)))
        return norm(x + self.dropout(sublayer(x)))

    def forward(self, x: Tensor, src_mask: Tensor | None = None) -> Tensor:
        x = self._residual(x, lambda h: self.self_attn(h, h, h, src_mask), self.norm1)
        return self._residual(x, self.ffn, self.norm2)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        residual_mode: str = "post",
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.residual_mode = residual_mode
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def _residual(self, x: Tensor, sublayer, norm: nn.LayerNorm) -> Tensor:
        if self.residual_mode == "pre":
            return x + self.dropout(sublayer(norm(x)))
        return norm(x + self.dropout(sublayer(x)))

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        x = self._residual(x, lambda h: self.self_attn(h, h, h, tgt_mask), self.norm1)
        x = self._residual(
            x,
            lambda h: self.cross_attn(h, encoder_output, encoder_output, src_mask),
            self.norm2,
        )
        return self._residual(x, self.ffn, self.norm3)


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model: int, layer_norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model: int, layer_norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Projection(nn.Linear):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbedding,
        tgt_embed: InputEmbedding,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection: Projection,
    ) -> None:
        super().__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_proj = projection

    def encode(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        return self.encoder(self.src_pos(self.src_embed(src)), src_mask)

    def decode(
        self,
        encoder_output: Tensor,
        src_mask: Tensor | None,
        tgt: Tensor,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        return self.decoder(self.tgt_pos(self.tgt_embed(tgt)), encoder_output, src_mask, tgt_mask)

    def project(self, x: Tensor) -> Tensor:
        return self.tgt_proj(x)

    def max_src_len(self) -> int:
        return int(self.src_pos.pe.shape[1])

    def max_tgt_len(self) -> int:
        return int(self.tgt_pos.pe.shape[1])

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        return self.project(decoder_output)


def build_transformer(config: ModelConfig) -> Transformer:
    if config.lang_embedding:
        raise NotImplementedError("lang_embedding is reserved for the mixed-direction mode")

    src_embed = InputEmbedding(config.d_model, config.src_vocab_size)
    tgt_embed = InputEmbedding(config.d_model, config.tgt_vocab_size)
    src_pos = PositionalEncoding(config.d_model, config.src_seq_len, config.dropout)
    tgt_pos = PositionalEncoding(config.d_model, config.tgt_seq_len, config.dropout)

    encoder_layers = nn.ModuleList(
        EncoderBlock(
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.dropout,
            config.residual_mode,
            config.layer_norm_eps,
        )
        for _ in range(config.n_layers)
    )
    decoder_layers = nn.ModuleList(
        DecoderBlock(
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.dropout,
            config.residual_mode,
            config.layer_norm_eps,
        )
        for _ in range(config.n_layers)
    )

    encoder = Encoder(encoder_layers, config.d_model, config.layer_norm_eps)
    decoder = Decoder(decoder_layers, config.d_model, config.layer_norm_eps)
    projection = Projection(config.d_model, config.tgt_vocab_size)

    model = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection)

    if config.tie_embeddings:
        if config.src_vocab_size != config.tgt_vocab_size:
            raise ValueError("tie_embeddings requires src_vocab_size == tgt_vocab_size")
        model.tgt_embed.weight = model.src_embed.weight
        model.tgt_proj.weight = model.src_embed.weight

    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)

    return model


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)