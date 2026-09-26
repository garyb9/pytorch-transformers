from __future__ import annotations

import torch
import torch.nn as nn

from pytorch_transformers import ModelConfig, build_transformer, count_parameters
from pytorch_transformers.model import (
    FeedForward,
    MultiHeadAttention,
    PositionalEncoding,
)


def small_config(**overrides) -> ModelConfig:
    values = {
        "src_vocab_size": 20,
        "tgt_vocab_size": 30,
        "src_seq_len": 8,
        "tgt_seq_len": 8,
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 32,
        "dropout": 0.0,
    }
    values.update(overrides)
    return ModelConfig(**values)


def causal_mask(size: int, batch: int = 1) -> torch.Tensor:
    return torch.tril(torch.ones(size, size, dtype=torch.bool)).expand(batch, 1, size, size)


def attention_keys(prefix: str) -> set[str]:
    names = ("w_q", "w_k", "w_v", "w_o")
    return {f"{prefix}.{name}.{suffix}" for name in names for suffix in ("weight", "bias")}


def linear_keys(prefix: str) -> set[str]:
    return {f"{prefix}.{suffix}" for suffix in ("weight", "bias")}


def expected_state_dict_keys(config: ModelConfig) -> set[str]:
    keys = {
        "src_embed.weight",
        "tgt_embed.weight",
        "encoder.norm.weight",
        "encoder.norm.bias",
        "decoder.norm.weight",
        "decoder.norm.bias",
        "tgt_proj.weight",
        "tgt_proj.bias",
    }
    for i in range(config.n_layers):
        base = f"encoder.layers.{i}"
        keys |= attention_keys(f"{base}.self_attn")
        keys |= linear_keys(f"{base}.ffn.linear1")
        keys |= linear_keys(f"{base}.ffn.linear2")
        keys |= linear_keys(f"{base}.norm1")
        keys |= linear_keys(f"{base}.norm2")

        base = f"decoder.layers.{i}"
        keys |= attention_keys(f"{base}.self_attn")
        keys |= attention_keys(f"{base}.cross_attn")
        keys |= linear_keys(f"{base}.ffn.linear1")
        keys |= linear_keys(f"{base}.ffn.linear2")
        keys |= linear_keys(f"{base}.norm1")
        keys |= linear_keys(f"{base}.norm2")
        keys |= linear_keys(f"{base}.norm3")
    return keys


def test_forward_shapes() -> None:
    config = small_config()
    model = build_transformer(config).eval()
    src = torch.randint(0, config.src_vocab_size, (3, config.src_seq_len))
    tgt = torch.randint(0, config.tgt_vocab_size, (3, config.tgt_seq_len))
    src_mask = torch.ones(3, 1, 1, config.src_seq_len, dtype=torch.bool)

    encoder_output = model.encode(src, src_mask)
    assert encoder_output.shape == (3, config.src_seq_len, config.d_model)

    logits = model(src, tgt, src_mask, causal_mask(config.tgt_seq_len, batch=3))
    assert logits.shape == (3, config.tgt_seq_len, config.tgt_vocab_size)


def test_state_dict_keys_match_contract() -> None:
    config = small_config()
    model = build_transformer(config)
    actual = set(model.state_dict())
    expected = expected_state_dict_keys(config)
    assert actual == expected


def test_positional_encoding_is_not_persistent() -> None:
    model = build_transformer(small_config())
    assert not any("pe" in key for key in model.state_dict())


def test_positional_encoding_shape() -> None:
    layer = PositionalEncoding(16, 8, 0.0)
    assert layer(torch.randn(3, 5, 16)).shape == (3, 5, 16)


def test_feedforward_shape() -> None:
    layer = FeedForward(16, 32, 0.0)
    assert layer(torch.randn(2, 5, 16)).shape == (2, 5, 16)


def test_attention_causal_mask_blocks_future() -> None:
    attn = MultiHeadAttention(8, 2, 0.0).eval()
    x = torch.randn(1, 4, 8)
    mask = torch.tril(torch.ones(1, 1, 4, 4, dtype=torch.bool))
    with torch.no_grad():
        attn(x, x, x, mask)
    scores = attn.attention_scores
    assert scores is not None
    scores = scores[0, 0]
    assert torch.triu(scores, diagonal=1).abs().max().item() < 1e-6
    assert scores[0, 0].item() > 0.5


def test_attention_padding_mask_is_zeroed() -> None:
    attn = MultiHeadAttention(8, 2, 0.0).eval()
    x = torch.randn(1, 3, 8)
    mask = torch.tensor([[[[True, True, False]]]])
    with torch.no_grad():
        attn(x, x, x, mask)
    scores = attn.attention_scores
    assert scores is not None
    assert scores[0, 0, :, 2].abs().max().item() < 1e-6


def test_tie_embeddings_shares_weights() -> None:
    config = small_config(src_vocab_size=20, tgt_vocab_size=20, tie_embeddings=True)
    model = build_transformer(config)
    assert model.tgt_embed.weight is model.src_embed.weight
    assert model.tgt_proj.weight is model.src_embed.weight


def test_pre_norm_mode_runs_and_backpropagates() -> None:
    config = small_config(residual_mode="pre")
    model = build_transformer(config)
    src = torch.randint(0, config.src_vocab_size, (2, config.src_seq_len))
    tgt = torch.randint(0, config.tgt_vocab_size, (2, config.tgt_seq_len))
    logits = model(src, tgt, None, causal_mask(config.tgt_seq_len, batch=2))
    logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum().item() > 0 for g in grads)


def test_count_parameters_positive() -> None:
    assert count_parameters(build_transformer(small_config())) > 0


def test_tiny_batch_overfits_copy_task() -> None:
    torch.manual_seed(0)
    vocab_size = 12
    seq_len = 8
    config = ModelConfig(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        src_seq_len=seq_len,
        tgt_seq_len=seq_len,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
    )
    model = build_transformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    bos = 2
    src = torch.randint(2, vocab_size, (16, seq_len))
    tgt_input = torch.cat([torch.full((16, 1), bos), src[:, :-1]], dim=1)
    label = src.clone()
    src_mask = torch.ones(16, 1, 1, seq_len, dtype=torch.bool)
    tgt_mask = causal_mask(seq_len, batch=16)

    loss_value = float("inf")
    for _ in range(600):
        optimizer.zero_grad()
        logits = model(src, tgt_input, src_mask, tgt_mask)
        loss = loss_fn(logits.reshape(-1, vocab_size), label.reshape(-1))
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        if loss_value < 0.02:
            break

    assert loss_value < 0.05, f"copy task did not converge, final loss {loss_value:.4f}"