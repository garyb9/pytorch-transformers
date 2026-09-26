from __future__ import annotations

from collections.abc import Sequence

import torch

from .data import normalize
from .dataset import causal_mask
from .model import Transformer
from .tokenizer import TokenizerWrapper


def content_ids(tokens: Sequence[int], tokenizer: TokenizerWrapper) -> list[int]:
    result: list[int] = []
    for token in tokens:
        if token == tokenizer.bos_id:
            continue
        if token == tokenizer.eos_id:
            break
        if token == tokenizer.pad_id:
            continue
        result.append(token)
    return result


def _source_tensor(
    source_ids: Sequence[int], tokenizer: TokenizerWrapper, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = [tokenizer.bos_id, *source_ids, tokenizer.eos_id]
    src = torch.tensor([tokens], dtype=torch.long, device=device)
    mask = torch.ones(1, 1, 1, src.size(1), dtype=torch.bool, device=device)
    return src, mask


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    tokenizer: TokenizerWrapper,
    source_ids: Sequence[int],
    device: torch.device,
    max_len: int,
) -> list[int]:
    model.eval()
    src, src_mask = _source_tensor(source_ids, tokenizer, device)
    encoder_output = model.encode(src, src_mask)
    ys = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)
    for _ in range(max_len):
        size = ys.size(1)
        tgt_mask = causal_mask(size).to(device).view(1, 1, size, size)
        output = model.decode(encoder_output, src_mask, ys, tgt_mask)
        next_id = int(model.project(output[:, -1]).argmax(dim=-1).item())
        ys = torch.cat([ys, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if next_id == tokenizer.eos_id:
            break
    return ys[0, 1:].tolist()


@torch.no_grad()
def beam_search(
    model: Transformer,
    tokenizer: TokenizerWrapper,
    source_ids: Sequence[int],
    device: torch.device,
    max_len: int,
    beam_size: int = 4,
    length_penalty: float = 0.6,
) -> list[int]:
    model.eval()
    src, src_mask = _source_tensor(source_ids, tokenizer, device)
    encoder_output = model.encode(src, src_mask)

    beams: list[tuple[list[int], float, bool]] = [([tokenizer.bos_id], 0.0, False)]
    completed: list[tuple[list[int], float]] = []

    for _ in range(max_len):
        if all(finished for _, _, finished in beams):
            break
        candidates: list[tuple[list[int], float, bool]] = []
        for tokens, log_prob, finished in beams:
            if finished:
                candidates.append((tokens, log_prob, True))
                continue
            size = len(tokens)
            ys = torch.tensor([tokens], dtype=torch.long, device=device)
            tgt_mask = causal_mask(size).to(device).view(1, 1, size, size)
            output = model.decode(encoder_output, src_mask, ys, tgt_mask)
            log_probs = torch.log_softmax(model.project(output[:, -1]), dim=-1)[0]
            top_log_probs, top_ids = torch.topk(log_probs, beam_size)
            pairs = zip(top_log_probs.tolist(), top_ids.tolist(), strict=False)
            for log_prob_value, token_id in pairs:
                new_tokens = [*tokens, token_id]
                new_log_prob = log_prob + log_prob_value
                is_finished = token_id == tokenizer.eos_id
                if is_finished:
                    completed.append((new_tokens, new_log_prob))
                else:
                    candidates.append((new_tokens, new_log_prob, False))

        def score(item: tuple[list[int], float, bool]) -> float:
            tokens, log_prob, _ = item
            return log_prob / (len(tokens) ** length_penalty)

        candidates.sort(key=score, reverse=True)
        beams = candidates[:beam_size]

    pool = completed or [(tokens, log_prob) for tokens, log_prob, _ in beams]
    best_tokens, _ = max(pool, key=lambda item: item[1] / (len(item[0]) ** length_penalty))
    return best_tokens[1:]


def translate_ids(
    model: Transformer,
    tokenizer: TokenizerWrapper,
    source_ids: Sequence[int],
    device: torch.device,
    max_len: int,
    beam: int = 1,
) -> list[int]:
    if beam > 1:
        return beam_search(model, tokenizer, source_ids, device, max_len, beam_size=beam)
    return greedy_decode(model, tokenizer, source_ids, device, max_len)


def translate_text(
    model: Transformer,
    tokenizer: TokenizerWrapper,
    text: str,
    device: torch.device,
    max_len: int = 256,
    beam: int = 1,
) -> str:
    source_ids = tokenizer.encode(normalize(text))
    if not source_ids:
        return ""
    output_ids = translate_ids(model, tokenizer, source_ids, device, max_len, beam=beam)
    return tokenizer.decode(output_ids).strip()