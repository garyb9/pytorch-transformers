from __future__ import annotations

import random

import torch
from torch.optim import Optimizer


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.98),
    eps: float = 1e-9,
) -> Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)


def build_scheduler(optimizer: Optimizer, warmup_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(1, warmup_steps)
    peak = warmup**-0.5

    def lr_lambda(step: int) -> float:
        current = step + 1
        scale = min(current**-0.5, current * warmup**-1.5)
        return scale / peak

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)