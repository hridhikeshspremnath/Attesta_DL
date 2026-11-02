"""Tiny feature-only MLP: main logit + 5 aux logits (aux loss weight set in config)."""
from __future__ import annotations

import torch
from torch import nn


class FeatureMLP(nn.Module):
    def __init__(self, n_in: int, hidden: list[int], dropout: float, n_aux: int = 5) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = n_in + 2  # + ecosystem one-hot
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.main = nn.Linear(d, 1)
        self.aux = nn.Linear(d, n_aux)

    def forward(self, x: torch.Tensor, eco: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(torch.cat([x, nn.functional.one_hot(eco, 2).float()], 1))
        return self.main(h).squeeze(1), self.aux(h)
