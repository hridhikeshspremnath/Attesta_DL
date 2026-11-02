"""MalConv-style gated byte CNN fused with a numeric-feature MLP. Heads: 1 main logit + 5 aux logits."""
from __future__ import annotations

import torch
from torch import nn


class MalConvFusion(nn.Module):
    def __init__(self, n_feats: int, embed_dim: int = 8, filters: int = 128, kernel: int = 128, stride: int = 64,
                 num_hidden: int = 64, fusion_hidden: int = 128, dropout: float = 0.2, n_aux: int = 5,
                 use_numeric: bool = True) -> None:
        super().__init__()
        self.kernel, self.stride, self.use_numeric = kernel, stride, use_numeric
        self.embed = nn.Embedding(257, embed_dim, padding_idx=0)  # 0 = pad, 1..256 = byte+1
        self.conv = nn.Conv1d(embed_dim, filters, kernel, stride)
        self.gate = nn.Conv1d(embed_dim, filters, kernel, stride)
        self.num = nn.Sequential(nn.Linear(n_feats + 2, num_hidden), nn.ReLU(), nn.Dropout(dropout))  # + eco one-hot
        d_in = filters + (num_hidden if use_numeric else 0)  # bytes-only ablation drops the numeric branch
        self.fuse = nn.Sequential(nn.Linear(d_in, fusion_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.main = nn.Linear(fusion_hidden, 1)
        self.aux = nn.Linear(fusion_hidden, n_aux)

    def _valid(self, lengths: torch.Tensor, n_pos: int) -> torch.Tensor:
        """(B, n_pos) mask: conv window p is valid if it starts inside the real (non-pad) bytes."""
        starts = torch.arange(n_pos, device=lengths.device) * self.stride
        return starts[None, :] < lengths[:, None].clamp(min=1)

    def forward(self, byt: torch.Tensor, x: torch.Tensor, eco: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = (byt > 0).sum(1)
        e = self.embed(byt).transpose(1, 2)  # (B, E, L)
        h = self.conv(e) * torch.sigmoid(self.gate(e))  # gated conv
        h = h.masked_fill(~self._valid(lengths, h.shape[2])[:, None, :], float("-inf"))
        cnn = h.max(2).values  # global max pool over positions
        if self.use_numeric:
            num = self.num(torch.cat([x, nn.functional.one_hot(eco, 2).to(x.dtype)], 1))
            cnn = torch.cat([cnn, num], 1)
        z = self.fuse(cnn)
        return self.main(z).squeeze(1), self.aux(z)
