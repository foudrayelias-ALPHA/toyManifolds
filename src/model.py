"""Small transformer for month×day concept encoding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import VOCAB_SIZE


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, t, d)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class ToyTransformer(nn.Module):
    """
    Minimal causal transformer: single-token input, residual stream activations
    exposed at each layer for manifold analysis.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        d_ff: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(
        self, token_ids: torch.Tensor, return_activations: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            token_ids: (batch,) or (batch, 1) token indices
            return_activations: if True, return residual stream after each layer
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(1)
        x = self.embed(token_ids)
        activations: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            activations.append(x[:, 0, :].detach().clone())
        x = self.ln_f(x)
        logits = self.head(x[:, 0, :]).squeeze(-1)
        if return_activations:
            return logits, activations
        return logits
