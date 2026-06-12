"""Group-sparse autoencoder (Atlas featurizer). See ATLAS-1_project_description.md §3.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class FeaturizerConfig:
    """Default hyperparameters (§10)."""

    ambient_dim: int = 128
    num_groups: int = 64
    group_size: int = 16
    lambda_g: float = 1e-2
    lambda_in: float = 1e-4
    learning_rate: float = 1e-3
    steps: int = 50_000


class GroupSparseAutoencoder(nn.Module):
    """Over-provisioned group-sparse autoencoder with unit-norm decoder atoms."""

    def __init__(self, config: FeaturizerConfig) -> None:
        super().__init__()
        self.config = config
        d = config.ambient_dim
        j = config.num_groups
        n = config.group_size
        latent_dim = j * n

        self.encoder = nn.Linear(d, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, d, bias=False)
        self._init_decoder_unit_norm()

    def _init_decoder_unit_norm(self) -> None:
        with torch.no_grad():
            weight = self.decoder.weight
            weight.copy_(weight / weight.norm(dim=0, keepdim=True).clamp_min(1e-8))

    @property
    def latent_dim(self) -> int:
        return self.config.num_groups * self.config.group_size

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        return self.decoder(a)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.encode(x)
        x_hat = self.decode(a)
        return x_hat, a

    def group_slices(self) -> list[slice]:
        n = self.config.group_size
        return [slice(i * n, (i + 1) * n) for i in range(self.config.num_groups)]

    def group_lasso_penalty(self, a: torch.Tensor) -> torch.Tensor:
        n = self.config.group_size
        groups = a.reshape(a.shape[0], self.config.num_groups, n)
        return (n**0.5) * groups.norm(dim=2).sum(dim=1).mean()

    def within_group_l1_penalty(self, a: torch.Tensor) -> torch.Tensor:
        return a.abs().mean()

    def loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        a: torch.Tensor,
        *,
        lambda_g: Optional[float] = None,
        lambda_in: Optional[float] = None,
    ) -> torch.Tensor:
        cfg = self.config
        lambda_g = cfg.lambda_g if lambda_g is None else lambda_g
        lambda_in = cfg.lambda_in if lambda_in is None else lambda_in
        recon = (x - x_hat).pow(2).mean()
        return recon + lambda_g * self.group_lasso_penalty(a) + lambda_in * self.within_group_l1_penalty(a)

    def live_groups(self, activations: torch.Tensor, theta_act: float) -> list[int]:
        """Groups with activation norm above threshold on the provided batch."""
        norms = []
        for sl in self.group_slices():
            norms.append(activations[:, sl].norm(dim=1).mean().item())
        return [i for i, norm in enumerate(norms) if norm > theta_act]


def train_featurizer(
    model: GroupSparseAutoencoder,
    data: torch.Tensor,
    *,
    steps: Optional[int] = None,
    learning_rate: Optional[float] = None,
) -> list[float]:
    """Train the featurizer; returns per-step reconstruction loss history."""
    cfg = model.config
    steps = cfg.steps if steps is None else steps
    learning_rate = cfg.learning_rate if learning_rate is None else learning_rate

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[float] = []

    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        x_hat, a = model(data)
        loss = model.loss(data, x_hat, a)
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    return history
