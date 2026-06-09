"""Train the toy transformer on the non-separable month×day task."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import MonthDayDataset
from .model import ToyTransformer


def train(
    epochs: int = 500,
    lr: float = 3e-3,
    d_model: int = 64,
    n_layers: int = 3,
    batch_size: int = 84,
    seed: int = 42,
    out_dir: Path | None = None,
) -> tuple[ToyTransformer, dict]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MonthDayDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ToyTransformer(d_model=d_model, n_layers=n_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history: dict[str, list[float]] = {"loss": [], "mse": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            tokens = batch["token_id"].to(device)
            targets = batch["target"].to(device)
            preds = model(tokens)
            loss = F.mse_loss(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * tokens.size(0)

        epoch_loss /= len(dataset)
        history["loss"].append(epoch_loss)
        history["mse"].append(epoch_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            tqdm.write(f"epoch {epoch + 1:4d}  mse={epoch_loss:.6f}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": {"d_model": d_model, "n_layers": n_layers},
                "history": history,
            },
            out_dir / "model.pt",
        )

    return model, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
