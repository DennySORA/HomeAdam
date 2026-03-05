"""Usage example for HomeAdam optimizers."""

from __future__ import annotations

import sys

import torch
from torch import nn

from homeadam import AdamSRF, HomeAdam, HomeAdamEW


def _select_device() -> torch.device:
    """Auto-select CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    """Train a small model on synthetic data with each optimizer."""
    torch.manual_seed(0)
    device = _select_device()

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        sys.stdout.write(f"Using device: {device} ({gpu_name})\n")
    else:
        sys.stdout.write("Using device: cpu\n")

    inputs = torch.randn(32, 4, device=device)
    targets = torch.randn(32, 1, device=device)
    criterion = nn.MSELoss()

    optimizer_configs: list[
        tuple[str, type[torch.optim.Optimizer], dict[str, object]]
    ] = [
        ("AdamSRF", AdamSRF, {"lr": 1e-2}),
        ("HomeAdam", HomeAdam, {"lr": 1e-2, "tau": 0.5}),
        ("HomeAdamEW", HomeAdamEW, {"lr": 1e-2, "tau": 0.5}),
    ]

    for name, opt_cls, kwargs in optimizer_configs:
        torch.manual_seed(0)
        model = nn.Linear(4, 1).to(device)
        optimizer = opt_cls(model.parameters(), **kwargs)  # type: ignore[arg-type]

        for _step in range(200):
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        final_loss = criterion(model(inputs), targets).item()
        sys.stdout.write(f"{name:>12s}: final loss = {final_loss:.6f}\n")


if __name__ == "__main__":
    main()
