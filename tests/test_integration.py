"""Integration tests: convergence on simple problems."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import Tensor, nn

from homeadam import AdamSRF, HomeAdam, HomeAdamEW


class XORNet(nn.Module):
    """Two-layer MLP for XOR."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.output(torch.relu(self.hidden(x)))


def _train_xor(
    optimizer_cls: type,
    xor_dataset: tuple[Tensor, Tensor],
    max_steps: int = 2000,
    **opt_kwargs: object,
) -> float:
    """Train XORNet and return final loss."""
    torch.manual_seed(42)
    model = XORNet()
    opt = optimizer_cls(model.parameters(), **opt_kwargs)
    inputs, targets = xor_dataset
    criterion = nn.MSELoss()

    final_loss = float("inf")
    for _ in range(max_steps):
        opt.zero_grad()
        pred = model(inputs)
        loss = criterion(pred, targets)
        loss.backward()
        opt.step()
        final_loss = loss.item()

    return final_loss


@pytest.mark.slow()
class TestConvergence:
    @pytest.fixture()
    def xor_dataset(self) -> tuple[Tensor, Tensor]:
        inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
        return inputs, targets

    def test_adam_srf_converges(self, xor_dataset: tuple[Tensor, Tensor]) -> None:
        loss = _train_xor(AdamSRF, xor_dataset, lr=0.01)
        assert loss < 0.05

    def test_homeadam_converges(self, xor_dataset: tuple[Tensor, Tensor]) -> None:
        loss = _train_xor(HomeAdam, xor_dataset, lr=0.01, tau=1e-12)
        assert loss < 0.05

    def test_homeadam_ew_converges(self, xor_dataset: tuple[Tensor, Tensor]) -> None:
        loss = _train_xor(HomeAdamEW, xor_dataset, lr=0.01, tau=1e-12)
        assert loss < 0.05

    def test_homeadam_ew_with_tau_converges(
        self, xor_dataset: tuple[Tensor, Tensor]
    ) -> None:
        loss = _train_xor(HomeAdamEW, xor_dataset, lr=0.01, tau=0.001)
        assert loss < 0.1

    def test_state_dict_resume_same_trajectory(
        self, xor_dataset: tuple[Tensor, Tensor]
    ) -> None:
        """Save/load mid-training produces same result as continuous training."""
        torch.manual_seed(42)
        model_continuous = XORNet()
        opt_continuous = HomeAdamEW(model_continuous.parameters(), lr=0.01, tau=0.001)
        inputs, targets = xor_dataset
        criterion = nn.MSELoss()

        # Train 100 steps
        for _ in range(100):
            opt_continuous.zero_grad()
            criterion(model_continuous(inputs), targets).backward()
            opt_continuous.step()

        # Save state
        torch.manual_seed(42)
        model_resumed = XORNet()
        opt_resumed = HomeAdamEW(model_resumed.parameters(), lr=0.01, tau=0.001)
        for _ in range(100):
            opt_resumed.zero_grad()
            criterion(model_resumed(inputs), targets).backward()
            opt_resumed.step()

        # Save and reload
        model_state = copy.deepcopy(model_resumed.state_dict())
        opt_state = copy.deepcopy(opt_resumed.state_dict())

        model_loaded = XORNet()
        model_loaded.load_state_dict(model_state)
        opt_loaded = HomeAdamEW(model_loaded.parameters(), lr=0.01, tau=0.001)
        opt_loaded.load_state_dict(opt_state)

        # Train 50 more steps on both
        for _ in range(50):
            opt_continuous.zero_grad()
            criterion(model_continuous(inputs), targets).backward()
            opt_continuous.step()

            opt_loaded.zero_grad()
            criterion(model_loaded(inputs), targets).backward()
            opt_loaded.step()

        for p_cont, p_load in zip(
            model_continuous.parameters(), model_loaded.parameters()
        ):
            torch.testing.assert_close(p_cont.data, p_load.data)
