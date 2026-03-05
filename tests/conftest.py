"""Shared test fixtures for HomeAdam tests."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn


@pytest.fixture()
def simple_param() -> Tensor:
    """A single 1-D parameter with a known gradient."""
    p = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
    p.grad = torch.tensor([0.1, 0.2, 0.3])
    return p


@pytest.fixture()
def linear_model() -> nn.Linear:
    """A tiny linear layer for optimizer-level tests."""
    torch.manual_seed(42)
    return nn.Linear(4, 2, bias=True)


@pytest.fixture()
def xor_dataset() -> tuple[Tensor, Tensor]:
    """XOR dataset for integration tests."""
    inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    return inputs, targets
