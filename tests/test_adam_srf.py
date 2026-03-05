"""Tests for the AdamSRF optimizer class (Algorithm 1)."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from homeadam import AdamSRF


class TestAdamSRFConstruction:
    def test_valid_defaults(self) -> None:
        model = nn.Linear(2, 1)
        opt = AdamSRF(model.parameters())
        assert opt.defaults["lr"] == 1e-3
        assert opt.defaults["betas"] == (0.9, 0.99)
        assert opt.defaults["eps"] == 1e-7
        assert opt.defaults["weight_decay"] == 0.0

    def test_negative_lr_raises(self) -> None:
        model = nn.Linear(2, 1)
        with pytest.raises(ValueError, match="learning rate"):
            AdamSRF(model.parameters(), lr=-0.1)

    def test_invalid_beta1_raises(self) -> None:
        model = nn.Linear(2, 1)
        with pytest.raises(ValueError, match="beta1"):
            AdamSRF(model.parameters(), betas=(1.0, 0.99))

    def test_invalid_beta2_raises(self) -> None:
        model = nn.Linear(2, 1)
        with pytest.raises(ValueError, match="beta2"):
            AdamSRF(model.parameters(), betas=(0.9, -0.1))

    def test_negative_eps_raises(self) -> None:
        model = nn.Linear(2, 1)
        with pytest.raises(ValueError, match="epsilon"):
            AdamSRF(model.parameters(), eps=-1e-7)

    def test_negative_weight_decay_raises(self) -> None:
        model = nn.Linear(2, 1)
        with pytest.raises(ValueError, match="weight_decay"):
            AdamSRF(model.parameters(), weight_decay=-0.01)


class TestAdamSRFStep:
    def test_single_step_updates_param(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        opt = AdamSRF(model.parameters(), lr=0.01)
        initial_weight = model.weight.data.clone()

        x = torch.randn(3, 4)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        assert not torch.equal(model.weight.data, initial_weight)

    def test_skip_none_grad(self) -> None:
        p1 = nn.Parameter(torch.tensor([1.0, 2.0]))
        p2 = nn.Parameter(torch.tensor([3.0, 4.0]))
        opt = AdamSRF([p1, p2], lr=0.01)

        # Only give gradient to p1
        p1.grad = torch.tensor([0.1, 0.2])
        # p2.grad stays None

        initial_p2 = p2.data.clone()
        opt.step()
        torch.testing.assert_close(p2.data, initial_p2)

    def test_weight_decay_shrinks_params(self) -> None:
        torch.manual_seed(0)
        model_wd = nn.Linear(4, 2)
        model_no_wd = nn.Linear(4, 2)
        model_no_wd.load_state_dict(model_wd.state_dict())

        opt_wd = AdamSRF(model_wd.parameters(), lr=0.01, weight_decay=0.1)
        opt_no_wd = AdamSRF(model_no_wd.parameters(), lr=0.01, weight_decay=0.0)

        x = torch.randn(3, 4)
        loss_wd = model_wd(x).sum()
        loss_wd.backward()
        opt_wd.step()

        loss_no_wd = model_no_wd(x).sum()
        loss_no_wd.backward()
        opt_no_wd.step()

        assert not torch.equal(model_wd.weight.data, model_no_wd.weight.data)

    def test_state_dict_roundtrip(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        opt = AdamSRF(model.parameters(), lr=0.01)

        x = torch.randn(3, 4)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        state = copy.deepcopy(opt.state_dict())

        opt2 = AdamSRF(model.parameters(), lr=0.01)
        opt2.load_state_dict(state)

        for group1, group2 in zip(opt.param_groups, opt2.param_groups):
            assert group1["lr"] == group2["lr"]
            assert group1["betas"] == group2["betas"]

    def test_closure(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        opt = AdamSRF(model.parameters(), lr=0.01)

        x = torch.randn(3, 4)

        def closure() -> float:
            opt.zero_grad()
            loss = model(x).sum()
            loss.backward()
            return loss.item()

        loss_val = opt.step(closure)
        assert loss_val is not None
        assert isinstance(loss_val, float)

    def test_multiple_param_groups(self) -> None:
        p1 = nn.Parameter(torch.randn(3, 2))
        p2 = nn.Parameter(torch.randn(2, 3))
        opt = AdamSRF(
            [
                {"params": [p1], "lr": 0.01},
                {"params": [p2], "lr": 0.001},
            ]
        )
        assert len(opt.param_groups) == 2
        assert opt.param_groups[0]["lr"] == 0.01
        assert opt.param_groups[1]["lr"] == 0.001
