"""Tests for the HomeAdamEW optimizer class (Algorithm 3)."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from homeadam import AdamSRF, HomeAdamEW


class TestHomeAdamEWConstruction:
    def test_valid_defaults(self) -> None:
        model = nn.Linear(2, 1)
        opt = HomeAdamEW(model.parameters())
        assert opt.defaults["lr"] == 1e-3
        assert opt.defaults["tau"] == 1.0

    def test_negative_tau_raises(self) -> None:
        model = nn.Linear(2, 1)
        with pytest.raises(ValueError, match="tau"):
            HomeAdamEW(model.parameters(), tau=-0.5)

    def test_zero_tau_raises(self) -> None:
        model = nn.Linear(2, 1)
        with pytest.raises(ValueError, match="tau"):
            HomeAdamEW(model.parameters(), tau=0.0)

    def test_negative_lr_raises(self) -> None:
        model = nn.Linear(2, 1)
        with pytest.raises(ValueError, match="learning rate"):
            HomeAdamEW(model.parameters(), lr=-0.01)

    def test_negative_eps_raises(self) -> None:
        model = nn.Linear(2, 1)
        with pytest.raises(ValueError, match="epsilon"):
            HomeAdamEW(model.parameters(), eps=-1e-8)

    def test_negative_weight_decay_raises(self) -> None:
        model = nn.Linear(2, 1)
        with pytest.raises(ValueError, match="weight_decay"):
            HomeAdamEW(model.parameters(), weight_decay=-0.01)


class TestHomeAdamEWStep:
    def test_single_step_updates_param(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        opt = HomeAdamEW(model.parameters(), lr=0.01)
        initial = model.weight.data.clone()

        x = torch.randn(3, 4)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        assert not torch.equal(model.weight.data, initial)

    def test_skip_none_grad(self) -> None:
        p1 = nn.Parameter(torch.tensor([1.0, 2.0]))
        p2 = nn.Parameter(torch.tensor([3.0, 4.0]))
        opt = HomeAdamEW([p1, p2], lr=0.01)

        p1.grad = torch.tensor([0.1, 0.2])
        initial_p2 = p2.data.clone()
        opt.step()
        torch.testing.assert_close(p2.data, initial_p2)

    def test_near_zero_tau_matches_adam_srf(self) -> None:
        """Near-zero tau means all elements adaptive → identical to AdamSRF."""
        torch.manual_seed(42)
        model1 = nn.Linear(4, 2)
        model2 = nn.Linear(4, 2)
        model2.load_state_dict(model1.state_dict())

        opt1 = AdamSRF(model1.parameters(), lr=0.01)
        opt2 = HomeAdamEW(model2.parameters(), lr=0.01, tau=1e-12)

        x = torch.randn(3, 4)
        model1(x).sum().backward()
        opt1.step()

        model2(x).sum().backward()
        opt2.step()

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(p1.data, p2.data)

    def test_large_tau_all_sgdm(self) -> None:
        """Very large tau → every element uses SGDM."""
        torch.manual_seed(42)
        model_a = nn.Linear(4, 2)
        model_s = nn.Linear(4, 2)
        model_s.load_state_dict(model_a.state_dict())

        opt_a = HomeAdamEW(model_a.parameters(), lr=0.01, tau=1e-12)
        opt_s = HomeAdamEW(model_s.parameters(), lr=0.01, tau=1e10)

        x = torch.randn(3, 4)
        model_a(x).sum().backward()
        opt_a.step()
        model_s(x).sum().backward()
        opt_s.step()

        any_different = any(
            not torch.equal(p1.data, p2.data)
            for p1, p2 in zip(model_a.parameters(), model_s.parameters())
        )
        assert any_different

    def test_state_dict_roundtrip(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        opt = HomeAdamEW(model.parameters(), lr=0.01, tau=0.5)

        x = torch.randn(3, 4)
        model(x).sum().backward()
        opt.step()

        state = copy.deepcopy(opt.state_dict())
        opt2 = HomeAdamEW(model.parameters(), lr=0.01, tau=0.5)
        opt2.load_state_dict(state)

        for group1, group2 in zip(opt.param_groups, opt2.param_groups):
            assert group1["lr"] == group2["lr"]
            assert group1["tau"] == group2["tau"]

    def test_weight_decay(self) -> None:
        torch.manual_seed(0)
        model_wd = nn.Linear(4, 2)
        model_no = nn.Linear(4, 2)
        model_no.load_state_dict(model_wd.state_dict())

        opt_wd = HomeAdamEW(model_wd.parameters(), lr=0.01, weight_decay=0.1)
        opt_no = HomeAdamEW(model_no.parameters(), lr=0.01, weight_decay=0.0)

        x = torch.randn(3, 4)
        model_wd(x).sum().backward()
        opt_wd.step()
        model_no(x).sum().backward()
        opt_no.step()

        assert not torch.equal(model_wd.weight.data, model_no.weight.data)

    def test_multiple_param_groups(self) -> None:
        p1 = nn.Parameter(torch.randn(3, 2))
        p2 = nn.Parameter(torch.randn(2, 3))
        opt = HomeAdamEW(
            [
                {"params": [p1], "lr": 0.01, "tau": 0.5},
                {"params": [p2], "lr": 0.001, "tau": 2.0},
            ]
        )
        assert len(opt.param_groups) == 2
        assert opt.param_groups[0]["tau"] == 0.5
        assert opt.param_groups[1]["tau"] == 2.0

    def test_closure(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        opt = HomeAdamEW(model.parameters(), lr=0.01)
        x = torch.randn(3, 4)

        def closure() -> float:
            opt.zero_grad()
            loss = model(x).sum()
            loss.backward()
            return loss.item()

        loss_val = opt.step(closure)
        assert loss_val is not None
        assert isinstance(loss_val, float)
