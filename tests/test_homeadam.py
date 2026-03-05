"""Tests for the HomeAdam optimizer class (Algorithm 2)."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from homeadam import AdamSRF, HomeAdam


class TestHomeAdamConstruction:
    def test_valid_defaults(self) -> None:
        model = nn.Linear(2, 1)
        opt = HomeAdam(model.parameters())
        assert opt.defaults["lr"] == 1e-3
        assert opt.defaults["tau"] == 1.0

    def test_negative_tau_raises(self) -> None:
        model = nn.Linear(2, 1)
        with pytest.raises(ValueError, match="tau"):
            HomeAdam(model.parameters(), tau=-0.5)

    def test_zero_tau_raises(self) -> None:
        model = nn.Linear(2, 1)
        with pytest.raises(ValueError, match="tau"):
            HomeAdam(model.parameters(), tau=0.0)

    def test_negative_lr_raises(self) -> None:
        model = nn.Linear(2, 1)
        with pytest.raises(ValueError, match="learning rate"):
            HomeAdam(model.parameters(), lr=-0.01)

    def test_invalid_betas_raises(self) -> None:
        model = nn.Linear(2, 1)
        with pytest.raises(ValueError, match="beta1"):
            HomeAdam(model.parameters(), betas=(1.5, 0.99))


class TestHomeAdamStep:
    def test_single_step_updates_param(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        opt = HomeAdam(model.parameters(), lr=0.01, tau=1.0)
        initial = model.weight.data.clone()

        x = torch.randn(3, 4)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        assert not torch.equal(model.weight.data, initial)

    def test_skip_none_grad(self) -> None:
        p1 = nn.Parameter(torch.tensor([1.0, 2.0]))
        p2 = nn.Parameter(torch.tensor([3.0, 4.0]))
        opt = HomeAdam([p1, p2], lr=0.01)

        p1.grad = torch.tensor([0.1, 0.2])
        initial_p2 = p2.data.clone()
        opt.step()
        torch.testing.assert_close(p2.data, initial_p2)

    def test_near_zero_tau_matches_adam_srf(self) -> None:
        """With near-zero tau, HomeAdam takes adaptive path → same as AdamSRF."""
        torch.manual_seed(42)
        model1 = nn.Linear(4, 2)
        model2 = nn.Linear(4, 2)
        model2.load_state_dict(model1.state_dict())

        opt1 = AdamSRF(model1.parameters(), lr=0.01)
        opt2 = HomeAdam(model2.parameters(), lr=0.01, tau=1e-12)

        x = torch.randn(3, 4)
        loss1 = model1(x).sum()
        loss1.backward()
        opt1.step()

        loss2 = model2(x).sum()
        loss2.backward()
        opt2.step()

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(p1.data, p2.data)

    def test_large_tau_uses_sgdm(self) -> None:
        """With huge tau, should use SGDM path (different from adaptive)."""
        torch.manual_seed(42)
        model_adaptive = nn.Linear(4, 2)
        model_sgdm = nn.Linear(4, 2)
        model_sgdm.load_state_dict(model_adaptive.state_dict())

        opt_a = HomeAdam(model_adaptive.parameters(), lr=0.01, tau=1e-12)
        opt_s = HomeAdam(model_sgdm.parameters(), lr=0.01, tau=1e10)

        x = torch.randn(3, 4)
        loss_a = model_adaptive(x).sum()
        loss_a.backward()
        opt_a.step()

        loss_s = model_sgdm(x).sum()
        loss_s.backward()
        opt_s.step()

        # Adaptive and SGDM should give different results
        any_different = any(
            not torch.equal(p1.data, p2.data)
            for p1, p2 in zip(model_adaptive.parameters(), model_sgdm.parameters())
        )
        assert any_different

    def test_global_switching_across_tensors(self) -> None:
        """Algorithm 2 switch is global across all parameters in a group."""
        p1 = nn.Parameter(torch.tensor([10.0]))
        p2 = nn.Parameter(torch.tensor([10.0]))
        opt = HomeAdam(
            [p1, p2],
            lr=0.1,
            betas=(0.0, 0.0),
            eps=0.0,
            tau=1.0,
            weight_decay=0.0,
        )

        # Step-1 with betas=(0,0): m_hat=g and v_hat=g^2.
        # p1 alone would satisfy adaptive (4 >= 1), p2 would not (0.25 < 1).
        # Global switch should force both parameters onto SGDM.
        p1.grad = torch.tensor([2.0])
        p2.grad = torch.tensor([0.5])
        opt.step()

        torch.testing.assert_close(p1.data, torch.tensor([9.8]))
        torch.testing.assert_close(p2.data, torch.tensor([9.95]))

    def test_state_dict_roundtrip(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        opt = HomeAdam(model.parameters(), lr=0.01)

        x = torch.randn(3, 4)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        state = copy.deepcopy(opt.state_dict())
        opt2 = HomeAdam(model.parameters(), lr=0.01)
        opt2.load_state_dict(state)

        for group1, group2 in zip(opt.param_groups, opt2.param_groups):
            assert group1["lr"] == group2["lr"]
            assert group1["tau"] == group2["tau"]

    def test_weight_decay(self) -> None:
        torch.manual_seed(0)
        model_wd = nn.Linear(4, 2)
        model_no = nn.Linear(4, 2)
        model_no.load_state_dict(model_wd.state_dict())

        opt_wd = HomeAdam(model_wd.parameters(), lr=0.01, weight_decay=0.1)
        opt_no = HomeAdam(model_no.parameters(), lr=0.01, weight_decay=0.0)

        x = torch.randn(3, 4)
        model_wd(x).sum().backward()
        opt_wd.step()
        model_no(x).sum().backward()
        opt_no.step()

        assert not torch.equal(model_wd.weight.data, model_no.weight.data)
