"""Feature tests for advanced optimizer behaviors.

Covers:
- FP32 state default / custom state dtype
- foreach parity
- HomeAdam capturable parity
- HomeAdamEW update mode parity
"""

from __future__ import annotations

import torch
from torch import nn

from homeadam import AdamSRF, HomeAdam, HomeAdamEW


def _single_param_step(
    opt: torch.optim.Optimizer, p: nn.Parameter, grad: torch.Tensor
) -> None:
    p.grad = grad.clone()
    opt.step()


def test_default_state_dtype_is_fp32_for_all_optimizers() -> None:
    p1 = nn.Parameter(torch.randn(8, dtype=torch.float64))
    p2 = nn.Parameter(torch.randn(8, dtype=torch.float64))
    p3 = nn.Parameter(torch.randn(8, dtype=torch.float64))

    o1 = AdamSRF([p1], lr=1e-3)
    o2 = HomeAdam([p2], lr=1e-3, tau=1e-6)
    o3 = HomeAdamEW([p3], lr=1e-3, tau=1e-6)

    g = torch.randn_like(p1)
    _single_param_step(o1, p1, g)
    _single_param_step(o2, p2, g)
    _single_param_step(o3, p3, g)

    assert o1.state[p1]["exp_avg"].dtype == torch.float32
    assert o2.state[p2]["exp_avg"].dtype == torch.float32
    assert o3.state[p3]["exp_avg"].dtype == torch.float32


def test_state_dtype_none_follows_parameter_dtype() -> None:
    p = nn.Parameter(torch.randn(8, dtype=torch.float64))
    opt = HomeAdamEW([p], lr=1e-3, tau=1e-6, state_dtype=None)
    _single_param_step(opt, p, torch.randn_like(p))

    assert opt.state[p]["exp_avg"].dtype == torch.float64
    assert opt.state[p]["exp_avg_sq"].dtype == torch.float64


def test_homeadam_ew_default_update_mode_is_denom() -> None:
    p = nn.Parameter(torch.randn(8, dtype=torch.float32))
    opt = HomeAdamEW([p], lr=1e-3, tau=1e-6)
    assert opt.param_groups[0]["update_mode"] == "denom"


def test_homeadam_ew_update_modes_are_numerically_close() -> None:
    torch.manual_seed(123)
    p_a = nn.Parameter(torch.randn(16, dtype=torch.float32))
    p_b = nn.Parameter(p_a.detach().clone())

    opt_a = HomeAdamEW([p_a], lr=1e-3, tau=0.1, update_mode="where_update")
    opt_b = HomeAdamEW([p_b], lr=1e-3, tau=0.1, update_mode="denom")

    for _ in range(5):
        grad = torch.randn_like(p_a)
        _single_param_step(opt_a, p_a, grad)
        _single_param_step(opt_b, p_b, grad)

    torch.testing.assert_close(p_a, p_b, atol=1e-6, rtol=1e-6)


def test_adam_srf_foreach_and_non_foreach_match() -> None:
    torch.manual_seed(7)
    p1_a = nn.Parameter(torch.randn(8))
    p2_a = nn.Parameter(torch.randn(8))
    p1_b = nn.Parameter(p1_a.detach().clone())
    p2_b = nn.Parameter(p2_a.detach().clone())

    opt_a = AdamSRF([p1_a, p2_a], lr=1e-3, foreach=True)
    opt_b = AdamSRF([p1_b, p2_b], lr=1e-3, foreach=False)

    g1 = torch.randn_like(p1_a)
    g2 = torch.randn_like(p2_a)

    p1_a.grad = g1.clone()
    p2_a.grad = g2.clone()
    p1_b.grad = g1.clone()
    p2_b.grad = g2.clone()

    opt_a.step()
    opt_b.step()

    torch.testing.assert_close(p1_a, p1_b)
    torch.testing.assert_close(p2_a, p2_b)


def test_homeadam_capturable_and_non_capturable_match() -> None:
    torch.manual_seed(9)
    p1_a = nn.Parameter(torch.randn(8))
    p2_a = nn.Parameter(torch.randn(8))
    p1_b = nn.Parameter(p1_a.detach().clone())
    p2_b = nn.Parameter(p2_a.detach().clone())

    opt_a = HomeAdam([p1_a, p2_a], lr=1e-3, tau=0.01, capturable=True)
    opt_b = HomeAdam([p1_b, p2_b], lr=1e-3, tau=0.01, capturable=False)

    g1 = torch.randn_like(p1_a)
    g2 = torch.randn_like(p2_a)

    p1_a.grad = g1.clone()
    p2_a.grad = g2.clone()
    p1_b.grad = g1.clone()
    p2_b.grad = g2.clone()

    opt_a.step()
    opt_b.step()

    torch.testing.assert_close(p1_a, p1_b)
    torch.testing.assert_close(p2_a, p2_b)
