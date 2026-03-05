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


def test_homeadam_foreach_and_non_foreach_match() -> None:
    torch.manual_seed(11)
    p1_a = nn.Parameter(torch.randn(8))
    p2_a = nn.Parameter(torch.randn(8))
    p1_b = nn.Parameter(p1_a.detach().clone())
    p2_b = nn.Parameter(p2_a.detach().clone())

    opt_a = HomeAdam([p1_a, p2_a], lr=1e-3, tau=1e-4, foreach=True)
    opt_b = HomeAdam([p1_b, p2_b], lr=1e-3, tau=1e-4, foreach=False)

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


def test_homeadam_ew_foreach_and_non_foreach_match() -> None:
    torch.manual_seed(13)
    p1_a = nn.Parameter(torch.randn(8))
    p2_a = nn.Parameter(torch.randn(8))
    p1_b = nn.Parameter(p1_a.detach().clone())
    p2_b = nn.Parameter(p2_a.detach().clone())

    opt_a = HomeAdamEW([p1_a, p2_a], lr=1e-3, tau=1e-4, foreach=True)
    opt_b = HomeAdamEW([p1_b, p2_b], lr=1e-3, tau=1e-4, foreach=False)

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


def test_homeadam_ew_creates_one_scalar_cache() -> None:
    p = nn.Parameter(torch.randn(8))
    opt = HomeAdamEW([p], lr=1e-3, tau=1e-6, update_mode="denom")
    _single_param_step(opt, p, torch.randn_like(p))

    one = opt.state[p]["one"]
    assert isinstance(one, torch.Tensor)
    assert one.numel() == 1
    assert one.dtype == opt.state[p]["exp_avg_sq"].dtype


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


def test_homeadam_switch_uses_bias_corrected_vhat() -> None:
    """Global switch must use v_hat (not raw v_t) as defined in Algorithm 2."""
    p = nn.Parameter(torch.tensor([10.0]))
    opt = HomeAdam(
        [p],
        lr=0.1,
        betas=(0.9, 0.99),
        eps=0.0,
        weight_decay=0.0,
        tau=1.0,
    )

    # Step-1:
    # v_t = (1-beta2) * g^2 = 0.01 * 4 = 0.04  (below tau)
    # v_hat = v_t / (1-beta2) = 4.0            (above tau)
    # Correct behavior (paper): adaptive branch, not SGDM branch.
    p.grad = torch.tensor([2.0])
    opt.step()

    # m_hat = 2.0, v_hat = 4.0 => update = lr * m_hat / v_hat = 0.05
    torch.testing.assert_close(p.data, torch.tensor([9.95]))


def test_homeadam_weight_decay_matches_adamw_when_grad_is_zero() -> None:
    """Decoupled weight decay should match AdamW exactly when grad contribution is zero."""
    p_home = nn.Parameter(torch.tensor([3.0, -2.0]))
    p_adamw = nn.Parameter(p_home.detach().clone())

    lr = 0.1
    wd = 0.2
    betas = (0.9, 0.99)
    eps = 1e-8

    opt_home = HomeAdam(
        [p_home],
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=wd,
        tau=1.0,
    )
    opt_adamw = torch.optim.AdamW(
        [p_adamw],
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=wd,
    )

    zero_grad = torch.zeros_like(p_home)
    _single_param_step(opt_home, p_home, zero_grad)
    _single_param_step(opt_adamw, p_adamw, zero_grad)

    expected = torch.tensor([3.0, -2.0]) * (1.0 - lr * wd)
    torch.testing.assert_close(p_home, expected)
    torch.testing.assert_close(p_adamw, expected)
    torch.testing.assert_close(p_home, p_adamw)


def test_homeadam_moment_buffers_match_adamw() -> None:
    """m_t and v_t recurrences should match AdamW; HomeAdam differs only in update rule."""
    torch.manual_seed(123)
    p_home = nn.Parameter(torch.randn(8))
    p_adamw = nn.Parameter(p_home.detach().clone())

    betas = (0.85, 0.97)
    lr = 1e-3
    eps = 1e-8
    wd = 1e-2

    opt_home = HomeAdam(
        [p_home], lr=lr, betas=betas, eps=eps, weight_decay=wd, tau=1e-12
    )
    opt_adamw = torch.optim.AdamW(
        [p_adamw], lr=lr, betas=betas, eps=eps, weight_decay=wd
    )

    for _ in range(5):
        grad = torch.randn_like(p_home)
        _single_param_step(opt_home, p_home, grad)
        _single_param_step(opt_adamw, p_adamw, grad)

    state_home = opt_home.state[p_home]
    state_adamw = opt_adamw.state[p_adamw]

    torch.testing.assert_close(state_home["exp_avg"], state_adamw["exp_avg"])
    torch.testing.assert_close(state_home["exp_avg_sq"], state_adamw["exp_avg_sq"])

    step_home = int(state_home["step"])
    step_adamw = int(torch.as_tensor(state_adamw["step"]).item())
    assert step_home == step_adamw

    beta1, beta2 = betas
    torch.testing.assert_close(
        state_home["beta1_power"],
        state_home["beta1_power"].new_tensor(beta1**step_home),
    )
    torch.testing.assert_close(
        state_home["beta2_power"],
        state_home["beta2_power"].new_tensor(beta2**step_home),
    )
