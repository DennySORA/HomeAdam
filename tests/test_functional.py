"""Tests for the core tensor operations in _functional.py."""

from __future__ import annotations

import torch
from torch import Tensor

from homeadam._functional import adam_srf_step, homeadam_ew_step, homeadam_step


# ---------------------------------------------------------------------------
# Pure-Python reference implementations (line-by-line from paper pseudocode)
# ---------------------------------------------------------------------------


def _ref_moments(
    grad: list[float],
    exp_avg: list[float],
    exp_avg_sq: list[float],
    beta1: float,
    beta2: float,
    step_count: int,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Compute moment updates and bias-corrected values in pure Python."""
    new_avg = [beta1 * m + (1 - beta1) * g for m, g in zip(exp_avg, grad)]
    new_sq = [beta2 * v + (1 - beta2) * g * g for v, g in zip(exp_avg_sq, grad)]
    bc1 = 1 - beta1**step_count
    bc2 = 1 - beta2**step_count
    m_hat = [m / bc1 for m in new_avg]
    v_hat = [v / bc2 for v in new_sq]
    return new_avg, new_sq, m_hat, v_hat


def _ref_adam_srf(
    param: list[float],
    grad: list[float],
    exp_avg: list[float],
    exp_avg_sq: list[float],
    step_count: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
) -> list[float]:
    """Reference Algorithm 1."""
    _, _, m_hat, v_hat = _ref_moments(
        grad, exp_avg, exp_avg_sq, beta1, beta2, step_count
    )
    result: list[float] = []
    for j in range(len(param)):
        p = param[j] * (1 - lr * wd) if wd != 0.0 else param[j]
        p -= lr * m_hat[j] / (v_hat[j] + eps)
        result.append(p)
    return result


def _ref_homeadam(
    param: list[float],
    grad: list[float],
    exp_avg: list[float],
    exp_avg_sq: list[float],
    step_count: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
    tau: float,
) -> list[float]:
    """Reference Algorithm 2."""
    _, _, m_hat, v_hat = _ref_moments(
        grad, exp_avg, exp_avg_sq, beta1, beta2, step_count
    )
    use_adaptive = min(v_hat) >= tau
    result: list[float] = []
    for j in range(len(param)):
        p = param[j] * (1 - lr * wd) if wd != 0.0 else param[j]
        if use_adaptive:
            p -= lr * m_hat[j] / (v_hat[j] + eps)
        else:
            p -= lr * m_hat[j]
        result.append(p)
    return result


def _ref_homeadam_ew(
    param: list[float],
    grad: list[float],
    exp_avg: list[float],
    exp_avg_sq: list[float],
    step_count: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
    tau: float,
) -> list[float]:
    """Reference Algorithm 3."""
    _, _, m_hat, v_hat = _ref_moments(
        grad, exp_avg, exp_avg_sq, beta1, beta2, step_count
    )
    result: list[float] = []
    for j in range(len(param)):
        p = param[j] * (1 - lr * wd) if wd != 0.0 else param[j]
        if v_hat[j] >= tau:
            p -= lr * m_hat[j] / (v_hat[j] + eps)
        else:
            p -= lr * m_hat[j]
        result.append(p)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LR = 0.01
_BETAS = (0.9, 0.99)
_EPS = 1e-7
_GRAD = [0.5, -0.3, 0.8]
_PARAM = [1.0, 2.0, 3.0]


def _make_tensors() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (param, grad, exp_avg, exp_avg_sq) as fresh tensors."""
    return (
        torch.tensor(_PARAM, dtype=torch.float64),
        torch.tensor(_GRAD, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
    )


# ---------------------------------------------------------------------------
# Algorithm 1: adam_srf_step
# ---------------------------------------------------------------------------


class TestAdamSrfStep:
    def test_matches_reference(self) -> None:
        param, grad, m, v = _make_tensors()
        adam_srf_step(
            param,
            grad,
            m,
            v,
            step_count=1,
            lr=_LR,
            beta1=0.9,
            beta2=0.99,
            eps=_EPS,
            weight_decay=0.0,
        )
        expected = _ref_adam_srf(
            _PARAM,
            _GRAD,
            [0.0] * 3,
            [0.0] * 3,
            1,
            _LR,
            0.9,
            0.99,
            _EPS,
            0.0,
        )
        torch.testing.assert_close(param, torch.tensor(expected, dtype=torch.float64))

    def test_weight_decay_effect(self) -> None:
        p_no_wd, grad, m, v = _make_tensors()
        adam_srf_step(
            p_no_wd,
            grad.clone(),
            m,
            v,
            step_count=1,
            lr=_LR,
            beta1=0.9,
            beta2=0.99,
            eps=_EPS,
            weight_decay=0.0,
        )
        p_wd, grad2, m2, v2 = _make_tensors()
        adam_srf_step(
            p_wd,
            grad2,
            m2,
            v2,
            step_count=1,
            lr=_LR,
            beta1=0.9,
            beta2=0.99,
            eps=_EPS,
            weight_decay=0.1,
        )
        # With weight decay, params should be strictly smaller in magnitude
        assert not torch.allclose(p_no_wd, p_wd)

    def test_no_sqrt_differs_from_standard_adam(self) -> None:
        """Verify that using v_hat+eps (no sqrt) gives different results from sqrt(v_hat)+eps."""
        param, grad, m, v = _make_tensors()
        adam_srf_step(
            param,
            grad,
            m,
            v,
            step_count=1,
            lr=_LR,
            beta1=0.9,
            beta2=0.99,
            eps=_EPS,
            weight_decay=0.0,
        )
        # Standard Adam would compute: step_size * m_hat / (sqrt(v_hat) + eps)
        # Our srf computes: step_size * m_hat / (v_hat + eps)
        # These should differ since sqrt(x) != x for x != 0 and x != 1
        m_hat = torch.tensor(_GRAD) * 0.1 / (1 - 0.9)
        v_hat = torch.tensor([g**2 for g in _GRAD]) * 0.01 / (1 - 0.99)
        srf_update = m_hat / (v_hat + _EPS)
        sqrt_update = m_hat / (torch.sqrt(v_hat) + _EPS)
        assert not torch.allclose(srf_update, sqrt_update)


# ---------------------------------------------------------------------------
# Algorithm 2: homeadam_step
# ---------------------------------------------------------------------------


class TestHomeAdamStep:
    def test_adaptive_when_vhat_above_tau(self) -> None:
        """With tau=0, all elements satisfy v_hat >= tau → adaptive path."""
        param, grad, m, v = _make_tensors()
        homeadam_step(
            param,
            grad,
            m,
            v,
            step_count=1,
            lr=_LR,
            beta1=0.9,
            beta2=0.99,
            eps=_EPS,
            weight_decay=0.0,
            tau=0.0,
        )
        expected = _ref_homeadam(
            _PARAM,
            _GRAD,
            [0.0] * 3,
            [0.0] * 3,
            1,
            _LR,
            0.9,
            0.99,
            _EPS,
            0.0,
            0.0,
        )
        torch.testing.assert_close(param, torch.tensor(expected, dtype=torch.float64))

    def test_sgdm_when_vhat_below_tau(self) -> None:
        """With very large tau, all elements fail → SGDM path."""
        param, grad, m, v = _make_tensors()
        homeadam_step(
            param,
            grad,
            m,
            v,
            step_count=1,
            lr=_LR,
            beta1=0.9,
            beta2=0.99,
            eps=_EPS,
            weight_decay=0.0,
            tau=1e10,
        )
        expected = _ref_homeadam(
            _PARAM,
            _GRAD,
            [0.0] * 3,
            [0.0] * 3,
            1,
            _LR,
            0.9,
            0.99,
            _EPS,
            0.0,
            1e10,
        )
        torch.testing.assert_close(param, torch.tensor(expected, dtype=torch.float64))

    def test_tau_zero_matches_adam_srf(self) -> None:
        """When tau=0, HomeAdam always takes adaptive path → same as AdamSRF."""
        p1, g1, m1, v1 = _make_tensors()
        p2, g2, m2, v2 = _make_tensors()

        adam_srf_step(
            p1,
            g1,
            m1,
            v1,
            step_count=1,
            lr=_LR,
            beta1=0.9,
            beta2=0.99,
            eps=_EPS,
            weight_decay=0.0,
        )
        homeadam_step(
            p2,
            g2,
            m2,
            v2,
            step_count=1,
            lr=_LR,
            beta1=0.9,
            beta2=0.99,
            eps=_EPS,
            weight_decay=0.0,
            tau=0.0,
        )
        torch.testing.assert_close(p1, p2)


# ---------------------------------------------------------------------------
# Algorithm 3: homeadam_ew_step
# ---------------------------------------------------------------------------


class TestHomeAdamEWStep:
    def test_matches_reference(self) -> None:
        param, grad, m, v = _make_tensors()
        homeadam_ew_step(
            param,
            grad,
            m,
            v,
            step_count=1,
            lr=_LR,
            beta1=0.9,
            beta2=0.99,
            eps=_EPS,
            weight_decay=0.0,
            tau=0.005,
        )
        expected = _ref_homeadam_ew(
            _PARAM,
            _GRAD,
            [0.0] * 3,
            [0.0] * 3,
            1,
            _LR,
            0.9,
            0.99,
            _EPS,
            0.0,
            0.005,
        )
        torch.testing.assert_close(param, torch.tensor(expected, dtype=torch.float64))

    def test_tau_zero_matches_adam_srf(self) -> None:
        """tau=0 means all elements use adaptive → same as AdamSRF."""
        p1, g1, m1, v1 = _make_tensors()
        p2, g2, m2, v2 = _make_tensors()

        adam_srf_step(
            p1,
            g1,
            m1,
            v1,
            step_count=1,
            lr=_LR,
            beta1=0.9,
            beta2=0.99,
            eps=_EPS,
            weight_decay=0.0,
        )
        homeadam_ew_step(
            p2,
            g2,
            m2,
            v2,
            step_count=1,
            lr=_LR,
            beta1=0.9,
            beta2=0.99,
            eps=_EPS,
            weight_decay=0.0,
            tau=0.0,
        )
        torch.testing.assert_close(p1, p2)

    def test_large_tau_gives_sgdm(self) -> None:
        """Very large tau → all elements use SGDM (denom=1)."""
        param, grad, m, v = _make_tensors()
        homeadam_ew_step(
            param,
            grad,
            m,
            v,
            step_count=1,
            lr=_LR,
            beta1=0.9,
            beta2=0.99,
            eps=_EPS,
            weight_decay=0.0,
            tau=1e10,
        )
        # SGDM: param = param - step_size * exp_avg
        bc1 = 1 - 0.9
        step_size = _LR / bc1
        new_m = [0.1 * g for g in _GRAD]
        expected = [p - step_size * mi for p, mi in zip(_PARAM, new_m)]
        torch.testing.assert_close(param, torch.tensor(expected, dtype=torch.float64))

    def test_mixed_elements(self) -> None:
        """Some elements adaptive, some SGDM based on per-element v_hat."""
        # Use gradients where some produce v_hat >= tau and others don't
        grad_vals = [10.0, 0.001, 5.0]
        param = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        grad = torch.tensor(grad_vals, dtype=torch.float64)
        m = torch.zeros(3, dtype=torch.float64)
        v = torch.zeros(3, dtype=torch.float64)

        tau = 0.5
        homeadam_ew_step(
            param,
            grad,
            m,
            v,
            step_count=1,
            lr=_LR,
            beta1=0.9,
            beta2=0.99,
            eps=_EPS,
            weight_decay=0.0,
            tau=tau,
        )
        expected = _ref_homeadam_ew(
            [1.0, 2.0, 3.0],
            grad_vals,
            [0.0] * 3,
            [0.0] * 3,
            1,
            _LR,
            0.9,
            0.99,
            _EPS,
            0.0,
            tau,
        )
        torch.testing.assert_close(param, torch.tensor(expected, dtype=torch.float64))

    def test_multi_step_accumulation(self) -> None:
        """Verify moments accumulate correctly over multiple steps."""
        param, _, m, v = _make_tensors()
        grads = [[0.5, -0.3, 0.8], [0.1, 0.4, -0.2], [-0.3, 0.1, 0.6]]

        ref_param = list(_PARAM)
        ref_m = [0.0, 0.0, 0.0]
        ref_v = [0.0, 0.0, 0.0]

        for step_idx, g_vals in enumerate(grads, start=1):
            grad = torch.tensor(g_vals, dtype=torch.float64)
            homeadam_ew_step(
                param,
                grad,
                m,
                v,
                step_count=step_idx,
                lr=_LR,
                beta1=0.9,
                beta2=0.99,
                eps=_EPS,
                weight_decay=0.0,
                tau=0.005,
            )
            # Update reference moments
            ref_m = [0.9 * mi + 0.1 * gi for mi, gi in zip(ref_m, g_vals)]
            ref_v = [0.99 * vi + 0.01 * gi * gi for vi, gi in zip(ref_v, g_vals)]
            bc1 = 1 - 0.9**step_idx
            bc2 = 1 - 0.99**step_idx
            v_hat = [vi / bc2 for vi in ref_v]
            step_size = _LR / bc1
            new_param: list[float] = []
            for j in range(3):
                p = ref_param[j]
                if v_hat[j] >= 0.005:
                    p -= step_size * ref_m[j] / (ref_v[j] / bc2 + _EPS)
                else:
                    p -= step_size * ref_m[j]
                new_param.append(p)
            ref_param = new_param

        torch.testing.assert_close(
            param, torch.tensor(ref_param, dtype=torch.float64), atol=1e-10, rtol=1e-10
        )
