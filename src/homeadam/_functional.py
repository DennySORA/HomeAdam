"""Pure tensor operations for HomeAdam optimizer variants.

Implements the core math for three algorithms from the paper
"HomeAdam: Adam and AdamW Algorithms Sometimes Go Home to Obtain
Better Provable Generalization" (arXiv:2603.02649v1).

All functions operate on single tensors (one parameter at a time),
following PyTorch optimizer conventions.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


EWUpdateMode = Literal["where_update", "denom"]


def _bias_correction_tensor(
    ref: Tensor,
    *,
    beta: float,
    step_count: int | None,
    beta_power: Tensor | float | None,
) -> Tensor:
    """Return scalar tensor ``1 - beta**step`` using either cache or step_count."""
    if beta_power is None:
        if step_count is None:
            raise ValueError("step_count is required when beta_power is not provided")
        return ref.new_tensor(1.0 - beta**step_count)

    if isinstance(beta_power, Tensor):
        if beta_power.device != ref.device or beta_power.dtype != ref.dtype:
            return (1.0 - beta_power).to(device=ref.device, dtype=ref.dtype)
        return 1.0 - beta_power

    return ref.new_tensor(1.0 - float(beta_power))


def _apply_update(
    param: Tensor,
    *,
    update: Tensor,
    step_size_t: Tensor,
    weight_decay: float,
    lr: float,
) -> None:
    """Apply decoupled WD and scaled update to parameter in-place.

    Supports mixed-dtype state by casting the final update to ``param.dtype``.
    """
    if weight_decay != 0.0:
        param.mul_(1.0 - lr * weight_decay)

    scaled_update = update * step_size_t
    if scaled_update.dtype != param.dtype:
        param.add_(scaled_update.to(dtype=param.dtype), alpha=-1.0)
        return

    param.add_(scaled_update, alpha=-1.0)


def _update_moments(
    *,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    grad: Tensor,
    beta1: float,
    beta2: float,
) -> None:
    """Update first/second moments in-place."""
    exp_avg.lerp_(grad, 1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)


def adam_srf_apply_step(
    param: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    *,
    step_count: int | None,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    beta1_power: Tensor | float | None = None,
    beta2_power: Tensor | float | None = None,
) -> None:
    """Apply Algorithm 1 update using precomputed moments."""
    bias_correction1 = _bias_correction_tensor(
        exp_avg,
        beta=beta1,
        step_count=step_count,
        beta_power=beta1_power,
    )
    bias_correction2 = _bias_correction_tensor(
        exp_avg_sq,
        beta=beta2,
        step_count=step_count,
        beta_power=beta2_power,
    )

    step_size_t = exp_avg.new_tensor(lr) / bias_correction1
    denom = exp_avg_sq / bias_correction2 + eps
    update = exp_avg / denom

    _apply_update(
        param,
        update=update,
        step_size_t=step_size_t,
        weight_decay=weight_decay,
        lr=lr,
    )


def adam_srf_step(
    param: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    *,
    step_count: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> None:
    """Algorithm 1: Adam(W)-srf — always-adaptive update without sqrt.

    Updates ``param``, ``exp_avg``, and ``exp_avg_sq`` in-place.
    """
    _update_moments(
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        grad=grad,
        beta1=beta1,
        beta2=beta2,
    )
    adam_srf_apply_step(
        param,
        exp_avg,
        exp_avg_sq,
        step_count=step_count,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
    )


def homeadam_apply_step(
    param: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    *,
    step_count: int | None,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    use_adaptive: bool | Tensor,
    beta1_power: Tensor | float | None = None,
    beta2_power: Tensor | float | None = None,
) -> None:
    """Apply HomeAdam(W) parameter update with precomputed moments."""
    bias_correction1 = _bias_correction_tensor(
        exp_avg,
        beta=beta1,
        step_count=step_count,
        beta_power=beta1_power,
    )
    bias_correction2 = _bias_correction_tensor(
        exp_avg_sq,
        beta=beta2,
        step_count=step_count,
        beta_power=beta2_power,
    )

    step_size_t = exp_avg.new_tensor(lr) / bias_correction1

    if isinstance(use_adaptive, bool):
        if use_adaptive:
            denom = exp_avg_sq / bias_correction2 + eps
            update = exp_avg / denom
        else:
            update = exp_avg
    else:
        use_adaptive_t = use_adaptive.to(device=exp_avg.device)
        denom = exp_avg_sq / bias_correction2 + eps
        adaptive_update = exp_avg / denom
        update = torch.where(use_adaptive_t, adaptive_update, exp_avg)

    _apply_update(
        param,
        update=update,
        step_size_t=step_size_t,
        weight_decay=weight_decay,
        lr=lr,
    )


def homeadam_step(
    param: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    *,
    step_count: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    tau: float,
    force_adaptive: bool | None = None,
) -> None:
    """Algorithm 2: HomeAdam(W) — global-switching update.

    For a single tensor, this computes local ``min(v_hat)``. Optimizer-level
    global switching (across tensors) should prefer ``homeadam_apply_step``
    with a precomputed group-level decision.
    """
    _update_moments(
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        grad=grad,
        beta1=beta1,
        beta2=beta2,
    )

    bias_correction2 = _bias_correction_tensor(
        exp_avg_sq,
        beta=beta2,
        step_count=step_count,
        beta_power=None,
    )

    if force_adaptive is None:
        use_adaptive: bool | Tensor = exp_avg_sq.amin().div(bias_correction2) >= tau
    else:
        use_adaptive = force_adaptive

    homeadam_apply_step(
        param,
        exp_avg,
        exp_avg_sq,
        step_count=step_count,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        use_adaptive=use_adaptive,
    )


def homeadam_ew_apply_step(
    param: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    *,
    step_count: int | None,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    tau: float,
    update_mode: EWUpdateMode = "denom",
    beta1_power: Tensor | float | None = None,
    beta2_power: Tensor | float | None = None,
) -> None:
    """Apply Algorithm 3 update using precomputed moments."""
    bias_correction1 = _bias_correction_tensor(
        exp_avg,
        beta=beta1,
        step_count=step_count,
        beta_power=beta1_power,
    )
    bias_correction2 = _bias_correction_tensor(
        exp_avg_sq,
        beta=beta2,
        step_count=step_count,
        beta_power=beta2_power,
    )

    step_size_t = exp_avg.new_tensor(lr) / bias_correction1
    v_hat = exp_avg_sq / bias_correction2

    if update_mode == "denom":
        denom = torch.where(v_hat >= tau, v_hat + eps, v_hat.new_tensor(1.0))
        update = exp_avg / denom
    elif update_mode == "where_update":
        adaptive_update = exp_avg / (v_hat + eps)
        update = torch.where(v_hat >= tau, adaptive_update, exp_avg)
    else:
        raise ValueError(f"Unknown update_mode: {update_mode}")

    _apply_update(
        param,
        update=update,
        step_size_t=step_size_t,
        weight_decay=weight_decay,
        lr=lr,
    )


def homeadam_ew_step(
    param: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    *,
    step_count: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    tau: float,
    update_mode: EWUpdateMode = "denom",
) -> None:
    """Algorithm 3: Element-wise HomeAdam(W) — per-element switching."""
    _update_moments(
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        grad=grad,
        beta1=beta1,
        beta2=beta2,
    )
    homeadam_ew_apply_step(
        param,
        exp_avg,
        exp_avg_sq,
        step_count=step_count,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        tau=tau,
        update_mode=update_mode,
    )
