"""Pure tensor operations for HomeAdam optimizer variants.

Implements the core math for three algorithms from the paper
"HomeAdam: Adam and AdamW Algorithms Sometimes Go Home to Obtain
Better Provable Generalization" (arXiv:2603.02649v1).

All functions operate on single tensors (one parameter at a time),
following PyTorch optimizer conventions.
"""

from __future__ import annotations

import torch
from torch import Tensor


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
    # Moment updates
    exp_avg.lerp_(grad, 1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    # Bias correction
    bias_correction1 = 1.0 - beta1**step_count
    bias_correction2 = 1.0 - beta2**step_count

    step_size = lr / bias_correction1

    # Decoupled weight decay (applied before gradient step)
    if weight_decay != 0.0:
        param.mul_(1.0 - lr * weight_decay)

    # Adaptive step: param -= step_size * m_hat / (v_hat + eps)
    # v_hat = exp_avg_sq / bias_correction2, so denominator = exp_avg_sq / bc2 + eps
    denom = exp_avg_sq.div(bias_correction2).add_(eps)
    param.addcdiv_(exp_avg, denom, value=-step_size)


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

    If ``min(v_hat) >= tau``, uses adaptive step; otherwise uses momentum-SGD.
    When ``force_adaptive`` is set, the branch decision is overridden.
    Updates ``param``, ``exp_avg``, and ``exp_avg_sq`` in-place.
    """
    # Moment updates
    exp_avg.lerp_(grad, 1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    if force_adaptive is None:
        bias_correction2 = 1.0 - beta2**step_count
        min_vhat = float(exp_avg_sq.amin().item()) / bias_correction2
        use_adaptive = min_vhat >= tau
    else:
        use_adaptive = force_adaptive

    homeadam_apply_step(
        param=param,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        step_count=step_count,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        use_adaptive=use_adaptive,
    )


def homeadam_apply_step(
    param: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    *,
    step_count: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    use_adaptive: bool,
) -> None:
    """Apply HomeAdam(W) parameter update with precomputed moments."""
    # Bias correction
    bias_correction1 = 1.0 - beta1**step_count
    bias_correction2 = 1.0 - beta2**step_count

    step_size = lr / bias_correction1

    # Decoupled weight decay
    if weight_decay != 0.0:
        param.mul_(1.0 - lr * weight_decay)

    if use_adaptive:
        denom = exp_avg_sq.div(bias_correction2).add_(eps)
        param.addcdiv_(exp_avg, denom, value=-step_size)
        return

    # Momentum-SGD step: param -= step_size * m_hat
    param.add_(exp_avg, alpha=-step_size)


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
) -> None:
    """Algorithm 3: Element-wise HomeAdam(W) — per-element switching.

    Each element independently uses adaptive or momentum-SGD based on
    whether ``v_hat_j >= tau``.
    Updates ``param``, ``exp_avg``, and ``exp_avg_sq`` in-place.
    """
    # Moment updates
    exp_avg.lerp_(grad, 1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    # Bias correction
    bias_correction1 = 1.0 - beta1**step_count
    bias_correction2 = 1.0 - beta2**step_count

    step_size = lr / bias_correction1

    # Decoupled weight decay
    if weight_decay != 0.0:
        param.mul_(1.0 - lr * weight_decay)

    v_hat = exp_avg_sq / bias_correction2

    # Per-element: adaptive where v_hat >= tau, else SGDM (denom=1)
    denom = torch.where(v_hat >= tau, v_hat + eps, v_hat.new_tensor(1.0))
    param.addcdiv_(exp_avg, denom, value=-step_size)
