"""Algorithm 2: HomeAdam(W) optimizer — global-switching variant."""

from __future__ import annotations

from collections.abc import Callable
from typing import overload

import torch
from torch.optim.optimizer import Optimizer, ParamsT

from homeadam._adam_srf import _validate_hyperparams
from homeadam._functional import homeadam_apply_step


def _update_group_min_vhat(
    min_vhat_by_device: dict[torch.device, torch.Tensor],
    *,
    exp_avg_sq: torch.Tensor,
    beta2: float,
    step_count: int,
) -> None:
    """Track per-device minimum v_hat for global branch selection."""
    bias_correction2 = 1.0 - beta2**step_count
    local_min_vhat = exp_avg_sq.amin().div(bias_correction2)
    prev_min = min_vhat_by_device.get(local_min_vhat.device)
    if prev_min is None:
        min_vhat_by_device[local_min_vhat.device] = local_min_vhat
    else:
        min_vhat_by_device[local_min_vhat.device] = torch.minimum(
            prev_min, local_min_vhat
        )


def _should_use_adaptive(
    min_vhat_by_device: dict[torch.device, torch.Tensor], tau: float
) -> bool:
    """Return True iff all tracked minima satisfy min(v_hat) >= tau."""
    for device_min_vhat in min_vhat_by_device.values():
        if bool((device_min_vhat < tau).item()):
            return False
    return True


class HomeAdam(Optimizer):
    """HomeAdam(W) with global switching between adaptive and SGDM.

    Implements Algorithm 2 from "HomeAdam: Adam and AdamW Algorithms
    Sometimes Go Home to Obtain Better Provable Generalization".

    Switches all parameters in each parameter group together:
    if any component has ``v_hat < tau``, the whole group uses SGDM;
    otherwise the whole group uses the adaptive (no-sqrt) step.

    Args:
        params: Iterable of parameters or parameter-group dicts.
        lr: Learning rate.
        betas: Coefficients for first and second moment estimates.
        eps: Term added to denominator for numerical stability.
        weight_decay: Decoupled weight-decay coefficient.
        tau: Switching threshold (>0) on bias-corrected second moment.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-7,
        weight_decay: float = 0.0,
        tau: float = 1.0,
    ) -> None:
        _validate_hyperparams(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, tau=tau
        )
        defaults: dict[str, float | tuple[float, float]] = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "tau": tau,
        }
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = ...) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step."""
        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]
            tau: float = group["tau"]

            params_with_grad: list[
                tuple[
                    torch.Tensor,
                    dict[str, torch.Tensor],
                    int,
                ]
            ] = []
            min_vhat_by_device: dict[torch.device, torch.Tensor] = {}

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = torch.tensor(0, dtype=torch.int64)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                step_count = int(state["step"].item())
                state["exp_avg"].lerp_(grad, 1.0 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                _update_group_min_vhat(
                    min_vhat_by_device,
                    exp_avg_sq=state["exp_avg_sq"],
                    beta2=beta2,
                    step_count=step_count,
                )

                params_with_grad.append((p, state, step_count))

            if not params_with_grad:
                continue

            use_adaptive = _should_use_adaptive(min_vhat_by_device, tau)

            for p, state, step_count in params_with_grad:
                homeadam_apply_step(
                    param=p,
                    exp_avg=state["exp_avg"],
                    exp_avg_sq=state["exp_avg_sq"],
                    step_count=step_count,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=weight_decay,
                    use_adaptive=use_adaptive,
                )

        return loss
