"""Algorithm 1: Adam(W)-srf optimizer — always-adaptive, no sqrt."""

from __future__ import annotations

from collections.abc import Callable
from typing import overload

import torch
from torch.optim.optimizer import Optimizer, ParamsT

from homeadam._functional import adam_srf_step


class AdamSRF(Optimizer):
    """Adam(W) without square-root in the denominator (srf variant).

    Implements Algorithm 1 from "HomeAdam: Adam and AdamW Algorithms
    Sometimes Go Home to Obtain Better Provable Generalization".

    Uses ``v_hat + eps`` instead of ``sqrt(v_hat) + eps`` in the
    denominator.  Weight decay is decoupled (AdamW-style).

    Args:
        params: Iterable of parameters or parameter-group dicts.
        lr: Learning rate.
        betas: Coefficients for first and second moment estimates.
        eps: Term added to denominator for numerical stability.
        weight_decay: Decoupled weight-decay coefficient.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-7,
        weight_decay: float = 0.0,
    ) -> None:
        _validate_hyperparams(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        defaults: dict[str, float | tuple[float, float]] = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
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
                step_count: int = int(state["step"].item())

                adam_srf_step(
                    param=p,
                    grad=grad,
                    exp_avg=state["exp_avg"],
                    exp_avg_sq=state["exp_avg_sq"],
                    step_count=step_count,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=weight_decay,
                )

        return loss


def _validate_hyperparams(
    *,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    tau: float | None = None,
) -> None:
    if lr < 0.0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if not 0.0 <= betas[0] < 1.0:
        raise ValueError(f"Invalid beta1: {betas[0]}")
    if not 0.0 <= betas[1] < 1.0:
        raise ValueError(f"Invalid beta2: {betas[1]}")
    if eps < 0.0:
        raise ValueError(f"Invalid epsilon: {eps}")
    if weight_decay < 0.0:
        raise ValueError(f"Invalid weight_decay: {weight_decay}")
    if tau is not None and tau <= 0.0:
        raise ValueError(f"Invalid tau: {tau}")
