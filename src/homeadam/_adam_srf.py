"""Algorithm 1: Adam(W)-srf optimizer — always-adaptive, no sqrt."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast, overload

import torch
from torch.optim.optimizer import Optimizer, ParamsT

from homeadam._functional import adam_srf_apply_step, adam_srf_scaled_update


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


def _validate_state_dtype(state_dtype: torch.dtype | None) -> None:
    if state_dtype is None:
        return
    if not state_dtype.is_floating_point:
        raise ValueError(f"state_dtype must be floating-point, got: {state_dtype}")


def _get_step(state: dict[str, Any]) -> int:
    step_obj = state.get("step", 0)
    if isinstance(step_obj, torch.Tensor):
        return int(step_obj.item())
    return int(step_obj)


def _resolve_state_dtype(
    param: torch.Tensor, state_dtype_opt: torch.dtype | None
) -> torch.dtype:
    return param.dtype if state_dtype_opt is None else state_dtype_opt


def _ensure_moment_state(
    *,
    state: dict[str, Any],
    param: torch.Tensor,
    beta1: float,
    beta2: float,
    state_dtype_opt: torch.dtype | None,
) -> None:
    """Initialize/normalize optimizer state for a parameter.

    Supports loading from previous state_dict formats by normalizing:
    - step as Python int
    - exp_avg / exp_avg_sq dtype/device
    - beta1_power / beta2_power cache tensors
    """
    step_count = _get_step(state)
    state["step"] = step_count

    state_dtype = _resolve_state_dtype(param, state_dtype_opt)
    device = param.device

    exp_avg = state.get("exp_avg")
    if not isinstance(exp_avg, torch.Tensor):
        exp_avg = torch.zeros_like(param, dtype=state_dtype)
    else:
        exp_avg = exp_avg.to(device=device, dtype=state_dtype)
    state["exp_avg"] = exp_avg

    exp_avg_sq = state.get("exp_avg_sq")
    if not isinstance(exp_avg_sq, torch.Tensor):
        exp_avg_sq = torch.zeros_like(param, dtype=state_dtype)
    else:
        exp_avg_sq = exp_avg_sq.to(device=device, dtype=state_dtype)
    state["exp_avg_sq"] = exp_avg_sq

    beta1_power = state.get("beta1_power")
    if not isinstance(beta1_power, torch.Tensor):
        beta1_power = torch.tensor(beta1**step_count, dtype=state_dtype, device=device)
    else:
        beta1_power = beta1_power.to(device=device, dtype=state_dtype)
    state["beta1_power"] = beta1_power

    beta2_power = state.get("beta2_power")
    if not isinstance(beta2_power, torch.Tensor):
        beta2_power = torch.tensor(beta2**step_count, dtype=state_dtype, device=device)
    else:
        beta2_power = beta2_power.to(device=device, dtype=state_dtype)
    state["beta2_power"] = beta2_power


class AdamSRF(Optimizer):
    """Adam(W) without square-root in the denominator (srf variant).

    Implements Algorithm 1 from "HomeAdam: Adam and AdamW Algorithms
    Sometimes Go Home to Obtain Better Provable Generalization".

    Uses ``v_hat + eps`` instead of ``sqrt(v_hat) + eps`` in the
    denominator. Weight decay is decoupled (AdamW-style).

    Args:
        params: Iterable of parameters or parameter-group dicts.
        lr: Learning rate.
        betas: Coefficients for first and second moment estimates.
        eps: Term added to denominator for numerical stability.
        weight_decay: Decoupled weight-decay coefficient.
        state_dtype: Dtype for optimizer state tensors. Defaults to fp32.
            Set to ``None`` to keep state dtype equal to parameter dtype.
        foreach: Use foreach/multi-tensor moment updates where possible.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-7,
        weight_decay: float = 0.0,
        *,
        state_dtype: torch.dtype | None = torch.float32,
        foreach: bool = True,
    ) -> None:
        _validate_hyperparams(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        _validate_state_dtype(state_dtype)
        defaults: dict[str, Any] = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "state_dtype": state_dtype,
            "foreach": foreach,
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
            lr = cast(float, group["lr"])
            beta1, beta2 = cast(tuple[float, float], group["betas"])
            eps = cast(float, group["eps"])
            weight_decay = cast(float, group["weight_decay"])
            state_dtype_opt = cast(torch.dtype | None, group["state_dtype"])
            foreach = cast(bool, group["foreach"])

            batch = _collect_group_batch(
                group=group,
                all_state=self.state,
                beta1=beta1,
                beta2=beta2,
                state_dtype_opt=state_dtype_opt,
            )
            if batch is None:
                continue

            _update_moments_batch(
                batch=batch, beta1=beta1, beta2=beta2, foreach=foreach
            )
            _increment_steps(batch.states)
            _apply_group_updates(
                batch=batch,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                weight_decay=weight_decay,
                foreach=foreach,
            )

        return loss


@dataclass
class _GroupBatch:
    params: list[torch.Tensor]
    grads: list[torch.Tensor]
    exp_avgs: list[torch.Tensor]
    exp_avg_sqs: list[torch.Tensor]
    beta1_powers: list[torch.Tensor]
    beta2_powers: list[torch.Tensor]
    states: list[dict[str, Any]]


def _collect_group_batch(
    *,
    group: dict[str, Any],
    all_state: dict[Any, Any],
    beta1: float,
    beta2: float,
    state_dtype_opt: torch.dtype | None,
) -> _GroupBatch | None:
    params: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []
    exp_avgs: list[torch.Tensor] = []
    exp_avg_sqs: list[torch.Tensor] = []
    beta1_powers: list[torch.Tensor] = []
    beta2_powers: list[torch.Tensor] = []
    states: list[dict[str, Any]] = []

    for p in group["params"]:
        if p.grad is None:
            continue
        if p.grad.is_sparse:
            raise RuntimeError("AdamSRF does not support sparse gradients")

        state = cast(dict[str, Any], all_state[p])
        _ensure_moment_state(
            state=state,
            param=p,
            beta1=beta1,
            beta2=beta2,
            state_dtype_opt=state_dtype_opt,
        )

        exp_avg = cast(torch.Tensor, state["exp_avg"])
        params.append(p)
        grads.append(p.grad.to(dtype=exp_avg.dtype))
        exp_avgs.append(exp_avg)
        exp_avg_sqs.append(cast(torch.Tensor, state["exp_avg_sq"]))
        beta1_powers.append(cast(torch.Tensor, state["beta1_power"]))
        beta2_powers.append(cast(torch.Tensor, state["beta2_power"]))
        states.append(state)

    if not params:
        return None

    return _GroupBatch(
        params=params,
        grads=grads,
        exp_avgs=exp_avgs,
        exp_avg_sqs=exp_avg_sqs,
        beta1_powers=beta1_powers,
        beta2_powers=beta2_powers,
        states=states,
    )


def _update_moments_batch(
    *,
    batch: _GroupBatch,
    beta1: float,
    beta2: float,
    foreach: bool,
) -> None:
    used_foreach = False
    if foreach:
        try:
            torch._foreach_lerp_(batch.exp_avgs, batch.grads, 1.0 - beta1)
            torch._foreach_mul_(batch.exp_avg_sqs, beta2)
            torch._foreach_addcmul_(
                batch.exp_avg_sqs, batch.grads, batch.grads, value=1.0 - beta2
            )
            torch._foreach_mul_(batch.beta1_powers, beta1)
            torch._foreach_mul_(batch.beta2_powers, beta2)
            used_foreach = True
        except RuntimeError:
            used_foreach = False

    if used_foreach:
        return

    for exp_avg, exp_avg_sq, grad, beta1_power, beta2_power in zip(
        batch.exp_avgs,
        batch.exp_avg_sqs,
        batch.grads,
        batch.beta1_powers,
        batch.beta2_powers,
        strict=True,
    ):
        exp_avg.lerp_(grad, 1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        beta1_power.mul_(beta1)
        beta2_power.mul_(beta2)


def _increment_steps(states: list[dict[str, Any]]) -> None:
    for state in states:
        state["step"] = _get_step(state) + 1


def _apply_group_updates(
    *,
    batch: _GroupBatch,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    foreach: bool,
) -> None:
    if foreach and _apply_group_updates_foreach(
        batch=batch,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
    ):
        return

    for p, state in zip(batch.params, batch.states, strict=True):
        adam_srf_apply_step(
            p,
            cast(torch.Tensor, state["exp_avg"]),
            cast(torch.Tensor, state["exp_avg_sq"]),
            step_count=None,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            beta1_power=cast(torch.Tensor, state["beta1_power"]),
            beta2_power=cast(torch.Tensor, state["beta2_power"]),
        )


def _apply_group_updates_foreach(
    *,
    batch: _GroupBatch,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> bool:
    if not batch.params:
        return False

    first_param = batch.params[0]
    if first_param.layout != torch.strided:
        return False

    for p in batch.params:
        if p.layout != torch.strided or p.device != first_param.device:
            return False

    scaled_updates: list[torch.Tensor] = []
    for p, state in zip(batch.params, batch.states, strict=True):
        scaled_update = adam_srf_scaled_update(
            cast(torch.Tensor, state["exp_avg"]),
            cast(torch.Tensor, state["exp_avg_sq"]),
            step_count=None,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            beta1_power=cast(torch.Tensor, state["beta1_power"]),
            beta2_power=cast(torch.Tensor, state["beta2_power"]),
        )
        if scaled_update.dtype != p.dtype:
            scaled_update = scaled_update.to(dtype=p.dtype)
        scaled_updates.append(scaled_update)

    try:
        if weight_decay != 0.0:
            torch._foreach_mul_(batch.params, 1.0 - lr * weight_decay)
        torch._foreach_add_(batch.params, scaled_updates, alpha=-1.0)
    except RuntimeError:
        return False

    return True
