"""Algorithm 3: Element-wise HomeAdam(W) — per-element switching."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast, overload

import torch
from torch.optim.optimizer import Optimizer, ParamsT

from homeadam._adam_srf import _ensure_moment_state, _validate_hyperparams
from homeadam._functional import (
    EWUpdateMode,
    homeadam_ew_apply_step,
    homeadam_ew_scaled_update,
)


class HomeAdamEW(Optimizer):
    """Element-wise HomeAdam(W) with per-dimension adaptive/SGDM switching.

    Implements Algorithm 3 from "HomeAdam: Adam and AdamW Algorithms
    Sometimes Go Home to Obtain Better Provable Generalization".

    Each element independently uses the adaptive step when
    ``v_hat_j >= tau`` and momentum-SGD otherwise.

    Args:
        params: Iterable of parameters or parameter-group dicts.
        lr: Learning rate.
        betas: Coefficients for first and second moment estimates.
        eps: Term added to denominator for numerical stability.
        weight_decay: Decoupled weight-decay coefficient.
        tau: Per-element switching threshold (>0) on bias-corrected second moment.
        state_dtype: Dtype for optimizer state tensors. Defaults to fp32.
            Set to ``None`` to keep state dtype equal to parameter dtype.
        foreach: Use foreach/multi-tensor moment updates where possible.
        update_mode: ``"denom"`` (paper-faithful default) or ``"where_update"``.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-7,
        weight_decay: float = 0.0,
        tau: float = 1.0,
        *,
        state_dtype: torch.dtype | None = torch.float32,
        foreach: bool = True,
        update_mode: EWUpdateMode = "denom",
    ) -> None:
        _validate_hyperparams(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, tau=tau
        )
        if state_dtype is not None and not state_dtype.is_floating_point:
            raise ValueError(f"state_dtype must be floating-point, got: {state_dtype}")
        if update_mode not in {"where_update", "denom"}:
            raise ValueError(f"Invalid update_mode: {update_mode}")

        defaults: dict[str, Any] = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "tau": tau,
            "state_dtype": state_dtype,
            "foreach": foreach,
            "update_mode": update_mode,
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
            tau = cast(float, group["tau"])
            state_dtype_opt = cast(torch.dtype | None, group["state_dtype"])
            foreach = cast(bool, group["foreach"])
            update_mode = cast(EWUpdateMode, group["update_mode"])

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
                tau=tau,
                update_mode=update_mode,
                foreach=foreach,
            )

        return loss


@dataclass
class _GroupBatch:
    params: list[torch.Tensor]
    grads: list[torch.Tensor]
    states: list[dict[str, Any]]
    exp_avgs: list[torch.Tensor]
    exp_avg_sqs: list[torch.Tensor]
    beta1_powers: list[torch.Tensor]
    beta2_powers: list[torch.Tensor]
    ones: list[torch.Tensor]


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
    states: list[dict[str, Any]] = []
    exp_avgs: list[torch.Tensor] = []
    exp_avg_sqs: list[torch.Tensor] = []
    beta1_powers: list[torch.Tensor] = []
    beta2_powers: list[torch.Tensor] = []
    ones: list[torch.Tensor] = []

    for p in group["params"]:
        if p.grad is None:
            continue
        if p.grad.is_sparse:
            raise RuntimeError("HomeAdamEW does not support sparse gradients")

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
        states.append(state)
        exp_avgs.append(exp_avg)
        exp_avg_sq = cast(torch.Tensor, state["exp_avg_sq"])
        exp_avg_sqs.append(exp_avg_sq)
        beta1_powers.append(cast(torch.Tensor, state["beta1_power"]))
        beta2_powers.append(cast(torch.Tensor, state["beta2_power"]))
        ones.append(_ensure_one_scalar(state=state, ref=exp_avg_sq))

    if not params:
        return None

    return _GroupBatch(
        params=params,
        grads=grads,
        states=states,
        exp_avgs=exp_avgs,
        exp_avg_sqs=exp_avg_sqs,
        beta1_powers=beta1_powers,
        beta2_powers=beta2_powers,
        ones=ones,
    )


def _ensure_one_scalar(*, state: dict[str, Any], ref: torch.Tensor) -> torch.Tensor:
    one = state.get("one")
    if not isinstance(one, torch.Tensor):
        one = ref.new_tensor(1.0)
    elif one.device != ref.device or one.dtype != ref.dtype:
        one = one.to(device=ref.device, dtype=ref.dtype)
    state["one"] = one
    return one


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
        state["step"] = int(cast(int, state["step"])) + 1


def _apply_group_updates(
    *,
    batch: _GroupBatch,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    tau: float,
    update_mode: EWUpdateMode,
    foreach: bool,
) -> None:
    if foreach and _apply_group_updates_foreach(
        batch=batch,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        tau=tau,
        update_mode=update_mode,
    ):
        return

    for p, state, one in zip(batch.params, batch.states, batch.ones, strict=True):
        homeadam_ew_apply_step(
            p,
            cast(torch.Tensor, state["exp_avg"]),
            cast(torch.Tensor, state["exp_avg_sq"]),
            step_count=None,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            tau=tau,
            update_mode=update_mode,
            beta1_power=cast(torch.Tensor, state["beta1_power"]),
            beta2_power=cast(torch.Tensor, state["beta2_power"]),
            one_tensor=one,
        )


def _apply_group_updates_foreach(
    *,
    batch: _GroupBatch,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    tau: float,
    update_mode: EWUpdateMode,
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
    for p, state, one in zip(batch.params, batch.states, batch.ones, strict=True):
        scaled_update = homeadam_ew_scaled_update(
            cast(torch.Tensor, state["exp_avg"]),
            cast(torch.Tensor, state["exp_avg_sq"]),
            step_count=None,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            tau=tau,
            update_mode=update_mode,
            beta1_power=cast(torch.Tensor, state["beta1_power"]),
            beta2_power=cast(torch.Tensor, state["beta2_power"]),
            one_tensor=one,
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
