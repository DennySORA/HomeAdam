"""Algorithm 2: HomeAdam(W) optimizer — global-switching variant."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast, overload

import torch
from torch.optim.optimizer import Optimizer, ParamsT

from homeadam._adam_srf import _ensure_moment_state, _validate_hyperparams
from homeadam._functional import homeadam_apply_step, homeadam_scaled_update


def _update_group_min_vhat(
    min_vhat_by_device: dict[torch.device, torch.Tensor],
    *,
    exp_avg_sq: torch.Tensor,
    beta2_power: torch.Tensor,
) -> None:
    """Track per-device minimum v_hat for global branch selection."""
    local_min_vhat = exp_avg_sq.amin().div(1.0 - beta2_power)
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
        state_dtype: Dtype for optimizer state tensors. Defaults to fp32.
            Set to ``None`` to keep state dtype equal to parameter dtype.
        foreach: Use foreach/multi-tensor moment updates where possible.
        capturable: If True, keeps global switch decision as device tensor and
            avoids host ``.item()`` sync in branch selection for single-device
            groups. Multi-device groups fall back to strict global bool logic.
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
        capturable: bool = False,
    ) -> None:
        _validate_hyperparams(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, tau=tau
        )
        if state_dtype is not None and not state_dtype.is_floating_point:
            raise ValueError(f"state_dtype must be floating-point, got: {state_dtype}")

        defaults: dict[str, Any] = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "tau": tau,
            "state_dtype": state_dtype,
            "foreach": foreach,
            "capturable": capturable,
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
            capturable = cast(bool, group["capturable"])

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
            use_adaptive_by_device = _compute_use_adaptive_by_device(
                batch=batch,
                tau=tau,
                capturable=capturable,
            )
            _apply_group_updates(
                batch=batch,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                weight_decay=weight_decay,
                use_adaptive_by_device=use_adaptive_by_device,
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

    for p in group["params"]:
        if p.grad is None:
            continue
        if p.grad.is_sparse:
            raise RuntimeError("HomeAdam does not support sparse gradients")

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
        exp_avg_sqs.append(cast(torch.Tensor, state["exp_avg_sq"]))
        beta1_powers.append(cast(torch.Tensor, state["beta1_power"]))
        beta2_powers.append(cast(torch.Tensor, state["beta2_power"]))

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
        state["step"] = int(cast(int, state["step"])) + 1


def _compute_use_adaptive_by_device(
    *,
    batch: _GroupBatch,
    tau: float,
    capturable: bool,
) -> dict[torch.device, bool | torch.Tensor]:
    min_vhat_by_device: dict[torch.device, torch.Tensor] = {}
    for exp_avg_sq, beta2_power in zip(
        batch.exp_avg_sqs, batch.beta2_powers, strict=True
    ):
        _update_group_min_vhat(
            min_vhat_by_device,
            exp_avg_sq=exp_avg_sq,
            beta2_power=beta2_power,
        )

    use_adaptive_by_device: dict[torch.device, bool | torch.Tensor] = {}
    if capturable and len(min_vhat_by_device) == 1:
        for device, device_min_vhat in min_vhat_by_device.items():
            use_adaptive_by_device[device] = device_min_vhat >= tau
        return use_adaptive_by_device

    # For multi-device groups, keep exact Algorithm 2 global semantics.
    # Cross-device tensor reduction would require extra communication/sync here.
    use_adaptive = _should_use_adaptive(min_vhat_by_device, tau)
    for device in min_vhat_by_device:
        use_adaptive_by_device[device] = use_adaptive
    return use_adaptive_by_device


def _apply_group_updates(
    *,
    batch: _GroupBatch,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    use_adaptive_by_device: dict[torch.device, bool | torch.Tensor],
    foreach: bool,
) -> None:
    if foreach and _apply_group_updates_foreach(
        batch=batch,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        use_adaptive_by_device=use_adaptive_by_device,
    ):
        return

    for p, state in zip(batch.params, batch.states, strict=True):
        homeadam_apply_step(
            p,
            cast(torch.Tensor, state["exp_avg"]),
            cast(torch.Tensor, state["exp_avg_sq"]),
            step_count=None,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            use_adaptive=use_adaptive_by_device[p.device],
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
    use_adaptive_by_device: dict[torch.device, bool | torch.Tensor],
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
        scaled_update = homeadam_scaled_update(
            cast(torch.Tensor, state["exp_avg"]),
            cast(torch.Tensor, state["exp_avg_sq"]),
            step_count=None,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            use_adaptive=use_adaptive_by_device[p.device],
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
