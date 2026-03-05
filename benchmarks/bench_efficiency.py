"""Efficiency benchmarks for HomeAdam optimizer variants.

Measures:
1. Micro-benchmark: EW legacy `denom` path vs `where_update` path
2. Per-step throughput across parameter scales
3. Peak memory comparison

Run: uv run python benchmarks/bench_efficiency.py
"""

from __future__ import annotations

import gc
import statistics
import sys
import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from homeadam import AdamSRF, HomeAdam, HomeAdamEW


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchResult:
    """Timing result for a single benchmark scenario."""

    name: str
    median_us: float
    iqr_us: float
    samples: int


@dataclass(frozen=True)
class MemResult:
    """Memory measurement for a single optimizer."""

    name: str
    state_bytes: int
    param_bytes: int


def _bench(
    fn: object,
    *,
    warmup: int = 20,
    repeats: int = 100,
) -> tuple[float, float]:
    """Return (median_us, iqr_us) of calling *fn* repeatedly."""
    if not callable(fn):
        raise TypeError(f"Expected callable, got {type(fn)}")
    for _ in range(warmup):
        fn()

    times_ns: list[int] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        times_ns.append(time.perf_counter_ns() - start)

    times_us = [t / 1_000 for t in times_ns]
    times_us.sort()
    median = statistics.median(times_us)
    q1 = times_us[len(times_us) // 4]
    q3 = times_us[3 * len(times_us) // 4]
    return median, q3 - q1


def _print_table(title: str, rows: list[BenchResult]) -> None:
    header = f"{'Name':<45s} {'Median (us)':>14s} {'IQR (us)':>12s} {'N':>6s}"
    sep = "-" * len(header)
    sys.stdout.write(f"\n{'=' * len(header)}\n")
    sys.stdout.write(f" {title}\n")
    sys.stdout.write(f"{'=' * len(header)}\n")
    sys.stdout.write(f"{header}\n{sep}\n")
    for r in rows:
        sys.stdout.write(
            f"{r.name:<45s} {r.median_us:>14.1f} {r.iqr_us:>12.1f} {r.samples:>6d}\n"
        )
    sys.stdout.write(f"{sep}\n")


def _make_param_with_grad(dim: int) -> nn.Parameter:
    """Create a parameter with a synthetic gradient attached."""
    p = nn.Parameter(torch.randn(dim))
    p.grad = torch.randn(dim)
    return p


# ---------------------------------------------------------------------------
# Benchmark 1: EW update path micro-benchmark
# ---------------------------------------------------------------------------


def _ew_kernel_denom(
    param: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    grad: Tensor,
    tau: float,
    eps: float,
    step_size: float,
    bc2: float,
) -> None:
    v_hat = exp_avg_sq / bc2
    denom = torch.where(v_hat >= tau, v_hat + eps, v_hat.new_tensor(1.0))
    param.addcdiv_(exp_avg, denom, value=-step_size)


def _ew_kernel_where_update(
    param: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    grad: Tensor,
    tau: float,
    eps: float,
    step_size: float,
    bc2: float,
) -> None:
    v_hat = exp_avg_sq / bc2
    adaptive_update = exp_avg / (v_hat + eps)
    update = torch.where(v_hat >= tau, adaptive_update, exp_avg)
    param.add_(update, alpha=-step_size)


def bench_ew_update_paths() -> list[BenchResult]:
    """Benchmark 1: Compare EW legacy vs fast update path."""
    results: list[BenchResult] = []
    repeats = 200

    for dim in [1_000, 100_000, 10_000_000]:
        label = f"d={dim:>10_d}"

        grad = torch.randn(dim)
        shared = {
            "grad": grad,
            "tau": 0.5,
            "eps": 1e-7,
            "step_size": 0.001,
            "bc2": 0.99,
        }

        # --- legacy denom path ---
        p1 = torch.randn(dim)
        m1 = torch.randn(dim)
        v1 = torch.rand(dim)

        def run_denom(
            _p: Tensor = p1,
            _m: Tensor = m1,
            _v: Tensor = v1,
            _kw: dict[str, object] = shared,
        ) -> None:
            _ew_kernel_denom(_p, _m, _v, **_kw)  # type: ignore[arg-type]

        med_denom, iqr_denom = _bench(run_denom, repeats=repeats)
        results.append(
            BenchResult(f"denom        {label}", med_denom, iqr_denom, repeats)
        )

        # --- where_update path ---
        p2 = torch.randn(dim)
        m2 = torch.randn(dim)
        v2 = torch.rand(dim)

        def run_where_update(
            _p: Tensor = p2,
            _m: Tensor = m2,
            _v: Tensor = v2,
            _kw: dict[str, object] = shared,
        ) -> None:
            _ew_kernel_where_update(_p, _m, _v, **_kw)  # type: ignore[arg-type]

        med_where, iqr_where = _bench(run_where_update, repeats=repeats)
        results.append(
            BenchResult(f"where_update {label}", med_where, iqr_where, repeats)
        )

    return results


# ---------------------------------------------------------------------------
# Benchmark 2: Optimizer step throughput
# ---------------------------------------------------------------------------


def bench_optimizer_step_throughput() -> list[BenchResult]:
    """Benchmark 2: Per-step wall-clock for each optimizer at various scales."""
    results: list[BenchResult] = []
    repeats = 100

    opt_configs: list[tuple[str, type[torch.optim.Optimizer], dict[str, object]]] = [
        ("AdamW (PyTorch)", torch.optim.AdamW, {"lr": 1e-3}),
        ("AdamSRF", AdamSRF, {"lr": 1e-3}),
        ("HomeAdam  tau=1e-10", HomeAdam, {"lr": 1e-3, "tau": 1e-10}),
        ("HomeAdam  tau=1e10", HomeAdam, {"lr": 1e-3, "tau": 1e10}),
        (
            "HomeAdamEW where_update",
            HomeAdamEW,
            {"lr": 1e-3, "tau": 0.5, "update_mode": "where_update"},
        ),
        (
            "HomeAdamEW denom",
            HomeAdamEW,
            {"lr": 1e-3, "tau": 0.5, "update_mode": "denom"},
        ),
    ]

    for dim in [1_000, 100_000, 10_000_000]:
        label = f"d={dim:>10_d}"

        for opt_name, opt_cls, kwargs in opt_configs:
            p = _make_param_with_grad(dim)
            opt = opt_cls([p], **kwargs)  # type: ignore[arg-type]

            def step(_opt: torch.optim.Optimizer = opt) -> None:
                _opt.step()

            med, iqr = _bench(step, warmup=10, repeats=repeats)
            results.append(BenchResult(f"{opt_name:<22s} {label}", med, iqr, repeats))

    return results


# ---------------------------------------------------------------------------
# Benchmark 3: Memory — optimizer state size
# ---------------------------------------------------------------------------


def _measure_state_memory(
    opt_cls: type[torch.optim.Optimizer],
    kwargs: dict[str, object],
    dim: int,
) -> int:
    """Return total bytes of optimizer state tensors after one step."""
    p = _make_param_with_grad(dim)
    opt = opt_cls([p], **kwargs)  # type: ignore[arg-type]
    opt.step()

    total = 0
    for state in opt.state.values():
        if isinstance(state, dict):
            for v in state.values():
                if isinstance(v, Tensor):
                    total += v.nelement() * v.element_size()
    return total


def bench_memory() -> list[MemResult]:
    """Benchmark 3: Optimizer state memory footprint."""
    dim = 10_000_000
    param_bytes = dim * 4  # float32

    configs: list[tuple[str, type[torch.optim.Optimizer], dict[str, object]]] = [
        ("AdamW (PyTorch)", torch.optim.AdamW, {"lr": 1e-3}),
        ("AdamSRF", AdamSRF, {"lr": 1e-3}),
        ("HomeAdam", HomeAdam, {"lr": 1e-3, "tau": 0.5}),
        ("HomeAdamEW", HomeAdamEW, {"lr": 1e-3, "tau": 0.5}),
    ]

    results: list[MemResult] = []
    for name, opt_cls, kwargs in configs:
        gc.collect()
        state_bytes = _measure_state_memory(opt_cls, kwargs, dim)
        results.append(MemResult(name, state_bytes, param_bytes))

    return results


def _print_mem_table(rows: list[MemResult]) -> None:
    header = (
        f"{'Optimizer':<22s} {'State (MB)':>12s} {'Param (MB)':>12s} {'Overhead':>10s}"
    )
    sep = "-" * len(header)
    sys.stdout.write(f"\n{'=' * len(header)}\n")
    sys.stdout.write(" Memory Footprint (d=10,000,000)\n")
    sys.stdout.write(f"{'=' * len(header)}\n")
    sys.stdout.write(f"{header}\n{sep}\n")
    for r in rows:
        state_mb = r.state_bytes / (1024 * 1024)
        param_mb = r.param_bytes / (1024 * 1024)
        overhead = f"{r.state_bytes / r.param_bytes:.1f}x"
        sys.stdout.write(
            f"{r.name:<22s} {state_mb:>12.1f} {param_mb:>12.1f} {overhead:>10s}\n"
        )
    sys.stdout.write(f"{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all benchmarks and print results."""
    sys.stdout.write("HomeAdam Efficiency Benchmarks\n")
    sys.stdout.write(f"PyTorch {torch.__version__} | CPU\n")
    sys.stdout.write(f"Threads: {torch.get_num_threads()}\n")

    sys.stdout.write("\n[1/3] EW denom vs where_update ...\n")
    _print_table(
        "Benchmark 1: EW Update Path (denom vs where_update)",
        bench_ew_update_paths(),
    )

    sys.stdout.write("\n[2/3] Optimizer step throughput ...\n")
    _print_table(
        "Benchmark 2: Per-Step Throughput",
        bench_optimizer_step_throughput(),
    )

    sys.stdout.write("\n[3/3] Memory footprint ...\n")
    _print_mem_table(bench_memory())

    sys.stdout.write("\nDone.\n")


if __name__ == "__main__":
    main()
