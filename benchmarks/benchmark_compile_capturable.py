"""Benchmark eager vs torch.compile and HomeAdam capturable behavior.

Run examples:
  uv run python benchmarks/benchmark_compile_capturable.py --device cuda
  uv run python benchmarks/benchmark_compile_capturable.py --device cpu
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from homeadam import HomeAdam, HomeAdamEW


@dataclass(frozen=True)
class BenchRow:
    name: str
    mode: str
    mean_ms: float | None
    median_ms: float | None
    samples_per_s: float | None
    status: str


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_case(
    *,
    name: str,
    mode: str,
    optimizer_factory: Callable[[list[nn.Parameter]], torch.optim.Optimizer],
    device: torch.device,
    compile_mode: str,
    repeats: int,
    warmup_steps: int,
    steps: int,
    batch_size: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> BenchRow:
    times_ms: list[float] = []
    for rep in range(repeats):
        torch.manual_seed(1234 + rep)
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        ).to(device)
        inputs = torch.randn(batch_size, input_dim, device=device)
        targets = torch.randn(batch_size, output_dim, device=device)
        criterion = nn.MSELoss()
        optimizer = optimizer_factory(list(model.parameters()))

        def step_fn(
            _optimizer: torch.optim.Optimizer = optimizer,
            _model: nn.Sequential = model,
            _inputs: torch.Tensor = inputs,
            _targets: torch.Tensor = targets,
            _criterion: nn.Module = criterion,
        ) -> None:
            _optimizer.zero_grad(set_to_none=True)
            outputs = _model(_inputs)
            loss = _criterion(outputs, _targets)
            loss.backward()
            _optimizer.step()

        run_step = step_fn
        if mode == "compile":
            if not hasattr(torch, "compile"):
                return BenchRow(
                    name=name,
                    mode=mode,
                    mean_ms=None,
                    median_ms=None,
                    samples_per_s=None,
                    status="SKIP (torch.compile unavailable)",
                )
            try:
                run_step = torch.compile(step_fn, mode=compile_mode, fullgraph=False)
            except Exception as exc:  # noqa: BLE001
                return BenchRow(
                    name=name,
                    mode=mode,
                    mean_ms=None,
                    median_ms=None,
                    samples_per_s=None,
                    status=f"FAIL (compile init: {type(exc).__name__})",
                )

        try:
            for _ in range(warmup_steps):
                run_step()
            _sync_if_cuda(device)

            start = time.perf_counter()
            for _ in range(steps):
                run_step()
            _sync_if_cuda(device)
            elapsed = time.perf_counter() - start
        except Exception as exc:  # noqa: BLE001
            return BenchRow(
                name=name,
                mode=mode,
                mean_ms=None,
                median_ms=None,
                samples_per_s=None,
                status=f"FAIL ({type(exc).__name__})",
            )

        times_ms.append((elapsed / steps) * 1000.0)

    mean_ms = statistics.mean(times_ms)
    median_ms = statistics.median(times_ms)
    samples_per_s = batch_size / (mean_ms / 1000.0)
    return BenchRow(
        name=name,
        mode=mode,
        mean_ms=mean_ms,
        median_ms=median_ms,
        samples_per_s=samples_per_s,
        status="OK",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Benchmark device.",
    )
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    args = parser.parse_args()

    device = _select_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print("Using device: cpu")

    print(
        "Settings: "
        f"compile_mode={args.compile_mode}, repeats={args.repeats}, "
        f"warmup_steps={args.warmup_steps}, steps={args.steps}, "
        f"batch={args.batch_size}, dims=({args.input_dim},{args.hidden_dim},{args.output_dim})"
    )

    factories: list[tuple[str, Callable[[list[nn.Parameter]], torch.optim.Optimizer]]] = [
        (
            "torch.AdamW",
            lambda params: torch.optim.AdamW(
                params,
                lr=args.lr,
                weight_decay=args.weight_decay,
                foreach=True,
            ),
        ),
        (
            "HomeAdam(capturable=False)",
            lambda params: HomeAdam(
                params,
                lr=args.lr,
                weight_decay=args.weight_decay,
                tau=1e-12,
                foreach=True,
                capturable=False,
            ),
        ),
        (
            "HomeAdam(capturable=True)",
            lambda params: HomeAdam(
                params,
                lr=args.lr,
                weight_decay=args.weight_decay,
                tau=1e-12,
                foreach=True,
                capturable=True,
            ),
        ),
        (
            "HomeAdamEW(denom)",
            lambda params: HomeAdamEW(
                params,
                lr=args.lr,
                weight_decay=args.weight_decay,
                tau=1e-12,
                foreach=True,
                update_mode="denom",
            ),
        ),
    ]

    rows: list[BenchRow] = []
    for name, factory in factories:
        rows.append(
            _run_case(
                name=name,
                mode="eager",
                optimizer_factory=factory,
                device=device,
                compile_mode=args.compile_mode,
                repeats=args.repeats,
                warmup_steps=args.warmup_steps,
                steps=args.steps,
                batch_size=args.batch_size,
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                output_dim=args.output_dim,
            )
        )
        rows.append(
            _run_case(
                name=name,
                mode="compile",
                optimizer_factory=factory,
                device=device,
                compile_mode=args.compile_mode,
                repeats=args.repeats,
                warmup_steps=args.warmup_steps,
                steps=args.steps,
                batch_size=args.batch_size,
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                output_dim=args.output_dim,
            )
        )

    print("\nResults:")
    print(
        f"{'Optimizer':30s} {'Mode':10s} {'Mean ms':>10s} {'Median ms':>10s} "
        f"{'Samples/s':>12s} {'Status':>24s}"
    )
    for row in rows:
        mean_s = f"{row.mean_ms:.3f}" if row.mean_ms is not None else "-"
        median_s = f"{row.median_ms:.3f}" if row.median_ms is not None else "-"
        sps_s = f"{row.samples_per_s:.2f}" if row.samples_per_s is not None else "-"
        print(
            f"{row.name:30s} {row.mode:10s} {mean_s:>10s} {median_s:>10s} "
            f"{sps_s:>12s} {row.status:>24s}"
        )


if __name__ == "__main__":
    main()
