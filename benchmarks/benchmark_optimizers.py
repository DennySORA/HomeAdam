"""Benchmark all optimizer variants on a shared synthetic workload.

Runs a short training loop and reports per-step latency statistics for:
- torch.optim.AdamW (baseline)
- AdamSRF (Algorithm 1)
- HomeAdam (Algorithm 2) with multiple tau settings
- HomeAdamEW (Algorithm 3) with multiple tau settings
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import torch
from torch import nn

from homeadam import AdamSRF, HomeAdam, HomeAdamEW


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_once(
    optimizer_factory: Callable[[list[nn.Parameter]], torch.optim.Optimizer],
    *,
    device: torch.device,
    warmup_steps: int,
    steps: int,
    batch_size: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> float:
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    ).to(device)
    inputs = torch.randn(batch_size, input_dim, device=device)
    targets = torch.randn(batch_size, output_dim, device=device)
    criterion = nn.MSELoss()
    optimizer = optimizer_factory(list(model.parameters()))

    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

    _sync_if_cuda(device)
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
    _sync_if_cuda(device)
    end = time.perf_counter()
    return (end - start) / steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device to benchmark on.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--output-dim", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="CPU thread count for more stable benchmarking.",
    )
    args = parser.parse_args()

    device = _select_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"Using device: {device} ({gpu_name})")
    else:
        print("Using device: cpu")

    configs: list[tuple[str, Callable[[list[nn.Parameter]], torch.optim.Optimizer]]] = [
        (
            "torch.AdamW (baseline)",
            lambda params: torch.optim.AdamW(
                params, lr=args.lr, weight_decay=args.weight_decay
            ),
        ),
        (
            "AdamSRF (Alg1)",
            lambda params: AdamSRF(params, lr=args.lr, weight_decay=args.weight_decay),
        ),
        (
            "HomeAdam (Alg2, tau=1e-12)",
            lambda params: HomeAdam(
                params, lr=args.lr, weight_decay=args.weight_decay, tau=1e-12
            ),
        ),
        (
            "HomeAdam (Alg2, tau=1.0)",
            lambda params: HomeAdam(
                params, lr=args.lr, weight_decay=args.weight_decay, tau=1.0
            ),
        ),
        (
            "HomeAdam (Alg2, tau=1e10)",
            lambda params: HomeAdam(
                params, lr=args.lr, weight_decay=args.weight_decay, tau=1e10
            ),
        ),
        (
            "HomeAdamEW (Alg3, tau=1e-12)",
            lambda params: HomeAdamEW(
                params, lr=args.lr, weight_decay=args.weight_decay, tau=1e-12
            ),
        ),
        (
            "HomeAdamEW (Alg3, tau=1.0)",
            lambda params: HomeAdamEW(
                params, lr=args.lr, weight_decay=args.weight_decay, tau=1.0
            ),
        ),
        (
            "HomeAdamEW (Alg3, tau=1e10)",
            lambda params: HomeAdamEW(
                params, lr=args.lr, weight_decay=args.weight_decay, tau=1e10
            ),
        ),
    ]

    print(
        "Benchmark settings: "
        f"repeats={args.repeats}, warmup_steps={args.warmup_steps}, "
        f"steps={args.steps}, batch={args.batch_size}, "
        f"dims=({args.input_dim},{args.hidden_dim},{args.output_dim}), "
        f"num_threads={args.num_threads}"
    )

    rows: list[tuple[str, float, float, float]] = []
    for name, optimizer_factory in configs:
        step_times = [
            _run_once(
                optimizer_factory,
                device=device,
                warmup_steps=args.warmup_steps,
                steps=args.steps,
                batch_size=args.batch_size,
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                output_dim=args.output_dim,
            )
            for _ in range(args.repeats)
        ]
        median_step_ms = statistics.median(step_times) * 1000
        mean_step_ms = statistics.mean(step_times) * 1000
        throughput = args.batch_size / (statistics.mean(step_times))
        rows.append((name, median_step_ms, mean_step_ms, throughput))

    print("\nResults:")
    print(
        f"{'Optimizer':34s} {'Median ms/step':>14s} {'Mean ms/step':>14s} "
        f"{'Samples/s':>12s}"
    )
    for name, median_step_ms, mean_step_ms, throughput in rows:
        print(
            f"{name:34s} {median_step_ms:14.3f} {mean_step_ms:14.3f} {throughput:12.2f}"
        )


if __name__ == "__main__":
    main()
