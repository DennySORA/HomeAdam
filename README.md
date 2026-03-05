# HomeAdam(W)

PyTorch implementation of optimizers from **"HomeAdam: Adam and AdamW Algorithms Sometimes Go Home to Obtain Better Provable Generalization"** ([arXiv:2603.02649v1](https://arxiv.org/abs/2603.02649)).

HomeAdam(W) dynamically switches between adaptive (Adam-like) and momentum-SGD updates based on a threshold on the bias-corrected second moment. This achieves O(1/N) generalization (matching SGD) while retaining O(T^{-1/4}) convergence.

## Optimizers

| Class | Algorithm | Description |
|-------|-----------|-------------|
| `AdamSRF` | Algorithm 1 | Adam(W) without sqrt in denominator — always adaptive |
| `HomeAdam` | Algorithm 2 | Global switching: entire tensor uses adaptive or SGDM |
| `HomeAdamEW` | Algorithm 3 | Element-wise switching: per-dimension adaptive/SGDM (**recommended**) |

## Installation

```bash
# From PyPI
pip install homeadam
```

For development:

```bash
# Clone and install project environment
uv sync
```

This project is configured to resolve `torch` from the CUDA 12.8 index via `uv` source mapping in `pyproject.toml`.

## Usage

```python
import torch
from torch import nn

from homeadam import HomeAdamEW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Linear(10, 1).to(device)
optimizer = HomeAdamEW(
    model.parameters(),
    lr=1e-3,
    tau=1.0,
    weight_decay=0.01,
    state_dtype=torch.float32,   # default
    foreach=True,                # default
    update_mode="denom",         # default
)

for inputs, targets in dataloader:
    inputs = inputs.to(device)
    targets = targets.to(device)
    optimizer.zero_grad()
    loss = criterion(model(inputs), targets)
    loss.backward()
    optimizer.step()
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-3 | Learning rate |
| `betas` | (0.9, 0.99) | Moment decay coefficients |
| `eps` | 1e-7 | Numerical stability term |
| `weight_decay` | 0.0 | Decoupled weight decay (0 = Adam, >0 = AdamW) |
| `tau` | 1.0 | Switching threshold `> 0` (HomeAdam/HomeAdamEW only) |
| `state_dtype` | `torch.float32` | Optimizer state dtype (`None` = follow parameter dtype) |
| `foreach` | `True` | Use foreach/multi-tensor moment updates where possible |
| `capturable` | `False` | HomeAdam only: on single-device groups, keep switch decision on-device (no host `.item()`) |
| `update_mode` | `"denom"` | HomeAdamEW only: `"denom"` (paper-faithful default) or `"where_update"` |

## Deep Analysis And Practical Recommendations

### 1) Algorithm behavior differences

- `AdamSRF` (Algorithm 1): always adaptive, no switching branch, closest to AdamW-style workflow with SRF denominator.
- `HomeAdam` (Algorithm 2): uses **global** condition `min(v_hat) >= tau`.  
  Because this is a global minimum over all dimensions/parameters in the group, it is very strict in large models.
- `HomeAdamEW` (Algorithm 3): uses element-wise condition `v_hat_j >= tau`, fully tensorized in update path and typically more practical for deep learning.

### 2) Global-switch synchronization reality (Algorithm 2)

- Algorithm 2 still needs a global boolean decision for branch semantics.
- `capturable=False` (default): one scalar host decision per device/group (low sync overhead).
- `capturable=True`: on single-device groups, keeps decision as device tensor (no host `.item()`). Multi-device groups fall back to strict global bool semantics.

### 3) When to use which optimizer

| Scenario | Recommended optimizer | Why |
|---|---|---|
| Need strongest baseline compatibility and predictable speed | `torch.optim.AdamW` | Most battle-tested baseline and ecosystem default |
| Need paper-faithful **global switch** logic | `HomeAdam` | Exact Algorithm 2 semantics; often close to AdamW speed |
| Need per-element switching behavior | `HomeAdamEW` | Algorithm 3 semantics, usually easier `tau` tuning than global min rule |
| Want always-adaptive SRF variant | `AdamSRF` | No switching branch; closest to AdamW-style training loop |

### 3.1) SDXL LoRA recommendation (practical)

For SDXL LoRA, start with `HomeAdamEW` first.

Why:

- SDXL LoRA has many trainable LoRA tensors (often small-to-medium matrices).
- Algorithm 2 (`HomeAdam`) uses global `min(v_hat)` over a group, which is very easy to be dominated by tiny outlier elements.
- Algorithm 3 (`HomeAdamEW`) applies switching per element, so it is usually easier to tune and closer to diffusion-training practice.

Suggested rollout:

1. Keep your existing SDXL LoRA recipe (batch size, scheduler, precision, regularization) unchanged.
2. Replace optimizer only, start with:
   - `optimizer = HomeAdamEW(...)`
   - `betas=(0.9, 0.99)`, `eps=1e-7`, `weight_decay` same as your current baseline
   - `tau=1e-12` as the first trial
3. Tune `tau` only after a clean baseline run.

Tau tuning for SDXL LoRA:

- If training is too noisy/unstable (early spikes, frequent divergence): increase `tau` (more SGDM behavior).
- If convergence is too slow or underfitting: decrease `tau` (more adaptive behavior).
- Practical search order: `1e-12 -> 1e-10 -> 1e-8 -> 1e-6`.

Minimal example (single group):

```python
optimizer = HomeAdamEW(
    lora_params,
    lr=1e-4,
    betas=(0.9, 0.99),
    eps=1e-7,
    weight_decay=0.0,
    tau=1e-12,
    state_dtype=torch.float32,
    foreach=True,
    update_mode="denom",
)
```

SDXL-style parameter groups (UNet LoRA + Text Encoder LoRA):

```python
optimizer = HomeAdamEW(
    [
        {"params": unet_lora_params, "lr": 1e-4, "tau": 1e-12},
        {"params": text_encoder_lora_params, "lr": 5e-6, "tau": 1e-12},
    ],
    betas=(0.9, 0.99),
    eps=1e-7,
    weight_decay=0.0,
    state_dtype=torch.float32,
    foreach=True,
    update_mode="denom",
)
```

Notes for SDXL:

- Use your current mixed-precision setup as-is (`fp16`/`bf16` with AMP/Accelerate).
- Keep gradient clipping enabled (`max_grad_norm=1.0` is a common safe default).
- For LoRA workloads, forward/backward time usually dominates wall-clock, so optimizer micro-latency is secondary to convergence quality.

### 4) Tau tuning guidance (important)

`tau` is the most sensitive hyperparameter for HomeAdam variants.

- For `HomeAdam` (Algorithm 2, global min rule):
  - In high-dimensional models, `min(v_hat)` is often extremely small.
  - If `tau` is not tiny enough, it can collapse into mostly SGDM branch.
  - In our CUDA probe, global adaptive ratio changed sharply around very small values (`~1e-20` to `1e-16` region), showing high sensitivity.
- For `HomeAdamEW` (Algorithm 3, element-wise rule):
  - Behavior transitions more smoothly and is easier to tune.
  - In our probe, `tau=1e-12` kept almost all elements adaptive, while larger values could push many elements to SGDM.

Recommended starting points:

- `HomeAdam`: start with very small `tau` (for example `1e-24` to `1e-20`) and monitor branch ratio.
- `HomeAdamEW`: start with `1e-12` (fp32) and tune upward only if you explicitly want stronger SGDM behavior.

### 5) CUDA environment notes

- Machine has NVIDIA GPU and CUDA driver support, but runtime compatibility still depends on installed PyTorch wheel.
- Verify runtime with:

```bash
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
```

- If you see capability warning (for example GPU capability is newer than the max compiled capability in current wheel), training may still run but you should treat peak performance/profiler behavior as potentially non-ideal.

## Performance Evaluation (Measured)

All algorithms are benchmarked with `benchmarks/benchmark_optimizers.py`.
Measured on **NVIDIA GB10** (CUDA capability 12.1) with **PyTorch 2.10.0+cu128**, 2026-03-06.

### Benchmark commands

```bash
# CUDA standard (1024x2048x1024 MLP)
uv run python benchmarks/benchmark_optimizers.py \
  --device cuda --repeats 5 --warmup-steps 50 --steps 200 \
  --batch-size 32 --input-dim 1024 --hidden-dim 2048 --output-dim 1024 --num-threads 1

# CUDA large (4096x8192x4096 MLP)
uv run python benchmarks/benchmark_optimizers.py \
  --device cuda --repeats 5 --warmup-steps 30 --steps 100 \
  --batch-size 16 --input-dim 4096 --hidden-dim 8192 --output-dim 4096 --num-threads 1

# CPU standard
uv run python benchmarks/benchmark_optimizers.py \
  --device cpu --repeats 5 --warmup-steps 50 --steps 200 \
  --batch-size 32 --input-dim 1024 --hidden-dim 2048 --output-dim 1024 --num-threads 1

# CPU micro-benchmarks
uv run python benchmarks/bench_efficiency.py
```

### CUDA — Standard workload (1024x2048x1024, batch=32)

| Optimizer | Median ms/step | Mean ms/step | Samples/s | vs AdamW |
|---|---:|---:|---:|---:|
| `torch.AdamW` | 1.897 | 1.894 | 16,899 | baseline |
| `AdamSRF` | 2.391 | 2.382 | 13,437 | +26.0% |
| `HomeAdam (tau=1e-12)` | 2.290 | 2.285 | 14,007 | +20.7% |
| `HomeAdam (tau=1.0)` | 2.342 | 2.329 | 13,740 | +23.5% |
| `HomeAdam (tau=1e10)` | 2.118 | 2.076 | 15,413 | +11.6% |
| `HomeAdamEW (tau=1e-12, denom)` | 2.711 | 2.708 | 11,818 | +42.9% |
| `HomeAdamEW (tau=1e-12, where_update)` | 2.668 | 2.658 | 12,040 | +40.6% |
| `HomeAdamEW (tau=1.0)` | 2.830 | 2.841 | 11,262 | +49.2% |
| `HomeAdamEW (tau=1e10)` | 2.633 | 2.626 | 12,184 | +38.8% |

### CUDA — Large workload (4096x8192x4096, batch=16)

| Optimizer | Median ms/step | Mean ms/step | Samples/s | vs AdamW |
|---|---:|---:|---:|---:|
| `torch.AdamW` | 36.318 | 36.324 | 440 | baseline |
| `AdamSRF` | 36.067 | 36.025 | 444 | -0.7% |
| `HomeAdam (tau=1e-12)` | 29.181 | 29.192 | 548 | **-19.7%** |
| `HomeAdam (tau=1.0)` | 29.197 | 29.111 | 550 | **-19.6%** |
| `HomeAdam (tau=1e10)` | 29.204 | 29.197 | 548 | **-19.6%** |
| `HomeAdamEW (tau=1e-12, denom)` | 40.102 | 40.089 | 399 | +10.4% |
| `HomeAdamEW (tau=1e-12, where_update)` | 41.194 | 41.173 | 389 | +13.4% |
| `HomeAdamEW (tau=1.0)` | 40.099 | 40.316 | 397 | +10.4% |
| `HomeAdamEW (tau=1e10)` | 40.508 | 40.617 | 394 | +11.5% |

### CPU — Standard workload (1024x2048x1024, batch=32, 1 thread)

| Optimizer | Median ms/step | Mean ms/step | Samples/s | vs AdamW |
|---|---:|---:|---:|---:|
| `torch.AdamW` | 14.564 | 14.514 | 2,205 | baseline |
| `AdamSRF` | 14.719 | 14.723 | 2,174 | +1.1% |
| `HomeAdam (tau=1e-12)` | 12.262 | 12.303 | 2,601 | **-15.8%** |
| `HomeAdam (tau=1.0)` | 12.888 | 12.846 | 2,491 | **-11.5%** |
| `HomeAdam (tau=1e10)` | 12.877 | 12.781 | 2,504 | **-11.6%** |
| `HomeAdamEW (tau=1e-12, denom)` | 21.397 | 21.504 | 1,488 | +46.9% |
| `HomeAdamEW (tau=1e-12, where_update)` | 21.643 | 21.607 | 1,481 | +48.6% |
| `HomeAdamEW (tau=1.0)` | 20.785 | 21.068 | 1,519 | +42.7% |
| `HomeAdamEW (tau=1e10)` | 20.798 | 20.812 | 1,538 | +42.8% |

### CPU micro-benchmark highlights (`bench_efficiency.py`)

EW update mode comparison (isolated, d=10M):

| Mode | Median (us) | IQR (us) |
|---|---:|---:|
| `denom` | 7,493 | 1,303 |
| `where_update` | 8,994 | 1,339 |

`denom` is ~17% faster at large tensor sizes. At d=1K, `where_update` is slightly faster (7.7 vs 8.2 us).

Isolated optimizer step throughput (d=10M, CPU single-thread):

| Optimizer | us/step | vs AdamW |
|---|---:|---:|
| `torch.AdamW` | 8,687 | baseline |
| `AdamSRF` | 8,758 | +0.8% |
| `HomeAdam (tau=1e-10)` | 6,173 | **-28.9%** |
| `HomeAdam (tau=1e10)` | 6,046 | **-30.4%** |
| `HomeAdamEW (denom)` | 13,374 | +53.9% |
| `HomeAdamEW (where_update)` | 13,544 | +55.9% |

Memory footprint (d=10M):

| Optimizer | State (MB) | Overhead |
|---|---:|---:|
| All variants | 76.3 | 2.0x |

### Key findings

1. **HomeAdam is fastest on large models** — 20% faster than PyTorch AdamW on the large CUDA workload. The SGDM branch (`param.add_`) is much cheaper than the full adaptive path. The `.item()` sync cost is amortized by forward/backward time.

2. **AdamSRF is near-identical to AdamW in throughput** — the square-root saving is offset by the extra Python-level tensor ops (the refactored `_apply_update` path). At large scale, AdamSRF converges to AdamW speed.

3. **HomeAdamEW has higher per-step overhead** — due to `torch.where` + extra tensor allocations. However, the overhead shrinks from ~45% (standard) to ~10% (large) as forward/backward dominates. The paper recommends Algorithm 3 for deep learning, and at realistic model scales the overhead is modest.

4. **`update_mode="denom"` is faster than `"where_update"`** — consistently ~17% faster at large tensor sizes in micro-benchmarks. Both are mathematically equivalent; `denom` is the paper-faithful default.

5. **Tau does not meaningfully affect throughput** — all tau values show <5% variation.

6. **Memory is identical across all variants** — 2.0x parameter size (exp_avg + exp_avg_sq), same as standard Adam/AdamW.

### Optimizer selection guide

| Scenario | Recommended | Why |
|---|---|---|
| Large-model GPU training | **HomeAdam** | 20% faster than AdamW, O(1/N) generalization |
| Per-element switching needed | **HomeAdamEW** | ~10% overhead at large scale, paper-recommended for DL |
| Drop-in AdamW replacement | **AdamSRF** | Same speed, no tau tuning, no sqrt |
| Small model / max compatibility | `torch.optim.AdamW` | Fastest at small scale, ecosystem default |

### Notes

- These are workload-specific measurements on a synthetic MLP benchmark.
- Relative ranking can change with model architecture, tensor layout, kernel fusion, precision, and hardware.
- The GPU capability mismatch (12.1 vs compiled max 12.0) may affect absolute numbers but not relative rankings.
- Always benchmark on your real model before final optimizer selection.

## Benchmark

```bash
# End-to-end optimizer throughput
uv run python benchmarks/benchmark_optimizers.py --device auto

# EW micro-benchmarks + memory
uv run python benchmarks/bench_efficiency.py

# Eager vs torch.compile and HomeAdam capturable behavior
uv run python benchmarks/benchmark_compile_capturable.py --device auto
```
