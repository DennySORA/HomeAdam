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
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python benchmarks/bench_efficiency.py
```

### CUDA — Standard workload (1024x2048x1024, batch=32)

| Optimizer | Median ms/step | Mean ms/step | Samples/s | vs AdamW |
|---|---:|---:|---:|---:|
| `torch.AdamW` | 1.867 | 1.865 | 17,157 | baseline |
| `AdamSRF` | 2.194 | 2.195 | 14,577 | +17.5% |
| `HomeAdam (tau=1e-12)` | 2.330 | 2.283 | 14,014 | +24.8% |
| `HomeAdam (tau=1.0)` | 2.312 | 2.310 | 13,854 | +23.8% |
| `HomeAdam (tau=1e10)` | 2.329 | 2.307 | 13,871 | +24.7% |
| `HomeAdamEW (tau=1e-12, denom)` | 2.778 | 2.773 | 11,539 | +48.8% |
| `HomeAdamEW (tau=1e-12, where_update)` | 2.812 | 2.813 | 11,375 | +50.6% |
| `HomeAdamEW (tau=1.0)` | 2.740 | 2.751 | 11,634 | +46.8% |
| `HomeAdamEW (tau=1e10)` | 2.757 | 2.745 | 11,659 | +47.7% |

### CUDA — Large workload (4096x8192x4096, batch=16)

| Optimizer | Median ms/step | Mean ms/step | Samples/s | vs AdamW |
|---|---:|---:|---:|---:|
| `torch.AdamW` | 33.568 | 33.370 | 479 | baseline |
| `AdamSRF` | 33.355 | 33.802 | 473 | -0.6% |
| `HomeAdam (tau=1e-12)` | 26.794 | 26.959 | 594 | **-20.2%** |
| `HomeAdam (tau=1.0)` | 26.637 | 26.632 | 601 | **-20.6%** |
| `HomeAdam (tau=1e10)` | 26.610 | 26.611 | 601 | **-20.7%** |
| `HomeAdamEW (tau=1e-12, denom)` | 37.293 | 37.297 | 429 | +11.1% |
| `HomeAdamEW (tau=1e-12, where_update)` | 38.632 | 38.608 | 414 | +15.1% |
| `HomeAdamEW (tau=1.0)` | 39.244 | 39.202 | 408 | +16.9% |
| `HomeAdamEW (tau=1e10)` | 40.048 | 40.044 | 400 | +19.3% |

### CPU — Standard workload (1024x2048x1024, batch=32, 1 thread)

| Optimizer | Median ms/step | Mean ms/step | Samples/s | vs AdamW |
|---|---:|---:|---:|---:|
| `torch.AdamW` | 14.439 | 14.448 | 2,215 | baseline |
| `AdamSRF` | 14.781 | 14.800 | 2,162 | +2.4% |
| `HomeAdam (tau=1e-12)` | 12.361 | 12.369 | 2,587 | **-14.4%** |
| `HomeAdam (tau=1.0)` | 12.329 | 12.354 | 2,590 | **-14.6%** |
| `HomeAdam (tau=1e10)` | 12.336 | 12.330 | 2,595 | **-14.6%** |
| `HomeAdamEW (tau=1e-12, denom)` | 20.319 | 20.038 | 1,597 | +40.7% |
| `HomeAdamEW (tau=1e-12, where_update)` | 21.016 | 21.065 | 1,519 | +45.6% |
| `HomeAdamEW (tau=1.0)` | 20.763 | 20.758 | 1,542 | +43.8% |
| `HomeAdamEW (tau=1e10)` | 21.181 | 21.149 | 1,513 | +46.7% |

### CPU micro-benchmark highlights (`bench_efficiency.py`)

EW update mode comparison (isolated, d=10M):

| Mode | Median (us) | IQR (us) |
|---|---:|---:|
| `denom` | 31,139 | 697 |
| `where_update` | 33,179 | 961 |

`denom` is ~6.1% faster at large tensor sizes (10M). At d=1K, `where_update` is slightly faster (7.7 vs 8.0 us).

Isolated optimizer step throughput (d=10M, CPU single-thread):

| Optimizer | us/step | vs AdamW |
|---|---:|---:|
| `torch.AdamW` | 18,305 | baseline |
| `AdamSRF` | 17,770 | -2.9% |
| `HomeAdam (tau=1e-10)` | 11,768 | **-35.7%** |
| `HomeAdam (tau=1e10)` | 11,773 | **-35.7%** |
| `HomeAdamEW (denom)` | 41,549 | +127.0% |
| `HomeAdamEW (where_update)` | 41,470 | +126.5% |

Memory footprint (d=10M):

| Optimizer | State (MB) | Overhead |
|---|---:|---:|
| All variants | 76.3 | 2.0x |

### Key findings

1. **HomeAdam is fastest on large models** — about 20% faster than PyTorch AdamW on the large CUDA workload, and ~14.6% faster on the CPU standard workload.

2. **AdamSRF is close to AdamW only on large CUDA** — -0.6% on the large CUDA workload, but +17.5% overhead on the CUDA standard workload and +2.4% on CPU standard.

3. **HomeAdamEW has higher per-step overhead** — around +47% to +51% on CUDA standard, and +11% to +19% on CUDA large as forward/backward dominates.

4. **`update_mode` tradeoff is small in end-to-end step time** — the isolated EW kernel shows `denom` ~6.1% faster at 10M, but full optimizer-step throughput at 10M is nearly tied (`where_update` ~0.2% faster).

5. **Tau impact depends on variant** — HomeAdam is very stable across tau in these runs, while HomeAdamEW can vary noticeably (up to ~8% on large CUDA).

6. **Memory is identical across all variants** — 2.0x parameter size (exp_avg + exp_avg_sq), same as standard Adam/AdamW.

### Optimizer selection guide

| Scenario | Recommended | Why |
|---|---|---|
| Large-model GPU training | **HomeAdam** | ~20% faster than AdamW in this benchmark, O(1/N) generalization |
| Paper-faithful DL variant | **HomeAdamEW / HomeAdamW** | Paper conclusion states element-wise variant is more suitable for deep learning; HomeAdamW is favored over HomeAdam in test metrics |
| Per-element switching needed | **HomeAdamEW** | +11% to +19% overhead at large scale, paper-recommended for DL |
| No-tau adaptive baseline | **AdamSRF** | No tau tuning, but throughput is workload-dependent |
| Small model / max compatibility | `torch.optim.AdamW` | Fastest at small scale, ecosystem default |

### Notes

- These are workload-specific measurements on a synthetic MLP benchmark.
- This section validates **throughput only**; it does not directly verify the paper's CV/NLP generalization claims (test accuracy/perplexity).
- Relative ranking can change with model architecture, tensor layout, kernel fusion, precision, and hardware.
- The GPU capability mismatch (12.1 vs compiled max 12.0) may affect absolute numbers but not relative rankings.
- Always benchmark on your real model before final optimizer selection.

## Benchmark

```bash
# End-to-end optimizer throughput
uv run python benchmarks/benchmark_optimizers.py --device auto

# EW micro-benchmarks + memory
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python benchmarks/bench_efficiency.py

# Eager vs torch.compile and HomeAdam capturable behavior
uv run python benchmarks/benchmark_compile_capturable.py --device auto
```
