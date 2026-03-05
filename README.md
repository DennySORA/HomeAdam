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
optimizer = HomeAdamEW(model.parameters(), lr=1e-3, tau=1.0, weight_decay=0.01)

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

## Deep Analysis And Practical Recommendations

### 1) Algorithm behavior differences

- `AdamSRF` (Algorithm 1): always adaptive, no switching branch, closest to AdamW-style workflow with SRF denominator.
- `HomeAdam` (Algorithm 2): uses **global** condition `min(v_hat) >= tau`.  
  Because this is a global minimum over all dimensions/parameters in the group, it is very strict in large models.
- `HomeAdamEW` (Algorithm 3): uses element-wise condition `v_hat_j >= tau`, fully tensorized in update path and typically more practical for deep learning.

### 2) Global-switch synchronization reality (Algorithm 2)

- Algorithm 2 still needs a global boolean decision for branch semantics.
- Current implementation reduces this to one scalar host decision per device per param-group (not per tensor element).
- This keeps Algorithm 2 semantics exact while avoiding excessive synchronization overhead.

### 3) When to use which optimizer

| Scenario | Recommended optimizer | Why |
|---|---|---|
| Need strongest baseline compatibility and expected behavior | `torch.optim.AdamW` | Most battle-tested baseline and ecosystem default |
| Want SRF variant while staying always-adaptive | `AdamSRF` | No switching branch; simple and stable |
| Need paper-faithful **global switch** logic | `HomeAdam` | Exact Algorithm 2 semantics |
| Deep learning training (CNN/Transformer/LLM) with practical switching | `HomeAdamEW` | Element-wise gating is more usable in practice and recommended by paper appendix |

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

Command:

```bash
uv run python benchmarks/benchmark_optimizers.py --device cuda --repeats 5 --warmup-steps 30 --steps 150 --batch-size 32 --input-dim 1024 --hidden-dim 2048 --output-dim 1024 --num-threads 1
```

CUDA results (measured on 2026-03-05):

| Optimizer | Median ms/step | Mean ms/step | Samples/s |
|---|---:|---:|---:|
| `torch.AdamW` | 1.679 | 1.683 | 19018.61 |
| `AdamSRF` | 1.599 | 1.650 | 19396.29 |
| `HomeAdam (tau=1e-12)` | 1.723 | 1.672 | 19143.53 |
| `HomeAdam (tau=1.0)` | 1.628 | 1.612 | 19850.44 |
| `HomeAdam (tau=1e10)` | 1.678 | 1.669 | 19175.30 |
| `HomeAdamEW (tau=1e-12)` | 2.115 | 2.126 | 15054.54 |
| `HomeAdamEW (tau=1.0)` | 2.167 | 2.219 | 14420.93 |
| `HomeAdamEW (tau=1e10)` | 2.337 | 2.313 | 13834.82 |

CPU reference results (same benchmark shape, `--device cpu --num-threads 1`):

| Optimizer | Median ms/step | Mean ms/step | Samples/s |
|---|---:|---:|---:|
| `torch.AdamW` | 14.566 | 14.605 | 2191.06 |
| `AdamSRF` | 13.362 | 13.335 | 2399.70 |
| `HomeAdam (tau=1e-12)` | 11.541 | 11.514 | 2779.22 |
| `HomeAdam (tau=1.0)` | 11.723 | 11.690 | 2737.49 |
| `HomeAdam (tau=1e10)` | 11.600 | 11.600 | 2758.52 |
| `HomeAdamEW (tau=1e-12)` | 17.155 | 17.115 | 1869.76 |
| `HomeAdamEW (tau=1.0)` | 17.325 | 17.209 | 1859.49 |
| `HomeAdamEW (tau=1e10)` | 17.371 | 17.361 | 1843.18 |

Notes:

- These are workload-specific measurements (synthetic MLP-shaped benchmark).
- Relative ranking can change with model architecture, tensor layout, kernel fusion, precision, and hardware.
- Always benchmark on your real model before final optimizer selection.

## Development

```bash
# Format
uv run ruff format --check .

# Lint
uv run ruff check .

# Type check
uv run mypy src

# Test (fast only)
uv run pytest -m "not slow"

# Test (all)
uv run pytest
```

## Benchmark

```bash
uv run python benchmarks/benchmark_optimizers.py --device auto
```
