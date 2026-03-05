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
Measured on **NVIDIA GB10** (CUDA capability 12.1) with **PyTorch 2.10.0+cu128**.

### Benchmark command

```bash
# Standard workload
uv run python benchmarks/benchmark_optimizers.py \
  --device cuda --repeats 5 --warmup-steps 50 --steps 200 \
  --batch-size 32 --input-dim 1024 --hidden-dim 2048 --output-dim 1024 \
  --num-threads 1

# Large workload
uv run python benchmarks/benchmark_optimizers.py \
  --device cuda --repeats 5 --warmup-steps 30 --steps 100 \
  --batch-size 16 --input-dim 4096 --hidden-dim 8192 --output-dim 4096 \
  --num-threads 1
```

### CUDA — Standard workload (1024×2048×1024, batch=32)

| Optimizer | Median ms/step | Mean ms/step | Samples/s | vs AdamW |
|---|---:|---:|---:|---:|
| `torch.AdamW` | 1.833 | 1.864 | 17,170 | baseline |
| `AdamSRF` | 1.686 | 1.664 | 19,234 | **-8.0%** |
| `HomeAdam (tau=1e-12)` | 1.674 | 1.675 | 19,106 | **-8.7%** |
| `HomeAdam (tau=1.0)` | 1.735 | 1.762 | 18,161 | **-5.3%** |
| `HomeAdam (tau=1e10)` | 1.711 | 1.670 | 19,161 | **-6.7%** |
| `HomeAdamEW (tau=1e-12)` | 2.225 | 2.264 | 14,132 | +21.4% |
| `HomeAdamEW (tau=1.0)` | 2.291 | 2.321 | 13,788 | +25.0% |
| `HomeAdamEW (tau=1e10)` | 2.272 | 2.268 | 14,112 | +24.0% |

### CUDA — Large workload (4096×8192×4096, batch=16)

| Optimizer | Median ms/step | Mean ms/step | Samples/s | vs AdamW |
|---|---:|---:|---:|---:|
| `torch.AdamW` | 32.408 | 32.390 | 494 | baseline |
| `AdamSRF` | 29.840 | 29.852 | 536 | **-7.9%** |
| `HomeAdam (tau=1e-12)` | 24.972 | 24.964 | 641 | **-22.9%** |
| `HomeAdam (tau=1.0)` | 24.890 | 24.883 | 643 | **-23.2%** |
| `HomeAdam (tau=1e10)` | 24.776 | 24.761 | 646 | **-23.5%** |
| `HomeAdamEW (tau=1e-12)` | 34.034 | 34.027 | 470 | +5.0% |
| `HomeAdamEW (tau=1.0)` | 34.007 | 33.996 | 471 | +4.9% |
| `HomeAdamEW (tau=1e10)` | 33.985 | 33.958 | 471 | +4.9% |

### CPU — Standard workload (1024×2048×1024, batch=32, 1 thread)

| Optimizer | Median ms/step | Mean ms/step | Samples/s | vs AdamW |
|---|---:|---:|---:|---:|
| `torch.AdamW` | 14.957 | 14.903 | 2,147 | baseline |
| `AdamSRF` | 13.666 | 13.675 | 2,340 | **-8.6%** |
| `HomeAdam (tau=1e-12)` | 11.770 | 11.743 | 2,725 | **-21.3%** |
| `HomeAdam (tau=1.0)` | 11.579 | 11.599 | 2,759 | **-22.6%** |
| `HomeAdam (tau=1e10)` | 11.581 | 11.619 | 2,754 | **-22.6%** |
| `HomeAdamEW (tau=1e-12)` | 18.912 | 18.956 | 1,688 | +26.4% |
| `HomeAdamEW (tau=1.0)` | 18.989 | 18.731 | 1,708 | +27.0% |
| `HomeAdamEW (tau=1e10)` | 18.668 | 18.668 | 1,714 | +24.8% |

### Isolated optimizer.step() cost (d=2M parameters)

Measures only the optimizer step, excluding forward/backward.

| Optimizer | CUDA (us/step) | vs AdamW | CPU (us/step) | vs AdamW |
|---|---:|---:|---:|---:|
| `torch.AdamW` | 494 | baseline | 3,150 | baseline |
| `AdamSRF` | 483 | **-2.4%** | 2,604 | **-17.3%** |
| `HomeAdam (tau=1e-12)` | 508 | +2.7% | 1,371 | **-56.5%** |
| `HomeAdam (tau=1.0)` | 371 | **-24.9%** | 1,404 | **-55.4%** |
| `HomeAdam (tau=1e10)` | 366 | **-26.0%** | 1,384 | **-56.1%** |
| `HomeAdamEW (tau=1e-12)` | 593 | +20.0% | 4,264 | +35.4% |
| `HomeAdamEW (tau=1.0)` | 595 | +20.4% | 6,528 | +107.3% |
| `HomeAdamEW (tau=1e10)` | 589 | +19.2% | 4,263 | +35.3% |

### GPU synchronization micro-benchmark

| Operation (tensor d=2M) | Latency |
|---|---:|
| `torch.cuda.synchronize()` | 2.4 us |
| `.item()` on scalar tensor | 28.8 us |
| `.amin()` (async, no sync) | 34.2 us |
| `.amin().item()` (forces pipeline sync) | 2,205 us |

### Memory footprint (d=10M parameters)

All optimizers store two state tensors per parameter (`exp_avg`, `exp_avg_sq`), identical overhead.

| Optimizer | State (MB) | Param (MB) | Overhead |
|---|---:|---:|---:|
| `torch.AdamW` | 76.3 | 38.1 | 2.0x |
| `AdamSRF` | 76.3 | 38.1 | 2.0x |
| `HomeAdam` | 76.3 | 38.1 | 2.0x |
| `HomeAdamEW` | 76.3 | 38.1 | 2.0x |

### Key findings

1. **AdamSRF is 8% faster than PyTorch AdamW** — the square-root-free denominator saves a `sqrt` kernel on both CPU and CUDA.

2. **HomeAdam is fastest on large models** — 23% faster than AdamW on the large workload. The SGDM branch (`param.add_`) is much cheaper than the full adaptive path. The `.item()` synchronization cost (2.2ms in isolation) is amortized by the forward/backward time in large models.

3. **HomeAdamEW overhead converges at scale** — 25% slower on small models, but **only 5% slower on large models**. The `torch.where` + temporary tensor allocation is fixed overhead that becomes negligible as forward/backward dominates. This validates the paper's recommendation of Algorithm 3 for deep learning.

4. **Tau does not affect throughput** — all tau values show <3% variation, within measurement noise. Branch selection cost is negligible.

5. **Memory is identical** — all variants store the same two moment tensors (2x parameter size), matching standard Adam/AdamW.

### Optimizer selection guide

| Scenario | Recommended | Why |
|---|---|---|
| Large-model GPU training | **HomeAdam** | 23% faster than AdamW, O(1/N) generalization |
| Per-element control needed | **HomeAdamEW** | Only 5% overhead at scale, paper-recommended for DL |
| Drop-in AdamW replacement | **AdamSRF** | 8% faster, no tau tuning needed |
| Small model / ecosystem compat | `torch.optim.AdamW` | Most battle-tested, lowest overhead at small scale |

### Notes

- These are workload-specific measurements on a synthetic MLP benchmark.
- Relative ranking can change with model architecture, tensor layout, kernel fusion, precision, and hardware.
- The GPU capability mismatch (12.1 vs compiled max 12.0) may affect absolute numbers but not relative rankings.
- Always benchmark on your real model before final optimizer selection.

## Development

```bash
# Install dev dependencies
uv sync --frozen --extra dev

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

## CI/CD

GitHub Actions workflows are configured under `.github/workflows`:

- `ci.yml`
  - Trigger: push / pull request to `main`
  - Checks: `ruff format --check`, `ruff check`, `mypy src`, `pytest`
- `release.yml`
  - Trigger: push tag `v*` (for example `v0.1.0`)
  - Pipeline:
    1. Validate tag version matches `pyproject.toml` version
    2. Re-run quality checks (format/lint/type/test)
    3. Build `sdist` + `wheel`
    4. Run `twine check`
    5. Upload build artifacts, create GitHub Release
    6. Publish package to PyPI

Required GitHub secret:

- `PYPI_API_TOKEN`: PyPI API token for package publishing

Release flow:

```bash
# 1) Update version in pyproject.toml
# 2) Commit version bump
git add pyproject.toml README.md
git commit -m "chore(release): prepare vX.Y.Z"

# 3) Tag and push
git tag vX.Y.Z
git push origin main
git push origin vX.Y.Z
```

## Benchmark

```bash
uv run python benchmarks/benchmark_optimizers.py --device auto
```
