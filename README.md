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

## Performance Evaluation (2026-03-06 snapshot)

Measured in this repository on this machine:

- GPU: `NVIDIA GB10` (CUDA capability `12.1`)
- PyTorch: `2.10.0+cu128`
- Note: PyTorch warns this wheel is built up to capability `12.0`; numbers are still useful for relative comparison on this setup.

### Commands used

```bash
# CUDA end-to-end step benchmark
uv run python benchmarks/benchmark_optimizers.py \
  --device cuda --repeats 3 --warmup-steps 5 --steps 30 \
  --batch-size 8 --input-dim 256 --hidden-dim 512 --output-dim 256

# CPU end-to-end step benchmark
uv run python benchmarks/benchmark_optimizers.py \
  --device cpu --repeats 3 --warmup-steps 5 --steps 30 \
  --batch-size 8 --input-dim 256 --hidden-dim 512 --output-dim 256 \
  --num-threads 4

# CPU micro-benchmarks (update path / step throughput / memory)
uv run python benchmarks/bench_efficiency.py
```

### CUDA end-to-end (MLP synthetic workload)

| Optimizer | Mean ms/step | Samples/s |
|---|---:|---:|
| `torch.AdamW` | 0.685 | 11,679.68 |
| `AdamSRF` | 0.926 | 8,635.29 |
| `HomeAdam (tau=1e-12)` | 0.756 | 10,575.52 |
| `HomeAdam (tau=1.0)` | 0.950 | 8,418.82 |
| `HomeAdam (tau=1e10)` | 0.927 | 8,631.30 |
| `HomeAdamEW (tau=1e-12, denom)` | 1.059 | 7,556.39 |
| `HomeAdamEW (tau=1e-12, where_update)` | 0.916 | 8,730.93 |
| `HomeAdamEW (tau=1.0)` | 0.983 | 8,139.34 |
| `HomeAdamEW (tau=1e10)` | 0.868 | 9,212.75 |

### CPU end-to-end (same workload)

| Optimizer | Mean ms/step | Samples/s |
|---|---:|---:|
| `torch.AdamW` | 1.957 | 4,088.57 |
| `AdamSRF` | 2.006 | 3,987.67 |
| `HomeAdam (tau=1e-12)` | 1.975 | 4,049.87 |
| `HomeAdam (tau=1.0)` | 1.917 | 4,172.33 |
| `HomeAdam (tau=1e10)` | 2.026 | 3,948.87 |
| `HomeAdamEW (tau=1e-12, denom)` | 2.106 | 3,797.77 |
| `HomeAdamEW (tau=1e-12, where_update)` | 2.108 | 3,794.28 |
| `HomeAdamEW (tau=1.0)` | 2.091 | 3,825.13 |
| `HomeAdamEW (tau=1e10)` | 2.041 | 3,919.13 |

### CPU micro-benchmark highlights (`bench_efficiency.py`)

EW path only (`Benchmark 1`):

- `d=1,000`: `where_update` slightly faster (`7.7us` vs `8.2us`)
- `d=100,000`: `denom` faster (`1591.6us` vs `1892.8us`)
- `d=10,000,000`: `denom` faster (`7704.4us` vs `8918.2us`)

Memory (`Benchmark 3`, `d=10,000,000`):

- `AdamW`, `AdamSRF`, `HomeAdam`, `HomeAdamEW` all use ~`76.3 MB` state (`2.0x` parameter size).

### Practical conclusions

1. In the current small end-to-end workload, `torch.AdamW` is fastest, with `HomeAdam` close behind depending on `tau`.
2. `HomeAdamEW` is generally slower than `HomeAdam`/`AdamW` here, but still the recommended choice when you specifically want Algorithm 3 per-element switching semantics.
3. `update_mode="denom"` remains the default because it is paper-faithful and faster at medium/large tensor sizes in the isolated CPU micro-benchmark.
4. Keep performance decisions workload-specific: benchmark on your real training setup (for example SDXL LoRA) before locking optimizer settings.

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

Published package:

- PyPI: https://pypi.org/project/homeadam/
- Latest verified release in this repository: `v0.1.3`

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

Manual fallback (when GitHub Actions trigger/dispatch is unavailable):

```bash
# Build and validate artifacts
uv run --with build python -m build
uv run --with twine twine check dist/*

# Upload directly to PyPI (only wheel + sdist)
TWINE_USERNAME=__token__ \
TWINE_PASSWORD="$PYPI_API_TOKEN" \
uv run --with twine twine upload dist/*.whl dist/*.tar.gz

# Create/update GitHub release assets manually if needed
gh release create vX.Y.Z dist/*.whl dist/*.tar.gz \
  --repo DennySORA/HomeAdam \
  --title "HomeAdam vX.Y.Z" \
  --generate-notes
```

## Benchmark

```bash
uv run python benchmarks/benchmark_optimizers.py --device auto
```
