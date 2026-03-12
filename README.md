# DP-FedSOFIM: Differentially Private Federated Learning via Second-Order Fisher Information Matrix Preconditioning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)](https://pytorch.org/)

This repository contains the official implementation accompanying the TMLR submission:

> **"Communication-Efficient Differentially Private Federated Learning via Second-Order Fisher Information Matrix Preconditioning"**
> Anonymous Authors (under review)

---

## Overview

**DP-FedSOFIM** is a differentially private federated optimization algorithm that incorporates second-order curvature information via Fisher Information Matrix (FIM) preconditioning at the server. It addresses two core challenges in private federated learning:

1. **Gradient noise amplification** under differential privacy degrades convergence, particularly under tight privacy budgets.
2. **Heterogeneous data distributions** across clients cause client drift and slow convergence in standard first-order methods.

DP-FedSOFIM applies a Sherman-Morrison rank-1 update to maintain an efficient approximation of the FIM inverse, enabling adaptive gradient preconditioning with the **same O(d) time and space complexity** as standard SGD with momentum.

### Key Contributions

- A **second-order server-side optimizer** for federated DP learning that uses momentum-based FIM approximation and a closed-form Sherman-Morrison preconditioner.
- **Exact privacy accounting** via the hockey-stick divergence formula for Gaussian mechanisms under composition, avoiding the looseness of RDP-based accountants.
- A **unified training framework** supporting four algorithms under a common codebase: DP-FedSOFIM, DP-FedGD, DP-FedFC, and DP-SCAFFOLD — enabling fair experimental comparison.
- Empirical evaluation on **CIFAR-10** and **PathMNIST** across 20 and 100 clients, under IID and non-IID (Dirichlet) partitioning, across privacy budgets ε ∈ {0.5, 1, 2, 5, 10}.

---

## Repository Structure

```
.
├── train.py                          # Core training loop (all algorithms)
├── main.py                           # CLI entry point with argument parsing
├── dataset.py                        # Federated data loading and partitioning
├── sanitizer.py                      # DP sanitizers (FedGD, SCAFFOLD, FedFC, FedNew)
├── hockey_stick_accountant.py        # Exact privacy accounting
├── run_dp_fedgd_experiments.py       # Sweep runner for all algorithm/epsilon combos
├── plot_comparison_accuracycurves.py # Accuracy curve plots across algorithms
├── plot_comparison_losscurves.py     # Loss curve plots across algorithms
├── LICENSE
└── README.md
```

---

## Method Details

### DP-FedSOFIM

Each round, clients compute gradients on their local data, clip them per-example for privacy, add calibrated Gaussian noise, and send the noisy averaged update to the server. The server aggregates these updates and applies a momentum-based second-order preconditioner: it maintains a running estimate of the gradient's curvature using a rank-1 Fisher Information Matrix approximation, inverted efficiently in closed form without ever forming a full matrix. This keeps the server update at the same computational cost as SGD with momentum while adapting the step direction to local curvature, which matters most when gradient signal is weak under tight privacy budgets.

### Privacy Accounting

Privacy is tracked using exact hockey-stick divergence accounting for the composition of Gaussian mechanisms across rounds, rather than the looser RDP-based approaches. Noise is calibrated via binary search to achieve a target (ε, δ)-DP guarantee for the full training run.

---

## Setup

### Requirements

```bash
pip install torch torchvision numpy scipy medmnist
```

- Python ≥ 3.8
- PyTorch ≥ 1.12
- CUDA recommended for 100-client experiments

### Data

CIFAR-10 downloads automatically via `torchvision`. PathMNIST and other MedMNIST datasets download automatically via `medmnist`. Pretrained ResNet backbones are loaded from `chenyaofo/pytorch-cifar-models` via `torch.hub` — an internet connection is required on first run.

---

## Reproducing Paper Experiments

All experiments use `run_dp_fedgd_experiments.py`. Edit the configuration block at the top of the file to select the algorithm and dataset, then run:

```bash
python run_dp_fedgd_experiments.py
```

The fixed experimental settings used across all paper experiments are:

| Setting | Value |
|---|---|
| Clients | 20 |
| Clients per round | 20 (full participation) |
| Local iterations | 1 |
| Batch size | 64 |
| Server learning rate | 0.1 |
| Gradient clip norm C_g | 10.0 |
| SOFIM β | 0.9 |
| SOFIM ρ | 0.5 |
| δ | 1e-5 |
| Seed | 42 |
| Rounds swept | 1, 2, 5, 10, 20, 30, 40, 50, 60, 70 |
| ε swept | 0.5, 1, 2, 5, 10 |

**Dataset selection** — set at the top of the sweep file:

```python
# CIFAR-10
SWEEP_DATASET = "cifar10"
SWEEP_BACKBONE = "cifar100_resnet20"

# PathMNIST
SWEEP_DATASET = "pathmnist"
SWEEP_BACKBONE = "cifar100_resnet20"
```

**Algorithm selection** — set exactly one flag to `True`:

```python
USE_SOFIM    = True   # DP-FedSOFIM (proposed)
USE_SCAFFOLD = False  # DP-SCAFFOLD baseline
USE_FEDFC    = False  # DP-FedFC baseline
USE_FEDNEW_FC = False # DP-FedNew-FC baseline
# All False   → DP-FedGD baseline
```

Results are saved as JSON files in `./results_sweep/` with filenames of the form `sofim_eps{ε}_rounds{T}.json`.

### Plotting

After completing sweeps for all algorithms:

```bash
python plot_comparison_accuracycurves.py
python plot_comparison_losscurves.py
```

The plotting scripts expect result directories named `./sofim_results/`, `./fedgd_results/`, `./fedfc_results/`, `./scaffold_results/`. Rename or symlink your sweep output directories accordingly before plotting.

---

## Key Arguments (`main.py`)

`main.py` provides a CLI for single runs. The values below reflect what is used in paper experiments — note that `run_dp_fedgd_experiments.py` overrides all of these explicitly.

### Model & Data

| Argument | Experiment value | Description |
|---|---|---|
| `--dataset` | `cifar10` / `pathmnist` | One of: `cifar10`, `pathmnist` |
| `--backbone` | `cifar100_resnet20` | Feature extractor backbone |
| `--partition_type` | `iid` | Data partition: `iid`, `dirichlet`, `non_iid_classes` |
| `--dirichlet_alpha` | `0.5` | Dirichlet concentration (lower = more heterogeneous) |

### Federated Learning

| Argument | Experiment value | Description |
|---|---|---|
| `--num_clients` | `20` | Total number of clients |
| `--clients_per_round` | `20` | Must equal `--num_clients` (full participation enforced) |
| `--federated_rounds` | swept | Number of communication rounds |
| `--local_iterations` | `1` | Local gradient steps per round (always 1) |
| `--batch_size` | `64` | Local batch size |
| `--learning_rate` | `0.1` | Server learning rate η |

### DP-FedSOFIM

| Argument | Experiment value | Description |
|---|---|---|
| `--use_sofim` | `True` | Enable SOFIM server optimizer |
| `--sofim_beta` | `0.9` | Momentum parameter β |
| `--sofim_rho` | `0.5` | FIM regularization ρ |
| `--sofim_warmup_rounds` | `0` | SGD warmup rounds before switching to SOFIM |

### Differential Privacy

| Argument | Experiment value | Description |
|---|---|---|
| `--epsilon` | swept | Target privacy budget ε |
| `--delta` | `1e-5` | Target privacy parameter δ |
| `--gradient_clip_norm` | `10.0` | Per-example gradient clipping norm C_g |
| `--no_dp` | `False` | Disable DP for non-private baseline |

---

## Privacy Accounting Details

`hockey_stick_accountant.py` implements:

- **`compute_delta(ε, Δ, σ, T)`**: Exact δ(ε) for T-fold Gaussian mechanism composition.
- **`find_epsilon(δ, Δ, σ, T)`**: Binary search for ε given target δ.
- **`RecordLevelDPFedGDAccountant`**: Wraps the above with the specific sensitivity and noise structure of DP-FedGD/SOFIM, accounting for clipping norm C_g, client count n, and dataset size |D_i|.

To verify privacy for a given configuration:

```python
from hockey_stick_accountant import RecordLevelDPFedGDAccountant

accountant = RecordLevelDPFedGDAccountant(
    C_g=10.0,
    n_clients=20,
    T_rounds=70,
    dataset_size=2500,
    neighbor_type="replace_one"
)
sigma_g = accountant.calibrate_noise(epsilon_target=5.0, delta_target=1e-5)
results = accountant.verify_privacy(epsilon=5.0, delta=1e-5)
print(results)
```

---

## Datasets

| Dataset | Task | Classes | Modality | Backbone |
|---|---|---|---|---|
| CIFAR-10 | Image classification | 10 | RGB 32×32 | CIFAR-100 ResNet-20 |
| PathMNIST | Pathology tissue classification | 9 | RGB 28×28 | CIFAR-100 ResNet-20 |

All experiments use frozen pretrained backbones. Only the final linear classifier is trained in the federated loop.

---

## Citation

Citation details will be provided upon acceptance.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
