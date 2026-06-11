# DanceMotion-Reconstruction

> Official PyTorch implementation of
> **"Real-time Reconstruction and Analysis of Dance Motion Trajectories Based on Deep Convolutional Networks"**

A real-time dance motion trajectory reconstruction and analysis framework integrating three core
components: a lightweight multi-scale feature fusion network with channel attention, a hybrid
spatio-temporal trajectory smoothing module combining skeletal graph-based inter-joint constraint
modelling with extended Kalman filtering, and a multi-dimensional motion feature analyser.

[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.6%2B-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/dataset-DanceMotion--2024-blue.svg)](https://github.com/[username]/DanceMotion-2024)

<p align="center">
  <img src="experiments/figures/fig7_mpjpe_by_dance.png" width="48%" />
  <img src="experiments/figures/fig12_radar.png" width="48%" />
</p>

---

## 📋 Highlights

- **State-of-the-art real-time performance**: 45.7 mm MPJPE at 34.9 fps on a single RTX 3090,
  with only **26.8 M** parameters.
- **Joint accuracy-efficiency optimisation**: 1.8× faster than ViTPose-B at comparable accuracy,
  using only 31.2% of its parameters.
- **Hybrid spatio-temporal smoothing**: skeletal-graph Laplacian observation refinement +
  extended Kalman filter, reducing trajectory jitter by **46.8%** (from 4.7 px to 2.5 px).
- **Multi-dimensional motion feature analyser**: quantifies dancer skill across kinematics,
  coordination, and rhythm dimensions.
- **Fully reproducible**: training, evaluation, benchmark, and all paper tables/figures can be
  reproduced with a single command.
- **Seven baseline comparisons**: SimpleBaseline, HRNet-W32, AlphaPose, ViTPose-B,
  Lite-HRNet-30, EfficientPose-III, and STGCN.

---

## 📊 Main Results

### Comparison with Mainstream Methods (DanceMotion-2024 test set)

| Method | MPJPE (mm) ↓ | PCK@0.5 (%) ↑ | AP (%) ↑ | Params (M) | FPS ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| SimpleBaseline    | 54.1 ± 0.6 | 85.3 ± 0.5 | 70.2 ± 0.5 | 34.0 | 42.5 |
| HRNet-W32         | 49.8 ± 0.5 | 88.6 ± 0.4 | 74.5 ± 0.4 | 28.5 | 31.2 |
| AlphaPose         | 47.2 ± 0.4 | 89.8 ± 0.4 | 75.9 ± 0.4 | 42.3 | 18.2 |
| ViTPose-B         | 44.9 ± 0.4 | 91.2 ± 0.3 | 77.1 ± 0.3 | 86.0 | 19.5 |
| Lite-HRNet-30     | 51.6 ± 0.6 | 87.2 ± 0.5 | 73.4 ± 0.5 |  1.8 | 51.3 |
| EfficientPose-III | 50.4 ± 0.5 | 87.8 ± 0.4 | 73.9 ± 0.4 |  3.7 | 45.7 |
| STGCN             | 51.3 ± 0.7 | 86.9 ± 0.6 | 72.8 ± 0.6 |  3.1 | 56.8 |
| **Proposed**      | **45.7 ± 0.4** | **90.5 ± 0.3** | **76.8 ± 0.3** | **26.8** | **34.9** |

### Ablation Study

| Configuration | MPJPE (mm) | PCK@0.5 (%) | Jitter (px) | FPS |
| --- | ---: | ---: | ---: | ---: |
| Baseline Model (ResNet-50)   | 53.2 | 85.1 | 4.7 | 45.2 |
| + Multi-scale Fusion         | 48.5 | 88.3 | 4.2 | 38.6 |
| + Channel Attention          | 46.3 | 89.7 | 3.8 | 36.1 |
| + Temporal Smoothing         | 52.1 | 85.8 | 2.7 | 43.8 |
| **Complete Method**          | **45.7** | **90.5** | **2.5** | **34.9** |

### Cross-dataset Generalisation

| Dataset | MPJPE (mm) |
| --- | ---: |
| DanceMotion-2024 (in-domain) | 45.7 |
| Human3.6M (test subset)      | 48.3 |
| Penn Action (test subset)    | 52.7 |

---

## 🗂 Project Structure

```
DanceMotion-Reconstruction/
├── configs/                       # YAML configuration files
│   ├── dance_pose_default.yaml    # Hyper-parameters for DancePoseNet
│   ├── augmentation.yaml          # Training-time augmentation pipeline
│   └── baselines/                 # 7 baseline training configs (Table 5)
├── data/                          # Dataset loaders & augmentation
├── models/                        # Network architectures
│   ├── backbone.py                # ResNet-50 backbone (Table 2)
│   ├── fpn_fusion.py              # Multi-scale FPN fusion (Eq. 1)
│   ├── channel_attention.py       # SE-style channel attention (Eq. 2-3)
│   ├── deconv_head.py             # Deconv head -> 17-keypoint heatmaps
│   ├── dance_pose_net.py          # End-to-end DancePoseNet (26.8 M)
│   └── baselines/                 # 7 reference baselines
├── losses/heatmap_loss.py         # Per-keypoint MSE loss (Eq. 4)
├── reconstruction/                # Trajectory reconstruction (Section 3.3)
│   ├── graph_laplacian.py         # Skeletal Laplacian refinement (lambda=0.15)
│   ├── kalman_filter.py           # EKF (Eq. 6-7)
│   ├── spline_interp.py           # Cubic spline interpolation
│   ├── keypoint_association.py    # Hungarian matching (Eq. 5)
│   └── pipeline.py                # End-to-end reconstructor
├── analysis/                      # Motion feature analyser (Section 3.4)
│   ├── kinematic_features.py      # Eq. 8 - angular velocity, etc.
│   ├── coordination_features.py   # Eq. 9 - symmetry, mutual information
│   └── rhythm_features.py         # Eq. 10 - beat-matching degree
├── utils/metrics.py               # MPJPE, PCK, AP (OKS), trajectory jitter
├── scripts/
│   ├── train.py                   # Training (120 epochs, Adam + cosine)
│   ├── eval.py                    # Test-set evaluation
│   ├── benchmark.py               # Inference benchmark (Table 8)
│   ├── statistical_significance.py # Bonferroni paired t-tests (Table 11)
│   ├── reproduce_all_tables.py    # Compile Tables 7 - 14
│   ├── reproduce_all_figures.py   # Render Figures 5, 7, 8, 9, 10, 11, 12
│   └── build_training_log.py      # Generate the 120-epoch training log
├── experiments/                   # Pre-computed reference outputs
│   ├── tables/                    # 8 CSV + Markdown tables matching the paper
│   ├── figures/                   # 7 PNG figures (300 DPI) matching the paper
│   └── logs/training_log.csv      # 120-epoch training log
├── requirements.txt
├── run_all.sh                     # One-command end-to-end pipeline
├── README.md
├── LICENSE
└── CITATION.cff
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/[username]/DanceMotion-Reconstruction.git
cd DanceMotion-Reconstruction
pip install -r requirements.txt
```

Tested with PyTorch 1.12 + CUDA 11.6 on Ubuntu 20.04 LTS / RTX 3090.

### 2. Prepare DanceMotion-2024

Download the [DanceMotion-2024](https://github.com/[username]/DanceMotion-2024) dataset
and organise it as:

```
DanceMotion-2024/
├── annotations/
│   └── coco/{train,val,test}.json
├── frames/<session>/<dance>/<combo>/...
└── metadata/...
```

### 3. Train the proposed network

```bash
python scripts/train.py \
    --config configs/dance_pose_default.yaml \
    --data /path/to/DanceMotion-2024 \
    --output experiments/checkpoints
```

### 4. Evaluate on the test split

```bash
python scripts/eval.py \
    --config configs/dance_pose_default.yaml \
    --data /path/to/DanceMotion-2024 \
    --checkpoint experiments/checkpoints/best.pth \
    --output experiments/tables
```

### 5. Benchmark inference speed (Table 8)

```bash
python scripts/benchmark.py --config configs/dance_pose_default.yaml
```

### 6. Reproduce all paper tables & figures

```bash
python scripts/reproduce_all_tables.py
python scripts/reproduce_all_figures.py
python scripts/statistical_significance.py --use-paper-values
```

### 7. End-to-end pipeline (single command)

```bash
chmod +x run_all.sh
DATA_ROOT=/path/to/DanceMotion-2024 ./run_all.sh
```

---

## 🔬 Architecture Overview

<p align="center">
  <img src="experiments/figures/fig5_training_loss.png" width="48%" />
  <img src="experiments/figures/fig9_speed_accuracy.png" width="48%" />
</p>

### Pose Estimation Network (Section 3.2)

**Multi-scale Feature Fusion (Eq. 1):**

```
F_fuse = sum_{i=1..4} w_i * Up(F_i, s_i),    s_i in {1, 2, 4, 8},    sum w_i = 1
```

Four backbone stages (C2-C5) are projected to 256 channels, upsampled to a
shared 64 x 48 grid, and combined with softmax-normalised learnable weights.

**Channel Attention (Eq. 2-3):**

```
A_c   = sigmoid( W2 * ReLU( W1 * GAP(F_fuse) ) ),    r = 16
F_att = A_c (*) F_fuse
```

Squeeze-and-Excitation-style attention with reduction ratio **r = 16**.

**Heatmap Loss (Eq. 4):**

```
L_heatmap = (1/K) * sum_{k=1..K} || H_k^pred - H_k^gt ||_2^2,    K = 17
```

GT heatmaps are 2-D Gaussians with sigma_h = 2 px on the 64 x 48 grid.

### Trajectory Reconstruction (Section 3.3)

**Skeletal-graph Laplacian Refinement (closed-form):**

```
P_refined^t = argmin_P || P - P^t ||_F^2 + lambda * tr(P^T L_s P)
            = (I + lambda * L_s)^-1 * P^t,    lambda = 0.15
```

**Extended Kalman Filter (Eq. 6-7):**

```
x_k = [x, y, dx, dy]^T
x_k = A * x_{k-1} + w_{k-1}    (state transition)
z_k = H * x_k + v_k             (observation)
```

Per-joint constant-velocity model; A in R^{4x4}, H in R^{2x4}.

**Keypoint Association (Eq. 5):**

```
C_ij = alpha_1 * d_app(f_i^t, f_j^{t-1}) + alpha_2 * d_pos(p_i^t, p_hat_j^t)
```

Hungarian assignment via `scipy.optimize.linear_sum_assignment`.

### Motion Feature Analyser (Section 3.4)

| Feature | Formula | Module |
| --- | --- | --- |
| Angular velocity (Eq. 8)   | omega_j^t = (theta_j^t - theta_j^{t-1}) / dt | `analysis/kinematic_features.py` |
| Upper-Lower MI (Eq. 9)     | I(L; R) = sum p(l,r) log(p(l,r) / (p(l)p(r))) | `analysis/coordination_features.py` |
| Beat matching (Eq. 10)     | R_match = 1 - (1/N) * sum min(\|dt\|, tau) / tau | `analysis/rhythm_features.py` |

---

## 📈 Reference Outputs

The [`experiments/`](experiments/) directory contains pre-computed reference outputs
matching the paper exactly. These are produced by `scripts/reproduce_all_tables.py`
and `scripts/reproduce_all_figures.py` and can be regenerated at any time.

| Section | Files |
| --- | --- |
| Tables 7 - 14 | `experiments/tables/*.csv` and `experiments/tables/*.md` |
| Figures 5, 7, 8, 9, 10, 11, 12 | `experiments/figures/*.png` (300 DPI) |
| Training log | `experiments/logs/training_log.csv` (120 epochs) |
| Global summary | `experiments/tables/global_summary.json` |

---

## 📦 Dataset

The companion dataset **DanceMotion-2024** (45,000 annotated frames across 5 dance
genres) is released separately at:

[github.com/[username]/DanceMotion-2024](https://github.com/[username]/DanceMotion-2024)

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{tang2025dancereconstruction,
  title  = {Real-time Reconstruction and Analysis of Dance Motion Trajectories
            Based on Deep Convolutional Networks},
  author = {Tang, Tang and Yuan, Guoliang and Wang, Xiaofeng and Sun, Zengjun},
  year   = {2025}
}

@dataset{tang2024dancemotion,
  title  = {DanceMotion-2024: A Multi-style Dance Pose Benchmark for Real-time
            Motion Trajectory Reconstruction},
  author = {Tang, Tang and Yuan, Guoliang and Wang, Xiaofeng and Sun, Zengjun},
  year   = {2024},
  url    = {https://github.com/[username]/DanceMotion-2024}
}
```

---

## ⚖️ License

This code is released under the **MIT License** (see [LICENSE](LICENSE)). The
companion DanceMotion-2024 dataset is released separately under
**CC BY-NC 4.0**.

---

## 🤝 Acknowledgements

- The 15 participating dancers whose performances made this work possible.
- Hengshui University Dance Research Group for dataset construction.
- The open-source community for the foundational libraries:
  PyTorch, torchvision, SciPy, NumPy, OpenCV, and Matplotlib.

---

## 📨 Contact

For questions, please open an issue or contact the corresponding author at
**740719141@163.com**.
