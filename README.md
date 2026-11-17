# ClearNight: Frequency-Decoupled Flare Suppression & Multi-Weather Restoration

[![Course](https://img.shields.io/badge/Course-DLCV--DA6021-blue.svg)](https://github.com/warsi1507/ClearNight_26_DLCV)
[![Paper](https://img.shields.io/badge/AAAI--2026-ClearNight-orange.svg)](https://arxiv.org/abs/2505.16479)
[![Dataset](https://img.shields.io/badge/HuggingFace-AllWeatherNight-yellow.svg)](https://huggingface.co/datasets/YuetongLiu/AllWeatherNight)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-green.svg)](LICENSE)

Implementation and experimental extension for **DA6021: Deep Learning for Computer Vision** (Assessment Project).

* **Team Name:** `aiddicts`
* **Team Members:**
  * Taes Padhihary (Roll No: 230101120)
  * Vatsalya Singh (Roll No: 230103114)
  * Vagisha Shrivastava (Roll No: 230101109)
  * Md Samar Warsi (Roll No: 230101064)
  * Himanshi Aggarwal (Roll No: 230106030)
* **Assigned Paper:** *Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration* (Liu et al., AAAI 2026) [[Paper](https://arxiv.org/abs/2505.16479)] [[Official Code](https://github.com/henlyta/ClearNight)]
* **Kaggle Reproduction Guide:** Detailed step-by-step instructions in [docs/training-kaggle.md](docs/training-kaggle.md).

---

## Table of Contents
1. [Project Overview & Core Motivation](#1-project-overview--core-motivation)
2. [Proposed Novelty & Technical Contributions](#2-proposed-novelty--technical-contributions)
3. [Theoretical & Mathematical Formulation](#3-theoretical--mathematical-formulation)
4. [Repository Layout](#4-repository-layout)
5. [Environment Setup & Installation](#5-environment-setup--installation)
6. [Dataset Preparation (AllWeatherNight)](#6-dataset-preparation-allweathernight)
7. [Training Harness](#7-training-harness)
8. [Inference & Evaluation Harness](#8-inference--evaluation-harness)
9. [Quantitative Results & Empirical Analysis](#9-quantitative-results--empirical-analysis)
10. [Reproducibility & Verification Suite](#10-reproducibility--verification-suite)

---

## 1. Project Overview & Core Motivation

Nighttime adverse weather scenarios simultaneously introduce complex, compounded degradations: atmospheric haze, rain streaks, surface raindrops, and falling snow, intertwined with non-uniform artificial illumination and intense point-source light flare.

* **Existing Limitation:** Conventional daytime restoration methods assume uniform ambient illumination and struggle with severe contrast degradation in dark regions. Prior nighttime methods predominantly tackle individual isolated degradations (e.g., night rain removal only) and fail when multiple weather effects coincide with severe light halos.
* **ClearNight Baseline:** ClearNight introduces a DehazeFormer-based backbone augmented with:
  1. **Retinex Priors:** Explicit decomposition into illumination ($I$) and reflectance ($R$) maps to guide spatial contrast handling.
  2. **Weather-Aware Dynamic Selection (WDS):** A lightweight mixture-of-experts (MoE) block that routes weather-specific convolutional units guided by a weather classification loss.
* **Our Investigation:** The original authors observe that flare removal is comparatively weak (e.g., 38.77 dB on flare vs. >43 dB in specialized models). We identify two distinct failure points:
  1. *Optimization Bug in Released Code:* In the official codebase, every WDS decoder convolution and candidate unit was initialized to absolute zero with intermediate ReLUs, trapping gradients ($\frac{\partial \mathcal{L}}{\partial W} = 0$) and causing the routing branch to learn only a trivial constant bias.
  2. *Spectral Leakage of Flare Halo:* Atmospheric flare halo is physically a low-frequency scattering phenomenon. In spatial pixel space, attempting to suppress flare removes sharp underlying textures and rain streaks.

---

## 2. Proposed Novelty & Technical Contributions

We introduce three targeted contributions that resolve the optimization bottleneck and directly address nighttime light flare:

```
[Input Feat F] ──► [1-Level Haar DWT] ──► LL Band (Low-Freq)  ──► LL - m ⊙ G(LL) ──┐
                                       ──► Detail Bands (LH,HL,HH) ────────────────┴──► [Haar IDWT] ──► Cleaned Feat F'
                                                                                            ▲
[Illumination Prior I] ──► [Adaptive Pool] ──► [Spatial Gate Network C] ────────────────────┘ (m ∈ [0,1])
```

1. **WDS Initialization & Gradient Flow Correction (`networks/model.py`, `networks/dsm.py`):**
   * *Problem in Official Code:* Zeroing all internal decoder convolutions forces $\delta_l = 0$ everywhere before the last layer.
   * *Our Fix:* Apply Kaiming normal initialization to intermediate convolutions and zero **only the final projection convolution**. This ensures the WDS block starts as an exact identity ($y = 0$ at step 0) while guaranteeing non-zero gradient flow $\frac{\partial \mathcal{L}}{\partial W_l} \neq 0$ from step 1 onward.
2. **Frequency-Decoupled Flare Suppression (FDFS) Module (`networks/flare.py`):**
   * Applies an orthonormal 2D Haar Discrete Wavelet Transform (DWT) to Stage-1 intermediate features ($F \in \mathbb{R}^{B \times C \times H \times W}$).
   * Isolates the diffuse halo in the approximation sub-band ($LL \in \mathbb{R}^{B \times C \times \frac{H}{2} \times \frac{W}{2}}$).
   * Predicts a spatial gate $m \in (0, 1)$ conditioned on $LL$ and downsampled illumination $I_\downarrow$, subtracting estimated glow $G_\theta(LL)$ while leaving high-frequency directional detail sub-bands ($LH, HL, HH$) pristine.
   * The terminal convolution of $G_\theta$ is zero-initialized, rendering the module a strict mathematical identity at initialization. Adds only 14K parameters (+0.5% overhead).
3. **Synthetic Flare Supervision Pipeline (`data/flare.py`, `data/dataset.py`):**
   * Generated physically grounded synthetic flare pairs using the paper's Atmospheric Point Spread Function (APSF) formulation (Eq. 1).
   * Injected 20% dedicated flare-only training crops ($p=0.2$) with empty weather labels, and introduced a dedicated scene-isolated flare validation set for PSNR-based model selection.

---

## 3. Theoretical & Mathematical Formulation

### 3.1 Atmospheric Point Spread Function & Low-Frequency Confinement
The paper models nighttime flare degradation as:
$$X_{\text{flare}} = \alpha X + \beta (L * K_{\text{APSF}})$$
where $X$ is the clean nighttime scene, $L = \mathbb{I}(X > \tau) \cdot X$ is the light source emission map, and $K_{\text{APSF}}$ is the atmospheric scattering kernel approximated by a mixture of Gaussians $K = w G_{\sigma_1} + (1-w) G_{\sigma_2}$ with $\sigma_1 \in [4, 15]$ px, $\sigma_2 \in [20, 60]$ px.

By the **Convolution Theorem**:
$$\mathcal{F}\{L * K\}(\omega) = \hat{L}(\omega) \cdot \hat{K}(\omega)$$
The continuous Fourier transform of a Gaussian kernel $G_\sigma(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{2\sigma^2}}$ satisfies:
$$\hat{G}_\sigma(\omega) = e^{-\frac{\sigma^2 \omega^2}{2}}$$
As the spatial kernel expands ($\sigma \gg 1$), its frequency envelope compresses exponentially toward $\omega = 0$. Consequently, the diffuse flare glow is mathematically confined to low spatial frequencies.

### 3.2 2D Haar DWT Detail Bands as Spatial Derivatives
A 2D Haar transform over non-overlapping $2 \times 2$ pixel neighbourhoods decomposes local features $[a, b; c, d]$ into four sub-bands:
$$LL = \frac{a+b+c+d}{2}, \quad LH = \frac{(a+b)-(c+d)}{2}, \quad HL = \frac{(a-b)+(c-d)}{2}, \quad HH = \frac{(a-b)-(c-d)}{2}$$

By Taylor expansion of a continuously differentiable luminance field $f(x, y)$ with pixel spacing $\Delta$:
$$LH \approx -\Delta \frac{\partial f}{\partial y} - \frac{\Delta^2}{2}\frac{\partial^2 f}{\partial x \partial y} \propto -\nabla_y f, \quad HL \approx -\Delta \frac{\partial f}{\partial x} - \frac{\Delta^2}{2}\frac{\partial^2 f}{\partial x \partial y} \propto -\nabla_x f$$
Because atmospheric flare halos are smooth and diffuse, their spatial gradients are negligible ($\nabla_x f \approx 0, \nabla_y f \approx 0$). Thus, flare energy produces near-zero response in $LH, HL, HH$. Operating exclusively on $LL$:
$$LL' = LL - m \odot G_\theta(LL), \quad m = \text{sigmoid}\big(C([LL, I_\downarrow])\big)$$
$$F' = W^\top [LL', LH, HL, HH]$$
guarantees that structural high frequencies (rain streaks, edges) are mathematically preserved.

---

## 4. Repository Layout

```text
ClearNight_26_DLCV/
├── configs/
│   ├── __init__.py
│   └── experiment_config.py      # baseline, baseline_fd (Ours-A), dwt_fd (Ours-B), official configs
├── data/                         # Data loading, synthesis, Retinex priors, and splits
│   ├── __init__.py
│   ├── augmentation.py           # Illumination-aware residual attenuation
│   ├── dataset.py                # WeatherPairDataset & FlareEvalDataset
│   ├── flare.py                  # Eq. 1 synthetic flare generator
│   ├── prepare_dataset.py        # Hugging Face downloader & precomputation pipeline
│   ├── retinex.py                # Single-scale Retinex (illumination & reflectance)
│   ├── splits.py                 # Scene-based 10% validation split manifest
│   └── weather_labels.py         # Multi-label weather parser from scene filenames
├── docs/
│   └── training-kaggle.md        # Comprehensive Kaggle training & reproduction guide
├── losses/                       # Training objectives
│   ├── __init__.py
│   ├── image_losses.py           # Depth consistency loss
│   ├── layers.py                 # Feature extractors
│   ├── perceptual.py             # VGG-16 perceptual feature loss
│   ├── depth/                    # ADDS-DepthNet architectures
│   └── weights/                  # Directory for pretrained loss weights
├── networks/                     # Model architectures
│   ├── __init__.py
│   ├── baseblocks.py             # DehazeFormer transformer layers, MPE, SKFusion
│   ├── dsm.py                    # Mixture of Units & dynamic selection router
│   ├── flare.py                  # Orthonormal Haar DWT & FDFS flare suppression module
│   └── model.py                  # Full ClearNight network with fixed WDS init
├── results/                      # Per-image evaluation JSONs (comparison_final.md is generated by compare.py)
│   ├── baseline_native.json
│   ├── baseline_official.json
│   ├── baseline_fd_native.json
│   ├── baseline_fd_official.json
│   ├── dwt_fd_native.json
│   └── dwt_fd_official.json
├── tests/                        # Automated unit tests and execution verifications
│   ├── test_correctness.py       # Mathematical tests (DWT orthonormality, init checks)
│   ├── test_training.py          # End-to-end pipeline test on synthetic fixtures
│   └── verify_run.py             # Standalone verification runner (1 epoch & evaluation)
├── utils/                        # Metrics and checkpoint management
│   ├── __init__.py
│   ├── checkpoints.py            # Atomic saving/loading with RNG state restoration
│   ├── evaluation.py             # Benchmark evaluation engine (native/official resolution)
│   └── metrics.py                # PSNR, SSIM, and illumination-stratified PSNR
├── compare.py                    # Paired statistical difference and Markdown table generator
├── evaluate.py                   # Evaluation CLI across 7 test subsets
├── requirements.txt              # Exact Python dependencies
├── run_comparison.py             # Script to train & evaluate all arms back-to-back
├── train.py                      # Main training CLI with validation tracking
├── visualize.py                  # Flare gate and output visualization script
├── LICENSE                       # BSD-3-Clause license
└── README.md                     # Project documentation
```

---

## 5. Environment Setup & Installation

### Hardware Requirements
* **GPU:** NVIDIA GPU with $\ge 8$ GB VRAM (e.g., T4, RTX 3060, A100).
* **Disk:** ~30 GB free storage (dataset archives + uncompressed pairs + precomputed Retinex priors).
* **System:** Linux / Windows 10/11 / WSL2.

### Installation
```bash
# Clone the repository
git clone https://github.com/warsi1507/ClearNight_26_DLCV.git
cd ClearNight_26_DLCV

# Create virtual environment (Python 3.10 - 3.12 recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install PyTorch with CUDA support (e.g. CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

---

## 6. Dataset Preparation (AllWeatherNight)

The AllWeatherNight benchmark contains **8,000 synthetic training pairs** and **1,000 synthetic test pairs** across 7 distinct weather subsets (`rain_scene`, `snow_scene`, `single_haze`, `single_rain`, `single_drop`, `single_snow`, `single_flare`).

Run the automated pipeline to download from Hugging Face and precompute Retinex illumination (`*1/`) and reflectance (`*2/`) priors:

```bash
# Download and unpack dataset
python -m data.prepare_dataset --precompute_retinex --out_dir data/AllWeatherNight

# Build 10% scene-grouped validation split (holding out 200 of 2000 scenes)
python -m data.splits --data_root data/AllWeatherNight --out data/AllWeatherNight/split_manifest.json
```

Directory structure produced:
```text
data/AllWeatherNight/
├── train/{rain,snow,drop}{,1,2,_gt}/          # 8,000 training pairs (1=illumination, 2=reflectance)
├── test/{rain_scene,snow_scene,...}{,1,2,_gt}/# 1,000 test pairs across 7 subsets
└── split_manifest.json                        # Scene-isolated validation partition
```

---

## 7. Training Harness

All models are trained with smooth-L1 reconstruction, weather classification BCE, load-balancing loss, and router L2 penalty:

$$\mathcal{L} = \mathcal{L}_{\text{smooth-L1}}(X_{\text{out}}, X_{\text{gt}}) + 10^{-3} \mathcal{L}_{\text{BCE}} + 10^{-2} \mathcal{L}_{\text{balance}} + \mathcal{L}_{\text{router-L2}}$$

Checkpoints are selected strictly based on the highest validation PSNR (never test data).

### Execute Training Commands

#### 1. Baseline (ClearNight with Fixed WDS Init)
```bash
python train.py --config baseline --epochs 20 --samples_per_branch 1200 --val_every 5 --no_aux_losses --out_dir checkpoints
```

#### 2. Ours-A (+ Synthetic Flare Supervision)
```bash
python train.py --config baseline_fd --epochs 20 --samples_per_branch 1200 --val_every 5 --no_aux_losses --out_dir checkpoints
```

#### 3. Ours-B (+ Flare Data + Frequency-Decoupled Haar DWT Module)
```bash
python train.py --config dwt_fd --epochs 20 --samples_per_branch 1200 --val_every 5 --no_aux_losses --out_dir checkpoints
```

*To resume an interrupted session:* Add `--resume checkpoints/<config>_seed20/latest.pth`.

---

## 8. Inference & Evaluation Harness

Evaluate trained checkpoints across all 7 test subsets at native full resolution:

```bash
# Evaluate baseline
python evaluate.py --checkpoint checkpoints/baseline_seed20/best.pth --data_root data/AllWeatherNight --out_json results/baseline_native.json

# Evaluate Ours-A
python evaluate.py --checkpoint checkpoints/baseline_fd_seed20/best.pth --data_root data/AllWeatherNight --out_json results/baseline_fd_native.json

# Evaluate Ours-B
python evaluate.py --checkpoint checkpoints/dwt_fd_seed20/best.pth --data_root data/AllWeatherNight --out_json results/dwt_fd_native.json

# Generate comparative Markdown table
python compare.py results/baseline_native.json results/baseline_fd_native.json results/dwt_fd_native.json --out results/comparison_final.md

# Visualize restored images and spatial flare gates
python visualize.py --checkpoint checkpoints/dwt_fd_seed20/best.pth --subset single_flare --n 6 --out_dir results/figures
```

---

## 9. Quantitative Results & Empirical Analysis

Evaluation was conducted on all 1,000 synthetic test images of AllWeatherNight at full original resolution ($640 \times 360$). All models were trained for an identical budget of 24,000 iterations (fp32, seed 20, best epoch 19).

### 9.1 Comparative Performance on AllWeatherNight (Native Resolution)

| Method | PSNR (dB $\uparrow$) | SSIM ($\uparrow$) | Flare PSNR (dB $\uparrow$) | Drop PSNR (dB $\uparrow$) | Params (M) | Latency (ms $\downarrow$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Input (No Processing) | 20.40 | 0.6300 | 32.59 | 28.86 | — | — |
| **Baseline (ClearNight)** | 29.68 | **0.8795** | 34.31 | 30.12 | **2.865** | **4499** |
| **Ours-A (+ Flare Data)** | **29.72** | 0.8786 | **36.30** | **30.71** | **2.865** | **4499** |
| **Ours-B (+ Flare Data + DWT)** | 29.65 | 0.8766 | 36.11 | 30.70 | 2.879 | 4602 |
| **$\Delta$ (Ours-A $-$ Baseline)** | **+0.04** | **$-$0.0009** | **+1.99** | **+0.60** | **0.000** | **0** |
| **$\Delta$ (Ours-B $-$ Baseline)** | **$-$0.03** | **$-$0.0029** | **+1.80** | **+0.58** | **+0.014** | **+103** |

### 9.2 Per-Subset PSNR / SSIM Breakdown

| Method | rain_scene | snow_scene | haze | rain | drop | snow | flare | Mean |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Input | 13.87 / .369 | 12.35 / .363 | 13.33 / .436 | 18.11 / .454 | 28.86 / .958 | 23.71 / .845 | 32.59 / .982 | 20.40 / .630 |
| Baseline | **27.70 / .824** | **25.83 / .789** | **31.37 / .915** | **29.18 / .832** | 30.12 / .941 | **29.22 / .892** | 34.31 / .964 | 29.68 / **.880** |
| **Ours-A** | 27.47 / .819 | 25.37 / .779 | 30.90 / .910 | 28.88 / .822 | **30.71 / .957** | 28.40 / .883 | **36.30 / .981** | **29.72** / .879 |
| **Ours-B** | 27.42 / .816 | 25.19 / .775 | 30.79 / .904 | 28.89 / .820 | 30.70 / .955 | 28.45 / .887 | 36.11 / .980 | 29.65 / .877 |

### 9.3 Key Empirical Findings & Discussion
1. **Targeted Flare Supervision Works:** Adding synthetic flare supervision (**Ours-A**) produces a massive **+1.99 dB gain on the flare subset** (improving 73% of flare test images) and **+0.60 dB on raindrop**, achieving the highest overall mean PSNR (29.72 dB). In dark regions of flare images, PSNR increases by +4.4 dB.
2. **Analysis of 1-Level DWT Module:** Adding the one-level Haar DWT module (**Ours-B**) yielded 36.11 dB on flare (+1.80 dB over baseline), but did not further improve beyond flare data alone ($-0.19$ dB relative to Ours-A). 
   * *Physical Reason:* The two-layer conv glow estimator $G_\theta$ has an effective receptive field of $\approx 10$ pixels. Empirical test residuals reveal the real atmospheric flare halo has $\sigma \in [12, 25]$ pixels. A single-level Haar DWT downsamples by a factor of 2, leaving the wider halo spanning beyond the kernel's spatial span. Furthermore, the spatial gate $m$ concentrates on intense light sources rather than the diffuse perimeter. A multi-level pyramid DWT is recommended for future exploration.

### 9.4 Trained Checkpoints, Logs and Results

All trained weights, configurations, training logs and test scores for the three runs (AllWeatherNight, 20 epochs × 1200 steps, fp32, seed 20, Kaggle T4) are publicly available at:

**https://drive.google.com/drive/folders/1FC4oONCjTzKoZDJjRJH6oE78fH6srQho**

| Folder | Contents |
|---|---|
| `baseline/` | Reproduced baseline: ClearNight with the WDS initialization fix |
| `baseline_fd/` | **Ours-A**: + synthetic flare-only training data |
| `dwt_fd/` | **Ours-B**: + flare data + Haar DWT flare suppression module |
| `comparison/` | Baseline-versus-proposed tables on the test set |

Each run folder contains `best.pth` (checkpoint with the best validation PSNR), `config.json`, `best.json` (best epoch and its validation scores), `history.jsonl` (per-epoch training losses and validation PSNR), `tensorboard/` event files (view with `tensorboard --logdir <run folder>`) and `test_results/` (scores on the 1,000 test images under both the native and official protocols). `dwt_fd/` additionally contains `figures/` (input / illumination / flare gate / output / ground-truth panels).

Load a checkpoint with:

```bash
python evaluate.py --checkpoint <run>/best.pth --data_root data/AllWeatherNight
```

---

## 10. Reproducibility & Verification Suite

The repository includes a comprehensive, standalone test suite executable on CPU in under 30 seconds without external datasets:

```bash
# Run complete test suite (15 unit tests covering DWT, init, data, training, evaluation)
python -m unittest discover -s tests -v

# Run end-to-end 1-epoch dry-run verification
python tests/verify_run.py
```

Expected output (numeric values vary per run):
```text
=== Step 1: Generating Sample Multi-Weather Nighttime Fixture ===
Sample data generated with valid scene splits and Retinex priors.

=== Step 2: Testing 1 Epoch of Training (DWT Flare Model) ===
dwt_fd seed=20 on cpu: 2.879M params, 2 steps/epoch, 1 epochs
epoch 0: val PSNR rain/snow/drop/flare = <v>/<v>/<v>/<v>, mean <v> (best)
epoch 0 done in <t>s, losses recon=<v>, bce=<v>, balance=<v>, router_l2=<v>
Training successfully completed 1 epoch! Saved checkpoint: <tmp>/checkpoints/dwt_fd_seed20/best.pth

=== Step 3: Testing Inference & Evaluation Harness ===
single_flare     PSNR <v>  SSIM <v>  (input <v> / <v>)  (dark/mid/bright <v>/<v>/<v>)  n=2
rain_scene       PSNR <v>  SSIM <v>  (input <v> / <v>)  (dark/mid/bright <v>/<v>/<v>)  n=2
Evaluation succeeded on subsets ['single_flare', 'rain_scene']!

=== All Training and Evaluation Verification Checks Passed! ===
```

---

## Citation

```bibtex
@inproceedings{liu2026clearnight,
  title     = {Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration},
  author    = {Liu, Yuetong and others},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2026}
}
```
