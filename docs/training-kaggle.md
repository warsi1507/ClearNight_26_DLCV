# Training ClearNight + DWT Flare Suppression on Kaggle (GPU)

This guide walks you through training the **ClearNight baseline** and our **proposed frequency-decoupled flare suppression models (Ours-A and Ours-B)** on a **free Kaggle GPU** (30 hours/week NVIDIA T4 x1 or P100). It covers dataset staging, Retinex prior precomputation, training execution, evaluation, and exporting trained checkpoints back to your local repository.

---

## 1. Experimental Setup & Study Arms

We evaluate three controlled experimental configurations under identical optimization budgets (20 epochs $\times$ 1,200 steps = 24,000 iterations, batch size 3 crops of $256\times256$, Adam $\text{lr}=2\times10^{-4}$, cosine restart period 10 epochs, seed 20):

| Model Arm | Config Name | Specificity Init | Synthetic Flare Data | DWT Flare Module | Parameters | Description |
|---|---|---|---|---|---|---|
| **Baseline** | `baseline` | `fixed` | $\times$ ($p=0.0$) | $\times$ | 2.865M | ClearNight architecture with working WDS gradient flow |
| **Ours-A** | `baseline_fd` | `fixed` | $\checkmark$ ($p=0.2$) | $\times$ | 2.865M | ClearNight + Eq. 1 synthetic flare training & validation pairs |
| **Ours-B** | `dwt_fd` | `fixed` | $\checkmark$ ($p=0.2$) | $\checkmark$ | 2.879M | Proposed full model: flare supervision + Haar DWT suppression |

### Key Artifacts Generated Per Run
Each training run writes to `checkpoints/<config>_seed<seed>/`:
- `best.pth`: Model weights selected strictly on held-out **validation split PSNR** (never the test set).
- `latest.pth`: Latest checkpoint containing model weights, optimizer, scheduler, epoch, and full RNG states for seamless resumption.
- `history.jsonl`: Step-by-step training loss metrics (`recon`, `bce`, `balance`, `router_l2`) and validation scores.
- `config.json`: Serialized hyperparameters ensuring exact reproducibility.
- `logs/`: TensorBoard execution logs.

---

## 2. Repository & Dataset Organization

On Kaggle, the directory structure mirrors the local repository root:

```text
/kaggle/working/
├── configs/                  # Experiment configurations (baseline, baseline_fd, dwt_fd)
├── data/
│   ├── dataset.py            # WeatherPairDataset & FlareEvalDataset
│   ├── flare.py              # Eq. 1 synthetic flare generator
│   ├── retinex.py            # Illumination & reflectance prior decomposition
│   ├── splits.py             # 10% scene-based validation split manifest generator
│   └── prepare_dataset.py    # Hugging Face downloader & auto-extractor
├── losses/                   # Perceptual & depth consistency losses
├── networks/
│   ├── model.py              # ClearNight backbone & WDS decoder init fix
│   ├── flare.py              # Frequency-decoupled Haar DWT module (Ours-B)
│   ├── dsm.py                # Mixture of Units & weather router
│   └── baseblocks.py         # DehazeFormer transformer stages
├── results/                  # Evaluation JSONs & comparative Markdown tables
├── utils/                    # Checkpoint management & evaluation metrics
├── train.py                  # Training CLI harness
├── evaluate.py               # 7-subset evaluation harness
├── compare.py                # Paired statistical comparison generator
└── visualize.py              # Flare gate & output visualization
```

---

## Part A — Staging Code & Cloud Storage

### Option 1: Direct Git Clone (Recommended)
If your repository is pushed to GitHub:
```bash
git clone https://github.com/warsi1507/ClearNight_26_DLCV.git /kaggle/working/ClearNight_26_DLCV
cd /kaggle/working/ClearNight_26_DLCV
```

### Option 2: Upload Project Zip Bundle via Google Drive
From your local Windows terminal, package the source files (excluding heavy caches and weights):
```powershell
Compress-Archive -Path 'configs','data','losses','networks','utils','tests','train.py','evaluate.py','compare.py','visualize.py','requirements.txt' -DestinationPath 'clearnight_bundle.zip' -Force
```
Upload `clearnight_bundle.zip` to Google Drive, set permissions to **Anyone with the link**, and extract its `FILE_ID`.

---

## Part B — Run Training on Kaggle

### B1. Create Notebook & Configure Accelerator
1. Navigate to [kaggle.com](https://www.kaggle.com) $\rightarrow$ **Create** $\rightarrow$ **New Notebook**.
2. In the right-hand **Notebook options** panel:
   - **Accelerator:** Select **GPU T4 x1** (or **GPU P100**).
   - **Internet:** Toggle to **On** *(mandatory for Hugging Face dataset download and package installation)*.
   - **Persistence:** Set to **Files only** (optional, preserves `/kaggle/working`).

---

### B2. Cell 1 — Install Dependencies
```bash
!pip install -q timm opencv-python huggingface_hub tensorboard
```

---

### B3. Cell 2 — Clone Repository / Extract Code
```bash
# Clone the repository
!git clone https://github.com/warsi1507/ClearNight_26_DLCV.git /kaggle/working/repo
%cd /kaggle/working/repo
```
*(Or if using Google Drive zip:)*
```bash
!pip install -q gdown
!gdown --id YOUR_DRIVE_FILE_ID -O /kaggle/working/clearnight_bundle.zip
!unzip -q -o /kaggle/working/clearnight_bundle.zip -d /kaggle/working/repo
%cd /kaggle/working/repo
```

---

### B4. Cell 3 — Download AllWeatherNight & Precompute Priors
The AllWeatherNight dataset (~7 GB zips) is hosted on Hugging Face (`YuetongLiu/AllWeatherNight`). The preprocessing script downloads, extracts, and precomputes the single-scale Retinex illumination and reflectance priors:

```bash
# 1. Download and structure dataset into data/AllWeatherNight/
!python -m data.prepare_dataset --precompute_retinex --out_dir data/AllWeatherNight

# 2. Generate 10% scene-based validation split (holding out 200 of 2000 training scenes)
!python -m data.splits --data_root data/AllWeatherNight --out data/AllWeatherNight/split_manifest.json
```

**Verification:** Confirm that directory counts match:
```bash
!ls data/AllWeatherNight/train/rain | wc -l
!ls data/AllWeatherNight/test/single_flare | wc -l
```
Expected output: 8,000 train images (rain, snow, drop) and 100 images per test subset.

---

### B5. Cell 4 — Environment & Hardware Verification
```bash
!nvidia-smi
!python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
```

---

### B6. Cell 5 — Execute Training

Kaggle instances allocate 2 vCPUs; setting `--num_workers 2` maximizes I/O pipeline throughput without CPU exhaustion. To match our reported budget without auxiliary VGG/depth networks:

#### Train Baseline (ClearNight with Fixed WDS Init)
```bash
!python train.py \
    --config baseline \
    --data_root data/AllWeatherNight \
    --epochs 20 \
    --samples_per_branch 1200 \
    --val_every 5 \
    --num_workers 2 \
    --no_aux_losses \
    --out_dir checkpoints \
    --seed 20
```

#### Train Ours-A (ClearNight + Synthetic Flare Supervision)
```bash
!python train.py \
    --config baseline_fd \
    --data_root data/AllWeatherNight \
    --epochs 20 \
    --samples_per_branch 1200 \
    --val_every 5 \
    --num_workers 2 \
    --no_aux_losses \
    --out_dir checkpoints \
    --seed 20
```

#### Train Ours-B (Full Proposed: Flare Data + Haar DWT Flare Module)
```bash
!python train.py \
    --config dwt_fd \
    --data_root data/AllWeatherNight \
    --epochs 20 \
    --samples_per_branch 1200 \
    --val_every 5 \
    --num_workers 2 \
    --no_aux_losses \
    --out_dir checkpoints \
    --seed 20
```

*Tip: To test execution before committing to a long run, run a quick 1-epoch dry run:*
```bash
!python train.py --config dwt_fd --epochs 1 --samples_per_branch 10 --val_every 1 --no_aux_losses --out_dir checkpoints/test_run
```

---

### B7. Cell 6 — Resuming Training After Interruption
If a Kaggle session reaches its 12-hour limit or is prematurely stopped, pass the `--resume` flag with the latest checkpoint:

```bash
!python train.py \
    --config dwt_fd \
    --data_root data/AllWeatherNight \
    --epochs 20 \
    --samples_per_branch 1200 \
    --val_every 5 \
    --num_workers 2 \
    --no_aux_losses \
    --out_dir checkpoints \
    --resume checkpoints/dwt_fd_seed20/latest.pth
```
The `--resume` flag automatically checks configuration integrity, restores model weights, optimizer state, CosineAnnealing scheduler cycle, and exact PyTorch/NumPy RNG states.

---

## Part C — Benchmark Evaluation & Result Comparison

### C1. Evaluate Checkpoints Across 7 Test Subsets
Run inference on the full-resolution AllWeatherNight test set (1,000 images):

```bash
# Evaluate Baseline
!python evaluate.py \
    --checkpoint checkpoints/baseline_seed20/best.pth \
    --data_root data/AllWeatherNight \
    --protocol native \
    --out_json results/baseline_native.json

# Evaluate Ours-A
!python evaluate.py \
    --checkpoint checkpoints/baseline_fd_seed20/best.pth \
    --data_root data/AllWeatherNight \
    --protocol native \
    --out_json results/baseline_fd_native.json

# Evaluate Ours-B
!python evaluate.py \
    --checkpoint checkpoints/dwt_fd_seed20/best.pth \
    --data_root data/AllWeatherNight \
    --protocol native \
    --out_json results/dwt_fd_native.json
```

### C2. Generate Comparative Markdown Tables
Compare performance across weather subsets and illumination strata (dark / mid / bright):

```bash
!python compare.py \
    results/baseline_native.json \
    results/baseline_fd_native.json \
    results/dwt_fd_native.json \
    --out results/comparison_final.md
```

### C3. Visualize Restored Images & Learned Flare Gates
```bash
!python visualize.py \
    --checkpoint checkpoints/dwt_fd_seed20/best.pth \
    --data_root data/AllWeatherNight \
    --subset single_flare \
    --n 6 \
    --out_dir results/figures
```

---

## Part D — Export Artifacts & Download

### D1. Package Checkpoints and Logs
Bundle all trained models, JSON evaluation scores, and figures into a single tar archive:

```bash
!tar -czvf /kaggle/working/clearnight_artifacts.tar.gz \
    checkpoints/*/best.pth \
    checkpoints/*/config.json \
    checkpoints/*/history.jsonl \
    results/*.json \
    results/*.md \
    results/figures/
```

Download `clearnight_artifacts.tar.gz` directly from the Kaggle right-hand **Output** panel.

### D2. Terminate the Session
To conserve your 30-hour weekly GPU quota:
1. Click the session menu in the top-right corner.
2. Select **Stop Session**.

---

## Troubleshooting Guide

| Issue | Root Cause | Solution |
|---|---|---|
| `CUDA out of memory` | Batch size or image crop exceeds GPU VRAM | Set `--crop_size 256` (default) and verify `--batch_size 1` (3 crops total across branches). |
| `DataLoader worker exited unexpectedly` | Kaggle CPU thread exhaustion | Specify `--num_workers 2` or `--num_workers 0` in `train.py`. |
| `FileNotFoundError: missing folder data/AllWeatherNight/...` | Extraction path mismatch | Verify `python -m data.prepare_dataset` finished completely and check folder structure. |
| `FloatingPointError: loss is nan/inf` | Gradient instability | Ensure learning rate is $2\times 10^{-4}$ and `--grad_clip_norm 1.0` is enabled if unscaled fp16 is attempted. |
| `Output files missing in sidebar` | Kaggle web UI stale | Click the refresh icon next to the **Output** heading in the right-hand panel. |
