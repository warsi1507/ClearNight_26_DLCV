```
All three are fp32, 20 × 1200, with best epoch 19, so they're fully comparable. I'll rename yesterday's 25-epoch tables so they don't get mixed up, then compare:

Now the exact code locations, for the list of changes:

## Final results (all fp32, 20 × 1200, best epoch 19)

**PSNR / SSIM, scored on the original-size images**

| | rain_scene | snow_scene | haze | rain | drop | snow | flare | mean |
|---|---|---|---|---|---|---|---|---|
| input (no processing) | 13.87 / .369 | 12.35 / .363 | 13.33 / .436 | 18.11 / .454 | 28.86 / .958 | 23.71 / .845 | 32.59 / .982 | 20.40 / .630 |
| **baseline** (paper's model) | **27.70 / .824** | **25.83 / .789** | **31.37 / .915** | **29.18 / .832** | 30.12 / .941 | **29.22** / .892 | 34.31 / .964 | 29.68 / **.880** |
| baseline_fd (+ flare data) | 27.47 / .819 | 25.37 / .779 | 30.90 / .910 | 28.88 / .822 | **30.71 / .957** | 28.40 / .883 | **36.30 / .981** | **29.72** / .879 |
| dwt_fd (+ flare data + DWT) | 27.42 / .816 | 25.19 / .775 | 30.79 / .904 | 28.89 / .820 | 30.70 / .955 | 28.45 / **.887** | 36.11 / .980 | 29.65 / .877 |

**Paired difference per image, PSNR dB ± standard error (% of images better)**

| | dwt_fd − baseline | baseline_fd − baseline | dwt_fd − baseline_fd |
|---|---|---|---|
| rain_scene | −0.27 ± 0.03 (21%) | −0.22 ± 0.03 (18%) | −0.05 ± 0.03 |
| snow_scene | −0.64 ± 0.05 (10%) | −0.46 ± 0.04 (12%) | −0.17 ± 0.04 |
| haze | −0.58 ± 0.10 (23%) | −0.47 ± 0.10 (32%) | −0.11 ± 0.11 |
| rain | −0.29 ± 0.04 (12%) | −0.31 ± 0.04 (12%) | +0.01 ± 0.02 |
| drop | **+0.58 ± 0.15 (70%)** | **+0.60 ± 0.15 (69%)** | −0.02 ± 0.03 |
| snow | −0.77 ± 0.12 (18%) | −0.82 ± 0.12 (20%) | +0.05 ± 0.05 |
| flare | **+1.80 ± 0.28 (73%)** | **+1.99 ± 0.31 (73%)** | −0.19 ± 0.07 |
| **all 1000** | −0.14 ± 0.04 (28%) | −0.06 ± 0.05 (29%) | −0.08 ± 0.02 (48%) |

**What the numbers show:**
- **The flare data drives the flare gains.** Flare improves by **+2.0 dB** (36.30, now 3.7 dB above the input) and raindrop by **+0.6 dB**. In the flare images, the bright regions where the glow sits gained +1.8 dB, and the dark regions +4.4 dB.
- **The other subsets lose a little.** Rain, snow and haze drop by 0.2–0.8 dB. Overall, the flare-data model is level with the baseline (mean 29.72 vs 29.68).
- **DWT adds nothing on top of the flare data** (−0.08 dB overall, −0.19 dB on flare). The DWT gains seen in the earlier fp16 and 25-epoch runs did not hold up in this clean comparison.
- **20 epochs beat 25 epochs**, as predicted: the baseline is +0.48 dB over the 25-epoch baseline, because epoch 19 is now at the end of a learning-rate cycle.
- **Against the paper (authors' scoring protocol):**

  | | paper | baseline | baseline_fd | dwt_fd |
  |---|---|---|---|---|
  | mean | 34.50 | 29.88 | 29.98 | 29.87 |
  | flare | 38.77 | 34.34 | 36.43 | 36.25 |

The tables are saved in `results/comparison_final.md` and `results/comparison_final_extra.md`. The 25-epoch tables were renamed to `old_25ep_*`. Nothing is committed to git yet.

---

## Every code change from the official repo, file by file (current `v2-flare-data` branch)

### Model
**`networks/model.py`** (rewrite of the official `ClearNight_model.py`)
- **WDS initialization fix:** `_init_weights`, lines 50–67. The official code zeroes every decoder conv and every unit, so the branch never trains. The fix uses Kaiming init and zeroes only the last decoder conv (line 67). `specificity_init='official'` keeps the old behaviour.
- **DWT flare module plugged in:** created at line 114 (`self.flare = FlareSuppression(...)` when `flare=True`) and applied at line 137, on the stage-1 features before the first WDS and transformer stage.
- **Padding:** the input and both priors are padded together (`pad`, line 125), with a shape check at line 165.

**`networks/dsm.py`** (rewrite of the official `DSM.py`)
- Units use Kaiming init instead of zeros (line 16). Zero init is optional (lines 10–13).
- Routing noise is added only during training (line 90).
- Unit count, top-k, noise and L2 weight are arguments. `top_k_softmax` is at line 63 and `load_balance_loss` at line 69.

**`networks/flare.py`** (new: the DWT module)
- `haar_kernels` (line 9) and `HaarDWT` (line 17): a fixed Haar transform and its exact inverse.
- `FlareSuppression` (line 34): `LL' = LL − m · G(LL)` (line 58). The last layer of `G` is zero-initialized (lines 49–50).

**`networks/baseblocks.py`:** only the `timm` import path changed (line 7).

### Data
**`data/flare.py`** (new: synthetic flare data, the paper's Eq. 1)
- `ALPHA = 0.995` (line 9), `_blur` (line 12), `add_flare` (line 22).

**`data/dataset.py`** (rewrite of the official `dataset_pairs.py`)
- **Flare pairs in training:** the `flare_probability` argument (line 57), the flare-pair block (lines 90–95), with the sample renamed to `<scene>_flare.png` so its weather label is empty.
- **`FlareEvalDataset`** (line 130): a fixed flare-only validation set, one image per validation scene.
- **Crops seeded** by (seed, epoch, index), so every config trains on the same patches (lines 105–107). This also fixes the official crop crash on 256-px images.
- Sorted file lists, RGB conversion, and a pair check that stops on missing files.

**`data/prepare_dataset.py`** (new): downloads AllWeatherNight from Hugging Face and unzips it.
- Fixes Windows-style backslash paths in the zips (line 35).
- Skips subfolders such as `no-exposure/` (line 38).

**`data/splits.py`** (new): the 10% validation split, grouped by scene (`build_manifest`, line 24).

**`data/weather_labels.py`** (new): weather labels from file names (line 18).

**`data/retinex.py`** (from the official `Retinex.py`): adds a guard for flat images (line 21) and an error on unreadable files (line 41).

### Training
**`train.py`** (rewrite of the official `training_ClearNight.py`)
- **Checkpoint selection:** chosen on the validation split, never on the test set. The flare validation set is added at line 116.
- **Same loss as the official code:** smooth-L1 plus BCE, load-balance and router L2 (line 159).
- **Full resume** with a config check (lines 129–131).
- **`--max_hours`** (line 50, logic at line 218) and **`best.json` written only at the end** (line 225).
- No fp16: it was added and then removed.

**`configs/experiment_config.py`:** every experiment setting lives here.
- `flare` (line 10) and `flare_data` (line 27) fields.
- Configs `dwt`, `baseline_fd` and `dwt_fd` (lines 42–44).

**`utils/checkpoints.py`:** the checkpoint stores the config so evaluation rebuilds the right model (lines 10–15). Saving is atomic (line 18).

### Evaluation
**`utils/evaluation.py`**
- Every test image is scored at its real size. The official resizing is kept only as the `official` protocol (line 18).
- **Input-vs-ground-truth score** (line 28).
- Dark / mid / bright region PSNR (line 31).

**`utils/metrics.py`** (from the official `UTILS.py`)
- `native` (clamped) and `official` protocols (lines 13–16).
- `region_psnr` GPU fix (line 29).

**`evaluate.py`, `compare.py`, `visualize.py`, `run_comparison.py`** (new)
- `compare.py` adds the **"input (no processing)" row** (line 38) and the **"beats input" column** (line 48).
- `visualize.py` draws the flare gate.

**`tests/`** (new): 15 tests.
```