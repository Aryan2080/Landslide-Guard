# Detection V2 - Phase 1 - Repository Audit

Date: 2026-09-10
Git HEAD at audit: `b9d5f86` on `main` (`Aryan2080/Landslide-Guard`)
Auditor: Detection V2 phase 1 (no code changes performed).

## 1. Stage 1 (FROZEN) - verification status

Loaded from `outputs/detection/data_verification/stage1_validation_report.txt`:

```
TOTAL CHECKS: 28  PASS: 28  FAIL: 0
```

Stage 1 remains **frozen** for V2. No file under `src/detection/preprocessing.py`,
`src/detection/dataset.py`, `outputs/detection/data_verification/`, or the raw
data at `data/raw/landslide4sense/` will be modified during V2.

Verified dataset facts (all from Stage 1, all **reused as-is**):

| Item | Value |
|------|-------|
| Image key | `img` (H5), shape `(128, 128, 14)`, dtype `float64` |
| Mask key  | `mask` (H5), shape `(128, 128)`, dtype `uint8`, values in `{0, 1}` |
| Train / Valid / Test | 3799 / 245 / 800 |
| Positive-pixel ratio (train) | ~2.32 % |
| NaN / Inf on train | 0 / 0 |
| Normalization | per-channel z-score, **train-only** statistics |
| Augmentation | hflip (p=0.5), vflip (p=0.5), rot90 x k in {0,1,2,3} - train only |
| Channels 0..11 | Sentinel-2 B1..B12 |
| Channel 12    | Slope (ALOS PALSAR DEM) |
| Channel 13    | DEM (ALOS PALSAR) |
| DataLoader tensor | image `(B, 14, 128, 128) float32`, mask `(B, 128, 128) float32` in `{0, 1}` |

## 2. Stage-2 V1 (existing) - what will be reused vs. superseded

### Files that will be REUSED unchanged in V2

| File | Purpose | Reason to keep |
|------|---------|----------------|
| `src/detection/__init__.py` | package init | trivial |
| `src/detection/preprocessing.py` | HDF5 IO + z-score normalization | Stage-1 frozen |
| `src/detection/dataset.py` | Landslide4SenseDataset + augmentation | Stage-1 frozen |
| `src/detection/metrics.py` | Dice / IoU / P / R / F1 / specificity / PR-AUC / threshold sweep | Metric definitions are locked; V2 uses the same accumulators |
| `src/detection/utils.py` | seeding, ExperimentTracker, EarlyStopping, device_summary | Fair-comparison plumbing |
| `src/detection/validate.py` | `compute_metrics` / `sweep_threshold` | Same interface expected by V2 |
| `outputs/detection/data_verification/normalization_statistics.json` | frozen train-only stats | Never recomputed |
| `configs/detection.yaml` | Stage-1 verified dataset+channel config | V2 references it, does not overwrite |

### Files that will be SUPERSEDED by V2 variants (V1 files kept for comparison, not deleted)

| V1 file | V2 successor | Reason for a new file |
|---------|--------------|-----------------------|
| `src/detection/model.py` | `src/detection/model_v2.py` | V2 adds configurable norm (BN/GN), residual double-conv option, initialization, and `base_features` sweep grid. V1 U-Net at `base_features=32` (7.77M params) remains available for identity checks. |
| `src/detection/losses.py` | `src/detection/losses_v2.py` | V2 makes all losses share the same call signature and returns a small metadata dict for logging. Adds Tversky and Focal Tversky. Weighted BCE+Dice uses train-only positive-pixel ratio for `pos_weight`. |
| `src/detection/train.py` | `src/detection/train_v2.py` | V2 mandates identical training protocol across all loss/architecture arms (same optimizer, LR schedule, epochs, patience, batch size, seed) so the comparison is fair. |
| N/A | `src/detection/evaluate_v2.py` | Test-set one-shot evaluator that refuses to run before the model + threshold have been locked (checks a `LOCKED` marker file). |
| `src/detection/postprocessing.py` | `src/detection/postprocess.py` | V2 keeps the small pixel-level cleanup helpers from V1 and adds a mask-to-polygon pipeline that emits GeoJSON in pixel coordinates (Landslide4Sense HDF5 files have no georeferencing - documented explicitly). |
| `src/detection/inference.py` | (updated in place) | Extended to load the V2 metadata bundle including channels + config. |
| `scripts/stage2_cpu_train.py` | (kept as-is) | This is the historical V1 CPU driver that produced the V1 numbers - keeping it lets anyone reproduce V1. |
| `scripts/stage2_kaggle_train.py` | `scripts/stage2_v2_kaggle_train.py` | V2 orchestrator: arch + loss experiment matrix under identical protocol, threshold sweep, one-shot test evaluation, and generation of every Phase-1..22 report file. |

### V1 results that MUST NOT be deleted or overwritten (per RULE 11)

```
outputs/detection/experiments/cpu_run_20260909_223053_results.csv
outputs/detection/experiments/cpu_run_20260909_223053_class_imbalance_summary.csv
outputs/detection/experiments/cpu_run_20260909_223053_threshold_sweep_validation.csv
outputs/detection/test/cpu_run_20260909_223053_test_metrics.json
outputs/detection/test/cpu_run_20260909_223053_stage2_report.json
outputs/detection/training/cpu_run_20260909_223053_baseline_history.csv
checkpoints/detection/best_model.pth                   (V1 export)
checkpoints/detection/final_model_config.json          (V1 metadata)
checkpoints/detection/cpu_run_20260909_223053/**/*     (per-arm V1 checkpoints)
configs/detection_final.yaml                           (V1 locked config)
```

V2 outputs go under `outputs/detection/v2_reports/`, `outputs/detection/v2_training/`,
`outputs/detection/v2_predictions/`, `models/detection/detection_v2_best.pth`, and
`configs/detection_v2_final.yaml`. **V1 artifacts are untouched.**

## 3. Detection V1 baseline (to be compared in Phase 21)

From `outputs/detection/test/cpu_run_20260909_223053_test_metrics.json` and the log:

| Metric | V1 (validation) | V1 (test @ threshold=0.60) |
|--------|-----------------|----------------------------|
| Dice        | 0.6555 | 0.6493 |
| IoU         | 0.4875 | 0.4807 |
| Precision   | 0.6203 | 0.6157 |
| Recall      | 0.6948 | 0.6868 |
| F1          | 0.6555 | 0.6493 |
| Specificity | -      | 0.9917 |
| Accuracy    | -      | 0.9860 |
| PR-AUC      | 0.6861 | 0.6683 |
| TP/FP/FN/TN | -      | 170015 / 106138 / 77516 / 12753531 |

**Caveats on the V1 loss comparison** (recorded honestly, per spec):

- Baseline BCE+Dice trained for 20 epochs.
- All four imbalance arms trained for only 5 epochs each.
- The comparison therefore favored the baseline through training budget rather than through loss design.

V2 **will fix this** by giving every loss arm the same number of epochs and the
same early-stopping criterion.

## 4. Configuration inventory

- `configs/detection.yaml`   - Stage-1 verified dataset + channel + augmentation config. **Read-only** for V2.
- `configs/detection_final.yaml` - V1 locked config (checkpoint, threshold=0.60). **Preserved.**
- `configs/detection_v2_final.yaml` - **Not created yet.** Will be produced at the end of Phase 12 after the V2 model + threshold are locked.

## 5. Notebook inventory

| Notebook | Role | V2 action |
|----------|------|-----------|
| `notebooks/00_kaggle_setup.ipynb` | env + Stage-1 verification | unchanged |
| `notebooks/01_detection_development.ipynb` | Stage-1 record | unchanged |
| `notebooks/02_detection_model_development.ipynb` | Stage-2 V1 model-dev notebook (28 sections) | unchanged; V1 development record |
| `notebooks/02_detection_v2_training.ipynb` | **NEW** - 23-section V2 notebook per spec | will be created in Phase 22 preparation |

## 6. Dependencies

Installed locally (`requirements.txt`, verified in Stage-1 report):

- python 3.13, numpy 2.2.6, pandas 2.3.0, matplotlib 3.10.3
- h5py 3.16.0, pyyaml 6.0.2, tqdm 4.67.1, scikit-image 0.26.0
- torch 2.11.0+cpu (local), torchvision (local)

For Kaggle GPU execution, `torch` and `torchvision` are pre-installed in the
Kaggle image (CUDA-enabled); `h5py` is also pre-installed on modern Kaggle
images. No `pip install` should be needed inside the notebook.

## 7. Compute posture for V2

- Local Windows box: CPU only (`torch.cuda.is_available() == False`).
- V2 training must run on **Kaggle GPU** (Tesla T4 x2 confirmed available;
  weekly quota reads ~25 hours as of this audit).
- V2 dataset: `aryanbanda/landslide4sense-full` (private, uploaded 2026-09-10)
  has masks for all three splits. This is what makes the V2 test-set evaluation
  possible on Kaggle - the community `tekbahadurkshetri/landslide4sense`
  dataset ships only training masks (competition split).

## 8. What Phase 1 changes on disk

Only this file:
`outputs/detection/v2_reports/repository_audit.md`

No source code or checkpoint was modified.
