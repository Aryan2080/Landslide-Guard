# LandslideGuard

## Current Status

- **Stage 1 — Data preparation:** COMPLETE (28/28 checks PASS).
- **Stage 2 — Detection model development:** code + 28-section Kaggle notebook shipped, actual training runs on Kaggle GPU.

Notebooks (numbered by lifecycle stage):

- [notebooks/00_kaggle_setup.ipynb](notebooks/00_kaggle_setup.ipynb) — Kaggle env + Stage-1 verification (**no training**). Run this first on Kaggle.
- [notebooks/01_detection_development.ipynb](notebooks/01_detection_development.ipynb) — Stage-1 data-preparation record (30 sections, executed).
- [notebooks/02_detection_model_development.ipynb](notebooks/02_detection_model_development.ipynb) — **Stage-2 model development** (28 sections): U-Net baseline, class-imbalance loss experiments, controlled HP sweep, threshold optimization, one-shot test evaluation, error analysis, model export + inference verification.

Docs / artifacts:

- [docs/kaggle_setup.md](docs/kaggle_setup.md) — step-by-step Kaggle instructions.
- [outputs/detection/data_verification/](outputs/detection/data_verification/) — Stage-1 statistics, figures, and PASS/FAIL log (committed).

## Project Objective

LandslideGuard is intended to develop an AI-based system for **Landslide Detection, Monitoring, and Prediction**. The first module under development is **Detection**: semantic segmentation of landslide-affected areas from multi-channel remote-sensing patches (Landslide4Sense).

Monitoring, Prediction, live ingestion, and frontend/backend are out of scope for the current stage.

## Detection Pipeline

```
Dataset  ->  HDF5 Reading  ->  14-Channel Verification  ->  Preprocessing
        ->  Normalization  ->  Dataset  ->  DataLoader  ->  U-Net (Stage 2)
        ->  Segmentation   ->  Post-processing  ->  Geospatial Polygon
```

## Repository Structure

```
LandslideGuard/
├── data/
│   ├── raw/landslide4sense/         Original dataset - NEVER committed to GitHub (.gitignored).
│   │   ├── TrainData/TrainData/{img,mask}/*.h5   3799 + 3799
│   │   ├── ValidData/ValidData/{img,mask}/*.h5    245 +  245
│   │   └── TestData/TestData/{img,mask}/*.h5      800 +  800
│   └── processed/detection/         Reserved for preprocessed arrays / stats / caches.
│
├── notebooks/
│   ├── 00_kaggle_setup.ipynb              Kaggle env + Stage-1 verification (no training).
│   ├── 01_detection_development.ipynb     Stage 1 - data preparation (executed).
│   └── 02_detection_model_development.ipynb  Stage 2 - full model development (28 sections).
│
├── src/detection/
│   ├── preprocessing.py             HDF5 loading + per-channel z-score normalization.
│   ├── dataset.py                   PyTorch Dataset + DataLoader + train-only augmentation.
│   ├── model.py                     U-Net (14 in / 1 out, ~7.77M params @ base_features=32).
│   ├── losses.py                    BCE / Dice / BCE+Dice / Focal / Focal+Dice (+ build_loss).
│   ├── metrics.py                   Exact Dice/IoU/P/R/F1/specificity/accuracy + PR-AUC + threshold sweep.
│   ├── train.py                     Trainer + fit() + evaluate() with best-val checkpointing.
│   ├── validate.py                  compute_metrics() / sweep_threshold() over a DataLoader.
│   ├── utils.py                     Seeding, device summary, ExperimentTracker CSV, EarlyStopping.
│   ├── postprocessing.py            Thresholding, small-component removal, hole fill, boundary extraction.
│   ├── inference.py                 DetectionInference bundle (checkpoint + stats + threshold).
│   └── geospatial.py                Mask -> polygon / GeoJSON export (planned).
│
├── models/detection/                Trained checkpoints (Stage 2; .gitignored).
├── outputs/detection/
│   ├── data_verification/           Stage-1 stats / reports / figs (committed to Git).
│   └── training/ predictions/ masks/ probability_maps/ overlays/ polygons/ reports/
│                                    Generated during Stage 2 (.gitignored).
├── configs/detection.yaml           Verified dataset + normalization config.
├── docs/
│   ├── detection/                   Detection-module documentation.
│   └── kaggle_setup.md              Kaggle setup guide.
├── tests/detection/                 Unit / integration tests (to be added).
├── requirements.txt                 Python dependencies.
├── README.md                        This file.
└── .gitignore                       Excludes datasets, checkpoints, envs, secrets.
```

## Verified Dataset Facts (Stage 1)

| Item | Value |
|------|-------|
| Image HDF5 key | `img` |
| Image shape / dtype | `(128, 128, 14)` / `float64` |
| Mask HDF5 key | `mask` |
| Mask shape / dtype | `(128, 128)` / `uint8`, values ∈ `{0, 1}` |
| Train / Valid / Test | 3799 / 245 / 800 files (image == mask counts in every split) |
| NaN / Inf on train | 0 / 0 |
| Positive-pixel ratio (train) | ~2.32 % |
| Normalization | per-channel z-score, **train-only** stats |
| Augmentation (train) | hflip (p=0.5), vflip (p=0.5), rot90 × k∈{0,1,2,3} |
| Augmentation (valid/test) | none |
| DataLoader tensor | image `(B, 14, 128, 128) float32`, mask `(B, 128, 128) float32` in {0.0, 1.0} |

### Channel mapping (authoritative)

| Idx | Feature | Source |
|----:|---------|--------|
| 0 | B1  – Coastal aerosol | Sentinel-2 |
| 1 | B2  – Blue            | Sentinel-2 |
| 2 | B3  – Green           | Sentinel-2 |
| 3 | B4  – Red             | Sentinel-2 |
| 4 | B5  – Red Edge 1      | Sentinel-2 |
| 5 | B6  – Red Edge 2      | Sentinel-2 |
| 6 | B7  – Red Edge 3      | Sentinel-2 |
| 7 | B8  – NIR             | Sentinel-2 |
| 8 | B8A – Narrow NIR      | Sentinel-2 |
| 9 | B9  – Water Vapor     | Sentinel-2 |
| 10 | B10 – Cirrus         | Sentinel-2 |
| 11 | B11 – SWIR 1         | Sentinel-2 |
| 12 | Slope                | ALOS PALSAR DEM |
| 13 | DEM                  | ALOS PALSAR |

## What is committed to GitHub, and what is not

**In Git:**

- All source code under `src/`.
- Both notebooks (`01_detection_development.ipynb`, `02_detection_kaggle_setup.ipynb`).
- `configs/detection.yaml`.
- The **small** Stage-1 verification artifacts under `outputs/detection/data_verification/` (JSON, CSV, TXT, and the six PNG figures — a few MB total). These make Stage 2 reproducible without re-running the full statistics pass.
- `requirements.txt`, `README.md`, `docs/**`, `.gitignore`.

**Not in Git (excluded by `.gitignore`):**

- `data/raw/` and any `*.h5` / `*.hdf5` file (the Landslide4Sense archive is 8.8 GB).
- Any model checkpoints (`*.pth`, `*.pt`, `*.ckpt`).
- All bulk generated outputs (`outputs/detection/{training,predictions,masks,probability_maps,overlays,polygons,reports}/`).
- Virtualenvs, `__pycache__/`, `.ipynb_checkpoints/`, logs, temp files.
- Anything credential-like: `.env`, `kaggle.json`, keys.

## Local development setup

```bash
git clone <REPO_URL> LandslideGuard
cd LandslideGuard
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the raw Landslide4Sense archive under:

```
data/raw/landslide4sense/
├── TrainData/TrainData/{img,mask}/*.h5
├── ValidData/ValidData/{img,mask}/*.h5
└── TestData/TestData/{img,mask}/*.h5
```

Re-run Stage-1 verification (optional, ~30 s):

```bash
python -c "from pathlib import Path; import runpy; runpy.run_path('notebooks/01_detection_development.ipynb')" || true
# or open the notebook in Jupyter and Run All.
```

Stage 2 (U-Net training) is intended to run on **Kaggle GPU** — see below.

## Kaggle GPU setup

Full step-by-step: **[docs/kaggle_setup.md](docs/kaggle_setup.md)**.

Short version:

1. Kaggle → **Create → New Notebook**.
2. Right sidebar → **Accelerator = GPU T4 x2** (or any GPU).
3. Right sidebar → **Add Input → Datasets → search "Landslide4Sense"** and attach the community dataset.
4. **First run** [notebooks/00_kaggle_setup.ipynb](notebooks/00_kaggle_setup.ipynb) — set `GITHUB_REPO = "https://github.com/Aryan2080/Landslide-Guard.git"` and `DATA_ROOT = "/kaggle/input/<slug>"`, Run All, wait for `KAGGLE ENVIRONMENT READY`.
5. **Then run** [notebooks/02_detection_model_development.ipynb](notebooks/02_detection_model_development.ipynb) — the 28-section Stage-2 notebook. Same two config values in Section 01. Runtime → Run All.
6. Training artifacts land under `/kaggle/working/`:
   - `checkpoints/detection/best_model.pth` — locked best model
   - `checkpoints/detection/final_model_config.yaml` — model + threshold + channels
   - `outputs/detection/training/` — model summary, loss/metric curves, history CSV
   - `outputs/detection/experiments/experiment_results.csv` — every experiment row
   - `outputs/detection/test/test_metrics.json` — one-shot test metrics
   - `outputs/detection/predictions/` — 6 test prediction figures
   - `outputs/detection/error_analysis/` — good / partial / hard test examples
   - `outputs/detection/validation/` — validation-side plots

**Roles:** GitHub carries the *code*, Kaggle Dataset carries the *raw HDF5 files*, Kaggle GPU runs *training* (Stage 2), and future training outputs live in `/kaggle/working/` inside the notebook.

## GPU verification (Stage-2 prerequisite)

Kaggle notebook Section 02 runs:

```python
import torch
print("cuda available :", torch.cuda.is_available())
print("device count  :", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device        :", torch.cuda.get_device_name(0))
```

If `cuda available: False`, switch the accelerator on and re-run — do not start Stage 2 without a GPU.

## Status

- **Stage 1 – data preparation:** COMPLETE (see notebook 01).
- **Stage 2 – U-Net training:** environment prepared, training **not started**.
- **Monitoring / Prediction / frontend / backend:** out of scope.

No model has been trained. No dataset files have been modified.
