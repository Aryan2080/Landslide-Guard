# Kaggle GPU setup — LandslideGuard Stage 2

This is a **setup + verification** guide. It does *not* train the U-Net.
Training belongs in a separate Stage-2 notebook.

## Role of each component

| Component | What it holds | Why |
|-----------|--------------|-----|
| **GitHub repo** | Project code, configs, small verification artifacts, notebooks | Reproducible, versioned |
| **Kaggle dataset (Landslide4Sense)** | Raw HDF5 image + mask patches | 8.8 GB — never in Git |
| **Kaggle GPU (T4 / P100 / L4)** | Compute | Local Windows box has no CUDA GPU |
| **`/kaggle/working/`** | Stage-2 training artifacts (checkpoints, logs, predictions) | Downloaded from the notebook run |

## Step-by-step

### 1. Enable GPU on your Kaggle account

- Open <https://www.kaggle.com/settings> → **Phone verification**. Required for GPU access.
- Verify email + phone once; then GPU quota is granted per week.

### 2. Attach the Landslide4Sense dataset to your notebook

- Kaggle → **Create → New Notebook**.
- Right sidebar → **+ Add Input → Datasets** → search `Landslide4Sense`.
- Attach the community-maintained dataset that contains `TrainData / ValidData / TestData`.
- Note the mount path shown in the sidebar. Typical values are:
  - `/kaggle/input/landslide4sense/`
  - `/kaggle/input/landslide4sense-competition/`
  - `/kaggle/input/landslide-detection/`

  The exact slug depends on the dataset the community maintainer chose. You will paste it into `DATA_ROOT` in the notebook.

### 3. Enable the GPU accelerator

- Right sidebar → **Session options → Accelerator → GPU T4 x2** (or another GPU).
- Reset the session so the change takes effect.

### 4. Bring in the LandslideGuard code

Two options — the notebook supports both:

- **Recommended:** git-clone the public GitHub repository. Set:
  ```python
  GITHUB_REPO = "https://github.com/<owner>/<repo>.git"
  ```
- **Fallback:** upload the LandslideGuard repo as a *Kaggle dataset* (a zip of the project source without `data/raw/`) and set:
  ```python
  GITHUB_REPO = None
  LOCAL_REPO_PATH = "/kaggle/input/<your-code-dataset>/LandslideGuard"
  ```

### 5. Upload and open `00_kaggle_setup.ipynb`

- In the Kaggle notebook editor, use **File → Upload notebook** and pick this file from the repo, or copy-paste the cells into a fresh notebook.

### 6. Set the two configuration variables

Open Section 03 (GitHub clone) and Section 06 (dataset location) and set:

```python
GITHUB_REPO = "https://github.com/<owner>/<repo>.git"
DATA_ROOT   = "/kaggle/input/<the-dataset-slug>"
```

`DATA_ROOT` is whatever directory *directly contains* `TrainData/`, `ValidData/`, `TestData/`. If the dataset uses the archive's nested layout (`TrainData/TrainData/{img,mask}/…`), point `DATA_ROOT` at the outer `TrainData/ValidData/TestData` level — the code handles either layout.

### 7. Run all cells

- **Runtime → Run All**.
- The last cell prints a PASS/FAIL table.
- Proceed to Stage 2 only when it prints:

  ```
  KAGGLE ENVIRONMENT READY
  ```

### 8. What "verified" means

At the end of Section 11 the notebook has:

- confirmed CUDA is available and a GPU is attached;
- git-cloned this repository and made `src.detection.*` importable;
- located the raw dataset at `DATA_ROOT`;
- loaded the frozen Stage-1 normalization statistics (**not** recomputed);
- instantiated `Landslide4SenseDataset` for train / valid / test;
- pulled one real batch of shape `(B, 14, 128, 128)` per split;
- moved that batch onto CUDA.

**No U-Net exists yet. No training loop has run. No checkpoints have been written.**

### 9. Next: Stage-2 training notebook

Create a separate notebook, e.g. `notebooks/03_detection_training_kaggle.ipynb`, that reuses the code cloned by `00_kaggle_setup.ipynb`:

```
GitHub code  +  Kaggle dataset  +  Stage-1 preprocessing
     ->  14-channel U-Net
     ->  training  ->  validation  ->  test
     ->  checkpoint in /kaggle/working/
```

That work is out of scope for the current setup task.

## Common pitfalls

- **`cuda available: False`** — Accelerator was left on CPU. Fix in Session options.
- **`DATA_ROOT` empty** — you attached the wrong dataset, or the mount path is one directory deeper/higher than you set.
- **`pip install -r requirements.txt` re-installs a CPU torch wheel** — Kaggle ships a CUDA-enabled PyTorch; do **not** force-reinstall it. The notebook checks each dependency and only installs what is genuinely missing (typically nothing beyond h5py, which is already present on modern Kaggle images).
- **Stage-1 normalization stats missing on Kaggle** — they live at `outputs/detection/data_verification/normalization_statistics.json` in the cloned repository. They must **not** be recomputed on Kaggle (that would leak validation/test information).
