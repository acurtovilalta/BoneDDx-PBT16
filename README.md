# BoneDDx-PBT16
This repository contains the code to train and evaluate a radiology-informed multi-task network for 16-entity primary bone tumor classification from routine radiographs. It uses a YAML config, performs 7-fold CV on the train split, and writes per-fold checkpoints plus aggregated metrics.


## 1) Reproduce the environment (with `uv`)

We use [`uv`](https://github.com/astral-sh/uv) for fast, reproducible Python environments.

### Prerequisites

* Python ≥ 3.10 (3.10/3.11 recommended)
* `uv` installed

### Create/sync the env

From the repo root:

```bash
# creates a virtualenv and installs pinned deps from pyproject + uv.lock
uv sync
```

> Tip: If you don’t have a lockfile yet (fresh repo), run `uv lock` once to resolve and pin all dependencies.


## 2) Data expectations

* An **Excel labels file** (see `configs/config.yaml: labels_path`) with at least:

  * `split` ∈ {`TRAIN`, `TEST`} (explicit split respected by the trainer)
  * `entity` (string class name; mapped across folds)
  * Auxiliary columns named exactly as in `aux_cols` (e.g., `lesion_pattern`, `cortical_destruction`, `periosteal_reaction`), already filled with categorical strings (including “UNK” if used). This is now hard coded. Sorry!

* An **image directory** (`img_dir`) containing the radiograph crops that `XrayDatasetMTL` knows how to resolve (via a filename path column. Ensure your dataset class and labels file agree on this).

Update the paths in `configs/config.yaml` to match your setup.

---

## 3) Configuration

All training/eval options live in `configs/config.yaml`. 

## 4) Running cross-validation

From the repo root:

```bash
uv run scripts/run_cv.py --config configs/config.yaml
```

If your repository is not package-structured and you see an import error, run with the repo root on `PYTHONPATH`:

```bash
uv run PYTHONPATH=. scripts/run_cv.py --config configs/config.yaml
```

What happens:

* The script reads the labels file, respects the explicit `TRAIN`/`TEST` split, and runs **7-fold CV on the TRAIN partition**.
* For each fold:

  * Trains until early stopping (by **validation balanced accuracy**).
  * Saves the **best checkpoint**.
  * Evaluates that checkpoint on **VAL** (the fold’s val split) and the **fixed TEST split**.

---

## 5) Outputs

All outputs are written under `save_dir/folds/`:

Per fold:

* `fold{K}_best.pt` — best checkpoint (by validation balanced accuracy)
* `fold{K}_VAL_report.txt` — text classification report for VAL
* `fold{K}_TEST_report.txt` — text classification report for TEST

Aggregates:

* `cv_per_fold_metrics.csv` — metrics per fold (VAL and TEST blocks)
* `cv_aggregate_metrics.csv` — mean ± sd across folds
* `cv_summary.xlsx` — same as above in Excel with two sheets (`per_fold`, `aggregate`) if Excel writer is available

Console logs (stdout):

* Epoch-wise training losses (main/aux) and validation metrics
* Early stopping notices
* Per-class counts per fold (to help diagnose imbalance)

---

## 8) Citing

TBA
