# Predictive Maintenance (PdM) — AI4I 2020

End-to-end predictive maintenance pipeline trained on the AI4I 2020 dataset.  
The project includes feature engineering, model training (Logistic Regression or HistGradientBoosting), optional probability calibration, and top-K alerting for maintenance capacity planning. A Streamlit UI is provided for batch scoring and interactive review.

---

## What this project does

- **Input**: AI4I 2020-style sensor data (CSV)
- **Output**:
  - A trained model artifact (`artifacts/pdm_model.joblib`)
  - Metadata report (`artifacts/pdm_meta.json`)
  - Batch scoring output (via CLI or Streamlit) with:
    - `p_failure` (predicted probability)
    - `rank` (descending risk rank)
    - `alert_topk` (True for top-K items)
    - engineered features (`TempDiff`, `Torque_x_RPM`)

---

## Repository Structure

```text
Predictive-maintenance/
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ app.py                   # Streamlit UI 
├─ artifacts/               # generated 
├─ data/                    # local datasets
├─ notebooks/
│  └─ notebook.ipynb
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ features.py
│  ├─ metrics.py
│  ├─ predict.py
│  └─ train.py
└─ venv/                     
```
## Setup
### 1) Create and activate a virtual environment

```
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```
pip install -r requirements.txt
```

## Data and Schema

### Required input columns (minimum for scoring)

- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

### Engineered features (computed automatically)

- **TempDiff** = `Process temperature [K] - Air temperature [K]`
- **Torque_x_RPM** = `Torque [Nm] * Rotational speed [rpm]`

### Training target

- `Machine failure` (binary)

### Optional columns

If present, the pipeline drops known non-feature/leakage columns (e.g., identifiers and failure-mode columns) based on configuration.

---

## Configuration

Defaults are defined in `src/config.py` via `TrainConfig`.

### Key settings

- `model_type`: `"hgb"` (default) or `"lr"`
- `calibrate`: `True/False`
- `calibration_method`: `"isotonic"` or `"sigmoid"`
- `decision_policy`: `"topk"` or `"f1"`
- `topk_k`: maintenance capacity per cycle (e.g., `50`)


## Train (CLI)

### Run training from the repository root
```powershell
python -m src.train
```

## Training outputs

- artifacts/pdm_model.joblib
- artifacts/pdm_meta.json

## Score (CLI)

### Score a batch CSV and create ranked output

```powershell
python -m src.predict --data-path data/raw/ai4i.csv --topk 50
```

### Scoring output

- `artifacts/pdm_scored.csv`

### Columns added during scoring

- `p_failure` — predicted probability of failure
- `rank` — descending risk rank (**1 = highest risk**)
- `alert_topk` — `True` for the **Top-K** selected records
- `TempDiff`, `Torque_x_RPM` — engineered features for interpretability

## Streamlit UI

### Run the UI from the repository root

```powershell
streamlit run app.py
```

### Expected behavior

- Loads the trained model from `artifacts/pdm_model.joblib`
- Scores an uploaded CSV using the same feature engineering and strict schema checks
- Supports filtering and exporting scored results

## Evaluation and Operational Use

### Reported model metrics

- PR-AUC and ROC-AUC are recorded in `artifacts/pdm_meta.json`

### Capacity-aware policy (Top-K)

- Each scoring cycle flags **K** records (ties may affect cutoff behavior)
- `precision@K` and `recall@K` quantify the operational trade-off for a given capacity

# 👥 Contributors

This is a personal project developed and maintained by:

- **Welederufeal Tadege** — [LinkedIn](https://www.linkedin.com/in/) | [GitHub](https://github.com/welde2001-bot)
