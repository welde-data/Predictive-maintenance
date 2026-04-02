# Predictive Maintenance Pipeline · AI4I 2020

End-to-end data engineering and ML pipeline built on Databricks, with an AI-powered Streamlit dashboard for batch scoring, top-K alerting, and natural language analysis via Gemini 2.0 Flash.

## Live Demo

[predmaint.streamlit.app](https://predictive-maintenance-gwxfcbgjkmlkgszgjkoqzq.streamlit.app/)

## Architecture

```
Raw CSV (UCI)
    ↓
01_bronze_ingest    →  ai4i_bronze_raw      (Delta)
    ↓
02_silver_clean     →  ai4i_silver_clean    (Delta · typed · validated)
    ↓
03_gold_features    →  ai4i_gold_features   (Delta · ML-ready feature vectors)
    ↓
04_train_and_score  →  ai4i_predictions     (Delta · RF predictions + probabilities)
    ↓
predictions.csv     →  Streamlit Dashboard  (top-K alerts · AI explanations)
```

## Stack

| Layer | Technology |
|---|---|
| Orchestration | Databricks Community Edition · DBR 14.x |
| Storage | Delta Lake |
| Processing | PySpark · Spark ML |
| Model | RandomForest · 80 trees · depth 8 |
| AI Assistant | Google Gemini 2.0 Flash |
| Dashboard | Streamlit · Plotly |

## Notebooks

| Notebook | Layer | Description |
|---|---|---|
| `01_bronze_ingest.py` | Bronze | Downloads AI4I 2020 CSV from UCI, sanitizes column names, writes Delta table. Idempotent — skips if row count matches. |
| `02_silver_clean.py` | Silver | Enforces schema types, runs 5 data quality checks (nulls, duplicates, label distribution, sensor ranges), writes Delta table. |
| `03_gold_features.py` | Gold | Spark ML pipeline: StringIndexer → OneHotEncoder (machine type) → VectorAssembler. Outputs feature vectors for ML. |
| `04_train_and_score.py` | ML · Serve | Trains RandomForest, evaluates AUC/F1/confusion matrix, writes predictions Delta table, exports predictions.csv with download button. |

## Dashboard Features

- **Top-K Alerts** — ranked table of highest-risk machines with inline probability bars
- **AI Machine Explainer** — select any UDI → Gemini explains risk and recommends action
- **AI Shift Report** — one-click maintenance briefing for the current shift
- **Ask the Data** — natural language questions answered by Gemini from the scoring results
- **Risk Distribution** — probability histogram, tier breakdown, calibration check
- **Model Performance** — confusion matrix, precision/recall/F1, false negative warnings
- **Raw Scores** — filterable full prediction table with CSV export

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
- 10,000 records · 14 features · ~3.4% failure rate
- Features: air/process temperature, rotational speed, torque, tool wear, 5 failure mode flags

## Running Locally

```bash
git clone https://github.com/yourname/predmaint
cd predmaint
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Add your Gemini key
echo "GEMINI_API_KEY=AIza..." > .env

streamlit run app.py
```

## Updating Data

After each Databricks pipeline run:

1. Run the export cell in `04_train_and_score.py`
2. Download `predictions.csv`
3. Copy to `data/predictions.csv`
4. `git add data/predictions.csv && git commit -m "update predictions" && git push`
5. Streamlit Cloud auto-redeploys in ~30 seconds

## Deploying to Streamlit Cloud

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select repo
3. Main file: `app.py`
4. Settings → Secrets → add:
   ```toml
   GEMINI_API_KEY = "AIza..."
   ```
5. Deploy

## Model Results

| Metric | Value |
|---|---|
| AUC-ROC | update after run |
| F1 Score | update after run |
| Precision | update after run |
| Recall | update after run |

## Project Structure

```
predmaint/
├── app.py                     #Streamlit dashboard
├── requirements.txt
├── README.md
├──LICENSE 
├── .gitignore
├── notebooks/
│   ├── 01_bronze_ingest.py
│   ├── 02_silver_clean.py
│   ├── 03_gold_features.py
│   └── 04_train_and_score.py
├── data/
│   ├── voice_maintenance_log.csv 
│   └── predictions.csv        #exported from Databricks
└── .streamlit/
    └── config.toml              # theme
```

#### Disclaimer

This project is built on the publicly available AI4I 2020 Predictive Maintenance Dataset (UCI). It is intended as a portfolio demonstration. No proprietary data, client data, or personal data is used or stored.
AI-generated responses are produced via the Google Gemini API using anonymised, aggregated scoring results only. Voice notes recorded in the Voice Logger tab are transcribed locally using OpenAI Whisper and are never transmitted to an external server. No audio data is persisted beyond the current session unless explicitly saved by the user.