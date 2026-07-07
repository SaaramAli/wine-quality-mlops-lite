# Wine Quality MLOps Lite

A compact, end-to-end machine learning pipeline demonstrating core MLOps
concepts — data validation, model serving, and drift monitoring — built
as a lightweight, fully runnable system rather than a production-scale
deployment.

## Motivation

Most portfolio ML projects stop at "train a model and report accuracy."
This project is deliberately scoped differently: the modelling task
(predicting wine quality from physicochemical properties) is intentionally
simple, because the point of the project is not the model — it's
demonstrating an understanding of what happens *around* the model in a
production setting. Specifically:

- How do you know your input data is trustworthy before it reaches a model?
- How do you serve a model as a usable service rather than a notebook cell?
- How do you know when a deployed model's input data has started to look
  different from what it was trained on — and needs retraining?

Each of these questions maps to one stage of the pipeline below.

## Architecture

Wine data (CSV)
│
▼
Validate (Great Expectations)
│
▼
Train model (RandomForest)
│
▼
Model artifact (joblib + reference stats)
│
▼
FastAPI /predict ──────────► Streamlit dashboard
▲
New batch data ──► Drift check ──────┘
(KS-test)

The pipeline has two paths that converge on the dashboard: the main
training/serving path (top), and a monitoring path that compares any new
incoming data batch against the distribution the model was trained on
(right). The monitoring path is what allows the system to flag *when* a
model might need retraining, rather than assuming it stays accurate
indefinitely.

## Design decisions: why "lite"

A full production MLOps stack for a project like this would typically
include Apache Airflow for orchestration, PostgreSQL for state and
prediction logging, and Grafana for monitoring dashboards. This project
replaces each of those with a lightweight equivalent that demonstrates
the same underlying concept without the operational overhead of standing
up and configuring that infrastructure:

| Production-scale tool | Lightweight equivalent used here | Concept preserved |
|---|---|---|
| Apache Airflow | A single ordered pipeline of scripts | Staged, repeatable pipeline execution |
| PostgreSQL | JSON reports written to disk | Persisting pipeline state and results |
| Grafana | A Streamlit dashboard | Visualizing monitoring output |
| — | KS-test drift check | Statistical detection of distribution shift |

The last row has no production-scale counterpart in the table because
it's the one genuinely original contribution of this project: automated,
statistical drift detection between a reference dataset and any new
incoming batch, using a two-sample Kolmogorov-Smirnov test per feature.

## Pipeline stages

1. **Data preparation** (`data/prepare_data.py`) — downloads the UCI Wine
   Quality (red) dataset and splits it into a reference set (used for
   training and as the baseline distribution for drift comparisons) and a
   simulated "new batch" (representing future incoming data). A
   deliberately corrupted copy of the new batch is also generated, to
   prove the validation step actually catches problems rather than
   trivially passing everything.

2. **Data validation** (`data/validate_data.py`) — defines a Great
   Expectations suite (value ranges for pH and quality, null checks on
   alcohol and quality) and validates both the reference set and the
   corrupted batch against it. The reference set passes in full; the
   corrupted batch fails on exactly the fields that were tampered with,
   confirming the validation logic is functioning correctly rather than
   passing by default.

3. **Model training** (`model/train_model.py`) — trains a RandomForest
   classifier predicting whether a wine is "good" (quality ≥ 6). Also
   computes and saves per-feature mean/std statistics from the reference
   set, which the drift check later compares new data against.

4. **Model serving** (`api/main.py`) — exposes the trained model as a
   FastAPI `/predict` endpoint, returning a prediction, label, and
   probability for a given set of input features.

5. **Drift monitoring** (`monitoring/check_drift.py`) — runs a two-sample
   KS-test per feature, comparing an incoming batch's distribution against
   the reference distribution. Any feature with p < 0.05 is flagged as
   drifted. Tested against both a genuinely clean held-out batch (expect
   no meaningful drift) and a synthetically shifted batch (expect drift
   correctly flagged on the shifted features).

6. **Dashboard** (`app/streamlit_app.py`) — a three-tab Streamlit app
   tying the above together: live predictions against the running API,
   the validation report, and the drift report, each re-runnable on
   demand from the UI.

## Results

- Baseline RandomForest: **78.5% accuracy**, 0.83 precision, 0.74 recall,
  0.78 F1 on a held-out test split.
- Model performance was not the focus of this project — no hyperparameter
  tuning or feature engineering was performed. The emphasis throughout is
  on pipeline structure, not model optimization.

## Known limitations

- **Multiple testing in the drift check.** Running an independent KS-test
  on 11 features at α = 0.05 means roughly a 1-in-2 chance of at least one
  false positive on any given clean batch, purely from chance (this was
  observed directly during development — a genuinely clean batch flagged
  one feature as drifted with a borderline p-value, while a synthetically
  shifted batch showed drift with p-values effectively at 0). A production
  version would apply a multiple-comparison correction (e.g. Bonferroni)
  or monitor drift trends over multiple batches rather than acting on a
  single flagged run.
- The "lite" infrastructure choices (file-based logging instead of
  Postgres, script sequencing instead of Airflow) are deliberate scope
  decisions for a demo project, not a claim that they would be
  sufficient at production scale.

## Running locally

```bash
conda create -n wine_mlops python=3.11 -y
conda activate wine_mlops
pip install -r requirements.txt

python data/prepare_data.py
python data/validate_data.py
python model/train_model.py

# In one terminal:
uvicorn api.main:app --reload --port 8000

# In a second terminal:
streamlit run app/streamlit_app.py
```

## Project structure

wine-quality-mlops-lite/
├── api/
│   └── main.py                  # FastAPI serving endpoint
├── app/
│   └── streamlit_app.py         # Dashboard: predict, validate, monitor
├── data/
│   ├── prepare_data.py          # Dataset download + splitting
│   ├── validate_data.py         # Great Expectations validation
│   ├── reference.csv
│   ├── new_batch.csv
│   └── bad_batch.csv
├── model/
│   ├── train_model.py           # Training script
│   └── artifacts/
│       ├── model.joblib
│       ├── metrics.json
│       └── reference_stats.json
├── monitoring/
│   └── check_drift.py           # KS-test drift detection
├── reports/
│   ├── validation_report.json
│   └── drift_report.json
├── requirements.txt
└── README.md

## Future work

Given more time, the natural next steps would be: automating the pipeline
with Airflow so that drift detection can trigger scheduled retraining;
persisting predictions and drift history to Postgres to track model
behavior over time rather than only a single snapshot; and replacing the
Streamlit dashboard with Grafana for production-grade alerting on drift
thresholds.