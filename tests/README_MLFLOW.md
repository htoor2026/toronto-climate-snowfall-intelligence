# MLflow Champion / Challenger Registry

This adds a local SQLite-backed MLflow registry for the Toronto snowfall project.

Flow:

```text
retrain_model.py
→ metrics + candidate model + decision
→ register_mlflow_models.py
→ MLflow experiment + model registry
→ @champion / @challenger aliases
```

Install:

```bash
python -m pip install -U mlflow
```

Run:

```bash
python -m pytest tests/ -v
python scripts/retrain_model.py --force
python scripts/register_mlflow_models.py
```

Expected current aliases:

```text
@champion   -> Historical Climatology
@challenger -> ElasticNet + NAO
```

Open UI:

```bash
mlflow server --backend-store-uri sqlite:///tracking/mlflow.db --host 127.0.0.1 --port 5000
```

Then open http://127.0.0.1:5000

The included dataset.py patch also stops old research-era gaps (1939–1950, 1995, 2014, 2022) from being reported as if they were newly missing production snapshots. Only seasons newer than the current training frontier are considered for automatic retraining.
