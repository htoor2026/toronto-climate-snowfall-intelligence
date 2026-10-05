# Operations Runbook

## Install

Python 3.11 is recommended.

```bash
python -m pip install -r requirements.txt
```

## Run tests

```bash
python -m pytest tests/ -v
```

## Refresh the current forecast

```bash
python scripts/refresh_forecast.py
```

Production-only dry run without NOAA/ECCC requests:

```bash
python scripts/refresh_forecast.py --skip-live-refresh
```

A refresh uses new predictor information. It does not retrain the model.

## Check whether retraining is needed

```bash
python scripts/retrain_model.py
```

Force a complete benchmark run:

```bash
python scripts/retrain_model.py --force
```

Retraining is only appropriate after a completed winter provides a new trustworthy
snowfall label.

## Update the MLflow registry

```bash
python scripts/register_mlflow_models.py
```

Start the local MLflow UI:

```bash
mlflow server \
  --backend-store-uri sqlite:///tracking/mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

Open `http://127.0.0.1:5000`.

## Run the operational agent

```bash
python scripts/run_agent.py
```

Generate a report even when there is no material event:

```bash
python scripts/run_agent.py --force-report
```

### Optional email alerts

Email is disabled by default. Configure `config/agent_config.json` and provide the
SMTP variables shown in `.env.example`. Never commit real credentials.

```bash
python scripts/run_agent.py --send-email
```

## Run the dashboard

```bash
streamlit run app.py
```

## Data policy

- Raw downloaded data is intentionally excluded from Git.
- Processed reproducibility artifacts used by the application are versioned.
- The live forecast refresh pulls current NOAA/ECCC inputs when run.
- Local MLflow databases and model binaries are excluded from Git.
