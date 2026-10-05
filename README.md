# Toronto Climate & Seasonal Snowfall Intelligence

An end-to-end climate analysis, probabilistic snowfall forecasting, monitoring, retraining, and MLOps project for Toronto.

This project goes beyond a forecasting notebook. It combines historical climate analysis, leakage-safe seasonal forecasting, uncertainty calibration, live predictor refreshes, challenger-model monitoring, retraining rules, MLflow model governance, automated tests, an operational agent, and a Streamlit dashboard.

---

## What the project does

The system answers five practical questions:

1. How has Toronto's climate changed historically?
2. How much snowfall is reasonable to expect for the coming winter?
3. How uncertain is that forecast?
4. Do new climate signals such as ENSO / El Niño, NAO, or autumn weather materially change the outlook?
5. Should a newer machine-learning model replace the current production model?

The system intentionally separates **forecast refresh** from **model retraining**.

```text
New climate predictors
        ↓
Refresh current-season forecast
        ↓
Detect material change
        ↓
Report / alert

Completed winter + actual snowfall
        ↓
Retrain challenger
        ↓
Walk-forward evaluation
        ↓
Promotion gate
        ↓
MLflow champion / challenger registry
```

---

## Business context

Seasonal snowfall affects winter-sensitive operations such as:

- municipal snow removal
- airport operations
- transportation and logistics
- workforce planning
- equipment allocation
- maintenance budgets
- seasonal inventory and procurement

The goal is not to claim perfect snowfall prediction. The goal is to provide a repeatable decision-support system that communicates expected snowfall, uncertainty, scenario probabilities, forecast changes, and model-governance decisions.

---

## Historical climate analysis

Historical Toronto climate analysis was performed primarily using Toronto City Centre climate data.

Key findings:

- Annual mean temperature trend: approximately **+0.218°C per decade**
- Winter warming trend: approximately **+0.301°C per decade**
- Extreme hot days using a 1991–2020 percentile threshold increased significantly
- Extreme cold days using a 1991–2020 percentile threshold decreased significantly
- No strong linear evidence was found for a long-term trend in total annual precipitation
- No strong simple linear evidence was found for long-term seasonal snowfall change

These results are descriptive and do not by themselves establish causal attribution.

---

## Snowfall target

Seasonal snowfall is built from Toronto Pearson daily snowfall observations.

The Pearson station thread combines:

- legacy station ID `5097`
- current station ID `51459`

The seasonal target is total snowfall across the Oct–Apr winter season.

Strict target-quality rules are used so questionable winters are not silently imputed into the supervised-learning target.

---

## Forecasting setup

Forecast issue date is approximately **November 1**.

For a winter ending in `2027`, the predictor year is `2026`.

Allowed pre-winter information includes:

- JAS ONI
- JAS NAO
- Sep–Oct Toronto temperature
- Sep–Oct Toronto precipitation
- previous-season snowfall
- rolling snowfall-history features

Information from the target winter itself is never used when producing that winter's forecast.

---

## Models

### Production champion

**Probabilistic Historical Climatology**

The production benchmark uses recent strict-quality snowfall history.

It provides:

- expected snowfall
- median snowfall
- below / near / above-normal probabilities
- empirical uncertainty interval
- conformalized uncertainty interval

### Machine-learning challenger

**ElasticNet + climate features**

Features include:

- ONI JAS
- NAO JAS
- Sep–Oct mean temperature
- Sep–Oct total precipitation
- previous-season snowfall
- 5-season rolling snowfall mean
- 10-season rolling snowfall mean
- 10-season rolling snowfall standard deviation
- time index

The challenger is deliberately not promoted just because it is more sophisticated.

---

## Validation

The project uses **one-step expanding walk-forward validation**.

For every historical forecast winter:

1. Train only on earlier winters
2. Forecast the next winter
3. Record the error
4. Expand the training window
5. Repeat

This is more realistic for time-series forecasting than a random train/test split.

### Current walk-forward results

| Model | MAE | RMSE | Bias |
|---|---:|---:|---:|
| Historical Climatology | 31.56 cm | 37.87 cm | +9.92 cm |
| ElasticNet + NAO | 31.30 cm | 38.05 cm | +8.82 cm |

The challenger improves MAE by only about **0.81%** and slightly worsens RMSE.

### Promotion gate

The challenger is promoted only if:

```text
MAE improvement vs climatology >= 3%
AND
RMSE does not degrade
```

Current result:

```text
Promote challenger: False
Champion: Historical Climatology
Challenger: ElasticNet + NAO
```

---

## Probabilistic forecasting

The production probabilistic forecast uses empirical climatology.

Historical evaluation:

- probabilistic climatology Brier score: approximately **0.666**
- challenger Brier score: approximately **0.716**

The challenger did not improve probabilistic performance enough to replace the benchmark.

---

## Uncertainty calibration

The original empirical 80% interval under-covered historically.

A conformal calibration layer was added.

Backtest result:

- original empirical interval coverage: **68.0%**
- conformalized empirical interval coverage: **82.9%**
- mean conformalized width: approximately **94.9 cm**

Selected method:

**Conformalized Empirical Interval**

---

## Current production snapshot

For the 2026–27 winter, the production system currently reports approximately:

- expected snowfall: **117.1 cm**
- median snowfall: **124.8 cm**
- calibrated 80% interval: **58.7–161.3 cm**
- below normal: **33.3%**
- near normal: **33.3%**
- above normal: **33.3%**

The climate-feature challenger remains unavailable until all required pre-winter predictors are complete.

---

## Forecast refresh pipeline

Run:

```bash
python scripts/refresh_forecast.py
```

The refresh pipeline:

```text
NOAA ONI / NAO
+ ECCC Sep–Oct Toronto weather
+ historical snowfall features
        ↓
current feature row
        ↓
production forecast
+ challenger forecast when inputs are complete
        ↓
change detection
        ↓
forecast history
```

A forecast refresh does **not** retrain the model.

---

## Change detection

The system compares the latest forecast with the previous snapshot.

It monitors:

- expected snowfall change
- below / near / above probability changes
- interval-bound changes
- challenger status transitions

Material changes can trigger reporting or future email alerts.

---

## Retraining pipeline

Normal run:

```bash
python scripts/retrain_model.py
```

Manual benchmark run:

```bash
python scripts/retrain_model.py --force
```

Retraining occurs only after a completed winter provides a new trustworthy snowfall label.

Pipeline:

```text
completed winter
+ archived pre-winter feature snapshot
        ↓
append new labelled row
        ↓
walk-forward benchmark
        ↓
fit challenger candidate
        ↓
promotion gate
        ↓
save decision + report
```

---

## MLflow model governance

Register the latest retraining snapshot:

```bash
python scripts/register_mlflow_models.py
```

Current registry:

```text
Registered model: toronto-seasonal-snowfall

@champion
→ version 1
→ Historical Climatology

@challenger
→ version 2
→ ElasticNet + NAO
```

The challenger can be registered without being promoted.

Run the local MLflow UI:

```bash
mlflow server \
  --backend-store-uri sqlite:///tracking/mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

Then open:

```text
http://127.0.0.1:5000
```

---

## Operational agent

The project includes a small deterministic agent layer.

Run:

```bash
python scripts/run_agent.py
```

Force a report for testing:

```bash
python scripts/run_agent.py --force-report
```

The agent can:

- detect material forecast events
- summarize retraining decisions
- check registry consistency
- generate operational reports
- optionally trigger email alerts

The agent cannot:

- change the target
- modify validation rules
- change promotion thresholds
- retrain on its own
- promote a challenger that failed the gate

---

## Dashboard

Run:

```bash
streamlit run app.py
```

The dashboard contains:

- Current Forecast
- Forecast History
- Model & MLOps
- Business Communication
- Technical Walkthrough
- System Status

The dashboard is intentionally read-only. Forecasting and retraining remain explicit, auditable workflows.

---

## Automated tests

Run:

```bash
python -m pytest tests/ -v
```

Current test coverage includes:

- production forecast change detection
- challenger change detection
- retraining decision logic
- MLflow registry helpers
- agent event policy

Current status:

```text
14 passed
```

---

## Project structure

```text
toronto-climate-analysis/
├── app.py
├── config/
│   ├── forecast_config.json
│   ├── retraining_config.json
│   ├── mlflow_config.json
│   └── agent_config.json
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── forecasts/
│   ├── latest_forecast.json
│   └── forecast_history.jsonl
├── models/
│   └── retraining/
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 08_forecast_data_collection.ipynb
│   ├── 09_forecast_feature_engineering.ipynb
│   ├── 10_forecast_modeling.ipynb
│   ├── 10b_forecast_feature_improvement.ipynb
│   ├── 10c_final_model_comparison.ipynb
│   ├── 11_probabilistic_snowfall_forecast.ipynb
│   └── 11b_probability_calibration.ipynb
├── reports/
├── scripts/
│   ├── refresh_forecast.py
│   ├── retrain_model.py
│   ├── register_mlflow_models.py
│   └── run_agent.py
├── src/
│   ├── forecasting/
│   ├── training/
│   ├── tracking/
│   └── agent/
├── tests/
├── tracking/
│   └── mlflow.db
└── README.md
```

---

## Key design choices

### Why climatology is still production

Because it is difficult to beat consistently with a small seasonal dataset.

The ML challenger currently offers only marginal MAE improvement and slightly worse RMSE.

### Why uncertainty matters

Seasonal snowfall is highly variable. A single point forecast can create false confidence.

The project therefore reports probabilities and calibrated intervals.

### Why forecast refresh and retraining are separate

A new climate signal gives new **input information**, not a new **label**.

New ONI / NAO / autumn data should refresh the forecast.

A completed winter provides the new observed snowfall label needed for supervised retraining.

### Why the project avoids unnecessary complexity

The system intentionally does not require:

- Kafka
- Airflow
- Kubernetes
- LangChain
- multiple autonomous agents
- paid LLM APIs

The emphasis is on correct forecasting methodology, uncertainty, monitoring, governance, reproducibility and business communication.

---

## Limitations

- Seasonal snowfall data is limited compared with large industrial ML datasets
- Climate indices do not guarantee predictive skill for Toronto snowfall
- A single local station does not represent all of the Greater Toronto Area
- Forecast uncertainty remains large
- Climate relationships can change over time
- This is a portfolio decision-support system, not an operational meteorological forecast service

---

## Main takeaway

The strongest result of this project is not that a complex ML model beat a simple benchmark.

It is that the system **tested that assumption and refused to promote the more complex model when the evidence was insufficient**.

That is the same principle expected in a real production ML system:

> use the simplest model that performs reliably, quantify uncertainty, monitor changes, retrain only when new labels arrive, and promote challengers only when they prove themselves out of sample.
