# Toronto Climate & Seasonal Snowfall Intelligence

An end-to-end portfolio project for **historical climate analysis, probabilistic seasonal snowfall forecasting, model monitoring, retraining, MLflow governance, operational reporting, and business communication**.

The project is intentionally conservative: a more complex model is **not** promoted unless it proves better than the production benchmark under leakage-safe walk-forward validation.

## Business question

> **How much snowfall should Toronto expect this winter, how uncertain is that estimate, and has new climate information changed the outlook enough to justify an operational response?**

This type of decision support is relevant to winter-sensitive operations such as municipal snow removal, airport operations, transportation and logistics, staffing, maintenance, equipment allocation, procurement, and contingency planning.

## What the system does

```text
Historical ECCC climate + snowfall
NOAA ONI / NAO + Toronto autumn weather
        |
        v
Leakage-safe features
        |
        +-----------------------------+
        |                             |
        v                             v
30-season climatology           ElasticNet challenger
(production champion)           (climate-signal model)
        |                             |
        +-------------+---------------+
                      |
                      v
     probabilities + calibrated interval
                      |
                      v
              change detection
                      |
          +-----------+-----------+
          |                       |
          v                       v
  forecast history          agent/reporting
          |                       |
          +-----------+-----------+
                      |
                      v
              Streamlit dashboard
```

When a winter finishes, the system follows a separate retraining path:

```text
completed winter + observed snowfall
        |
        v
new labelled season
        |
        v
walk-forward benchmark
        |
        v
promotion gate
        |
        v
MLflow @champion / @challenger
```

**New El Niño / ENSO or NAO information refreshes a forecast. It does not retrain the model.**
Retraining requires a new completed winter and therefore a new target label.

## Current saved forecast snapshot

The repository includes a saved **2026-10-01** forecast snapshot for the 2026-27 winter:

| Output | Value |
|---|---:|
| Expected snowfall | 117.1 cm |
| Median snowfall | 124.8 cm |
| Calibrated 80% interval | 58.7-161.3 cm |
| Below normal | 33.3% |
| Near normal | 33.3% |
| Above normal | 33.3% |

The climate-feature challenger is intentionally blocked until its required JAS climate indices and Sep-Oct Toronto weather inputs are complete.

## Historical climate findings

Historical Toronto analysis found:

- annual mean temperature trend of about **+0.218 C per decade**
- winter temperature trend of about **+0.301 C per decade**
- increasing hot extremes using a 1991-2020 percentile threshold
- decreasing cold extremes using a 1991-2020 percentile threshold
- no strong simple linear evidence for a long-term annual precipitation trend
- no strong simple linear evidence for a long-term seasonal snowfall trend

These are observational results and do not, by themselves, establish causal attribution.

## Data and target

### Historical climate
Long-term Toronto climate analysis uses Environment and Climate Change Canada data, including homogenized temperature and precipitation products.

### Snowfall target
The seasonal snowfall target uses the Toronto Pearson station thread:

- legacy station ID: `5097`
- current station ID: `51459`

The target is Oct-Apr seasonal snowfall. Strict completeness rules are used so questionable seasons are not silently imputed into supervised training.

### Climate predictors
The challenger uses pre-winter information only:

- JAS Oceanic Nino Index (ONI)
- JAS North Atlantic Oscillation (NAO)
- Sep-Oct Toronto mean temperature
- Sep-Oct Toronto precipitation
- previous-season snowfall
- 5-season and 10-season rolling snowfall features
- 10-season snowfall variability
- time index

For a winter ending in 2027, the predictor year is 2026. Information from winter 2026-27 itself is not allowed into that forecast.

## Validation and model governance

The project uses **one-step expanding walk-forward validation** rather than a random split.

For each historical forecast winter:

1. train only on earlier winters
2. forecast the next winter
3. record the error
4. expand the training window
5. repeat

The production benchmark in retraining now uses the same definition as the live system: the latest **30 strict-quality prior winters**.

### Current benchmark results

| Model | MAE | RMSE | Bias |
|---|---:|---:|---:|
| 30-season Historical Climatology | 31.50 cm | 37.43 cm | +5.45 cm |
| ElasticNet + NAO | 31.30 cm | 38.05 cm | +8.82 cm |

Challenger skill versus production climatology:

- MAE skill: **+0.61%**
- RMSE skill: **-1.66%**

### Promotion gate

The challenger is promoted only when:

```text
MAE improvement vs production climatology >= 3%
AND
RMSE does not degrade
```

Current decision:

```text
Champion:   Historical Climatology
Challenger: ElasticNet + NAO
Promote:    No
```

This is a deliberate result: greater model complexity is not treated as evidence of better production performance.

## Probabilistic forecast and uncertainty

The production probability layer uses empirical climatology for below-, near-, and above-normal snowfall categories.

Historical probability evaluation favored climatology over the ML challenger:

- probabilistic climatology Brier score: about **0.666**
- challenger Brier score: about **0.716**

The original nominal 80% empirical interval under-covered historically, so a conformal calibration layer was added:

- original interval coverage: **68.0%**
- conformalized empirical interval coverage: **82.9%**
- selected method: **Conformalized Empirical Interval**

## MLOps and monitoring

The project includes:

- live NOAA/ECCC predictor refresh
- production/challenger change detection
- forecast history
- label-triggered retraining
- ElasticNetCV hyperparameter tuning during retraining
- objective promotion rules
- MLflow experiment tracking and model registry
- `@champion` and `@challenger` aliases
- deterministic operational agent/report generation
- optional SMTP email alerts
- Streamlit dashboard
- automated tests and GitHub Actions CI

The agent cannot change the target, validation design, promotion threshold, or promote a failed challenger.

## Quick start

Python 3.11 is recommended.

```bash
git clone https://github.com/htoor2026/toronto-climate-snowfall-intelligence.git
cd toronto-climate-snowfall-intelligence

python -m pip install -r requirements.txt
python -m pytest tests/ -v
```

Run the dashboard:

```bash
streamlit run app.py
```

Refresh the latest forecast:

```bash
python scripts/refresh_forecast.py
```

Check whether a new completed season requires retraining:

```bash
python scripts/retrain_model.py
```

Force a complete benchmark run:

```bash
python scripts/retrain_model.py --force
```

Update the MLflow registry after retraining:

```bash
python scripts/register_mlflow_models.py
```

Run the agent:

```bash
python scripts/run_agent.py
```

See [docs/OPERATIONS.md](docs/OPERATIONS.md) for the full runbook and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the system design.

## Dashboard

The Streamlit dashboard contains six views:

1. Current Forecast
2. Forecast History
3. Model & MLOps
4. Business Communication
5. Technical Walkthrough
6. System Status

The dashboard is read-only by design. Forecasting, retraining, registry updates, and alerts remain explicit and auditable workflows.

## Research notebooks

Key notebooks include:

- `01_data_collection.ipynb`
- `04a_precipitation_analysis.ipynb`
- `04b_snowfall_analysis.ipynb`
- `07_final_analysis.ipynb`
- `08_forecast_data_collection.ipynb`
- `09_forecast_feature_engineering.ipynb`
- `10_forecast_modeling.ipynb`
- `10b_forecast_feature_improvement.ipynb`
- `10c_final_model_comparison.ipynb`
- `11_probabilistic_snowfall_forecast.ipynb`
- `11b_probability_calibration.ipynb`

Processed research artifacts and figures are versioned in the repository; large raw downloads are intentionally excluded from Git.

## Repository structure

```text
toronto-climate-snowfall-intelligence/
├── app.py
├── config/
├── data/
│   └── processed/
├── docs/
├── figures/
├── forecasts/
├── models/
├── notebooks/
├── reports/
├── scripts/
│   ├── refresh_forecast.py
│   ├── retrain_model.py
│   ├── register_mlflow_models.py
│   └── run_agent.py
├── src/
│   ├── agent/
│   ├── forecasting/
│   ├── tracking/
│   └── training/
├── tests/
├── requirements.txt
└── README.md
```

## Design choices

### Why not deep learning?
The model-ready seasonal sample is small. A neural network would add parameter complexity without enough labelled seasons to justify it. The project therefore prioritizes strong baselines, regularization, leakage-safe validation, and uncertainty.

### Why not retrain when El Nino changes?
El Nino / ONI is a predictor, not a new target label. A new predictor value should refresh the current forecast. Retraining becomes statistically justified when a completed winter supplies new observed snowfall.

### Why keep climatology in production?
Because the ML challenger has not passed the pre-specified promotion gate. The system prefers evidence over complexity.

## Future work

The current system is designed so future data can be incorporated without changing the forecasting methodology.

### During the pre-winter period

As new climate information becomes available, especially updated **ENSO / El Nino (ONI)**, **NAO**, and Sep-Oct Toronto weather:

```text
new ONI / NAO / autumn weather
        |
        v
refresh current-season predictors
        |
        v
update challenger forecast when inputs are complete
        |
        v
compare with previous forecast
        |
        v
report / alert only if the change is material
```

This is a **forecast refresh**, not model retraining. El Nino / ONI is an input feature; a new value does not create a new supervised-learning label.

### After the winter is complete

When the season finishes and a strict-quality observed snowfall total becomes available:

```text
archived pre-winter feature snapshot
+ completed observed snowfall
        |
        v
append one new labelled season
        |
        v
retrain ElasticNetCV challenger
        |
        v
rerun expanding walk-forward evaluation
        |
        v
compare against 30-season production climatology
        |
        v
register new MLflow version
        |
        v
promote only if the configured gate is passed
```

This lets the system learn from new winters while preserving leakage-safe historical features and objective model governance.

### Possible later extensions

- schedule forecast refreshes automatically during the pre-winter period
- trigger the agent/email workflow only after material forecast changes
- add more winters before reconsidering higher-capacity models
- test additional climate predictors only when they can be justified and evaluated out of sample
- deploy the Streamlit dashboard publicly for demonstration

## Limitations

- seasonal snowfall provides a small effective supervised-learning sample
- a local station does not represent every part of the Greater Toronto Area
- teleconnection indices do not guarantee local predictive skill
- forecast uncertainty remains wide
- climate relationships can change over time
- this is a portfolio decision-support system, not an operational meteorological service

## Main takeaway

The strongest part of this project is not that a complex model wins.

It is that the system **tests whether complexity adds value, quantifies uncertainty, monitors changing inputs, retrains only when new labels arrive, and refuses to promote a challenger that has not proved itself out of sample**.
