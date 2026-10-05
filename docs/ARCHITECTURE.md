# Architecture

## System goal

This project is a seasonal snowfall decision-support system for Toronto. It separates
three concerns that are often mixed together in portfolio projects:

1. **Forecast refresh** when new predictors arrive.
2. **Retraining** only when a new completed winter provides a label.
3. **Model promotion** only when a challenger beats the production benchmark.

## High-level flow

```text
Historical ECCC climate + snowfall data
NOAA ONI / NAO
        |
        v
Feature construction
        |
        +-----------------------------+
        |                             |
        v                             v
Production climatology          ElasticNet challenger
        |                             |
        +-------------+---------------+
                      |
                      v
           Probabilities + interval
                      |
                      v
              Change detection
                      |
                      v
           Forecast history / agent
                      |
                      v
              Streamlit dashboard
```

## Forecast refresh

```text
new ONI / NAO / Sep-Oct weather
        |
        v
scripts/refresh_forecast.py
        |
        +--> production forecast
        +--> challenger forecast when all inputs are ready
        +--> material-change detection
        +--> forecasts/latest_forecast.json
        +--> forecasts/forecast_history.jsonl
```

A forecast refresh does **not** retrain the model.

## Retraining

```text
completed winter + observed snowfall
        |
        v
archived pre-winter feature snapshot
        |
        v
scripts/retrain_model.py
        |
        v
expanding walk-forward evaluation
        |
        v
promotion gate
        |
        v
candidate artifact + decision report
```

The retraining benchmark uses the same **30-season strict-quality climatology**
definition as the production point forecast.

## Model governance

The challenger is promoted only when:

```text
MAE skill vs production climatology >= 3%
AND
RMSE skill vs production climatology >= 0%
```

MLflow stores both versions and maintains:

- `@champion`
- `@challenger`

Registration is not the same as promotion.

## Agent boundary

The agent can summarize outputs and optionally send an email. It cannot change the
target, validation design, promotion threshold, or deploy a failed challenger.

## Dashboard

`app.py` is read-only. It presents:

- current forecast
- forecast history
- model/MLOps state
- business communication
- technical walkthrough
- operational status
