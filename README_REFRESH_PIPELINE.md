# Forecast Refresh Pipeline

Copy these files into the root of `toronto-climate-analysis`.

The pipeline assumes these research outputs already exist:

- `data/processed/forecast_features_full.csv`
- `data/processed/final_model_dataset.csv`
- `models/interval_calibration_metadata.json`

## Run

```bash
python scripts/refresh_forecast.py
```

Production-only dry run without NOAA/ECCC calls:

```bash
python scripts/refresh_forecast.py --skip-live-refresh
```

Explicit season:

```bash
python scripts/refresh_forecast.py --season-year 2027
```

## Current logic

- Production = calibrated probabilistic climatology.
- Challenger = ElasticNet + ONI + NAO + Sep-Oct Toronto climate.
- New ENSO/NAO/autumn data refreshes the challenger.
- New climate predictors do **not** retrain the model.
- Retraining happens after a completed winter provides a new snowfall label.

## Output

- `forecasts/latest_forecast.json`
- `forecasts/forecast_history.jsonl`

The challenger is intentionally `not_ready` until Sep-Oct local weather coverage passes the configured threshold.
