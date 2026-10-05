# Retraining Pipeline

This pipeline is intentionally separate from forecast refresh.

## Forecast refresh

New ENSO / NAO / autumn weather:

```text
new predictors
→ refresh current-season forecast
```

This does **not** retrain the model.

## Retraining

A completed winter adds a new labeled example:

```text
pre-winter archived feature row
+ completed trustworthy snowfall total
→ append training row
→ walk-forward benchmark
→ champion vs challenger
→ promotion gate
```

The script only appends a season when:

1. the actual snowfall exists in the strict-quality history,
2. the season is not already in the training dataset,
3. the forecast archive contains a `challenger_forecast.status == "ready"` snapshot,
4. all model features are present in that archived pre-winter snapshot.

This protects against rebuilding historical predictors using information that was not actually available at forecast time.

## Run today

No new completed season exists after 2026, so normal mode should skip:

```bash
python scripts/retrain_model.py
```

To test the full benchmark pipeline on the existing training data:

```bash
python scripts/retrain_model.py --force
```

## Promotion gate

Current configuration:

- challenger MAE skill vs climatology >= 3%
- challenger RMSE skill vs climatology >= 0%

Otherwise climatology remains champion.

The pipeline saves a fitted challenger candidate even when it is not promoted.
Actual model registry promotion is handled in the later MLflow stage.
