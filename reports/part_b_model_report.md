# Toronto Snowfall Forecasting — Walk-Forward Model Report

- Model-ready seasons: 70
- Training/evaluation range: 1951–2026
- First out-of-sample forecast: 1971
- Number of out-of-sample winters: 50

## Model comparison

- Historical Climatology: MAE=31.56 cm, RMSE=37.87 cm, bias=+9.92 cm, skill vs climatology=+0.0%
- ElasticNet: MAE=31.64 cm, RMSE=37.97 cm, bias=+8.84 cm, skill vs climatology=-0.2%
- Rolling 10-Season Mean: MAE=31.68 cm, RMSE=37.72 cm, bias=+0.64 cm, skill vs climatology=-0.4%
- Linear Regression: MAE=34.24 cm, RMSE=42.10 cm, bias=+0.92 cm, skill vs climatology=-8.5%

## Selected candidate

**Historical Climatology**

- Walk-forward MAE: 31.56 cm
- Walk-forward RMSE: 37.87 cm

The selected method is based on historical walk-forward performance. It is not yet a probabilistic forecast and has not yet entered the retraining/champion-challenger workflow.