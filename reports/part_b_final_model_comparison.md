# Toronto Seasonal Snowfall — Final Point-Forecast Model Comparison

- Model-ready winters: 70
- Evaluation range: 1971–2026
- Out-of-sample forecasts per available method: 50

## Results

- ElasticNet + NAO: MAE=31.30 cm, RMSE=38.05 cm, bias=+8.82 cm, MAE skill=+0.8%, RMSE skill=-0.5%
- Historical Climatology: MAE=31.56 cm, RMSE=37.87 cm, bias=+9.92 cm, MAE skill=+0.0%, RMSE skill=+0.0%
- Rolling 10-Season Mean: MAE=31.68 cm, RMSE=37.72 cm, bias=+0.64 cm, MAE skill=-0.4%, RMSE skill=+0.4%

## Decision

**Production candidate: Historical Climatology**

No complex model passes the practical skill gate; retain climatology as the benchmark production method

The next stage should convert the selected benchmark/candidate into a probabilistic forecast rather than presenting a single snowfall total as certain.