# Toronto Snowfall Forecast — Climate Feature Screening

- Common modeling winters: 70
- Common range: 1951–2026
- First walk-forward test winter: 1971
- Out-of-sample winters: 50

## Full-period results

- Base + NAO: MAE=31.30 cm, RMSE=38.05 cm, bias=+8.82 cm, skill vs climatology=+0.8%
- Historical Climatology: MAE=31.56 cm, RMSE=37.87 cm, bias=+9.92 cm, skill vs climatology=+0.0%
- Base + PDO: MAE=31.60 cm, RMSE=37.93 cm, bias=+8.92 cm, skill vs climatology=-0.1%
- Base: MAE=31.64 cm, RMSE=37.97 cm, bias=+8.84 cm, skill vs climatology=-0.2%
- Base + PNA: MAE=31.67 cm, RMSE=37.95 cm, bias=+8.99 cm, skill vs climatology=-0.3%
- Base + AO + NAO + PNA + PDO: MAE=31.91 cm, RMSE=38.44 cm, bias=+9.20 cm, skill vs climatology=-1.1%
- Base + AO: MAE=31.92 cm, RMSE=38.17 cm, bias=+9.20 cm, skill vs climatology=-1.1%

## Best full-period method

**Base + NAO**

These results are exploratory feature screening. Because several candidate climate indices were tested on a small dataset, the best result should be treated as a candidate for confirmation rather than proof of stable predictive skill.