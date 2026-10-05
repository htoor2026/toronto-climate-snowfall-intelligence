# Toronto Seasonal Snowfall — Probabilistic Forecast Report

- Walk-forward winters evaluated: 50
- Climatology window: latest 30 prior trustworthy winters
- Prediction interval: 80%

## Probabilistic backtest

- ElasticNet + NAO Residual: Brier=0.716, category accuracy=40.0%, 80% interval coverage=64.0%, mean interval width=75.8 cm, Brier skill vs climatology=-7.6%
- Probabilistic Climatology: Brier=0.666, category accuracy=44.0%, 80% interval coverage=68.0%, mean interval width=81.2 cm, Brier skill vs climatology=+0.0%

## Production decision

**Probabilistic Climatology**

Probabilistic challenger does not pass the practical skill gate; retain climatology as production benchmark.

## Next-season climatological baseline

- Expected snowfall: 117.1 cm
- 80% interval: 65.3–154.6 cm
- Below-normal probability: 33.3%
- Near-normal probability: 33.3%
- Above-normal probability: 33.3%