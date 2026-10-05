import pandas as pd

from src.training.evaluate import production_climatology_prediction


def test_production_climatology_uses_latest_strict_seasons():
    strict = pd.DataFrame(
        {
            "season_year": [2000, 2001, 2002, 2003, 2004],
            "snowfall_cm": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    prediction = production_climatology_prediction(
        strict,
        season_year=2005,
        climatology_window=3,
    )

    assert prediction == 40.0


def test_production_climatology_ignores_target_and_future_seasons():
    strict = pd.DataFrame(
        {
            "season_year": [2001, 2002, 2003, 2004],
            "snowfall_cm": [10.0, 20.0, 30.0, 999.0],
        }
    )

    prediction = production_climatology_prediction(
        strict,
        season_year=2004,
        climatology_window=3,
    )

    assert prediction == 20.0
