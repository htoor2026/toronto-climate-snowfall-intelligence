from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FINAL_FEATURES = [
    "oni_jas",
    "nao_jas",
    "sep_oct_mean_temp_c",
    "sep_oct_total_precip_mm",
    "snow_lag1_cm",
    "snow_roll5_mean_cm",
    "snow_roll10_mean_cm",
    "snow_roll10_std_cm",
    "year_index",
]


def make_elasticnet(n_train: int) -> Pipeline:
    n_splits = min(5, max(2, n_train // 6))

    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                alphas=np.logspace(-3, 2, 60),
                cv=TimeSeriesSplit(n_splits=n_splits),
                max_iter=20000,
            ),
        ),
    ])


def _rmse(y_true, y_pred) -> float:
    return float(
        np.sqrt(
            mean_squared_error(y_true, y_pred)
        )
    )


def walk_forward_compare(
    data: pd.DataFrame,
    *,
    min_train_size: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare production benchmark climatology vs ElasticNet challenger
    on the exact same expanding-window test winters.
    """
    required = ["season_year", "snowfall_cm"] + FINAL_FEATURES

    df = (
        data.dropna(subset=required)
        .sort_values("season_year")
        .reset_index(drop=True)
    )

    if len(df) <= min_train_size:
        raise ValueError(
            f"Need more than {min_train_size} complete rows; found {len(df)}."
        )

    rows = []

    for test_idx in range(min_train_size, len(df)):
        train = df.iloc[:test_idx]
        test = df.iloc[[test_idx]]

        actual = float(test["snowfall_cm"].iloc[0])
        season_year = int(test["season_year"].iloc[0])

        # Production benchmark: expanding historical mean.
        climatology_pred = float(
            train["snowfall_cm"].mean()
        )

        rows.append({
            "season_year": season_year,
            "model": "Historical Climatology",
            "actual_cm": actual,
            "predicted_cm": climatology_pred,
        })

        # Challenger: ElasticNet + climate features.
        model = make_elasticnet(len(train))
        model.fit(
            train[FINAL_FEATURES],
            train["snowfall_cm"],
        )

        challenger_pred = float(
            model.predict(
                test[FINAL_FEATURES]
            )[0]
        )

        rows.append({
            "season_year": season_year,
            "model": "ElasticNet + NAO",
            "actual_cm": actual,
            "predicted_cm": challenger_pred,
        })

    predictions = pd.DataFrame(rows)
    predictions["error_cm"] = (
        predictions["predicted_cm"]
        - predictions["actual_cm"]
    )
    predictions["abs_error_cm"] = (
        predictions["error_cm"].abs()
    )

    metrics = (
        predictions.groupby("model")
        .apply(
            lambda g: pd.Series({
                "n_forecasts": len(g),
                "MAE_cm": mean_absolute_error(
                    g["actual_cm"],
                    g["predicted_cm"],
                ),
                "RMSE_cm": _rmse(
                    g["actual_cm"],
                    g["predicted_cm"],
                ),
                "Bias_cm": float(
                    g["error_cm"].mean()
                ),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    clim = metrics[
        metrics["model"] == "Historical Climatology"
    ].iloc[0]

    metrics["MAE_skill_vs_climatology_pct"] = (
        1 - metrics["MAE_cm"] / float(clim["MAE_cm"])
    ) * 100

    metrics["RMSE_skill_vs_climatology_pct"] = (
        1 - metrics["RMSE_cm"] / float(clim["RMSE_cm"])
    ) * 100

    return predictions, metrics


def recent_period_metrics(
    predictions: pd.DataFrame,
    *,
    n_recent: int = 15,
) -> pd.DataFrame:
    years = sorted(
        predictions["season_year"].unique()
    )
    recent_years = years[-min(n_recent, len(years)):]

    recent = predictions[
        predictions["season_year"].isin(recent_years)
    ]

    return (
        recent.groupby("model")
        .apply(
            lambda g: pd.Series({
                "n_forecasts": len(g),
                "MAE_cm": mean_absolute_error(
                    g["actual_cm"],
                    g["predicted_cm"],
                ),
                "RMSE_cm": _rmse(
                    g["actual_cm"],
                    g["predicted_cm"],
                ),
                "Bias_cm": float(
                    (g["predicted_cm"] - g["actual_cm"]).mean()
                ),
            }),
            include_groups=False,
        )
        .reset_index()
    )
