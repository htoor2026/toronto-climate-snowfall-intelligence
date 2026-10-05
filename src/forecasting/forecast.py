from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import apply_calibrated_interval, empirical_tercile_thresholds
from .feature_builder import FINAL_FEATURES, missing_challenger_features


def _make_elasticnet(n_train: int) -> Pipeline:
    n_splits = min(5, max(2, n_train // 6))
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            alphas=np.logspace(-3, 2, 60),
            cv=TimeSeriesSplit(n_splits=n_splits),
            max_iter=20000,
        )),
    ])


def build_production_climatology_forecast(*, strict_history: pd.DataFrame, season_year: int, climatology_window: int, calibration_metadata: dict) -> dict:
    history = strict_history[["season_year", "snowfall_cm"]].dropna().drop_duplicates("season_year").sort_values("season_year")
    prior = history[history["season_year"] < season_year]["snowfall_cm"].tail(climatology_window).to_numpy(dtype=float)
    if len(prior) < 20:
        raise ValueError(f"Only {len(prior)} prior trustworthy winters available; need at least 20.")

    expected = float(np.mean(prior))
    median = float(np.median(prior))
    lower_empirical = float(np.quantile(prior, 0.10))
    upper_empirical = float(np.quantile(prior, 0.90))
    q33, q67 = empirical_tercile_thresholds(prior)
    lower_cal, upper_cal = apply_calibrated_interval(
        expected_cm=expected,
        lower_empirical_cm=lower_empirical,
        upper_empirical_cm=upper_empirical,
        metadata=calibration_metadata,
    )

    return {
        "status": "ready",
        "role": "production",
        "method": "Probabilistic Climatology",
        "season_year": int(season_year),
        "history_n": int(len(prior)),
        "expected_snowfall_cm": expected,
        "median_snowfall_cm": median,
        "empirical_lower80_cm": lower_empirical,
        "empirical_upper80_cm": upper_empirical,
        "calibrated_lower80_cm": lower_cal,
        "calibrated_upper80_cm": upper_cal,
        "below_normal_threshold_cm": q33,
        "above_normal_threshold_cm": q67,
        "p_below": float(np.mean(prior <= q33)),
        "p_near": float(np.mean((prior > q33) & (prior <= q67))),
        "p_above": float(np.mean(prior > q67)),
        "calibration_method": calibration_metadata["selected_interval_method"],
        "conformal_qhat_cm": float(calibration_metadata["final_qhat_cm"]),
    }


def _distribution_summary(samples: np.ndarray, *, q33: float, q67: float) -> dict:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    return {
        "median_snowfall_cm": float(np.median(samples)),
        "lower80_cm": float(np.quantile(samples, 0.10)),
        "upper80_cm": float(np.quantile(samples, 0.90)),
        "p_below": float(np.mean(samples <= q33)),
        "p_near": float(np.mean((samples > q33) & (samples <= q67))),
        "p_above": float(np.mean(samples > q67)),
    }


def build_challenger_forecast(*, model_data: pd.DataFrame, feature_row: dict, production_forecast: dict) -> dict:
    missing = missing_challenger_features(feature_row)
    if missing:
        return {
            "status": "not_ready",
            "role": "challenger",
            "method": "ElasticNet + NAO Residual",
            "season_year": int(feature_row["season_year"]),
            "missing_features": missing,
        }

    data = model_data.dropna(subset=["snowfall_cm"] + FINAL_FEATURES).sort_values("season_year").reset_index(drop=True)
    if len(data) < 25:
        return {
            "status": "not_ready",
            "role": "challenger",
            "method": "ElasticNet + NAO Residual",
            "season_year": int(feature_row["season_year"]),
            "reason": "insufficient_training_rows",
            "training_rows": int(len(data)),
        }

    calibration_size = min(15, max(10, int(round(0.25 * len(data)))))
    proper = data.iloc[:-calibration_size]
    calibration = data.iloc[-calibration_size:]

    residual_model = _make_elasticnet(len(proper))
    residual_model.fit(proper[FINAL_FEATURES], proper["snowfall_cm"])
    calibration_pred = residual_model.predict(calibration[FINAL_FEATURES])
    residuals = calibration["snowfall_cm"].to_numpy(dtype=float) - calibration_pred

    final_model = _make_elasticnet(len(data))
    final_model.fit(data[FINAL_FEATURES], data["snowfall_cm"])
    x_current = pd.DataFrame([feature_row])[FINAL_FEATURES]
    point = float(final_model.predict(x_current)[0])
    samples = point + residuals

    summary = _distribution_summary(
        samples,
        q33=float(production_forecast["below_normal_threshold_cm"]),
        q67=float(production_forecast["above_normal_threshold_cm"]),
    )

    return {
        "status": "ready",
        "role": "challenger",
        "method": "ElasticNet + NAO Residual",
        "season_year": int(feature_row["season_year"]),
        "training_rows": int(len(data)),
        "calibration_rows": int(calibration_size),
        "expected_snowfall_cm": point,
        **summary,
    }
