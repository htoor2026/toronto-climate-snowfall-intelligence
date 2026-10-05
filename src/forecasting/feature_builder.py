from __future__ import annotations

import pandas as pd


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


def _strict_history_map(strict_history: pd.DataFrame) -> pd.Series:
    hist = strict_history[["season_year", "snowfall_cm"]].dropna().drop_duplicates("season_year").sort_values("season_year")
    return hist.set_index("season_year")["snowfall_cm"]


def build_current_feature_row(*, season_year: int, live_predictors: dict, strict_history: pd.DataFrame) -> dict:
    strict_map = _strict_history_map(strict_history)
    lag1 = strict_map.get(season_year - 1, float("nan"))
    prior5 = strict_map.reindex(range(season_year - 5, season_year)).dropna()
    prior10 = strict_map.reindex(range(season_year - 10, season_year)).dropna()

    return {
        "season_year": int(season_year),
        "oni_jas": live_predictors.get("oni_jas"),
        "nao_jas": live_predictors.get("nao_jas"),
        "sep_oct_mean_temp_c": live_predictors.get("sep_oct_mean_temp_c"),
        "sep_oct_total_precip_mm": live_predictors.get("sep_oct_total_precip_mm"),
        "snow_lag1_cm": None if pd.isna(lag1) else float(lag1),
        "snow_roll5_mean_cm": float(prior5.mean()) if len(prior5) >= 3 else None,
        "snow_roll10_mean_cm": float(prior10.mean()) if len(prior10) >= 5 else None,
        "snow_roll10_std_cm": float(prior10.std()) if len(prior10) >= 5 else None,
        "year_index": float(season_year - 1939),
    }


def missing_challenger_features(feature_row: dict) -> list[str]:
    missing = []
    for feature in FINAL_FEATURES:
        value = feature_row.get(feature)
        if value is None or pd.isna(value):
            missing.append(feature)
    return missing
