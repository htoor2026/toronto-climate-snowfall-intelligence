#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.calibration import load_calibration_metadata
from src.forecasting.change_detection import (
    detect_challenger_change,
    detect_material_change,
)
from src.forecasting.config import ForecastConfig
from src.forecasting.data_refresh import refresh_live_predictors
from src.forecasting.feature_builder import build_current_feature_row
from src.forecasting.forecast import (
    build_challenger_forecast,
    build_production_climatology_forecast,
)
from src.forecasting.storage import load_latest, save_forecast_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the Toronto seasonal snowfall forecast."
    )
    parser.add_argument(
        "--season-year",
        type=int,
        default=None,
        help=(
            "Snow season ending year. "
            "Default: latest trustworthy season + 1."
        ),
    )
    parser.add_argument(
        "--skip-live-refresh",
        action="store_true",
        help=(
            "Do not call NOAA/ECCC. Production climatology will still run, "
            "but challenger will be unavailable."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path = PROJECT_ROOT / "config" / "forecast_config.json"
    config = ForecastConfig.from_json(config_path)

    strict_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "forecast_features_full.csv"
    )
    model_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "final_model_dataset.csv"
    )
    calibration_path = (
        PROJECT_ROOT
        / "models"
        / "interval_calibration_metadata.json"
    )

    if not strict_path.exists():
        raise FileNotFoundError(
            f"Missing {strict_path}. Run the research notebooks first."
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Run notebook 10c first."
        )

    strict = pd.read_csv(strict_path)
    strict = (
        strict[["season_year", "snowfall_cm"]]
        .dropna()
        .drop_duplicates("season_year")
        .sort_values("season_year")
        .reset_index(drop=True)
    )

    model_data = pd.read_csv(model_path)
    calibration_metadata = load_calibration_metadata(
        calibration_path
    )

    season_year = (
        args.season_year
        if args.season_year is not None
        else int(strict["season_year"].max() + 1)
    )
    predictor_year = season_year - 1

    # ---------------------------------------------------------
    # Production forecast
    # ---------------------------------------------------------
    production = build_production_climatology_forecast(
        strict_history=strict,
        season_year=season_year,
        climatology_window=config.climatology_window,
        calibration_metadata=calibration_metadata,
    )

    # ---------------------------------------------------------
    # Challenger refresh
    # ---------------------------------------------------------
    live_predictors = None
    feature_row = None

    challenger = {
        "status": "not_run",
        "role": "challenger",
        "method": "ElasticNet + NAO Residual",
        "season_year": season_year,
    }

    if (
        not args.skip_live_refresh
        and config.challenger_enabled
    ):
        try:
            live_predictors = refresh_live_predictors(
                predictor_year=predictor_year,
                station_id=config.target_station_id,
                autumn_min_coverage_pct=(
                    config.autumn_min_coverage_pct
                ),
            )

            feature_row = build_current_feature_row(
                season_year=season_year,
                live_predictors=live_predictors,
                strict_history=strict,
            )

            if not live_predictors["autumn_features_usable"]:
                challenger = {
                    "status": "not_ready",
                    "role": "challenger",
                    "method": "ElasticNet + NAO Residual",
                    "season_year": season_year,
                    "reason": "sep_oct_features_not_complete",
                    "temp_coverage_pct": live_predictors[
                        "sep_oct_temp_coverage_pct"
                    ],
                    "precip_coverage_pct": live_predictors[
                        "sep_oct_precip_coverage_pct"
                    ],
                }
            else:
                challenger = build_challenger_forecast(
                    model_data=model_data,
                    feature_row=feature_row,
                    production_forecast=production,
                )

        except Exception as exc:
            challenger = {
                "status": "error",
                "role": "challenger",
                "method": "ElasticNet + NAO Residual",
                "season_year": season_year,
                "error": str(exc),
            }

    # ---------------------------------------------------------
    # Previous forecast snapshot
    # ---------------------------------------------------------
    latest_path = PROJECT_ROOT / config.latest_forecast_path
    history_path = PROJECT_ROOT / config.forecast_history_path

    previous_snapshot = load_latest(latest_path)

    previous_production = None
    previous_challenger = None

    if previous_snapshot is not None:
        previous_production = previous_snapshot.get(
            "production_forecast"
        )
        previous_challenger = previous_snapshot.get(
            "challenger_forecast"
        )

    # ---------------------------------------------------------
    # Production change detection
    # ---------------------------------------------------------
    production_change = detect_material_change(
        previous_production,
        production,
        expected_change_cm=config.material_expected_change_cm,
        probability_change=config.material_probability_change,
        interval_bound_change_cm=(
            config.material_interval_bound_change_cm
        ),
    )

    # ---------------------------------------------------------
    # Challenger change detection
    # ---------------------------------------------------------
    challenger_change = detect_challenger_change(
        previous_challenger,
        challenger,
        expected_change_cm=config.material_expected_change_cm,
        probability_change=config.material_probability_change,
        interval_bound_change_cm=(
            config.material_interval_bound_change_cm
        ),
    )

    overall_material_change = (
        production_change.material
        or challenger_change.material
    )

    payload = {
        "season_year": season_year,
        "predictor_year": predictor_year,
        "production_forecast": production,
        "challenger_forecast": challenger,
        "live_predictors": live_predictors,
        "current_feature_row": feature_row,
        "change_detection": {
            "material": overall_material_change,
            "production": {
                "material": production_change.material,
                "reasons": production_change.reasons,
                "metrics": production_change.metrics,
            },
            "challenger": {
                "material": challenger_change.material,
                "reasons": challenger_change.reasons,
                "metrics": challenger_change.metrics,
            },
        },
    }

    save_forecast_snapshot(
        latest_path=latest_path,
        history_path=history_path,
        payload=payload,
    )

    print(json.dumps(payload, indent=2))

    print("\nSUMMARY")
    print("-------")
    print(
        f"Season: {season_year - 1}-"
        f"{str(season_year)[-2:]}"
    )
    print(
        "Production expected snowfall: "
        f"{production['expected_snowfall_cm']:.1f} cm"
    )
    print(
        "Calibrated interval: "
        f"{production['calibrated_lower80_cm']:.1f}-"
        f"{production['calibrated_upper80_cm']:.1f} cm"
    )
    print(
        "Production probabilities: "
        f"below={production['p_below']:.1%}, "
        f"near={production['p_near']:.1%}, "
        f"above={production['p_above']:.1%}"
    )
    print(
        f"Challenger status: {challenger['status']}"
    )

    if challenger.get("status") == "ready":
        print(
            "Challenger expected snowfall: "
            f"{challenger['expected_snowfall_cm']:.1f} cm"
        )
        print(
            "Challenger probabilities: "
            f"below={challenger['p_below']:.1%}, "
            f"near={challenger['p_near']:.1%}, "
            f"above={challenger['p_above']:.1%}"
        )

    print(
        "Material production change: "
        f"{production_change.material}"
    )
    print(
        "Material challenger change: "
        f"{challenger_change.material}"
    )
    print(
        "Overall material forecast change: "
        f"{overall_material_change}"
    )

    if production_change.reasons:
        print(
            "Production reasons:",
            ", ".join(production_change.reasons),
        )

    if challenger_change.reasons:
        print(
            "Challenger reasons:",
            ", ".join(challenger_change.reasons),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
