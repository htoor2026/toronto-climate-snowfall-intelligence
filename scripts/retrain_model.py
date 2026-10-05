#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.dataset import (
    append_completed_seasons_from_archive,
)
from src.training.retrain import (
    evaluate_retraining,
    fit_full_challenger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain and benchmark the snowfall challenger "
            "after new completed winters become available."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Run benchmark even when no new completed training row "
            "has been added."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path = (
        PROJECT_ROOT
        / "config"
        / "retraining_config.json"
    )

    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    base_training_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "final_model_dataset.csv"
    )

    strict_history_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "forecast_features_full.csv"
    )

    forecast_history_path = (
        PROJECT_ROOT
        / "forecasts"
        / "forecast_history.jsonl"
    )

    if not base_training_path.exists():
        raise FileNotFoundError(
            f"Missing {base_training_path}"
        )

    if not strict_history_path.exists():
        raise FileNotFoundError(
            f"Missing {strict_history_path}"
        )

    base_training = pd.read_csv(
        base_training_path
    )

    strict_history = pd.read_csv(
        strict_history_path
    )

    updated, appended, skipped = (
        append_completed_seasons_from_archive(
            base_training,
            strict_history,
            forecast_history_path,
        )
    )

    if not appended and not args.force:
        print(
            "No new completed model-ready season was found."
        )
        print(
            "Retraining skipped. "
            "Use --force to rerun the benchmark manually."
        )

        if skipped:
            print(
                "Completed seasons missing an archived READY "
                f"feature snapshot: {skipped}"
            )

        return 0

    predictions, metrics, recent, decision = (
        evaluate_retraining(
            updated,
            min_train_size=int(
                config["min_train_size"]
            ),
            minimum_mae_skill_pct=float(
                config[
                    "minimum_mae_skill_vs_climatology_pct"
                ]
            ),
            minimum_rmse_skill_pct=float(
                config[
                    "minimum_rmse_skill_vs_climatology_pct"
                ]
            ),
        )
    )

    challenger_model = fit_full_challenger(
        updated
    )

    output_dir = (
        PROJECT_ROOT
        / "models"
        / "retraining"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
    )

    reports_dir = (
        PROJECT_ROOT
        / "reports"
        / "retraining"
    )
    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    updated_path = (
        processed_dir
        / "production_training_dataset.csv"
    )

    predictions_path = (
        processed_dir
        / "retraining_walk_forward_predictions.csv"
    )

    metrics_path = (
        processed_dir
        / "retraining_model_comparison.csv"
    )

    recent_path = (
        processed_dir
        / "retraining_recent_comparison.csv"
    )

    candidate_model_path = (
        output_dir
        / "elasticnet_nao_candidate.joblib"
    )

    decision_path = (
        output_dir
        / "latest_retraining_decision.json"
    )

    updated.to_csv(
        updated_path,
        index=False,
    )
    predictions.to_csv(
        predictions_path,
        index=False,
    )
    metrics.to_csv(
        metrics_path,
        index=False,
    )
    recent.to_csv(
        recent_path,
        index=False,
    )

    joblib.dump(
        challenger_model,
        candidate_model_path,
    )

    result = {
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "training_rows": int(
            len(updated)
        ),
        "latest_training_season": int(
            updated["season_year"].max()
        ),
        "appended_seasons": appended,
        "skipped_missing_ready_snapshot": skipped,
        "champion": "Historical Climatology",
        "challenger": "ElasticNet + NAO",
        "promotion_gate": {
            "minimum_mae_skill_vs_climatology_pct": float(
                config[
                    "minimum_mae_skill_vs_climatology_pct"
                ]
            ),
            "minimum_rmse_skill_vs_climatology_pct": float(
                config[
                    "minimum_rmse_skill_vs_climatology_pct"
                ]
            ),
        },
        "challenger_mae_skill_pct": (
            decision.challenger_mae_skill_pct
        ),
        "challenger_rmse_skill_pct": (
            decision.challenger_rmse_skill_pct
        ),
        "promote_challenger": (
            decision.promote_challenger
        ),
        "decision_reason": (
            decision.reason
        ),
    }

    decision_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "# Snowfall Model Retraining Report",
        "",
        f"- Training rows: {result['training_rows']}",
        f"- Latest training season: {result['latest_training_season']}",
        f"- Newly appended seasons: {appended or 'none'}",
        "",
        "## Champion vs challenger",
        "",
    ]

    for _, row in metrics.iterrows():
        report_lines.append(
            f"- {row['model']}: "
            f"MAE={row['MAE_cm']:.2f} cm, "
            f"RMSE={row['RMSE_cm']:.2f} cm, "
            f"bias={row['Bias_cm']:+.2f} cm, "
            f"MAE skill={row['MAE_skill_vs_climatology_pct']:+.1f}%, "
            f"RMSE skill={row['RMSE_skill_vs_climatology_pct']:+.1f}%"
        )

    report_lines.extend([
        "",
        "## Promotion decision",
        "",
        (
            "**PROMOTE CHALLENGER**"
            if decision.promote_challenger
            else "**RETAIN CLIMATOLOGY CHAMPION**"
        ),
        "",
        decision.reason,
    ])

    report_path = (
        reports_dir
        / "latest_retraining_report.md"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nMODEL COMPARISON")
    print("----------------")
    print(
        metrics.to_string(
            index=False
        )
    )

    print("\nRECENT 15-WINTER COMPARISON")
    print("---------------------------")
    print(
        recent.to_string(
            index=False
        )
    )

    print("\nDECISION")
    print("--------")
    print(
        "Promote challenger:",
        decision.promote_challenger,
    )
    print(decision.reason)

    print("\nSaved:")
    print(updated_path)
    print(predictions_path)
    print(metrics_path)
    print(candidate_model_path)
    print(decision_path)
    print(report_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
