from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .evaluate import (
    FINAL_FEATURES,
    make_elasticnet,
    recent_period_metrics,
    walk_forward_compare,
)


@dataclass
class RetrainingDecision:
    promote_challenger: bool
    reason: str
    challenger_mae_skill_pct: float
    challenger_rmse_skill_pct: float


def evaluate_retraining(
    data: pd.DataFrame,
    strict_history: pd.DataFrame,
    *,
    min_train_size: int,
    climatology_window: int,
    minimum_mae_skill_pct: float,
    minimum_rmse_skill_pct: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    RetrainingDecision,
]:
    predictions, metrics = walk_forward_compare(
        data,
        strict_history,
        min_train_size=min_train_size,
        climatology_window=climatology_window,
    )

    recent = recent_period_metrics(
        predictions,
        n_recent=15,
    )

    challenger = metrics[
        metrics["model"] == "ElasticNet + NAO"
    ].iloc[0]

    mae_skill = float(
        challenger["MAE_skill_vs_climatology_pct"]
    )
    rmse_skill = float(
        challenger["RMSE_skill_vs_climatology_pct"]
    )

    promote = (
        mae_skill >= minimum_mae_skill_pct
        and rmse_skill >= minimum_rmse_skill_pct
    )

    if promote:
        reason = (
            "Challenger passes the configured walk-forward "
            "MAE and RMSE skill gates."
        )
    else:
        reason = (
            "Challenger does not pass the configured walk-forward "
            "skill gates; retain climatology as champion."
        )

    decision = RetrainingDecision(
        promote_challenger=promote,
        reason=reason,
        challenger_mae_skill_pct=mae_skill,
        challenger_rmse_skill_pct=rmse_skill,
    )

    return predictions, metrics, recent, decision


def fit_full_challenger(
    data: pd.DataFrame,
):
    required = ["snowfall_cm"] + FINAL_FEATURES

    train = (
        data.dropna(subset=required)
        .sort_values("season_year")
        .reset_index(drop=True)
    )

    model = make_elasticnet(
        len(train)
    )

    model.fit(
        train[FINAL_FEATURES],
        train["snowfall_cm"],
    )

    return model
