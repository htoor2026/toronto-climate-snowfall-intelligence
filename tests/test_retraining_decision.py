import pandas as pd

from src.training.retrain import (
    RetrainingDecision,
)


def test_retraining_decision_dataclass():
    decision = RetrainingDecision(
        promote_challenger=False,
        reason="retain champion",
        challenger_mae_skill_pct=0.8,
        challenger_rmse_skill_pct=-0.5,
    )

    assert decision.promote_challenger is False
    assert decision.challenger_mae_skill_pct == 0.8
    assert decision.challenger_rmse_skill_pct == -0.5
