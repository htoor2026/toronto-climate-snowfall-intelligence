from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChangeResult:
    material: bool
    reasons: list[str]
    metrics: dict


def detect_material_change(
    previous: dict | None,
    current: dict,
    *,
    expected_change_cm: float,
    probability_change: float,
    interval_bound_change_cm: float,
    lower_key: str = "calibrated_lower80_cm",
    upper_key: str = "calibrated_upper80_cm",
) -> ChangeResult:
    if previous is None:
        return ChangeResult(True, ["first_forecast"], {})

    metrics = {}
    reasons = []

    expected_delta = (
        float(current["expected_snowfall_cm"])
        - float(previous["expected_snowfall_cm"])
    )
    metrics["expected_change_cm"] = expected_delta

    if abs(expected_delta) >= expected_change_cm:
        reasons.append("expected_snowfall_changed")

    for key in ["p_below", "p_near", "p_above"]:
        delta = float(current[key]) - float(previous[key])
        metrics[f"{key}_change"] = delta
        if abs(delta) >= probability_change:
            reasons.append(f"{key}_changed")

    if (
        lower_key in previous and lower_key in current
        and upper_key in previous and upper_key in current
    ):
        lower_delta = float(current[lower_key]) - float(previous[lower_key])
        upper_delta = float(current[upper_key]) - float(previous[upper_key])

        metrics[f"{lower_key}_change_cm"] = lower_delta
        metrics[f"{upper_key}_change_cm"] = upper_delta

        if abs(lower_delta) >= interval_bound_change_cm:
            reasons.append(f"{lower_key}_changed")
        if abs(upper_delta) >= interval_bound_change_cm:
            reasons.append(f"{upper_key}_changed")

    return ChangeResult(bool(reasons), reasons, metrics)


def detect_challenger_change(
    previous: dict | None,
    current: dict,
    *,
    expected_change_cm: float,
    probability_change: float,
    interval_bound_change_cm: float,
) -> ChangeResult:
    current_status = current.get("status")
    previous_status = previous.get("status") if previous else None

    # No usable challenger forecast yet.
    if current_status != "ready":
        reasons = []
        if previous_status == "ready":
            reasons.append("challenger_became_unavailable")

        return ChangeResult(
            material=False,
            reasons=reasons,
            metrics={
                "previous_status": previous_status,
                "current_status": current_status,
            },
        )

    # First time the challenger becomes usable.
    if previous is None or previous_status != "ready":
        return ChangeResult(
            material=True,
            reasons=["challenger_became_ready"],
            metrics={
                "previous_status": previous_status,
                "current_status": current_status,
            },
        )

    # Both are ready: compare challenger forecast values.
    return detect_material_change(
        previous,
        current,
        expected_change_cm=expected_change_cm,
        probability_change=probability_change,
        interval_bound_change_cm=interval_bound_change_cm,
        lower_key="lower80_cm",
        upper_key="upper80_cm",
    )
