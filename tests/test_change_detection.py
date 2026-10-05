from src.forecasting.change_detection import detect_material_change


def test_first_forecast_is_material():
    current = {
        "expected_snowfall_cm": 117.0,
        "p_below": 0.33,
        "p_near": 0.33,
        "p_above": 0.34,
        "calibrated_lower80_cm": 55.0,
        "calibrated_upper80_cm": 170.0,
    }
    result = detect_material_change(None, current, expected_change_cm=10.0, probability_change=0.10, interval_bound_change_cm=15.0)
    assert result.material is True
    assert "first_forecast" in result.reasons


def test_small_change_is_not_material():
    previous = {
        "expected_snowfall_cm": 117.0,
        "p_below": 0.33,
        "p_near": 0.33,
        "p_above": 0.34,
        "calibrated_lower80_cm": 55.0,
        "calibrated_upper80_cm": 170.0,
    }
    current = {
        "expected_snowfall_cm": 121.0,
        "p_below": 0.37,
        "p_near": 0.31,
        "p_above": 0.32,
        "calibrated_lower80_cm": 58.0,
        "calibrated_upper80_cm": 174.0,
    }
    result = detect_material_change(previous, current, expected_change_cm=10.0, probability_change=0.10, interval_bound_change_cm=15.0)
    assert result.material is False


def test_probability_change_is_material():
    previous = {
        "expected_snowfall_cm": 117.0,
        "p_below": 0.33,
        "p_near": 0.33,
        "p_above": 0.34,
        "calibrated_lower80_cm": 55.0,
        "calibrated_upper80_cm": 170.0,
    }
    current = {
        "expected_snowfall_cm": 120.0,
        "p_below": 0.48,
        "p_near": 0.30,
        "p_above": 0.22,
        "calibrated_lower80_cm": 55.0,
        "calibrated_upper80_cm": 170.0,
    }
    result = detect_material_change(previous, current, expected_change_cm=10.0, probability_change=0.10, interval_bound_change_cm=15.0)
    assert result.material is True
    assert "p_below_changed" in result.reasons
