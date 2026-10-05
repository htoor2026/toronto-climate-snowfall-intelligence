from src.forecasting.change_detection import detect_challenger_change


def test_challenger_not_ready_is_not_material():
    current = {
        "status": "not_ready",
        "reason": "sep_oct_features_not_complete",
    }

    result = detect_challenger_change(
        None,
        current,
        expected_change_cm=10.0,
        probability_change=0.10,
        interval_bound_change_cm=15.0,
    )

    assert result.material is False


def test_challenger_becoming_ready_is_material():
    previous = {"status": "not_ready"}

    current = {
        "status": "ready",
        "expected_snowfall_cm": 108.0,
        "p_below": 0.50,
        "p_near": 0.30,
        "p_above": 0.20,
        "lower80_cm": 60.0,
        "upper80_cm": 155.0,
    }

    result = detect_challenger_change(
        previous,
        current,
        expected_change_cm=10.0,
        probability_change=0.10,
        interval_bound_change_cm=15.0,
    )

    assert result.material is True
    assert "challenger_became_ready" in result.reasons


def test_probability_shift_is_material():
    previous = {
        "status": "ready",
        "expected_snowfall_cm": 115.0,
        "p_below": 0.35,
        "p_near": 0.35,
        "p_above": 0.30,
        "lower80_cm": 60.0,
        "upper80_cm": 165.0,
    }

    current = {
        "status": "ready",
        "expected_snowfall_cm": 113.0,
        "p_below": 0.52,
        "p_near": 0.29,
        "p_above": 0.19,
        "lower80_cm": 60.0,
        "upper80_cm": 165.0,
    }

    result = detect_challenger_change(
        previous,
        current,
        expected_change_cm=10.0,
        probability_change=0.10,
        interval_bound_change_cm=15.0,
    )

    assert result.material is True
    assert "p_below_changed" in result.reasons
