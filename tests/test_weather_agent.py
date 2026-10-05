from src.agent.weather_agent import (
    decide_event,
)


def test_no_material_change_means_no_report():
    forecast = {
        "change_detection": {
            "material": False
        }
    }

    decision = decide_event(
        forecast,
        None,
        None,
    )

    assert decision.should_report is False
    assert decision.should_email is False


def test_material_forecast_change_triggers_report_and_email():
    forecast = {
        "change_detection": {
            "material": True,
            "production": {
                "reasons": [
                    "expected_snowfall_changed"
                ]
            },
            "challenger": {
                "reasons": []
            },
        }
    }

    decision = decide_event(
        forecast,
        None,
        None,
    )

    assert decision.should_report is True
    assert decision.should_email is True
    assert (
        "material_forecast_change"
        in decision.event_types
    )


def test_failed_promotion_does_not_trigger_alert():
    retraining = {
        "promote_challenger": False
    }

    registry = {
        "promotion_gate_passed": False
    }

    decision = decide_event(
        None,
        retraining,
        registry,
    )

    assert decision.should_report is False
    assert decision.should_email is False


def test_promotion_triggers_report_and_email():
    retraining = {
        "promote_challenger": True
    }

    registry = {
        "promotion_gate_passed": True
    }

    decision = decide_event(
        None,
        retraining,
        registry,
    )

    assert decision.should_report is True
    assert decision.should_email is True
    assert (
        "challenger_passed_promotion_gate"
        in decision.event_types
    )


def test_force_report_does_not_force_email():
    decision = decide_event(
        None,
        None,
        None,
        force_report=True,
    )

    assert decision.should_report is True
    assert decision.should_email is False
