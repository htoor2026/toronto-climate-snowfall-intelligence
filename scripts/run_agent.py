#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.agent.weather_agent import (
    build_report,
    decide_event,
    load_json,
    save_report,
    send_email,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a minimal operational report from the "
            "latest forecast, retraining, and MLflow outputs."
        )
    )

    parser.add_argument(
        "--force-report",
        action="store_true",
        help="Generate a report even if no material event occurred.",
    )

    parser.add_argument(
        "--send-email",
        action="store_true",
        help=(
            "Send email when the agent marks the event as email-worthy."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = json.loads(
        (
            PROJECT_ROOT
            / "config"
            / "agent_config.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    forecast = load_json(
        PROJECT_ROOT
        / "forecasts"
        / "latest_forecast.json"
    )

    retraining = load_json(
        PROJECT_ROOT
        / "models"
        / "retraining"
        / "latest_retraining_decision.json"
    )

    registry = load_json(
        PROJECT_ROOT
        / "models"
        / "retraining"
        / "latest_mlflow_registry.json"
    )

    decision = decide_event(
        forecast,
        retraining,
        registry,
        force_report=args.force_report,
    )

    print("AGENT")
    print("-----")
    print(
        "Should report:",
        decision.should_report,
    )
    print(
        "Should email:",
        decision.should_email,
    )
    print(
        "Events:",
        decision.event_types or "none",
    )

    if not decision.should_report:
        print(
            "No material event. "
            "Nothing else to do."
        )
        return 0

    report = build_report(
        forecast,
        retraining,
        registry,
        decision,
    )

    md_path, json_path = save_report(
        report,
        PROJECT_ROOT
        / "reports"
        / "agent",
        decision,
    )

    print(
        "Saved report:",
        md_path,
    )
    print(
        "Saved event:",
        json_path,
    )

    email_enabled = bool(
        config.get(
            "email_enabled",
            False,
        )
    )

    if (
        args.send_email
        and email_enabled
        and decision.should_email
    ):
        to_email = config.get(
            "email_to"
        )

        if not to_email:
            raise ValueError(
                "email_to is missing from config/agent_config.json"
            )

        send_email(
            subject=(
                "Toronto snowfall system alert"
            ),
            body=report,
            to_email=to_email,
        )

        print(
            "Email sent to:",
            to_email,
        )

    elif args.send_email and not email_enabled:
        print(
            "Email not sent: email_enabled=false."
        )

    elif args.send_email and not decision.should_email:
        print(
            "Email not sent: event is not email-worthy."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
