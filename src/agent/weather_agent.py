from __future__ import annotations

import json
import os
import smtplib
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any


@dataclass
class AgentDecision:
    should_report: bool
    should_email: bool
    event_types: list[str]
    reasons: list[str]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def decide_event(
    forecast_payload: dict[str, Any] | None,
    retraining_payload: dict[str, Any] | None,
    registry_payload: dict[str, Any] | None,
    *,
    force_report: bool = False,
) -> AgentDecision:
    """
    Deterministic agent policy.

    The agent may summarize and notify, but it does not:
    - change the model
    - alter the promotion gate
    - modify validation rules
    - retrain automatically
    """
    events: list[str] = []
    reasons: list[str] = []

    if force_report:
        events.append("manual_report")
        reasons.append("manual report requested")

    if forecast_payload:
        change = forecast_payload.get(
            "change_detection", {}
        )

        if bool(change.get("material")):
            events.append("material_forecast_change")

            production = change.get(
                "production", {}
            )
            challenger = change.get(
                "challenger", {}
            )

            for reason in production.get(
                "reasons", []
            ):
                reasons.append(
                    f"production: {reason}"
                )

            for reason in challenger.get(
                "reasons", []
            ):
                reasons.append(
                    f"challenger: {reason}"
                )

    if retraining_payload:
        if bool(
            retraining_payload.get(
                "promote_challenger"
            )
        ):
            events.append(
                "challenger_passed_promotion_gate"
            )
            reasons.append(
                "challenger passed configured promotion gate"
            )

    # Registry should agree with retraining decision.
    # A mismatch is worth reporting because it is an operational issue.
    if (
        retraining_payload
        and registry_payload
    ):
        expected_promoted = bool(
            retraining_payload.get(
                "promote_challenger"
            )
        )

        registered_promoted = bool(
            registry_payload.get(
                "promotion_gate_passed"
            )
        )

        if expected_promoted != registered_promoted:
            events.append(
                "registry_decision_mismatch"
            )
            reasons.append(
                "MLflow registry state disagrees with retraining decision"
            )

    should_report = bool(events)

    # Email only for operationally meaningful events.
    should_email = any(
        event in {
            "material_forecast_change",
            "challenger_passed_promotion_gate",
            "registry_decision_mismatch",
        }
        for event in events
    )

    return AgentDecision(
        should_report=should_report,
        should_email=should_email,
        event_types=events,
        reasons=reasons,
    )


def build_report(
    forecast_payload: dict[str, Any] | None,
    retraining_payload: dict[str, Any] | None,
    registry_payload: dict[str, Any] | None,
    decision: AgentDecision,
) -> str:
    now = datetime.now(
        timezone.utc
    ).isoformat()

    lines = [
        "# Toronto Snowfall System Report",
        "",
        f"Generated: {now}",
        "",
        "## Event",
        "",
        (
            ", ".join(decision.event_types)
            if decision.event_types
            else "No material event"
        ),
        "",
    ]

    if decision.reasons:
        lines.extend([
            "## Reasons",
            "",
        ])

        for reason in decision.reasons:
            lines.append(
                f"- {reason}"
            )

        lines.append("")

    if forecast_payload:
        production = forecast_payload.get(
            "production_forecast", {}
        )

        challenger = forecast_payload.get(
            "challenger_forecast", {}
        )

        lines.extend([
            "## Current Forecast",
            "",
            (
                f"- Season: "
                f"{forecast_payload.get('season_year')}"
            ),
            (
                f"- Production method: "
                f"{production.get('method', 'unknown')}"
            ),
            (
                f"- Expected snowfall: "
                f"{_fmt(production.get('expected_snowfall_cm'))} cm"
            ),
            (
                f"- 80% interval: "
                f"{_fmt(production.get('calibrated_lower80_cm'))}"
                f"–{_fmt(production.get('calibrated_upper80_cm'))} cm"
            ),
            (
                f"- Below / Near / Above: "
                f"{_pct(production.get('p_below'))} / "
                f"{_pct(production.get('p_near'))} / "
                f"{_pct(production.get('p_above'))}"
            ),
            (
                f"- Challenger status: "
                f"{challenger.get('status', 'unknown')}"
            ),
            "",
        ])

        if challenger.get("status") == "ready":
            lines.extend([
                (
                    f"- Challenger expected snowfall: "
                    f"{_fmt(challenger.get('expected_snowfall_cm'))} cm"
                ),
                "",
            ])

    if retraining_payload:
        lines.extend([
            "## Latest Retraining Decision",
            "",
            (
                f"- Champion: "
                f"{retraining_payload.get('champion', 'unknown')}"
            ),
            (
                f"- Challenger: "
                f"{retraining_payload.get('challenger', 'unknown')}"
            ),
            (
                f"- Challenger MAE skill: "
                f"{_fmt(retraining_payload.get('challenger_mae_skill_pct'))}%"
            ),
            (
                f"- Challenger RMSE skill: "
                f"{_fmt(retraining_payload.get('challenger_rmse_skill_pct'))}%"
            ),
            (
                f"- Promote challenger: "
                f"{bool(retraining_payload.get('promote_challenger'))}"
            ),
            "",
        ])

    if registry_payload:
        champion = registry_payload.get(
            "champion", {}
        )
        challenger = registry_payload.get(
            "challenger", {}
        )

        lines.extend([
            "## MLflow Registry",
            "",
            (
                f"- Champion: version "
                f"{champion.get('version', '?')} "
                f"({champion.get('method', 'unknown')})"
            ),
            (
                f"- Challenger: version "
                f"{challenger.get('version', '?')} "
                f"({challenger.get('method', 'unknown')})"
            ),
            "",
        ])

    lines.extend([
        "## Agent Policy",
        "",
        (
            "This agent only summarizes existing forecast, retraining, "
            "and registry outputs. It does not override validation, "
            "change promotion thresholds, or deploy a failed challenger."
        ),
        "",
    ])

    return "\n".join(lines)


def save_report(
    report_text: str,
    output_dir: Path,
    decision: AgentDecision,
) -> tuple[Path, Path]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    md_path = (
        output_dir
        / "latest_agent_report.md"
    )

    json_path = (
        output_dir
        / "latest_agent_event.json"
    )

    md_path.write_text(
        report_text,
        encoding="utf-8",
    )

    event_payload = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "should_report": decision.should_report,
        "should_email": decision.should_email,
        "event_types": decision.event_types,
        "reasons": decision.reasons,
    }

    json_path.write_text(
        json.dumps(
            event_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    return md_path, json_path


def send_email(
    *,
    subject: str,
    body: str,
    to_email: str,
) -> None:
    """
    Optional SMTP email sender.

    Required environment variables:
    SMTP_HOST
    SMTP_PORT
    SMTP_USERNAME
    SMTP_PASSWORD
    SMTP_FROM_EMAIL
    """
    host = os.environ["SMTP_HOST"]
    port = int(
        os.environ.get(
            "SMTP_PORT",
            "587",
        )
    )
    username = os.environ["SMTP_USERNAME"]
    password = os.environ["SMTP_PASSWORD"]
    from_email = os.environ["SMTP_FROM_EMAIL"]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP(
        host,
        port,
        timeout=30,
    ) as server:
        server.starttls()
        server.login(
            username,
            password,
        )
        server.send_message(msg)


def _fmt(
    value: Any,
) -> str:
    if value is None:
        return "n/a"

    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def _pct(
    value: Any,
) -> str:
    if value is None:
        return "n/a"

    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return str(value)
