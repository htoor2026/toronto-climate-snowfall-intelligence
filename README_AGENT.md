# Minimal Weather Agent

This is intentionally small.

It is not an autonomous model-building agent and does not use an LLM.

It reads three existing outputs:

```text
forecasts/latest_forecast.json
models/retraining/latest_retraining_decision.json
models/retraining/latest_mlflow_registry.json
```

Then it applies deterministic rules:

```text
material forecast change?
    → create report
    → optionally email

challenger passed promotion gate?
    → create report
    → optionally email

registry disagrees with retraining decision?
    → create report
    → optionally email

nothing important changed?
    → do nothing
```

The agent does NOT:

- retrain a model
- alter validation rules
- change the target
- change promotion thresholds
- promote a failed challenger
- call paid APIs
- require an LLM

## Install

No new Python dependency is required.

## Test

```bash
python -m pytest tests/ -v
```

## Run normally

```bash
python scripts/run_agent.py
```

With the current project state, this will probably do nothing because the
forecast has no material change and the challenger did not pass the promotion
gate.

## Force a report for testing

```bash
python scripts/run_agent.py --force-report
```

This creates:

```text
reports/agent/latest_agent_report.md
reports/agent/latest_agent_event.json
```

## Email later

Email is disabled by default in:

```text
config/agent_config.json
```

To enable it, set:

```json
{
  "email_enabled": true,
  "email_to": "your-email@example.com"
}
```

and provide SMTP credentials via environment variables:

```text
SMTP_HOST
SMTP_PORT
SMTP_USERNAME
SMTP_PASSWORD
SMTP_FROM_EMAIL
```

Then:

```bash
python scripts/run_agent.py --send-email
```

Do not hard-code passwords in the repository.

## Why this stays simple

The forecasting, retraining, validation, and MLflow layers already make the
project technically strong. This agent only performs orchestration and
notification. A dashboard can read the same JSON outputs later without adding
another backend service.
