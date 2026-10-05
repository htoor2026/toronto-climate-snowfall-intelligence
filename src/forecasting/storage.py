from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def load_latest(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_forecast_snapshot(*, latest_path: Path, history_path: Path, payload: dict) -> None:
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    stamped = {**payload, "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    latest_path.write_text(json.dumps(stamped, indent=2), encoding="utf-8")
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(stamped) + "\n")
