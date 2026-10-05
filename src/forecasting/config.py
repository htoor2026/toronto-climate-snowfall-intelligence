from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ForecastConfig:
    target_station_id: int
    target_station_name: str
    climatology_window: int
    autumn_min_coverage_pct: float
    material_expected_change_cm: float
    material_probability_change: float
    material_interval_bound_change_cm: float
    challenger_enabled: bool
    forecast_history_path: str
    latest_forecast_path: str

    @classmethod
    def from_json(cls, path: Path) -> "ForecastConfig":
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)
