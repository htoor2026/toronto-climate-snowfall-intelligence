from __future__ import annotations

from pathlib import Path
import json
import numpy as np


def load_calibration_metadata(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Calibration metadata not found: {path}. Run notebook 11b first.")
    return json.loads(path.read_text(encoding="utf-8"))


def apply_calibrated_interval(*, expected_cm: float, lower_empirical_cm: float, upper_empirical_cm: float, metadata: dict) -> tuple[float, float]:
    method = metadata["selected_interval_method"]
    qhat = float(metadata["final_qhat_cm"])
    if method == "Conformalized Empirical Interval":
        lower = lower_empirical_cm - qhat
        upper = upper_empirical_cm + qhat
    elif method == "Symmetric Point-Error Conformal":
        lower = expected_cm - qhat
        upper = expected_cm + qhat
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    return max(0.0, float(lower)), float(upper)


def empirical_tercile_thresholds(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 1/3)), float(np.quantile(values, 2/3))
