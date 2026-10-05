from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor

from src.forecasting.feature_builder import FINAL_FEATURES


def sqlite_tracking_uri(path: Path) -> str:
    """Return an absolute SQLite URI suitable for MLflow."""
    resolved = Path(path).expanduser().resolve()
    return f"sqlite:///{resolved.as_posix()}"


def sha256_file(path: Path) -> str:
    """Hash a file in chunks so large artifacts do not need to fit in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_hash(frame: pd.DataFrame) -> str:
    """Produce a deterministic hash for a dataframe snapshot."""
    normalized = frame.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1).reset_index(drop=True)
    payload = normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_model_comparison(path: Path) -> dict[str, dict[str, float]]:
    """Load the model-comparison CSV into MLflow-friendly metric dictionaries."""
    frame = pd.read_csv(path)
    if "model" not in frame.columns:
        raise ValueError(f"Expected a model column in {path}.")

    result: dict[str, dict[str, float]] = {}
    for _, row in frame.iterrows():
        model_name = str(row["model"])
        metrics: dict[str, float] = {}
        for column in frame.columns:
            if column == "model":
                continue
            value = pd.to_numeric(row[column], errors="coerce")
            if pd.notna(value):
                metrics[column] = float(value)
        result[model_name] = metrics
    return result


def build_climatology_estimator(
    strict_history: pd.DataFrame,
    training: pd.DataFrame,
    *,
    window: int,
):
    """Build the production champion as a constant sklearn estimator."""
    if window <= 0:
        raise ValueError("climatology window must be positive.")

    required_history = {"season_year", "snowfall_cm"}
    if not required_history.issubset(strict_history.columns):
        raise ValueError("strict_history must contain season_year and snowfall_cm.")

    if "season_year" not in training.columns:
        raise ValueError("training must contain season_year.")

    training_years = pd.to_numeric(training["season_year"], errors="coerce").dropna()
    if training_years.empty:
        raise ValueError("training contains no valid season_year values.")

    latest_labelled_season = int(training_years.max())

    history = strict_history[["season_year", "snowfall_cm"]].copy()
    history["season_year"] = pd.to_numeric(history["season_year"], errors="coerce")
    history["snowfall_cm"] = pd.to_numeric(history["snowfall_cm"], errors="coerce")
    history = (
        history.dropna()
        .drop_duplicates("season_year", keep="last")
        .sort_values("season_year")
    )
    history = history[history["season_year"] <= latest_labelled_season].tail(window)

    if len(history) < window:
        raise ValueError(f"Need {window} strict-quality winters; found {len(history)}.")

    expected_snowfall_cm = float(history["snowfall_cm"].mean())

    complete_training = (
        training.dropna(subset=FINAL_FEATURES + ["snowfall_cm"])
        .sort_values("season_year")
    )
    if complete_training.empty:
        raise ValueError("No model-ready training rows are available.")

    X = complete_training[FINAL_FEATURES]
    y = complete_training["snowfall_cm"]

    estimator = DummyRegressor(strategy="constant", constant=expected_snowfall_cm)
    estimator.fit(X, y)

    return estimator, expected_snowfall_cm, history.reset_index(drop=True)


def find_version_by_tags(
    client,
    *,
    model_name: str,
    role: str,
    snapshot_hash: str,
):
    """Return an existing model version with the same role and snapshot hash."""
    versions = client.search_model_versions(f"name='{model_name}'")
    matches = [
        version
        for version in versions
        if (version.tags or {}).get("role") == role
        and (version.tags or {}).get("snapshot_hash") == snapshot_hash
    ]
    if not matches:
        return None
    return max(matches, key=lambda version: int(version.version))


def set_version_tags(
    client,
    *,
    model_name: str,
    version: str | int,
    tags: dict,
) -> None:
    """Apply normalized string tags to one MLflow model version."""
    for key, value in tags.items():
        client.set_model_version_tag(
            name=model_name,
            version=str(version),
            key=str(key),
            value=str(value),
        )
