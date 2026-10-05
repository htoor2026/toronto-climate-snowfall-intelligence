from pathlib import Path

import pandas as pd

from src.tracking.mlflow_registry import dataframe_hash, sqlite_tracking_uri


def test_sqlite_tracking_uri_is_absolute(tmp_path):
    uri = sqlite_tracking_uri(tmp_path / 'mlflow.db')
    assert uri.startswith('sqlite:////')
    assert uri.endswith('mlflow.db')


def test_dataframe_hash_is_deterministic():
    df = pd.DataFrame({'season_year':[2025,2026],'snowfall_cm':[100.0,120.0]})
    assert dataframe_hash(df) == dataframe_hash(df.copy())
