from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import FINAL_FEATURES


def load_latest_ready_feature_snapshot(history_path: Path, *, season_year: int) -> dict | None:
    if not history_path.exists():
        return None
    matches = []
    for raw_line in history_path.read_text(encoding='utf-8').splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if int(payload.get('season_year', -1)) != int(season_year):
            continue
        challenger = payload.get('challenger_forecast') or {}
        feature_row = payload.get('current_feature_row')
        if challenger.get('status') == 'ready' and feature_row:
            matches.append((payload.get('generated_at_utc', ''), feature_row))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return matches[-1][1]


def append_completed_seasons_from_archive(base_training: pd.DataFrame, strict_history: pd.DataFrame, forecast_history_path: Path) -> tuple[pd.DataFrame, list[int], list[int]]:
    data = base_training.copy().sort_values('season_year').reset_index(drop=True)
    numeric_years = pd.to_numeric(data['season_year'], errors='coerce').dropna().astype(int)
    if numeric_years.empty:
        raise ValueError('Training dataset has no valid season_year values.')
    latest_existing_season = int(numeric_years.max())
    strict = (strict_history[['season_year','snowfall_cm']].dropna().drop_duplicates('season_year').sort_values('season_year'))
    strict = strict[pd.to_numeric(strict['season_year'], errors='coerce') > latest_existing_season]
    appended, skipped, new_rows = [], [], []
    for _, target in strict.iterrows():
        season_year = int(target['season_year'])
        snapshot = load_latest_ready_feature_snapshot(forecast_history_path, season_year=season_year)
        if snapshot is None:
            skipped.append(season_year)
            continue
        row = {'season_year': season_year, 'snowfall_cm': float(target['snowfall_cm'])}
        complete = True
        for feature in FINAL_FEATURES:
            value = snapshot.get(feature)
            if value is None or pd.isna(value):
                complete = False
                break
            row[feature] = float(value)
        if not complete:
            skipped.append(season_year)
            continue
        new_rows.append(row)
        appended.append(season_year)
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        for col in data.columns:
            if col not in new_df.columns:
                new_df[col] = np.nan
        for col in new_df.columns:
            if col not in data.columns:
                data[col] = np.nan
        new_df = new_df[data.columns]
        data = pd.concat([data, new_df], ignore_index=True)
    data = data.sort_values('season_year').drop_duplicates('season_year', keep='last').reset_index(drop=True)
    return data, appended, skipped
