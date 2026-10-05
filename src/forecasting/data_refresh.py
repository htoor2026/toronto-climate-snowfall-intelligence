from __future__ import annotations

from io import StringIO

import pandas as pd
import requests


ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
NAO_URL = (
    "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/"
    "pna/norm.nao.monthly.b5001.current.ascii"
)
ECCC_DAILY_URL = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"


def _request_text(url: str, *, params: dict | None = None, timeout: int = 60) -> str:
    headers = {"User-Agent": "toronto-snowfall-forecast/1.0"}
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


def fetch_oni_jas(predictor_year: int) -> float | None:
    text = _request_text(ONI_URL)
    table = pd.read_csv(StringIO(text), sep=r"\s+")
    required = {"SEAS", "YR", "ANOM"}
    if not required.issubset(table.columns):
        raise ValueError(f"Unexpected ONI schema: {table.columns.tolist()}")
    row = table[(table["SEAS"].astype(str) == "JAS") & (pd.to_numeric(table["YR"], errors="coerce") == predictor_year)]
    if row.empty:
        return None
    value = pd.to_numeric(row.iloc[-1]["ANOM"], errors="coerce")
    return None if pd.isna(value) else float(value)


def fetch_nao_jas(predictor_year: int) -> tuple[float | None, int]:
    text = _request_text(NAO_URL)
    table = pd.read_csv(StringIO(text), sep=r"\s+", header=None, names=["year", "month", "nao"], usecols=[0, 1, 2])
    for col in ["year", "month", "nao"]:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    table = table.dropna(subset=["year", "month", "nao"]).copy()
    table["year"] = table["year"].astype(int)
    table["month"] = table["month"].astype(int)
    subset = table[(table["year"] == predictor_year) & (table["month"].isin([7, 8, 9]))].copy()
    valid = int(subset["nao"].notna().sum())
    if valid != 3:
        return None, valid
    return float(subset["nao"].mean()), valid


def fetch_eccc_daily(station_id: int, year: int) -> pd.DataFrame:
    params = {
        "format": "csv",
        "stationID": station_id,
        "Year": year,
        "Month": 1,
        "Day": 1,
        "timeframe": 2,
        "submit": "Download Data",
    }
    text = _request_text(ECCC_DAILY_URL, params=params)
    df = pd.read_csv(StringIO(text))
    if df.empty:
        raise ValueError(f"ECCC returned no rows for station {station_id}, year {year}.")
    return df


def _first_matching_column(df: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    for original in df.columns:
        lower = str(original).strip().lower()
        if any(candidate.strip().lower() in lower for candidate in candidates):
            return original
    raise KeyError(f"Could not find any of {candidates}. Columns: {df.columns.tolist()}")


def build_sep_oct_features(daily: pd.DataFrame, *, predictor_year: int, min_coverage_pct: float = 95.0) -> dict:
    date_col = _first_matching_column(daily, ["Date/Time", "Date/Time (LST)", "LOCAL_DATE", "date"])
    temp_col = _first_matching_column(daily, ["Mean Temp (°C)", "Mean Temp", "mean temp"])
    precip_col = _first_matching_column(daily, ["Total Precip (mm)", "Total Precip", "total precip"])

    df = daily.copy()
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["_temp"] = pd.to_numeric(df[temp_col], errors="coerce")
    df["_precip"] = pd.to_numeric(df[precip_col], errors="coerce")
    autumn = df[(df["_date"].dt.year == predictor_year) & (df["_date"].dt.month.isin([9, 10]))].copy()

    expected_days = 61
    temp_valid = int(autumn["_temp"].notna().sum())
    precip_valid = int(autumn["_precip"].notna().sum())
    temp_coverage = 100.0 * temp_valid / expected_days
    precip_coverage = 100.0 * precip_valid / expected_days
    usable = temp_coverage >= min_coverage_pct and precip_coverage >= min_coverage_pct

    return {
        "sep_oct_mean_temp_c": float(autumn["_temp"].mean()) if temp_valid else None,
        "sep_oct_total_precip_mm": float(autumn["_precip"].sum(min_count=1)) if precip_valid else None,
        "sep_oct_temp_coverage_pct": float(temp_coverage),
        "sep_oct_precip_coverage_pct": float(precip_coverage),
        "autumn_features_usable": bool(usable),
        "autumn_observed_rows": int(len(autumn)),
    }


def refresh_live_predictors(*, predictor_year: int, station_id: int, autumn_min_coverage_pct: float) -> dict:
    oni_jas = fetch_oni_jas(predictor_year)
    nao_jas, nao_valid_months = fetch_nao_jas(predictor_year)
    daily = fetch_eccc_daily(station_id, predictor_year)
    autumn = build_sep_oct_features(daily, predictor_year=predictor_year, min_coverage_pct=autumn_min_coverage_pct)
    return {
        "predictor_year": predictor_year,
        "oni_jas": oni_jas,
        "nao_jas": nao_jas,
        "nao_jas_valid_months": nao_valid_months,
        **autumn,
    }
