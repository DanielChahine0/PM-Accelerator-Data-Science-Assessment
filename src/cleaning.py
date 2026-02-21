"""
cleaning.py – Data cleaning & preprocessing pipeline.

Steps:
1. Standardize column names (snake_case)
2. Convert last_updated → datetime, extract daily date
3. Remove duplicate daily records per location (keep latest)
4. Forward/backward fill per location for numeric columns
5. Detect outliers via IQR and add flag columns
6. Save cleaned dataset → data/processed/weather_clean.parquet
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "GlobalWeatherRepository.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
CLEAN_PATH = PROCESSED_DIR / "weather_clean.parquet"


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case, replace spaces/dots with underscores."""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s\.\-]+", "_", regex=True)
    )
    return df


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert *last_updated* to datetime and extract a daily date column."""
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    df["date"] = df["last_updated"].dt.date
    df["date"] = pd.to_datetime(df["date"])
    return df


def remove_daily_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the latest record per (location_name, country, date)."""
    df = df.sort_values("last_updated")
    df = df.drop_duplicates(subset=["location_name", "country", "date"], keep="last")
    return df.reset_index(drop=True)


def fill_missing_per_location(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then backward-fill numeric columns within each location."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df[numeric_cols] = (
        df.groupby("location_name")[numeric_cols]
        .transform(lambda g: g.ffill().bfill())
    )
    return df


def flag_iqr_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    factor: float = 1.5,
) -> pd.DataFrame:
    """Add boolean *_outlier* flag columns using the IQR method."""
    if columns is None:
        columns = [
            "temperature_celsius",
            "humidity",
            "precip_mm",
            "wind_kph",
            "pressure_mb",
        ]
    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        df[f"{col}_outlier"] = (df[col] < lower) | (df[col] > upper)
    return df


def run_cleaning(raw_path: str | Path | None = None) -> pd.DataFrame:
    """Execute the full cleaning pipeline and save to parquet."""
    path = Path(raw_path) if raw_path else RAW_PATH
    print(f"[cleaning] Reading {path} ...")
    df = pd.read_csv(path)
    print(f"[cleaning] Raw shape: {df.shape}")

    df = standardize_columns(df)
    df = parse_datetime(df)
    df = remove_daily_duplicates(df)
    df = fill_missing_per_location(df)
    df = flag_iqr_outliers(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_PATH, index=False)
    print(f"[cleaning] Saved cleaned data → {CLEAN_PATH}  shape={df.shape}")
    return df


if __name__ == "__main__":
    run_cleaning()
