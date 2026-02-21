#!/usr/bin/env python
"""
main.py – Weather Trend Forecasting: end-to-end pipeline.

Stages
------
1. Data cleaning  →  data/processed/weather_clean.parquet
2. EDA plots      →  reports/figures/
3. Feature engineering
4. Train/val/test split (70/15/15 chronological)
5. Forecasting models (Naive, Seasonal Naive, SARIMA, Prophet, ML, Ensemble)
6. Evaluation (MAE, RMSE, sMAPE) for 7-day & 14-day horizons
7. Advanced analytics (STL + Isolation Forest anomalies, spatial map,
   monthly climate, feature importance)
"""

from __future__ import annotations

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for server / CI
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ----- project imports -----
from src.cleaning import run_cleaning, CLEAN_PATH
from src.features import build_features
from src.anomalies import stl_anomaly_detection, isolation_forest_anomalies
from src.evaluation import evaluate
from src.models.baseline import naive_forecast, seasonal_naive_forecast
from src.models.arima import fit_sarima
from src.models.prophet_model import fit_prophet
from src.models.ml_regression import (
    train_ml_model,
    predict_ml,
    feature_importance,
    FEATURE_COLS,
)
from src.models.ensemble import inverse_mae_weights, weighted_ensemble

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Major cities used for per-city time-series forecasting
MAJOR_CITIES = [
    "London",
    "Paris",
    "Tokyo",
    "Cairo",
    "Moscow",
]

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ============================================================
# 1. DATA CLEANING
# ============================================================
def stage_clean() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STAGE 1 – DATA CLEANING")
    print("=" * 60)
    if CLEAN_PATH.exists():
        print(f"[clean] Loading existing {CLEAN_PATH}")
        df = pd.read_parquet(CLEAN_PATH)
    else:
        df = run_cleaning()
    print(f"  Cleaned shape: {df.shape}")
    return df


# ============================================================
# 2. EDA  (plots saved to reports/figures/)
# ============================================================
def stage_eda(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("STAGE 2 – EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # 2a – Missing values bar chart
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        missing.plot.bar(ax=ax, color="salmon")
        ax.set_title("Missing Values per Column")
        ax.set_ylabel("Count")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "missing_values.png", dpi=150)
        plt.close(fig)
        print("  Saved missing_values.png")
    else:
        print("  No missing values to plot.")

    # 2b – Temperature + Precipitation distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df["temperature_celsius"].dropna().hist(
        bins=60, ax=axes[0], color="steelblue", edgecolor="white"
    )
    axes[0].set_title("Temperature (°C) Distribution")
    axes[0].set_xlabel("Temperature (°C)")
    df["precip_mm"].dropna().hist(
        bins=60, ax=axes[1], color="teal", edgecolor="white"
    )
    axes[1].set_title("Precipitation (mm) Distribution")
    axes[1].set_xlabel("Precipitation (mm)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "temp_precip_distributions.png", dpi=150)
    plt.close(fig)
    print("  Saved temp_precip_distributions.png")

    # 2c – Time series for major cities
    fig, ax = plt.subplots(figsize=(14, 6))
    for city in MAJOR_CITIES:
        sub = df[df["location_name"] == city].sort_values("date")
        if len(sub) == 0:
            continue
        ax.plot(sub["date"], sub["temperature_celsius"], label=city, alpha=0.8)
    ax.set_title("Daily Temperature – Major Cities")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "timeseries_major_cities.png", dpi=150)
    plt.close(fig)
    print("  Saved timeseries_major_cities.png")

    # 2d – Correlation heatmap (numeric only)
    numeric = df.select_dtypes(include="number")
    # Keep only the most important columns to keep heatmap readable
    keep_cols = [
        "temperature_celsius", "feels_like_celsius", "humidity",
        "precip_mm", "wind_kph", "pressure_mb", "cloud",
        "visibility_km", "uv_index", "gust_kph",
        "air_quality_pm2_5", "air_quality_pm10",
    ]
    keep_cols = [c for c in keep_cols if c in numeric.columns]
    corr = numeric[keep_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved correlation_heatmap.png")


# ============================================================
# 3. FORECASTING PIPELINE
# ============================================================

def _chrono_split(
    series: pd.Series,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
):
    """Split a 1-D series chronologically into train / val / test."""
    n = len(series)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return series.iloc[:t1], series.iloc[t1:t2], series.iloc[t2:]


def forecast_city(
    df: pd.DataFrame,
    city: str,
    horizons: list[int] = [7, 14],
) -> list[dict]:
    """
    Run all forecasting models for a single city at multiple horizons.
    Returns a list of evaluation dicts.
    """
    sub = df[df["location_name"] == city].sort_values("date").reset_index(drop=True)
    series = sub["temperature_celsius"].dropna()
    if len(series) < 60:
        print(f"  [skip] {city}: only {len(series)} records")
        return []

    results = []
    for horizon in horizons:
        train_s, val_s, test_s = _chrono_split(series)

        # Use validation set's first *horizon* values as ground truth
        gt = val_s.values[:horizon]
        if len(gt) < horizon:
            continue

        preds: dict[str, np.ndarray] = {}

        # --- Naive ---
        preds["Naive"] = naive_forecast(train_s, horizon)

        # --- Seasonal Naive ---
        preds["SeasonalNaive"] = seasonal_naive_forecast(train_s, horizon, season=7)

        # --- SARIMA ---
        preds["SARIMA"] = fit_sarima(train_s, horizon=horizon)

        # --- Prophet ---
        train_df_prophet = sub.iloc[: len(train_s)][["date", "temperature_celsius"]]
        preds["Prophet"] = fit_prophet(train_df_prophet, horizon=horizon)

        # --- ML Regression ---
        feat_df = build_features(sub)
        feat_train = feat_df.iloc[: len(train_s)]
        feat_val = feat_df.iloc[len(train_s) : len(train_s) + horizon]
        ml_model, used_cols = train_ml_model(feat_train)
        if len(feat_val) >= horizon:
            preds["ML_GBR"] = predict_ml(ml_model, feat_val, used_cols)
        else:
            preds["ML_GBR"] = naive_forecast(train_s, horizon)

        # --- Ensemble ---
        weights = inverse_mae_weights(gt, preds)
        preds["Ensemble"] = weighted_ensemble(preds, weights)

        for name, pred in preds.items():
            result = evaluate(gt, pred[: len(gt)], model_name=name)
            result["city"] = city
            result["horizon"] = horizon
            results.append(result)

    return results


def stage_forecast(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STAGE 3 – FORECASTING")
    print("=" * 60)

    all_results = []
    for city in MAJOR_CITIES:
        print(f"  Forecasting: {city} ...")
        city_res = forecast_city(df, city)
        all_results.extend(city_res)

    results_df = pd.DataFrame(all_results)
    if len(results_df) > 0:
        print("\n  === Model Comparison ===")
        summary = (
            results_df.groupby(["model", "horizon"])[["MAE", "RMSE", "sMAPE"]]
            .mean()
            .round(4)
        )
        print(summary.to_string())

        results_df.to_csv(ROOT / "reports" / "forecast_results.csv", index=False)
        print(f"\n  Saved reports/forecast_results.csv")

        # -- Comparison bar chart --
        for h in [7, 14]:
            sub = results_df[results_df["horizon"] == h]
            if sub.empty:
                continue
            avg = sub.groupby("model")["MAE"].mean().sort_values()
            fig, ax = plt.subplots(figsize=(8, 5))
            avg.plot.barh(ax=ax, color="steelblue")
            ax.set_xlabel("MAE (°C)")
            ax.set_title(f"Model Comparison – {h}-day Horizon (mean MAE)")
            plt.tight_layout()
            fig.savefig(FIG_DIR / f"model_comparison_{h}day.png", dpi=150)
            plt.close(fig)
            print(f"  Saved model_comparison_{h}day.png")

    return results_df


# ============================================================
# 4. ADVANCED ANALYTICS
# ============================================================
def stage_advanced(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("STAGE 4 – ADVANCED ANALYTICS")
    print("=" * 60)

    # ---- 4a. STL + Isolation Forest anomaly detection ----
    print("  4a. Anomaly detection ...")
    city = MAJOR_CITIES[0]
    sub = df[df["location_name"] == city].sort_values("date").set_index("date")
    ts = sub["temperature_celsius"].dropna()
    if len(ts) >= 14:
        stl_df = stl_anomaly_detection(ts, period=7)
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        axes[0].plot(ts.index, ts.values, color="steelblue")
        axes[0].set_ylabel("Observed")
        axes[0].set_title(f"STL Decomposition – {city}")
        axes[1].plot(stl_df.index, stl_df["trend"], color="orange")
        axes[1].set_ylabel("Trend")
        axes[2].plot(stl_df.index, stl_df["seasonal"], color="green")
        axes[2].set_ylabel("Seasonal")
        axes[3].plot(stl_df.index, stl_df["resid"], color="grey", alpha=0.6)
        anom_idx = stl_df[stl_df["anomaly"]].index
        axes[3].scatter(
            anom_idx,
            stl_df.loc[anom_idx, "resid"],
            color="red",
            zorder=5,
            label="Anomaly",
        )
        axes[3].set_ylabel("Residual")
        axes[3].legend()
        plt.tight_layout()
        fig.savefig(FIG_DIR / "stl_anomaly_detection.png", dpi=150)
        plt.close(fig)
        print("    Saved stl_anomaly_detection.png")

    # Isolation Forest on full dataset
    df["iso_forest_anomaly"] = isolation_forest_anomalies(df)
    n_anom = df["iso_forest_anomaly"].sum()
    print(f"    Isolation Forest anomalies: {n_anom} / {len(df)}")

    # ---- 4b. Spatial temperature map ----
    print("  4b. Spatial temperature map ...")
    latest_date = df["date"].max()
    snap = df[df["date"] == latest_date].drop_duplicates("location_name")
    if len(snap) > 0:
        fig, ax = plt.subplots(figsize=(14, 7))
        sc = ax.scatter(
            snap["longitude"],
            snap["latitude"],
            c=snap["temperature_celsius"],
            cmap="RdYlBu_r",
            s=20,
            alpha=0.7,
        )
        plt.colorbar(sc, ax=ax, label="Temperature (°C)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Global Temperature Map – {latest_date.date()}")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "spatial_temperature_map.png", dpi=150)
        plt.close(fig)
        print("    Saved spatial_temperature_map.png")

    # ---- 4c. Monthly climate comparison by continent ----
    print("  4c. Monthly climate by continent ...")
    continent_map = _country_to_continent()
    df["continent"] = df["country"].map(continent_map).fillna("Other")
    df["month_num"] = df["date"].dt.month
    monthly = (
        df.groupby(["continent", "month_num"])["temperature_celsius"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    for cont in monthly["continent"].unique():
        c = monthly[monthly["continent"] == cont]
        ax.plot(c["month_num"], c["temperature_celsius"], marker="o", label=cont)
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Temperature (°C)")
    ax.set_title("Monthly Average Temperature by Continent")
    ax.set_xticks(range(1, 13))
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "monthly_climate_continent.png", dpi=150)
    plt.close(fig)
    print("    Saved monthly_climate_continent.png")

    # ---- 4d. Feature importance (ML model) ----
    print("  4d. Feature importance ...")
    city = MAJOR_CITIES[0]
    sub = df[df["location_name"] == city].sort_values("date")
    feat_df = build_features(sub)
    model, used_cols = train_ml_model(feat_df)
    imp = feature_importance(model, used_cols)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=imp.head(15), x="importance", y="feature", ax=ax, color="steelblue")
    ax.set_title(f"Top 15 Feature Importances – {city}")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    print("    Saved feature_importance.png")
    print(imp.head(15).to_string(index=False))

    # ---- 4e. Air-quality correlation ----
    print("  4e. Air quality & weather correlation ...")
    aq_cols = [c for c in df.columns if c.startswith("air_quality")]
    weather_cols = ["temperature_celsius", "humidity", "wind_kph", "precip_mm"]
    combined = [c for c in aq_cols + weather_cols if c in df.columns]
    if len(combined) > 3:
        corr = df[combined].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Air Quality vs Weather – Correlation")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "air_quality_weather_corr.png", dpi=150)
        plt.close(fig)
        print("    Saved air_quality_weather_corr.png")


def _country_to_continent() -> dict:
    """A lightweight mapping of common countries → continent."""
    mapping = {}
    asia = [
        "Afghanistan", "Armenia", "Azerbaijan", "Bahrain", "Bangladesh",
        "Bhutan", "Brunei", "Cambodia", "China", "Georgia", "India",
        "Indonesia", "Iran", "Iraq", "Israel", "Japan", "Jordan",
        "Kazakhstan", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon",
        "Malaysia", "Maldives", "Mongolia", "Myanmar", "Nepal", "Oman",
        "Pakistan", "Philippines", "Qatar", "Saudi Arabia", "Singapore",
        "South Korea", "Sri Lanka", "Syria", "Taiwan", "Tajikistan",
        "Thailand", "Timor-Leste", "Turkey", "Turkmenistan",
        "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen",
    ]
    europe = [
        "Albania", "Andorra", "Austria", "Belarus", "Belgium",
        "Bosnia and Herzegovina", "Bosnia And Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
        "Czechia", "Denmark", "Estonia", "Finland", "France", "Germany",
        "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Kosovo",
        "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta",
        "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia",
        "Norway", "Poland", "Portugal", "Romania", "Russia", "San Marino",
        "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland",
        "Ukraine", "United Kingdom",
    ]
    africa = [
        "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso",
        "Burundi", "Cameroon", "Cape Verde", "Central African Republic",
        "Chad", "Comoros", "Congo", "Democratic Republic of the Congo",
        "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini",
        "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau",
        "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya",
        "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius",
        "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda",
        "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone",
        "Somalia", "South Africa", "South Sudan", "Sudan", "Tanzania",
        "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe",
    ]
    north_america = [
        "Antigua and Barbuda", "Bahamas", "Barbados", "Belize", "Canada",
        "Costa Rica", "Cuba", "Dominica", "Dominican Republic",
        "El Salvador", "Grenada", "Guatemala", "Haiti", "Honduras",
        "Jamaica", "Mexico", "Nicaragua", "Panama",
        "Saint Kitts and Nevis", "Saint Lucia",
        "Saint Vincent and the Grenadines", "Trinidad and Tobago",
        "United States of America", "United States",
    ]
    south_america = [
        "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
        "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela",
    ]
    oceania = [
        "Australia", "Fiji", "Kiribati", "Marshall Islands", "Micronesia",
        "Nauru", "New Zealand", "Palau", "Papua New Guinea", "Samoa",
        "Solomon Islands", "Tonga", "Tuvalu", "Vanuatu",
    ]
    for c in asia:
        mapping[c] = "Asia"
    for c in europe:
        mapping[c] = "Europe"
    for c in africa:
        mapping[c] = "Africa"
    for c in north_america:
        mapping[c] = "North America"
    for c in south_america:
        mapping[c] = "South America"
    for c in oceania:
        mapping[c] = "Oceania"
    return mapping


# ============================================================
# MAIN
# ============================================================
def main():
    # 1. Cleaning
    df = stage_clean()

    # 2. EDA
    stage_eda(df)

    # 3. Forecasting
    results_df = stage_forecast(df)

    # 4. Advanced
    stage_advanced(df)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE ✓")
    print("=" * 60)
    print(f"  Figures  → {FIG_DIR}")
    print(f"  Results  → reports/forecast_results.csv")


if __name__ == "__main__":
    main()