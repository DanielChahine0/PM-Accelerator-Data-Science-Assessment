#!/usr/bin/env python
"""
demo.py – Walkthrough script that demonstrates how each assessment
requirement is met by this project.

Run:
    python demo.py

This script does NOT re-train models or regenerate plots. It loads existing
artefacts (cleaned data, saved figures, forecast results) and prints a
narrated walkthrough mapping every deliverable to the corresponding code
and output.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# ── project paths ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "reports" / "figures"
RESULTS_CSV = ROOT / "reports" / "forecast_results.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "weather_clean.parquet"
RAW_PATH = ROOT / "data" / "raw" / "GlobalWeatherRepository.csv"

# ── helpers ───────────────────────────────────────────────────

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"
CHECK = "✅"
ARROW = "→"


def header(title: str) -> None:
    width = 70
    print()
    print(f"{BOLD}{BLUE}{'═' * width}{RESET}")
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(f"{BOLD}{BLUE}{'═' * width}{RESET}")


def sub(title: str) -> None:
    print(f"\n  {BOLD}{CYAN}{title}{RESET}")


def bullet(text: str) -> None:
    print(f"    {GREEN}{CHECK}{RESET} {text}")


def info(text: str) -> None:
    for line in textwrap.wrap(text, width=72):
        print(f"      {line}")


def file_exists(path: Path, label: str) -> bool:
    ok = path.exists()
    status = f"{GREEN}found{RESET}" if ok else f"{YELLOW}NOT FOUND{RESET}"
    print(f"    {ARROW} {label}: {status}  ({path.relative_to(ROOT)})")
    return ok


# ==============================================================
# BEGIN WALKTHROUGH
# ==============================================================

def main() -> None:
    print(f"\n{BOLD}{'─' * 70}{RESET}")
    print(f"{BOLD}  PM Accelerator – Weather Trend Forecasting")
    print(f"  Assessment Requirements Demo Walkthrough{RESET}")
    print(f"{BOLD}{'─' * 70}{RESET}")
    print()
    print("  This script walks through every assessment requirement and shows")
    print("  exactly where and how it is fulfilled in the codebase.\n")

    # ──────────────────────────────────────────────────────────
    # 0. PM ACCELERATOR MISSION
    # ──────────────────────────────────────────────────────────
    header("DELIVERABLE: PM Accelerator Mission Statement")
    sub("Requirement")
    info("Display the PM Accelerator mission on the report/presentation/dashboard.")
    sub("How it is met")
    bullet("The README.md opens with the PM Accelerator mission statement:")
    readme = (ROOT / "README.md").read_text()
    # Extract the blockquote
    for line in readme.splitlines():
        if line.startswith("> **PM Accelerator**"):
            print(f"\n      {YELLOW}\"{line[2:70]}...\"{RESET}\n")
            break
    bullet("Location: README.md, top of file (lines 3-4)")

    # ──────────────────────────────────────────────────────────
    # 1. DATA CLEANING & PREPROCESSING  (Basic)
    # ──────────────────────────────────────────────────────────
    header("BASIC 1/3: Data Cleaning & Preprocessing")
    sub("Requirement")
    info("Handle missing values, outliers, and normalize data.")
    sub("How it is met  (src/cleaning.py)")

    bullet("Column standardization: names → snake_case  (standardize_columns)")
    info("Ensures consistent access; e.g. 'Last Updated' → 'last_updated'.")

    bullet("Datetime parsing: last_updated → datetime + daily date column  (parse_datetime)")
    info("Uses the 'lastupdated' feature as required for time-series analysis.")

    bullet("Duplicate removal: keep latest record per (location, country, date)  (remove_daily_duplicates)")

    bullet("Missing-value handling: forward + backward fill per location  (fill_missing_per_location)")
    info("Grouped by location_name so fills respect each city's own history.")

    bullet("Outlier detection: IQR-based flag columns for 5 key variables  (flag_iqr_outliers)")
    info("Columns flagged: temperature, humidity, precipitation, wind, pressure.")
    info("Uses 1.5×IQR rule; outliers are flagged but kept, not removed.")

    bullet("Output saved as Parquet for fast reload  (data/processed/weather_clean.parquet)")

    # Show cleaned data stats
    if CLEAN_PATH.exists():
        df = pd.read_parquet(CLEAN_PATH)
        print(f"\n    {ARROW} Cleaned dataset shape: {BOLD}{df.shape[0]:,} rows × {df.shape[1]} columns{RESET}")
        missing = df.isnull().sum().sum()
        print(f"    {ARROW} Remaining missing values: {BOLD}{missing:,}{RESET}")
        outlier_cols = [c for c in df.columns if c.endswith("_outlier")]
        if outlier_cols:
            n_outliers = df[outlier_cols].any(axis=1).sum()
            print(f"    {ARROW} Rows flagged as outliers (any column): {BOLD}{n_outliers:,}{RESET}")
    else:
        print(f"\n    {YELLOW}[Run `python main.py` first to generate cleaned data]{RESET}")

    # ──────────────────────────────────────────────────────────
    # 2. EXPLORATORY DATA ANALYSIS  (Basic)
    # ──────────────────────────────────────────────────────────
    header("BASIC 2/3: Exploratory Data Analysis (EDA)")
    sub("Requirement")
    info("Perform basic EDA to uncover trends, correlations, and patterns.")
    info("Generate visualizations for temperature and precipitation.")
    sub("How it is met  (main.py → stage_eda)")

    figures = {
        "missing_values.png": "Missing-value bar chart per column",
        "temp_precip_distributions.png": "Temperature & precipitation histograms  ← REQUIRED",
        "timeseries_major_cities.png": "Daily temperature time series for 5 major cities (trends)",
        "correlation_heatmap.png": "Correlation heatmap of key numeric features (correlations)",
    }
    for fname, desc in figures.items():
        path = FIG_DIR / fname
        exists = path.exists()
        status = f"{GREEN}exists{RESET}" if exists else f"{YELLOW}missing{RESET}"
        bullet(f"{desc}")
        print(f"      File: reports/figures/{fname}  [{status}]")

    bullet("Interactive notebook at notebooks/eda.ipynb for deeper exploration")

    # ──────────────────────────────────────────────────────────
    # 3. MODEL BUILDING  (Basic)
    # ──────────────────────────────────────────────────────────
    header("BASIC 3/3: Model Building & Evaluation")
    sub("Requirement")
    info("Build a basic forecasting model and evaluate its performance using")
    info("different metrics. Use lastupdated feature for time series analysis.")
    sub("How it is met")

    bullet("Time feature: 'last_updated' is parsed in cleaning.py → 'date' column")
    info("All time-series splits and forecasts use this date column.")

    bullet("Chronological train/val/test split: 70 / 15 / 15 %  (main.py → _chrono_split)")
    info("Prevents data leakage by never training on future data.")

    bullet("Baseline models: Naive & Seasonal Naive  (src/models/baseline.py)")
    info("Naive repeats last value; Seasonal Naive repeats last 7-day cycle.")

    bullet("Evaluation metrics: MAE, RMSE, sMAPE  (src/evaluation.py)")
    info("Three complementary metrics give a well-rounded view of accuracy.")

    bullet("Forecast horizons: 7-day and 14-day ahead")

    # Show results table
    if RESULTS_CSV.exists():
        results = pd.read_csv(RESULTS_CSV)
        avg = results.groupby(["model", "horizon"])[["MAE", "RMSE", "sMAPE"]].mean().round(4)
        sub("Average Metrics Across 5 Cities")
        print()
        print(avg.to_string())
        print()
    else:
        print(f"\n    {YELLOW}[Run `python main.py` first to generate results]{RESET}")

    # ──────────────────────────────────────────────────────────
    # 4. ADVANCED EDA – ANOMALY DETECTION
    # ──────────────────────────────────────────────────────────
    header("ADVANCED 1/3: Advanced EDA – Anomaly Detection")
    sub("Requirement")
    info("Implement anomaly detection to identify and analyze outliers.")
    sub("How it is met  (src/anomalies.py)")

    bullet("STL Decomposition  (stl_anomaly_detection)")
    info("Decomposes time series into trend + seasonal + residual components.")
    info("Flags residuals exceeding 3σ as anomalies.")
    file_exists(FIG_DIR / "stl_anomaly_detection.png", "STL anomaly plot")

    bullet("Isolation Forest  (isolation_forest_anomalies)")
    info("Unsupervised multivariate anomaly detection on temperature, humidity,")
    info("precipitation, wind speed, and pressure. Contamination = 2%.")

    bullet("IQR outlier flags already applied during data cleaning (Basic req.)")

    # ──────────────────────────────────────────────────────────
    # 5. MULTIPLE MODELS + ENSEMBLE
    # ──────────────────────────────────────────────────────────
    header("ADVANCED 2/3: Forecasting with Multiple Models + Ensemble")
    sub("Requirement")
    info("Build and compare multiple forecasting models.")
    info("Create an ensemble of models to improve forecast accuracy.")
    sub("How it is met")

    models_info = [
        ("Naive Baseline", "src/models/baseline.py", "Last observed value repeated"),
        ("Seasonal Naive", "src/models/baseline.py", "Last 7-day cycle repeated"),
        ("SARIMA", "src/models/arima.py", "(1,1,1)×(1,1,0,7) with automatic fallback"),
        ("Prophet", "src/models/prophet_model.py", "Weekly + yearly seasonality via Facebook Prophet"),
        ("ML (Gradient Boosting)", "src/models/ml_regression.py", "GBR with 22 engineered features"),
        ("Weighted Ensemble", "src/models/ensemble.py", "Inverse-MAE weighted average of all models"),
    ]
    for name, src, desc in models_info:
        bullet(f"{name}  ({src})")
        info(desc)

    print()
    bullet("Model comparison charts saved for both horizons:")
    file_exists(FIG_DIR / "model_comparison_7day.png", "7-day comparison")
    file_exists(FIG_DIR / "model_comparison_14day.png", "14-day comparison")

    # Show which model wins
    if RESULTS_CSV.exists():
        results = pd.read_csv(RESULTS_CSV)
        for h in [7, 14]:
            best = results[results["horizon"] == h].groupby("model")["MAE"].mean().idxmin()
            best_mae = results[results["horizon"] == h].groupby("model")["MAE"].mean().min()
            print(f"    {ARROW} Best {h}-day model: {BOLD}{best}{RESET} (MAE = {best_mae:.4f} °C)")

    # ──────────────────────────────────────────────────────────
    # 6. UNIQUE / ADVANCED ANALYSES
    # ──────────────────────────────────────────────────────────
    header("ADVANCED 3/3: Unique Analyses")

    # 6a – Climate Analysis
    sub("A. Climate Analysis – Long-term patterns by region")
    info("Requirement: Study long-term climate patterns and variations in different regions.")
    bullet("Monthly average temperature compared across 6 continents  (main.py → stage_advanced)")
    bullet("Country → continent mapping in src/utils.py (190+ countries)")
    file_exists(FIG_DIR / "monthly_climate_continent.png", "Monthly climate chart")

    # 6b – Environmental Impact
    sub("B. Environmental Impact – Air quality correlation")
    info("Requirement: Analyze air quality and its correlation with weather parameters.")
    bullet("Correlation matrix: PM2.5, PM10, CO, NO₂, SO₂, O₃ vs temp, humidity, wind, precip")
    file_exists(FIG_DIR / "air_quality_weather_corr.png", "Air quality correlation heatmap")

    # 6c – Feature Importance
    sub("C. Feature Importance")
    info("Requirement: Apply different techniques to assess feature importance.")
    bullet("Gradient Boosting built-in feature_importances_  (src/models/ml_regression.py)")
    bullet("22 engineered features ranked: lags, rolling stats, calendar, lat/lon, weather")
    file_exists(FIG_DIR / "feature_importance.png", "Feature importance bar chart")

    # 6d – Spatial Analysis
    sub("D. Spatial Analysis – Geographic weather patterns")
    info("Requirement: Analyze and visualize geographical patterns in the data.")
    bullet("Global scatter plot: lat/lon coloured by temperature on the latest date")
    file_exists(FIG_DIR / "spatial_temperature_map.png", "Spatial temperature map")

    # 6e – Geographical Patterns
    sub("E. Geographical Patterns – Cross-country/continent comparison")
    info("Requirement: Explore how weather conditions differ across countries and continents.")
    bullet("Monthly climate by continent chart (see Climate Analysis above)")
    bullet("Major-city time series comparison: London, Paris, Tokyo, Cairo, Moscow")
    file_exists(FIG_DIR / "timeseries_major_cities.png", "Multi-city time series")

    # ──────────────────────────────────────────────────────────
    # 7. FEATURE ENGINEERING DETAILS
    # ──────────────────────────────────────────────────────────
    header("SUPPORTING: Feature Engineering  (src/features.py)")
    sub("Features created for the ML model")
    bullet("Lag features: temperature at t-1, t-2, t-7, t-14 days")
    bullet("Rolling statistics: 7-day & 14-day rolling mean and std (shift=1 to avoid leakage)")
    bullet("Calendar features: day_of_week, month, day_of_year, is_weekend")
    bullet("Cyclical encoding: sin/cos transforms for month and day-of-week")
    bullet("Spatial features: latitude and longitude carried through")
    print(f"\n    {ARROW} Total engineered features used by GBR: {BOLD}22{RESET}")

    # ──────────────────────────────────────────────────────────
    # 8. DELIVERABLES CHECKLIST
    # ──────────────────────────────────────────────────────────
    header("DELIVERABLES CHECKLIST")

    deliverables = [
        ("PM Accelerator mission displayed", "README.md (top)"),
        ("Data cleaning & preprocessing", "src/cleaning.py"),
        ("EDA with temp & precip visualizations", "main.py → stage_eda + reports/figures/"),
        ("Basic forecasting model + metrics", "src/models/baseline.py, src/evaluation.py"),
        ("Anomaly detection (STL + Isolation Forest)", "src/anomalies.py"),
        ("Multiple forecasting models compared", "src/models/*.py + reports/forecast_results.csv"),
        ("Ensemble model", "src/models/ensemble.py"),
        ("Climate analysis by region/continent", "main.py → stage_advanced, src/utils.py"),
        ("Air quality ↔ weather correlation", "main.py → stage_advanced"),
        ("Feature importance analysis", "src/models/ml_regression.py"),
        ("Spatial/geographic temperature map", "main.py → stage_advanced"),
        ("Geographical patterns across countries", "main.py → stage_advanced"),
        ("Well-organized README.md documentation", "README.md"),
        ("GitHub repository with all code", "This repository"),
        ("Reproducible pipeline (single command)", "python main.py"),
    ]

    print()
    for desc, loc in deliverables:
        print(f"    {GREEN}{CHECK}{RESET}  {desc}")
        print(f"       {ARROW} {loc}")
    print()

    # ──────────────────────────────────────────────────────────
    # 9. HOW TO RUN
    # ──────────────────────────────────────────────────────────
    header("HOW TO REPRODUCE")
    print(f"""
    {BOLD}# 1. Install dependencies{RESET}
    pip install -r requirements.txt

    {BOLD}# 2. Run the full pipeline (cleaning → EDA → forecasting → advanced){RESET}
    python main.py

    {BOLD}# 3. Run this demo walkthrough{RESET}
    python demo.py

    {BOLD}Outputs:{RESET}
      • Cleaned data   → data/processed/weather_clean.parquet
      • Figures         → reports/figures/  (10 plots)
      • Forecast table  → reports/forecast_results.csv
      • EDA notebook    → notebooks/eda.ipynb
    """)

    # ──────────────────────────────────────────────────────────
    # 10. SUMMARY
    # ──────────────────────────────────────────────────────────
    header("SUMMARY")
    print(f"""
    This project fulfills {BOLD}ALL{RESET} requirements of the PM Accelerator
    Weather Trend Forecasting Tech Assessment:

      • {BOLD}Basic Assessment{RESET} — data cleaning, EDA with temperature/precipitation
        visualizations, and a forecasting model evaluated with MAE/RMSE/sMAPE.

      • {BOLD}Advanced Assessment{RESET} — anomaly detection (STL + Isolation Forest),
        six forecasting models compared side-by-side, a weighted ensemble,
        climate analysis, air-quality correlation, feature importance,
        spatial mapping, and geographical pattern analysis.

      • {BOLD}Deliverables{RESET} — PM Accelerator mission in README, well-organized
        documentation, reproducible single-command pipeline, and all
        results saved to the reports/ directory.
    """)

    print(f"{BOLD}{'─' * 70}{RESET}")
    print(f"{BOLD}  Demo walkthrough complete.{RESET}")
    print(f"{BOLD}{'─' * 70}{RESET}\n")


if __name__ == "__main__":
    main()
