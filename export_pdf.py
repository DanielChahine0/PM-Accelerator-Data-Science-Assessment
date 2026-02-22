#!/usr/bin/env python
"""
export_pdf.py – Export the demo walkthrough + figures as a polished PDF report.

Run:
    python export_pdf.py

Output:
    reports/PM_Accelerator_Weather_Report.pdf
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from fpdf import FPDF

# ── project paths ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "reports" / "figures"
RESULTS_CSV = ROOT / "reports" / "forecast_results.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "weather_clean.parquet"
OUTPUT_PDF = ROOT / "reports" / "PM_Accelerator_Weather_Report.pdf"


# ── colour palette (RGB) ─────────────────────────────────────
C_PRIMARY = (25, 60, 120)       # dark blue – titles
C_SECTION = (40, 90, 160)       # medium blue – section headers
C_CHECK   = (34, 139, 34)       # green – check marks
C_BODY    = (40, 40, 40)        # near-black – body text
C_GREY    = (100, 100, 100)     # grey – captions / secondary
C_LIGHT   = (230, 237, 246)     # light blue – table header bg


def _sanitize(text: str) -> str:
    """Replace Unicode characters that latin-1 cannot encode."""
    replacements = {
        "\u2013": "-",   # en-dash
        "\u2014": "--",  # em-dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u2192": "->",  # →
        "\u2713": "v",   # ✓ (we'll use a v; overridden below with symbol font)
        "\u2715": "x",
        "\u00d7": "x",   # ×
        "\u03c3": "sigma",
        "\u2194": "<->",
        "\u00b0": " ",   # degree
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    # Final fallback: encode to latin-1, replacing anything left
    return text.encode("latin-1", errors="replace").decode("latin-1")


class ReportPDF(FPDF):
    """Custom FPDF subclass with header/footer and helper methods."""

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*C_GREY)
        self.cell(0, 6, "PM Accelerator  |  Weather Trend Forecasting  |  Tech Assessment Report", align="C")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*C_GREY)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # ── convenience helpers ───────────────────────────────────

    def section_title(self, text: str):
        """Large coloured section heading with a ruled line."""
        self.ln(4)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*C_PRIMARY)
        self.cell(0, 9, _sanitize(text))
        self.ln(9)
        self.set_draw_color(*C_SECTION)
        self.set_line_width(0.6)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def sub_heading(self, text: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*C_SECTION)
        self.cell(0, 7, _sanitize(text))
        self.ln(7)

    def body(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*C_BODY)
        self.multi_cell(0, 5.5, _sanitize(text))
        self.ln(1)

    def bullet(self, text: str, indent: float = 10):
        x0 = self.get_x()
        self.set_x(x0 + indent)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*C_CHECK)
        self.cell(5, 5.5, ">")  # bullet marker
        self.set_text_color(*C_BODY)
        self.multi_cell(0, 5.5, _sanitize(text))
        self.ln(0.5)

    def info_line(self, text: str, indent: float = 18):
        x0 = self.get_x()
        self.set_x(x0 + indent)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(*C_GREY)
        self.multi_cell(0, 5, _sanitize(text))
        self.set_x(x0)
        self.ln(0.5)

    def arrow_line(self, text: str, indent: float = 10):
        x0 = self.get_x()
        self.set_x(x0 + indent)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*C_BODY)
        self.cell(5, 5.5, "->")  # arrow
        self.multi_cell(0, 5.5, _sanitize(text))
        self.ln(0.5)

    def add_figure(self, path: Path, caption: str = "", max_w: float = 170):
        """Embed a PNG image centred on the page with an optional caption."""
        if not path.exists():
            self.info_line(f"[Figure not found: {path.name}]")
            return
        # Check if enough space; otherwise add page
        if self.get_y() > 200:
            self.add_page()
        x = (self.w - max_w) / 2
        self.image(str(path), x=x, w=max_w)
        if caption:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*C_GREY)
            self.cell(0, 5, caption, align="C")
            self.ln(6)

    def metrics_table(self, df: pd.DataFrame):
        """Render a small metrics table."""
        col_widths = [35, 18, 18, 18, 18]
        headers = ["Model", "Horizon", "MAE", "RMSE", "sMAPE"]

        # header row
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*C_LIGHT)
        self.set_text_color(*C_PRIMARY)
        for w, h in zip(col_widths, headers):
            self.cell(w, 6, h, border=1, fill=True, align="C")
        self.ln()

        # data rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*C_BODY)
        for _, row in df.iterrows():
            vals = [
                str(row["model"]),
                str(int(row["horizon"])),
                f"{row['MAE']:.4f}",
                f"{row['RMSE']:.4f}",
                f"{row['sMAPE']:.4f}",
            ]
            for w, v in zip(col_widths, vals):
                self.cell(w, 5.5, v, border=1, align="C")
            self.ln()
        self.ln(3)


# ==============================================================
# BUILD THE PDF
# ==============================================================

def build_report():
    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Title page ────────────────────────────────────────────
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(*C_PRIMARY)
    pdf.cell(0, 14, "Weather Trend Forecasting", align="C")
    pdf.ln(16)
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(*C_SECTION)
    pdf.cell(0, 10, "PM Accelerator  -  Tech Assessment Report", align="C")
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(*C_GREY)
    pdf.cell(0, 8, "Assessment Requirements Demo Walkthrough", align="C")
    pdf.ln(10)
    pdf.cell(0, 8, "Daniel Chahine", align="C")
    pdf.ln(30)

    # Mission statement
    readme = (ROOT / "README.md").read_text()
    mission = ""
    for line in readme.splitlines():
        if line.startswith("> **PM Accelerator**"):
            mission = line[2:]
            break
    if mission:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*C_BODY)
        pdf.set_x(30)
        pdf.multi_cell(pdf.w - 60, 6, _sanitize(mission))

    # ── Page 2+: content ──────────────────────────────────────
    pdf.add_page()

    # ============ BASIC 1/3 ============
    pdf.section_title("BASIC 1/3: Data Cleaning & Preprocessing")
    pdf.sub_heading("Requirement")
    pdf.body("Handle missing values, outliers, and normalize data.")
    pdf.sub_heading("How it is met  (src/cleaning.py)")

    pdf.bullet("Column standardization: names to snake_case  (standardize_columns)")
    pdf.info_line("Ensures consistent access; e.g. 'Last Updated' -> 'last_updated'.")
    pdf.bullet("Datetime parsing: last_updated -> datetime + daily date column  (parse_datetime)")
    pdf.info_line("Uses the 'lastupdated' feature as required for time-series analysis.")
    pdf.bullet("Duplicate removal: keep latest record per (location, country, date)  (remove_daily_duplicates)")
    pdf.bullet("Missing-value handling: forward + backward fill per location  (fill_missing_per_location)")
    pdf.info_line("Grouped by location_name so fills respect each city's own history.")
    pdf.bullet("Outlier detection: IQR-based flag columns for 5 key variables  (flag_iqr_outliers)")
    pdf.info_line("Columns flagged: temperature, humidity, precipitation, wind, pressure.")
    pdf.info_line("Uses 1.5x IQR rule; outliers are flagged but kept, not removed.")
    pdf.bullet("Output saved as Parquet for fast reload  (data/processed/weather_clean.parquet)")

    if CLEAN_PATH.exists():
        df = pd.read_parquet(CLEAN_PATH)
        pdf.ln(2)
        pdf.arrow_line(f"Cleaned dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        missing = df.isnull().sum().sum()
        pdf.arrow_line(f"Remaining missing values: {missing:,}")
        outlier_cols = [c for c in df.columns if c.endswith("_outlier")]
        if outlier_cols:
            n_out = df[outlier_cols].any(axis=1).sum()
            pdf.arrow_line(f"Rows flagged as outliers (any column): {n_out:,}")

    # ============ BASIC 2/3 ============
    pdf.add_page()
    pdf.section_title("BASIC 2/3: Exploratory Data Analysis (EDA)")
    pdf.sub_heading("Requirement")
    pdf.body("Perform basic EDA to uncover trends, correlations, and patterns.\n"
             "Generate visualizations for temperature and precipitation.")
    pdf.sub_heading("How it is met  (main.py -> stage_eda)")

    eda_figs = [
        ("missing_values.png",           "Missing-value bar chart per column"),
        ("temp_precip_distributions.png", "Temperature & precipitation histograms (REQUIRED)"),
        ("timeseries_major_cities.png",   "Daily temperature time series for 5 major cities"),
        ("correlation_heatmap.png",       "Correlation heatmap of key numeric features"),
    ]
    for fname, desc in eda_figs:
        status = "exists" if (FIG_DIR / fname).exists() else "missing"
        pdf.bullet(f"{desc}  [{status}]")

    pdf.bullet("Interactive notebook at notebooks/eda.ipynb for deeper exploration")

    # Embed key EDA figures
    for fname, caption in [
        ("temp_precip_distributions.png", "Figure: Temperature & Precipitation Distributions"),
        ("correlation_heatmap.png",       "Figure: Correlation Heatmap"),
        ("timeseries_major_cities.png",   "Figure: Daily Temperature - Major Cities"),
    ]:
        path = FIG_DIR / fname
        if path.exists():
            pdf.ln(3)
            pdf.add_figure(path, caption)

    # ============ BASIC 3/3 ============
    pdf.add_page()
    pdf.section_title("BASIC 3/3: Model Building & Evaluation")
    pdf.sub_heading("Requirement")
    pdf.body("Build a basic forecasting model and evaluate its performance using "
             "different metrics. Use lastupdated feature for time series analysis.")
    pdf.sub_heading("How it is met")

    pdf.bullet("Time feature: 'last_updated' is parsed in cleaning.py -> 'date' column")
    pdf.info_line("All time-series splits and forecasts use this date column.")
    pdf.bullet("Chronological train/val/test split: 70 / 15 / 15 %  (main.py -> _chrono_split)")
    pdf.info_line("Prevents data leakage by never training on future data.")
    pdf.bullet("Baseline models: Naive & Seasonal Naive  (src/models/baseline.py)")
    pdf.info_line("Naive repeats last value; Seasonal Naive repeats last 7-day cycle.")
    pdf.bullet("Evaluation metrics: MAE, RMSE, sMAPE  (src/evaluation.py)")
    pdf.info_line("Three complementary metrics give a well-rounded view of accuracy.")
    pdf.bullet("Forecast horizons: 7-day and 14-day ahead")

    if RESULTS_CSV.exists():
        results = pd.read_csv(RESULTS_CSV)
        avg = results.groupby(["model", "horizon"])[["MAE", "RMSE", "sMAPE"]].mean().round(4).reset_index()
        pdf.ln(3)
        pdf.sub_heading("Average Metrics Across 5 Cities")
        pdf.metrics_table(avg)

    # ============ ADVANCED 1/3 ============
    pdf.add_page()
    pdf.section_title("ADVANCED 1/3: Anomaly Detection")
    pdf.sub_heading("Requirement")
    pdf.body("Implement anomaly detection to identify and analyze outliers.")
    pdf.sub_heading("How it is met  (src/anomalies.py)")

    pdf.bullet("STL Decomposition  (stl_anomaly_detection)")
    pdf.info_line("Decomposes time series into trend + seasonal + residual components.")
    pdf.info_line("Flags residuals exceeding 3 sigma as anomalies.")
    pdf.bullet("Isolation Forest  (isolation_forest_anomalies)")
    pdf.info_line("Unsupervised multivariate anomaly detection on temperature, humidity,")
    pdf.info_line("precipitation, wind speed, and pressure. Contamination = 2%.")
    pdf.bullet("IQR outlier flags already applied during data cleaning (Basic req.)")

    stl_fig = FIG_DIR / "stl_anomaly_detection.png"
    if stl_fig.exists():
        pdf.ln(3)
        pdf.add_figure(stl_fig, "Figure: STL Anomaly Detection")

    # ============ ADVANCED 2/3 ============
    pdf.add_page()
    pdf.section_title("ADVANCED 2/3: Multiple Models + Ensemble")
    pdf.sub_heading("Requirement")
    pdf.body("Build and compare multiple forecasting models.\n"
             "Create an ensemble of models to improve forecast accuracy.")
    pdf.sub_heading("How it is met")

    models = [
        ("Naive Baseline  (src/models/baseline.py)",      "Last observed value repeated"),
        ("Seasonal Naive  (src/models/baseline.py)",       "Last 7-day cycle repeated"),
        ("SARIMA  (src/models/arima.py)",                  "(1,1,1)x(1,1,0,7) with automatic fallback"),
        ("Prophet  (src/models/prophet_model.py)",         "Weekly + yearly seasonality via Facebook Prophet"),
        ("ML Gradient Boosting  (src/models/ml_regression.py)", "GBR with 22 engineered features"),
        ("Weighted Ensemble  (src/models/ensemble.py)",    "Inverse-MAE weighted average of all models"),
    ]
    for name, desc in models:
        pdf.bullet(name)
        pdf.info_line(desc)

    pdf.ln(2)
    pdf.bullet("Model comparison charts saved for both horizons:")

    if RESULTS_CSV.exists():
        results = pd.read_csv(RESULTS_CSV)
        for h in [7, 14]:
            best = results[results["horizon"] == h].groupby("model")["MAE"].mean().idxmin()
            best_mae = results[results["horizon"] == h].groupby("model")["MAE"].mean().min()
            pdf.arrow_line(f"Best {h}-day model: {best} (MAE = {best_mae:.4f} C)")

    for fname, caption in [
        ("model_comparison_7day.png",  "Figure: 7-Day Model Comparison"),
        ("model_comparison_14day.png", "Figure: 14-Day Model Comparison"),
    ]:
        path = FIG_DIR / fname
        if path.exists():
            pdf.ln(3)
            pdf.add_figure(path, caption)

    # ============ ADVANCED 3/3 ============
    pdf.add_page()
    pdf.section_title("ADVANCED 3/3: Unique Analyses")

    # A – Climate Analysis
    pdf.sub_heading("A. Climate Analysis - Long-term patterns by region")
    pdf.body("Requirement: Study long-term climate patterns and variations in different regions.")
    pdf.bullet("Monthly average temperature compared across 6 continents  (main.py -> stage_advanced)")
    pdf.bullet("Country -> continent mapping in src/utils.py (190+ countries)")
    fig_climate = FIG_DIR / "monthly_climate_continent.png"
    if fig_climate.exists():
        pdf.ln(2)
        pdf.add_figure(fig_climate, "Figure: Monthly Climate by Continent")

    # B – Environmental Impact
    pdf.sub_heading("B. Environmental Impact - Air quality correlation")
    pdf.body("Requirement: Analyze air quality and its correlation with weather parameters.")
    pdf.bullet("Correlation matrix: PM2.5, PM10, CO, NO2, SO2, O3 vs temp, humidity, wind, precip")
    fig_aq = FIG_DIR / "air_quality_weather_corr.png"
    if fig_aq.exists():
        pdf.ln(2)
        pdf.add_figure(fig_aq, "Figure: Air Quality vs Weather Correlation")

    # C – Feature Importance
    pdf.add_page()
    pdf.sub_heading("C. Feature Importance")
    pdf.body("Requirement: Apply different techniques to assess feature importance.")
    pdf.bullet("Gradient Boosting built-in feature_importances_  (src/models/ml_regression.py)")
    pdf.bullet("22 engineered features ranked: lags, rolling stats, calendar, lat/lon, weather")
    fig_fi = FIG_DIR / "feature_importance.png"
    if fig_fi.exists():
        pdf.ln(2)
        pdf.add_figure(fig_fi, "Figure: Feature Importance")

    # D – Spatial Analysis
    pdf.sub_heading("D. Spatial Analysis - Geographic weather patterns")
    pdf.body("Requirement: Analyze and visualize geographical patterns in the data.")
    pdf.bullet("Global scatter plot: lat/lon coloured by temperature on the latest date")
    fig_sp = FIG_DIR / "spatial_temperature_map.png"
    if fig_sp.exists():
        pdf.ln(2)
        pdf.add_figure(fig_sp, "Figure: Spatial Temperature Map")

    # E – Geographical Patterns
    pdf.sub_heading("E. Geographical Patterns - Cross-country/continent comparison")
    pdf.body("Requirement: Explore how weather conditions differ across countries and continents.")
    pdf.bullet("Monthly climate by continent chart (see Climate Analysis above)")
    pdf.bullet("Major-city time series comparison: London, Paris, Tokyo, Cairo, Moscow")

    # ============ FEATURE ENGINEERING ============
    pdf.add_page()
    pdf.section_title("Feature Engineering  (src/features.py)")
    pdf.sub_heading("Features created for the ML model")
    pdf.bullet("Lag features: temperature at t-1, t-2, t-7, t-14 days")
    pdf.bullet("Rolling statistics: 7-day & 14-day rolling mean and std (shift=1 to avoid leakage)")
    pdf.bullet("Calendar features: day_of_week, month, day_of_year, is_weekend")
    pdf.bullet("Cyclical encoding: sin/cos transforms for month and day-of-week")
    pdf.bullet("Spatial features: latitude and longitude carried through")
    pdf.arrow_line("Total engineered features used by GBR: 22")

    # ============ DELIVERABLES CHECKLIST ============
    pdf.ln(6)
    pdf.section_title("Deliverables Checklist")

    deliverables = [
        ("PM Accelerator mission displayed",           "README.md (top)"),
        ("Data cleaning & preprocessing",              "src/cleaning.py"),
        ("EDA with temp & precip visualizations",      "main.py -> stage_eda + reports/figures/"),
        ("Basic forecasting model + metrics",          "src/models/baseline.py, src/evaluation.py"),
        ("Anomaly detection (STL + Isolation Forest)", "src/anomalies.py"),
        ("Multiple forecasting models compared",       "src/models/*.py + reports/forecast_results.csv"),
        ("Ensemble model",                             "src/models/ensemble.py"),
        ("Climate analysis by region/continent",       "main.py -> stage_advanced, src/utils.py"),
        ("Air quality <-> weather correlation",        "main.py -> stage_advanced"),
        ("Feature importance analysis",                "src/models/ml_regression.py"),
        ("Spatial/geographic temperature map",         "main.py -> stage_advanced"),
        ("Geographical patterns across countries",     "main.py -> stage_advanced"),
        ("Well-organized README.md documentation",     "README.md"),
        ("GitHub repository with all code",            "This repository"),
        ("Reproducible pipeline (single command)",     "python main.py"),
    ]
    for desc, loc in deliverables:
        pdf.bullet(f"{desc}  ->  {loc}")

    # ============ HOW TO REPRODUCE ============
    pdf.add_page()
    pdf.section_title("How to Reproduce")
    pdf.body("# 1. Install dependencies\n"
             "    pip install -r requirements.txt\n\n"
             "# 2. Run the full pipeline (cleaning -> EDA -> forecasting -> advanced)\n"
             "    python main.py\n\n"
             "# 3. Run the demo walkthrough\n"
             "    python demo.py\n\n"
             "# 4. Export this PDF report\n"
             "    python export_pdf.py")

    pdf.ln(4)
    pdf.sub_heading("Outputs")
    pdf.bullet("Cleaned data   ->  data/processed/weather_clean.parquet")
    pdf.bullet("Figures         ->  reports/figures/  (10 plots)")
    pdf.bullet("Forecast table  ->  reports/forecast_results.csv")
    pdf.bullet("EDA notebook    ->  notebooks/eda.ipynb")
    pdf.bullet("PDF report      ->  reports/PM_Accelerator_Weather_Report.pdf")

    # ============ SUMMARY ============
    pdf.ln(6)
    pdf.section_title("Summary")
    pdf.body(
        "This project fulfills ALL requirements of the PM Accelerator "
        "Weather Trend Forecasting Tech Assessment:\n\n"
        "Basic Assessment - data cleaning, EDA with temperature/precipitation "
        "visualizations, and a forecasting model evaluated with MAE/RMSE/sMAPE.\n\n"
        "Advanced Assessment - anomaly detection (STL + Isolation Forest), "
        "six forecasting models compared side-by-side, a weighted ensemble, "
        "climate analysis, air-quality correlation, feature importance, "
        "spatial mapping, and geographical pattern analysis.\n\n"
        "Deliverables - PM Accelerator mission in README, well-organized "
        "documentation, reproducible single-command pipeline, and all "
        "results saved to the reports/ directory."
    )

    # ── save ──────────────────────────────────────────────────
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT_PDF))
    print(f"\n  PDF saved to: {OUTPUT_PDF.relative_to(ROOT)}")
    print(f"  Pages: {pdf.page_no()}\n")


if __name__ == "__main__":
    build_report()
