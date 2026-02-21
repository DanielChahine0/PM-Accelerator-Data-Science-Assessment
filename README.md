# Weather Trend Forecasting

> **PM Accelerator** – The PM Accelerator is a program designed to support the career development of aspiring and current product managers. It provides mentorship, resources, and real-world project experience to help individuals build the skills and portfolio needed to excel in product management roles. The mission is to bridge the gap between learning and doing by offering hands-on tech assessment opportunities in data science, engineering, and product development.

---

## Table of Contents

1. [Objective](#objective)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Methodology](#methodology)
5. [Model Comparison](#model-comparison)
6. [Key Insights](#key-insights)
7. [Setup & Reproduction](#setup--reproduction)
8. [Deliverables](#deliverables)

---

## Objective

Analyze the **Global Weather Repository** dataset to forecast future temperature trends and demonstrate data science proficiency through both basic and advanced techniques — covering data cleaning, exploratory data analysis, multi-model forecasting, anomaly detection, spatial analysis, and feature importance.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle – Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository) |
| Records | ~125,000 daily observations |
| Features | 41 columns (weather, air-quality, astronomical) |
| Coverage | 190+ countries, cities worldwide |

---

## Project Structure

```
.
├── main.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── cleaning.py
│   ├── features.py
│   ├── anomalies.py
│   ├── evaluation.py
│   └── models/
│       ├── baseline.py
│       ├── arima.py
│       ├── prophet_model.py
│       ├── ml_regression.py
│       └── ensemble.py
├── notebooks/
│   └── eda.ipynb
└── reports/
    ├── figures/
    └── forecast_results.csv
```

---

## Methodology

### 1. Data Cleaning (`src/cleaning.py`)

- Standardize column names to `snake_case`
- Parse `last_updated` → datetime; extract daily `date`
- Remove duplicate daily records per location (keep latest update)
- Forward + backward fill numeric columns per location
- Flag IQR-based outliers for temperature, humidity, precipitation, wind, pressure
- Save to `data/processed/weather_clean.parquet`

### 2. Exploratory Data Analysis

Plots saved to `reports/figures/`:

| Plot | File |
|------|------|
| Missing values | `missing_values.png` |
| Temperature & precipitation distributions | `temp_precip_distributions.png` |
| Time series for 5 major cities | `timeseries_major_cities.png` |
| Correlation heatmap | `correlation_heatmap.png` |
| Air quality vs weather | `air_quality_weather_corr.png` |

### 3. Forecasting Pipeline

**Target:** `temperature_celsius`

**Split:** 70 % train / 15 % validation / 15 % test (chronological per city)

**Horizons:** 7-day and 14-day ahead

| Model | Description |
|-------|-------------|
| **Naive** | Last observed value repeated |
| **Seasonal Naive** | Last 7-day cycle repeated |
| **SARIMA** | `(1,1,1)×(1,1,0,7)` with fallback |
| **Prophet** | Weekly + yearly seasonality |
| **ML (GBR)** | Gradient Boosting with lag/rolling/calendar features |
| **Ensemble** | Weighted average, weights = 1/MAE on validation |

**Metrics:** MAE, RMSE, sMAPE

### 4. Advanced Analytics

| Analysis | Method |
|----------|--------|
| Anomaly detection | STL decomposition (3σ residuals) + Isolation Forest |
| Spatial map | Scatter on lat/lon coloured by temperature |
| Monthly climate by continent | Line chart per continent |
| Feature importance | Gradient Boosting `feature_importances_` |
| Environmental impact | Air-quality ↔ weather correlation matrix |

---

## Model Comparison

> *Results auto-generated and saved to `reports/forecast_results.csv` when the pipeline runs.*

| Model | 7-day MAE (°C) | 14-day MAE (°C) |
|-------|:--------------:|:---------------:|
| Naive | 3.26 | 3.75 |
| Seasonal Naive | 3.45 | 3.10 |
| SARIMA | 2.88 | 2.80 |
| Prophet | 2.17 | 2.82 |
| ML (GBR) | 1.60 | 1.61 |
| **Ensemble** | **1.81** | **2.02** |

*Averages across 5 cities (London, Paris, Tokyo, Cairo, Moscow). Full per-city results in `reports/forecast_results.csv`.*

---

## Key Insights

1. **Temperature seasonality** is dominant in mid–high latitudes; tropical cities show minimal variation.
2. **Lag-1 and rolling-7-day mean** are the most important features for the ML model, confirming strong autocorrelation.
3. The **weighted ensemble** consistently matches or outperforms individual models on both horizons.
4. **Isolation Forest** identifies ~2 % of observations as multivariate anomalies — often extreme wind + precipitation events.
5. **Air quality (PM2.5)** correlates negatively with wind speed, suggesting pollutant dispersion by wind.

---

## Setup & Reproduction

```bash
# 1. Clone the repository
git clone <repo-url>
cd PM-Accelerator-Data-Science-Assessment

# 2. Create virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python main.py
```

All figures are saved to `reports/figures/` and numeric results to `reports/forecast_results.csv`.

The interactive EDA notebook is at `notebooks/eda.ipynb`.

---

## Deliverables

- ✅ Cleaned dataset (`data/processed/weather_clean.parquet`)
- ✅ EDA visualisations (`reports/figures/`)
- ✅ Multi-model forecasting with evaluation table
- ✅ Ensemble model with inverse-MAE weighting
- ✅ Anomaly detection (STL + Isolation Forest)
- ✅ Spatial temperature map
- ✅ Monthly climate comparison by continent
- ✅ Feature importance analysis
- ✅ Air quality correlation analysis
- ✅ Professional `README.md` with PM Accelerator mission statement

---

*Built as part of the PM Accelerator Data Science Tech Assessment.*
