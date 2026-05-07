# ☀️ Solar Power Forecasting — 1 to 10 Days Ahead

**A Comparative Study of ML & Deep Learning Models for Multi-Horizon Solar Energy Prediction**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

> An end-to-end machine learning pipeline that trains, evaluates, and compares four forecasting models — **Linear Regression, Polynomial Regression, Random Forest, and an Artificial Neural Network (ANN)** — for predicting solar power plant output at 1, 3, 5, and 10-day forecast horizons.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Pipeline](#-pipeline)
- [Models](#-models)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Future Work](#-future-work)
- [Citation](#-citation)
- [License](#-license)

---

## 🔍 Overview

Accurate solar power forecasting is essential for grid operators, energy traders, and renewable integration planners. Short-horizon forecasts (1 day) support daily dispatch decisions; medium-horizon forecasts (3–10 days) enable maintenance scheduling, reserve planning, and storage optimization.

This project builds a **systematic, reproducible forecasting pipeline** that:

- Loads real-world solar plant data directly from a public dataset
- Performs correlation analysis to identify the most predictive meteorological features
- Trains four models of increasing complexity on an 80/20 chronological split
- Evaluates each model across four forecast horizons (1, 3, 5, 10 days) using RMSE and R²
- Visualizes forecast traces and performance comparisons across all models and horizons

---

## 📦 Dataset

| Property | Value |
|---|---|
| **Source** | [Solar Power Plant Data — GitHub](https://raw.githubusercontent.com/rayhaneeeruet/Solar-Data/refs/heads/main/Solar%20Power%20Plant%20Data.csv) |
| **Format** | CSV, loaded directly via URL (no download needed) |
| **Target** | `SystemProduction` — hourly solar power output (kW) |
| **Features used** | `Radiation`, `AirTemperature`, `WindSpeed`, `RelativeAirHumidity` |

### Feature Correlations

| Feature | Correlation with SystemProduction |
|---|---|
| `Radiation` | **+0.786** (strongest predictor) |
| `AirTemperature` | Moderate positive |
| `WindSpeed` | Weak positive |
| `RelativeAirHumidity` | **−0.545** (negative — cloud cover proxy) |

> The dataset is loaded automatically at runtime — no manual download is required.

---

## 🔧 Pipeline

The notebook follows a 10-step structured pipeline:

```
Raw CSV (URL)
      │
      ▼
┌─────────────────────────────────────┐
│  1. Load Dataset                     │
│  2. Feature Selection (4 variables)  │
│  3. Correlation Matrix               │
│  4. Normalize (StandardScaler)       │
│  5. Train/Test Split (80/20)         │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌────────────┐   ┌──────────────────────┐
│  Training  │   │  Evaluation Loop     │
│            │   │                      │
│  • LR      │   │  Horizons: 1,3,5,10  │
│  • PR      │   │  Metrics: RMSE, R²   │
│  • RF      │   │                      │
│  • ANN     │   └──────────────────────┘
└────────────┘
       │
       ▼
  Plots & Results Table
```

**Key design choice — chronological split:** `shuffle=False` is used in `train_test_split` to preserve temporal order and prevent data leakage, which is critical for time series forecasting.

---

## 🤖 Models

### 1. Linear Regression (Baseline)
Standard OLS regression with z-score standardised features. Provides a linear baseline to quantify non-linear improvement from more complex models.

### 2. Polynomial Regression (degree = 2)
Extends linear regression with all second-degree feature interactions via `PolynomialFeatures(degree=2)`. Captures nonlinear relationships between meteorological variables and power output without requiring a neural network.

### 3. Random Forest Regressor
```
n_estimators : 100
max_depth    : 20
random_state : 42
```
Ensemble of 100 decision trees with bootstrap aggregation. Handles nonlinear feature interactions natively and provides implicit feature importance rankings. Best suited for capturing the complex, threshold-like behavior of solar irradiance.

### 4. Artificial Neural Network (ANN)
```
Input → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1)
Optimizer : Adam
Loss      : MSE
Epochs    : 50  |  Batch size : 32
```
A fully-connected feed-forward network. Learns hierarchical nonlinear representations of the meteorological inputs without any explicit feature engineering.

---

## 📊 Results

### RMSE Comparison Across Forecast Horizons

| Model | 1 Day | 3 Days | 5 Days | 10 Days |
|---|---|---|---|---|
| Linear Regression | — | — | — | — |
| Polynomial Regression | — | — | — | — |
| **Random Forest** | — | — | — | — |
| ANN | — | — | — | — |

> Run the notebook to populate this table with your results. Values vary with dataset version and random seeds.

The notebook also produces:
- **Correlation heatmap** — meteorological feature relationships
- **Forecast vs. actual traces** — 1, 3, 5, and 10-day horizon plots for Random Forest
- **RMSE and R² bar charts** — side-by-side model comparison at horizon 1
- **Full RMSE table** — all models × all horizons

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/solar-power-forecasting.git
cd solar-power-forecasting
pip install -r requirements.txt
```

### Dependencies

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
tensorflow>=2.12
jupyter>=1.0
```

Or install manually:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
```

---

## 🚀 Usage

### Run in Jupyter

```bash
jupyter notebook Solar_Power_Forecasting_10_Days_Ahead.ipynb
```

Then run all cells: **Kernel → Restart & Run All**

The notebook will:
1. Automatically download the dataset from GitHub
2. Train all four models
3. Evaluate across all forecast horizons
4. Display all plots and the final results table inline

### Run in Google Colab

Upload `Solar_Power_Forecasting_10_Days_Ahead.ipynb` to [Google Colab](https://colab.research.google.com) and run all cells. All dependencies are pre-installed in the Colab environment.

---

## 🗂️ Project Structure

```
solar-power-forecasting/
│
├── Solar_Power_Forecasting_10_Days_Ahead.ipynb   # Main notebook
├── requirements.txt                               # Python dependencies
└── README.md                                      # This file
```

The dataset is loaded at runtime via URL — no `data/` folder is needed.

---

## 🔭 Future Work

- **HGBR / XGBoost** — gradient-boosted frameworks with native missing-value handling and stronger tabular performance
- **LSTM / Transformer** — sequence-aware architectures that exploit temporal autocorrelation in irradiance data
- **Feature engineering** — cyclic time encoding (sin/cos of hour, day-of-year), lag features, rolling means
- **Night-time zero filtering** — excluding zero-production hours reduces distortion of R² and RMSE metrics
- **Log-transform target** — reduces right-skew in `SystemProduction`, improving loss function behavior
- **Probabilistic forecasting** — quantile regression or Monte Carlo dropout for uncertainty bounds
- **Multi-site generalization** — transfer learning across geographically distinct solar plants

---

## 📄 Citation

If you use this project in your research, please cite:

```bibtex
@misc{solar_forecasting_2025,
  author       = {[Your Name]},
  title        = {Solar Power Forecasting: 1 to 10 Days Ahead},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/YOUR_USERNAME/solar-power-forecasting}
}
```

---

## 📬 Contact

**[Your Name]**
Department of Electrical Engineering, [University Name]
📧 [your.email@university.edu]
🔗 [LinkedIn Profile](https://linkedin.com/in/yourprofile)

---

## 📃 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

⭐ **If this project was useful to you, please consider giving it a star!** ⭐

</div>
