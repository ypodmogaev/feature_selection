# 🌾 Feature Selection for CME Wheat Futures Forecasting

**Systematic comparison of 15+ feature selection methods for weekly forecasting of CME Wheat Futures (ZW) using Lasso regression.**

> This repository contains the complete codebase and experimental results for identifying the most robust, accurate, and interpretable feature selection configuration. The research demonstrates that **Spearman rank correlation** provides the optimal production-ready balance: **2.49% MAPE**, **59.5% Directional Accuracy**, and **>2x higher feature stability** (Jaccard 0.46) compared to complex ML wrappers like XGBoost.

---

## 🎯 Project Goal

The main objective is to **isolate and evaluate the effect of feature selection** on a fixed baseline model (Lasso regression) across 198 distinct configurations. The pipeline balances four critical pillars:
1. **Predictive Accuracy** (MAPE, Directional Accuracy)
2. **Temporal Stability** (Jaccard similarity of selected features across rolling windows)
3. **Economic Diversity** (Shannon Entropy of selected factor groups)
4. **Computational Efficiency** (Execution time ranging from 0.015 to 113.1 minutes)

---

## 📊 Data

- **Timeframe:** 2011 – 2025 (704 weekly observations)
- **Target Variable:** Weekly Typical Price `(High + Low + Close) / 3`. *Chosen to smooth fixing noise and non-monetizable gaps, reflecting pure weekly price consensus.*
- **Base Factors:** 93 macroeconomic and market indicators across 6 categories.
- **Final Design Matrix:** **~3,200+ features** generated via strict no-leakage feature engineering.

### Factor Groups (Prefixes)
| Prefix | Group | Prefix | Group |
| :--- | :--- | :--- | :--- |
| `e` | Stock indexes | `fp` | Futures positions (CFTC COT) |
| `c` | Commodity indexes | `sd` | Supply & demand fundamentals |
| `$` | FX rates | `b` | Freight indexes |

---

## 🛠 Feature Engineering

Features are constructed globally, then sliced strictly by train/test boundaries to prevent data leakage. Twelve analytical categories are generated:

| # | Category | What it captures |
| :--- | :--- | :--- |
| 1 | 🧠 **Memory (Lags)** | `lag_1, 2, 3, 4, 12, 52` (inertia & annual seasonality) |
| 2 | 🚀 **Momentum** | `diff_1, 4, 52`, acceleration, smoothed momentum |
| 3 | ⚖️ **Relative Strength** | `pr/e`, `pr/c`, `pr/$`, `pr/b` (price cleaned from market beta) |
| 4 | 📊 **Channels & Levels** | Relative position, distance to SMA, SMA slope (12/52/104w) |
| 5 | 📉 **Volatility** | Spread, Z-score, annualized vol, vol % change |
| 6 | 🩺 **Trend Health** | Up-weeks count, Efficiency Ratio |
| 7 | 🔗 **Cross-Asset Correlations** | Rolling correlation (13w) between price and factors |
| 8 | 💥 **Distribution Tails** | Weeks since max, outlier flag (`|Z| > 2.5`) |
| 9 | 🌀 **Fractality** | Fractal dimension / Hurst exponent proxy |
| 10 | 📆 **Seasonality** | Expanding mean by month (strict no-leakage) |
| 11 | 🎚️ **Spread Dynamics** | Rate of convergence with benchmarks |
| 12 | 📏 **Stationarity** | Multi-period log returns (1w, 2w, 4w, 12w, 52w) |

*Plus synergy features: Volatility × Volume interaction, Momentum spreads, Historical channel breakouts, Cumulative position flow (COT).*

---

## 🔄 Methodology

### Rolling-Window Validation
- **Start period:** Week 264 (~5 years of history before the first forecast)
- **Rolling step:** 20 weeks
- **Training depths:** 2 years, 7 years, All available history
- **Horizon:** Next-week log return → inverse-transformed to price levels

### Strict No-Leakage Pipeline (per window)
```text
1. Slice dataset by current rolling period
2. Apply Feature Selection method (fit on TRAIN only)
3. StandardScaler (fit on TRAIN only)
4. LassoCV (TimeSeriesSplit, 150 alphas, eps=1e-15)
5. Inverse-transform log-returns → price levels
6. Compute multi-dimensional metrics & log execution time
