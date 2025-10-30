# 📈 Hull Tactical - Market Prediction Challenge

![Kaggle Competition](https://img.shields.io/badge/Kaggle-Featured%20Competition-blue.svg)
![Metric](https://img.shields.io/badge/Metric-Custom%20Sharpe%20Ratio-yellow.svg)
![Prize](https://img.shields.io/badge/Prize-%24100,000-brightgreen.svg)
![Type](https://img.shields.io/badge/Type-Forecasting%20(Code)-orange.svg)

This repository contains my solution for the **Hull Tactical - Market Prediction** competition on Kaggle. The goal is not just to predict market returns, but to develop a robust *investment strategy* that outperforms the S&P 500 while managing volatility.

This competition directly challenges the **Efficient Market Hypothesis (EMH)** by providing proprietary data to see if a repeatable "edge" can be found.

* **Competition Page:** [https://www.kaggle.com/competitions/hull-tactical-market-prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction)

---

## 🎯 The Challenge: Beyond Prediction

This is a two-part challenge:

1.  **Predict Returns:** Forecast the excess returns of the S&P 500.
2.  **Create a Strategy:** For each trading day, determine an optimal allocation to the S&P 500. This allocation is a value between **0.0 (all cash) and 2.0 (2x leverage)**.

The primary constraint is to stay within a **120% volatility limit** relative to the S&P 500.

---

## 📊 Evaluation Metric

The competition uses a **variant of the Sharpe ratio**. This custom metric is designed to:
* Reward strategies that successfully outperform the market's return.
* Penalize strategies that take on excessive volatility (i.e., significantly more than the underlying market).

The official metric code is available on the competition's "Overview" page.

---

## 🗓️ Competition Timeline (Forecasting)

This is a **forecasting competition** with two distinct phases:

* **Phase 1: Training (Sept 16 - Dec 15, 2025)**
    * Build, train, and validate models using the provided historical data.
    * Final notebooks must be submitted by December 15, 2025.

* **Phase 2: Forecasting (Dec 2025 - June 16, 2026)**
    * Submitted notebooks will be **re-run periodically against new, live market data** as it becomes available.
    * The final leaderboard is determined by performance on this future data, rewarding robust and generalizable models.

---

## 💾 The Dataset

The data is a time-series dataset combining:
1.  **Public Market Information:** Standard daily market data.
2.  **Proprietary Dataset:** A unique, proprietary dataset provided by Hull Tactical, giving participants the raw material to uncover patterns others might miss.

---

## ⚙️ Code & Submission Requirements

This is a **Code Competition** with strict requirements:

* **Notebooks Only:** Submissions must be made via Kaggle Notebooks.
* **Time-Series API:** The notebook must use the provided `iter_test` API to generate predictions day-by-day and prevent "lookahead" bias.
* **Internet Access:** **Must be disabled** in the submitted notebook.
* **External Data:** **Allowed**, but must be freely and publicly available and added as a Kaggle dataset.
* **Runtime Limit (Training):** 8 hours.
* **Runtime Limit (Forecasting):** 9 hours (to accommodate the growing test set).

---

## 🚀 My Solution

*(This is where you detail your approach!)*

### 1. Data Preprocessing
* (e.g., How I handled time-series indexing, merged proprietary and public data, and normalized features.)

### 2. Feature Engineering
* (e.g., Created rolling window features like 20-day volatility, 50-day moving averages, MACD, RSI, and interaction terms between proprietary features.)

### 3. Validation Strategy
* (e.g., This is critical. I used a **walk-forward validation** strategy with a rolling origin to simulate the live forecasting environment and prevent overfitting.)

### 4. Modeling
* **(Part 1: The Prediction Model)** (e.g., I used an `LGBMRegressor` / `CatBoostRegressor` trained to predict the next day's excess return `y_true`.)
* **(Part 2: The Allocation Strategy)** (e.g., This is the "betting" logic. Based on the model's prediction $\hat{y}$, I used a scaled sigmoid or `tanh` function to map the prediction to the required `[0, 2]` allocation range. For example: `allocation = 1 + tanh(k * \hat{y})`, where `k` is a tuned scaling factor.)

### 5. Final Model
* (e.g., My final submission is an ensemble of 5 models trained on different time-series folds.)

---

## 🏁 Final Result

* **Public Leaderboard Score:** `[Your Public LB Score]`
* **Private Leaderboard Score:** `[Your Final Private LB Score After Forecasting Phase]`

---

## 🔗 Citation

> Blair Hull, Petra Bakosova, Laurent Lanteigne, Aishvi Shah, Euan C Sinclair, Petri Fast, Will Raj, Harold Janecek, Sohier Dane, and Addison Howard. Hull Tactical - Market Prediction. https://kaggle.com/competitions/hull-tactical-market-prediction, 2025. Kaggle.
