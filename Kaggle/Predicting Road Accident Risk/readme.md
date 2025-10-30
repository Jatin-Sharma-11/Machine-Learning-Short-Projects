# 🚦 Predicting Road Accident Risk - Kaggle Playground S5E10

![Kaggle Competition](https://img.shields.io/badge/Kaggle-Playground%20S5E10-blue.svg)
![Evaluation Metric](https://img.shields.io/badge/Metric-RMSE-yellow.svg)
![Language](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20Scikit--Learn%20%7C%20LGBM-orange.svg)

This repository contains my solution for the **Kaggle Playground Series - Season 5, Episode 10: Predicting Road Accident Risk**.

---

## 🏆 Competition Overview

The goal of this competition is to predict the likelihood of road accidents (`accident_risk`) based on various road-related features.

This competition is the first part of a two-part challenge in collaboration with **Stack Overflow**. Completing both this challenge and the subsequent web application challenge on Stack Overflow earns a special “Code Scientist” badge.

* **Competition Page:** [https://www.kaggle.com/competitions/playground-series-s5e10](https://www.kaggle.com/competitions/playground-series-s5e10)
* **Deadline:** October 31, 2025

---

## 📉 Evaluation Metric

Submissions are evaluated using the **Root Mean Squared Error (RMSE)**. A lower RMSE score indicates a more accurate model.

> $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

Where $y_i$ is the actual `accident_risk` and $\hat{y}_i$ is the predicted `accident_risk`.

---

## 📁 The Dataset

The dataset is synthetically generated but based on a real-world problem. It includes the following files:

* `train.csv`: The training set, including the `id`, features, and the target variable (`accident_risk`).
* `test.csv`: The test set, containing `id` and features, but without the target.
* `sample_submission.csv`: A file showing the correct format for submission.

---

## 🚀 My Solution & Approach

This section details the methodology used to preprocess the data, engineer features, and train the final model.

### 1. Exploratory Data Analysis (EDA)
*(Describe your findings here. What did you learn about the data?)*
* Checked for missing values (e.g., "There were no missing values in the dataset.").
* Analyzed the distribution of the target variable, `accident_risk`.
* Examined correlations between features and the target.
* Visualized distributions of categorical and numerical features.

### 2. Data Preprocessing
*(How did you clean the data?)*
* **Scaling:** Applied `StandardScaler` / `MinMaxScaler` to the numerical features.
* **Encoding:** Used `OneHotEncoder` / `LabelEncoder` for categorical features like `road_type` or `weather_conditions`.

### 3. Feature Engineering
*(What new features did you create to improve the model?)*
* **Interaction Features:** (e.g., `speed_limit * traffic_density`).
* **Polynomial Features:** (e.g., `feature^2`).
* **Risk Ratios:** (e.g., `feature_A / feature_B`).

### 4. Modeling
*(What model(s) did you use?)*
I used a **LightGBM (Light Gradient Boosting Machine)** model, which is well-suited for tabular data due to its speed and performance.

* **Model:** `LGBMRegressor`
* **Validation Strategy:** 5-Fold Cross-Validation (`KFold(n_splits=5, shuffle=True)`). This helps ensure the model generalizes well and avoids overfitting.
* **Hyperparameter Tuning:** (e.g., "I used Optuna / GridSearchCV to find the optimal parameters...").
    * `n_estimators`: 2000
    * `learning_rate`: 0.01
    * `num_leaves`: 31
    * *(...add your other key parameters...)*

The final test predictions are an average of the predictions from all 5 folds.

---

## 🛠️ Tools & Libraries

* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Preprocessing:** `sklearn.preprocessing` (StandardScaler, OneHotEncoder)
* **Modeling:** `lightgbm`
* **Validation:** `sklearn.model_selection` (KFold)
* **Metric:** `sklearn.metrics` (mean_squared_error)

---

## 🏁 Final Result

* **Public Leaderboard Score:** `[Your Public RMSE]`
* **Private Leaderboard Score:** `[Your Private RMSE]`

---

## 📄 Submission File

The model outputs a `submission.csv` file with predictions clipped between 0 and 1, as required.

```csv
id,accident_risk
517754,0.352
517755,0.992
517756,0.021
etc.

🔗 Citation
Walter Reade and Elizabeth Park. Predicting Road Accident Risk. https://kaggle.com/competitions/playground-series-s5e10, 2025. Kaggle.
