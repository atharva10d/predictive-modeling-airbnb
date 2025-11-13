# 📊 Predictive Modeling with Inside Airbnb

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-active-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

An end-to-end data science pipeline that uses the **Inside Airbnb** `listings.csv` to predict listing prices (regression) and Superhost status (classification), plus an advanced XGBoost probability model to identify high-potential non-Superhosts.

---

## 🔭 Table of contents
- [Project overview](#project-overview)
- [Goals](#goals)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Setup & installation](#setup--installation)
- [Usage](#usage)
  - [Run main pipeline (Tasks 1–4)](#run-main-pipeline-tasks-1-4)
  - [Run advanced XGBoost analysis (Task 5)](#run-advanced-xgboost-analysis-task-5)
- [Notebook summaries](#notebook-summaries)
- [Key techniques](#key-techniques)
- [Outputs & expected files](#outputs--expected-files)
- [Contributing](#contributing)
- [License](#license)

---

## 📝 Project overview

This repo provides a cleaned, reproducible workflow for:
- Data cleaning & feature engineering (including sentiment with VADER)
- Price prediction (regression)
- Superhost classification (binary classification)
- Advanced probability scoring (XGBoost) to surface high-potential non-Superhosts

---

## 🎯 Goals

- **Task 1 — Regression:** Predict listing price using location, amenities, property type, and engineered features.  
- **Task 2 — Classification:** Predict whether a host becomes a **Superhost**.  
- **Task 3 — Advanced:** Produce calibrated probability scores (XGBoost) and identify non-Superhosts with high Superhost potential.

> **Note:** Replace `listings.csv` with the Inside Airbnb file for your chosen country/city before running.

---

## 🛠 Tech stack

- **Core:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`  
- **Modeling:** `xgboost`, `sklearn` (Linear, Ridge, Lasso, ElasticNet, DecisionTree, SVC, LogisticRegression)  
- **NLP:** `nltk` (VADER)  
- **Imbalance handling:** `imblearn` (SMOTE), `scale_pos_weight` for XGBoost  
- **Explainability:** `shap`  
- **Environment:** `conda`, `jupyter` / `nbviewer`

---

## 📁 Project structure

airbnb-predictive-modeling/
│
├── code/
│ └── code.py # Runs advanced probability model (Task 5)
│
├── data/
│ ├── listings.csv # Raw dataset (add this file)
│ └── listings_cleaned.csv # Generated after preprocessing
│
├── notebooks/
│ ├── 001_data_preprocessing.ipynb
│ ├── 002_regression_model.ipynb
│ ├── 003_classification_model.ipynb
│ ├── 004_model_comparison.ipynb
│ └── 005_advanced_predictive_probability.ipynb
│
├── utils/
│ ├── init.py
│ ├── model_helpers.py # Model training & evaluation utilities
│ └── preprocessing.py # Data cleaning & feature engineering
│
├── visuals/ # Notebook output plots
│ ├── 1/
│ ├── 2/
│ ├── 3/
│ ├── 4/
│ └── 5/
│
├── main.py # Runs Tasks 1–4
├── requirements.txt
├── setup_structure.py
└── README.md

yaml
Copy code

---

## 🚀 Setup & installation

```bash
# 1. Clone
git clone https://[your-repository-url]/airbnb-predictive-modeling.git
cd airbnb-predictive-modeling

# 2. Create conda environment
conda create -n airbnb_env python=3.10 -y
conda activate airbnb_env

# 3. Install dependencies
pip install -r requirements.txt
🏃 Usage
Run main pipeline (Tasks 1–4)
This runs preprocessing, regression and classification comparisons.

bash
Copy code
python main.py
What this does

Loads data/listings.csv

Cleans and engineers features → saves data/listings_cleaned.csv

Trains & compares 4 regression models (Linear, Ridge, Lasso, ElasticNet)

Trains & compares 4 classification models (Logistic, DecisionTree, SVM, GaussianNB) with SMOTE + tuning

Run advanced XGBoost analysis (Task 5)
bash
Copy code
python code/code.py
What this does

Loads data/listings_cleaned.csv

Applies a robust 5E-1 string hotfix for corrupted numeric values

Trains/tunes an XGBoost probability model (uses scale_pos_weight)

Produces high-confidence probability scores and lists non-Superhosts with >80% predicted probability

🔬 Notebook summaries
001_data_preprocessing.ipynb
Read ~79 columns, dropped 30+ irrelevant ones

Engineered:

host_duration_days

amenities_count

description_sentiment (VADER)

Handled outliers/missing values and fixed recurring 5E-1 string error

Observed class imbalance in host_is_superhost

002_regression_model.ipynb
ColumnTransformer (scaling + OHE)

Models tried: Linear, Ridge, Lasso, ElasticNet

Best: Ridge (R² ≈ 0.70, RMSE ≈ $113)

003_classification_model.ipynb
Baseline Logistic Regression

SMOTE inside ImbPipeline + stratified split

Result: Recall ≈ 74%, Precision ≈ 54%

004_model_comparison.ipynb
Compared Decision Tree, SVM, Logistic Regression, GaussianNB (GridSearchCV)

Best: Tuned Decision Tree (F1 ≈ 0.64)

Worst: GaussianNB (violated independence assumption)

005_advanced_predictive_probability.ipynb
Tuned XGBoost with scale_pos_weight

Results: F1 ≈ 0.75, AUC ≈ 0.90

Found a group of non-Superhosts (high reviews, low response rate) with >80% Superhost probability

🛠 Key techniques & methods
Area	Techniques
Preprocessing	ColumnTransformer, OneHotEncoder, StandardScaler, Pipeline
Feature engineering	Sentiment (VADER), amenities count, date parsing, outlier handling
Regression	Linear, Ridge (L2), Lasso (L1), ElasticNet
Classification	LogisticRegression, DecisionTree, SVM, GaussianNB, XGBoost
Imbalance	SMOTE, scale_pos_weight, stratified sampling
Tuning	GridSearchCV
Explainability	SHAP

✅ Outputs / expected files
data/listings_cleaned.csv — cleaned dataset (produced by preprocessing)

visuals/ — saved charts and figures per notebook

notebooks/ — interactive experiments and results

