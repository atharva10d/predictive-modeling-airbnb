# predictive-modeling-airbnb
An end-to-end data science project analyzing the Inside Airbnb dataset. This project includes data cleaning, feature engineering, price prediction (regression), and Superhost status prediction (classification).

📊 Predictive Modeling with Inside Airbnb

An end-to-end data science project using the Inside Airbnb dataset to perform regression, classification, and advanced probability modeling. It covers data cleaning, feature engineering, exploratory analysis, model comparison, and identification of high-potential Superhosts.

📝 Project Overview

This project builds and evaluates machine learning models using the Inside Airbnb dataset for your selected country.

Goals

Task 1 — Regression: Predict listing prices using features such as location, amenities, and property type.

Task 2 — Classification: Predict whether a host becomes a Superhost.

Task 3 — Advanced: Use XGBoost to generate probability scores and identify non-Superhosts with high potential.

🛠 Tech Stack

Core Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

ML Models: XGBoost, Logistic Regression, Decision Tree, SVM

NLP: NLTK (VADER sentiment analysis)

Imbalanced Data: SMOTE, scale_pos_weight

Explainability: SHAP

Environment: Conda, Jupyter Notebook

📂 Project Structure
airbnb-predictive-modeling/
│
├── code/
│   └── code.py                     # Runs advanced probability model (Task 5)
│
├── data/
│   ├── listings.csv                # Raw dataset (to be added by user)
│   └── listings_cleaned.csv        # Generated after preprocessing
│
├── notebooks/
│   ├── 001_data_preprocessing.ipynb
│   ├── 002_regression_model.ipynb
│   ├── 003_classification_model.ipynb
│   ├── 004_model_comparison.ipynb
│   └── 005_advanced_predictive_probability.ipynb
│
├── utils/
│   ├── __init__.py
│   ├── model_helpers.py            # Model training & evaluation functions
│   └── preprocessing.py            # Data cleaning & feature engineering
│
├── visuals/                        # Plots from each notebook
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   └── 5/
│
├── main.py                         # Runs Tasks 1–4
├── requirements.txt
├── setup_structure.py
└── README.md

🚀 Setup & Installation
1. Clone the Repository
git clone https://[your-repository-url]/airbnb-predictive-modeling.git
cd airbnb-predictive-modeling

2. Create the Conda Environment
conda create -n airbnb_env python=3.10
conda activate airbnb_env

3. Install Dependencies
pip install -r requirements.txt

4. Add the Dataset

Download listings.csv for your country from the Inside Airbnb website and place it into:

/data/listings.csv

🏃 How to Run the Project
1. Run Tasks 1–4 (Main Pipeline)

This performs preprocessing, regression modeling, and classification modeling.

python main.py


This script will:

Load and clean listings.csv

Create listings_cleaned.csv

Train and compare 4 regression models

Train and compare 4 classification models (with SMOTE + tuning)

2. Run Advanced XGBoost Analysis (Task 5)
python code/code.py


This script will:

Load listings_cleaned.csv

Fix the recurring 5E-1 string error in numeric columns

Train + tune an advanced XGBoost probability model

Compute high-confidence Superhost probabilities

Identify non-Superhosts with >80% probability of becoming one

🔬 Notebook-by-Notebook Summary
001_data_preprocessing.ipynb

Loaded 79 columns, removed 30+ irrelevant ones

Engineered features:

host_duration_days

amenities_count

description_sentiment (VADER)

Fixed the recurring 5E-1 string issue

Noted strong imbalance in host_is_superhost

002_regression_model.ipynb

Applied ColumnTransformer (scaling + OHE)

Trained and compared:

Linear Regression

Ridge

Lasso

ElasticNet

Best Model: Ridge Regression

R² ≈ 0.70

RMSE ≈ $113

L2 penalty handled multicollinearity well

003_classification_model.ipynb

Baseline Logistic Regression

Used SMOTE to handle imbalance

Stratified splitting

Result:

Recall: 74%

Precision: 54%

004_model_comparison.ipynb

Models Compared:

Decision Tree

SVM

Logistic Regression

GaussianNB

Used GridSearchCV for tuning.

Best Model:
✔ Tuned Decision Tree (F1 = 0.64)

Worst Model:
✘ GaussianNB (correlated features violate independence assumption)

005_advanced_predictive_probability.ipynb

Built advanced XGBoost probability model

Used scale_pos_weight instead of SMOTE

Achieved:

F1-Score: 0.75

AUC: 0.90

Key Insight:
Non-Superhosts with perfect review scores but low response rates have high Superhost potential (>80%).

🛠 Key Techniques Used
Category	Techniques
Preprocessing	ColumnTransformer, OneHotEncoder, StandardScaler, Pipelines
Feature Engineering	Sentiment analysis, date parsing, outlier handling
Regression Models	Linear, Ridge (L2), Lasso (L1), ElasticNet
Classification Models	Logistic Regression, Decision Tree, SVM, GaussianNB, XGBoost
Imbalanced Data	SMOTE, scale_pos_weight, stratified sampling
Model Tuning	GridSearchCV
Explainability	SHAP
