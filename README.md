📝 Project Overview

This repository contains notebooks and scripts that build and evaluate machine learning models on the Inside Airbnb listings.csv for your chosen country/city.

Goals

Task 1 — Regression: Predict listing price using features such as location, amenities, and property type.

Task 2 — Classification: Predict whether a host becomes a Superhost.

Task 3 — Advanced: Build an XGBoost probability model to identify non-Superhosts with high potential to become Superhosts.

🛠 Tech Stack

Core: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Modeling: XGBoost, Logistic Regression, Decision Tree, SVM

NLP: NLTK (VADER sentiment)

Imbalanced Data: SMOTE, scale_pos_weight

Explainability: SHAP

Env: Conda, Jupyter Notebooks

📂 Project Structure
airbnb-predictive-modeling/
│
├── code/
│   └── code.py                     # Runs advanced probability model (Task 5)
│
├── data/
│   ├── listings.csv                # Raw dataset (add this file)
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
├── visuals/                        # Notebook output plots
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   └── 5/
│
├── main.py                         # Runs Tasks 1–4
├── requirements.txt
├── setup_structure.py              # Builds folder structure
└── README.md

🚀 Setup & Installation
1. Clone repository
git clone https://[your-repository-url]/airbnb-predictive-modeling.git
cd airbnb-predictive-modeling

2. Create Conda environment
conda create -n airbnb_env python=3.10 -y
conda activate airbnb_env

3. Install dependencies
pip install -r requirements.txt

4. Add data

Download listings.csv for your country/city from Inside Airbnb and place it at:

/data/listings.csv

🏃 How to run
Run preprocessing, regression and classification (Tasks 1–4)
python main.py


What this does

Loads data/listings.csv and runs the preprocessing pipeline.

Saves cleaned features to data/listings_cleaned.csv.

Trains & compares 4 regression models.

Trains & compares 4 classification models (SMOTE + tuning).

Run the advanced XGBoost analysis (Task 5)
python code/code.py


What this does

Loads data/listings_cleaned.csv.

Applies robust hotfix for a recurring 5E-1 string issue in numeric columns.

Trains and tunes an XGBoost probability model (uses scale_pos_weight).

Produces probability scores and lists non-Superhosts with >80% Superhost probability.

🔬 Notebook-by-Notebook Summary
001_data_preprocessing.ipynb

Loaded ~79 columns and dropped 30+ irrelevant ones.

Engineered features:

host_duration_days

amenities_count

description_sentiment (VADER)

Handled outliers and missing values.

Fixed recurring 5E-1 string error in numeric columns.

Noted strong imbalance in host_is_superhost.

002_regression_model.ipynb

Built ColumnTransformer (scaling + one-hot encoding).

Trained and compared:

Linear Regression

Ridge

Lasso

ElasticNet

Best model: Ridge Regression

R² ≈ 0.70

RMSE ≈ $113

003_classification_model.ipynb

Baseline Logistic Regression.

Used SMOTE in an ImbPipeline, with stratified split.

Result:

Recall: 74%

Precision: 54%

004_model_comparison.ipynb

Compared Decision Tree, SVM, Logistic Regression, GaussianNB (with GridSearchCV tuning).

Best model: Tuned Decision Tree (F1 ≈ 0.64)
Worst model: GaussianNB (assumption of feature independence violated)

005_advanced_predictive_probability.ipynb

Tuned XGBoost probability model (used scale_pos_weight for imbalance).

Performance:

F1 ≈ 0.75

AUC ≈ 0.90

Key insight: A group of non-Superhosts (often perfect review scores but low response rates) showed >80% probability of becoming Superhosts.

🛠 Key techniques
Category	Techniques
Preprocessing	ColumnTransformer, OneHotEncoder, StandardScaler, Pipelines
Feature engineering	Sentiment analysis (VADER), date parsing, amenities count, outlier handling
Regression models	Linear, Ridge (L2), Lasso (L1), ElasticNet
Classification models	Logistic Regression, Decision Tree, SVM, GaussianNB, XGBoost
Imbalance handling	SMOTE, scale_pos_weight, stratified sampling
Model tuning	GridSearchCV
Explainability	SHAP
Outputs / Expected files

data/listings_cleaned.csv — cleaned dataset produced by preprocessing.

visuals/ — plots and figures saved per notebook.

Notebook HTML/PNG exports (optional) for report sharing.

Notes & tips

If Markdown renders compacted: ensure you paste the file into README.md (root) and save. GitHub will render headings and code blocks automatically.

For long lines that should wrap in .md, add blank lines between sections to create clear paragraph breaks.

Use relative links for internal files (e.g., notebooks/001_data_preprocessing.ipynb) so links work across branches and clones.
