# 📊 Predictive Modeling with Inside Airbnb

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-active-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

An end-to-end data science pipeline using the **Inside Airbnb** `listings.csv` to predict listing prices (regression), Superhost status (classification), and build an advanced XGBoost probability model to identify high-potential non-Superhosts.

---

## 🔭 Table of contents

* [Project overview](#project-overview)
* [Goals](#goals)
* [Tech stack](#tech-stack)
* [Project structure](#project-structure)
* [Setup & installation](#setup--installation)
* [🏃 Usage](#-usage)

  * [Run main pipeline (Tasks 1–4)](#run-main-pipeline-tasks-1-4)
  * [Run advanced XGBoost analysis (Task 5)](#run-advanced-xgboost-analysis-task-5)
* [🔬 Notebook summaries](#-notebook-summaries)
* [🛠 Key techniques & methods](#-key-techniques--methods)
* [✅ Outputs / expected files](#-outputs--expected-files)
* [🧭 Tips & GitHub Markdown hints](#-tips--github-markdown-hints)
* [Contributing](#contributing)
* [License](#license)

---

## 📝 Project overview

This repository contains reproducible notebooks and scripts that implement:

* Data cleaning & feature engineering (including sentiment with VADER)
* Price prediction (regression)
* Superhost prediction (classification)
* Advanced XGBoost probability estimation for future Superhosts

---

## 🎯 Goals

* **Task 1 — Regression:** Predict listing price using engineered features.
* **Task 2 — Classification:** Predict whether a host becomes a Superhost.
* **Task 3 — Advanced:** Generate reliable probability scores using XGBoost to identify high-potential non-Superhosts.

> **Note:** Add `listings.csv` from Inside Airbnb before running any scripts.

---

## 🛠 Tech stack

* **Core:** pandas, numpy, scikit-learn, matplotlib, seaborn
* **Modeling:** XGBoost, Logistic Regression, Decision Tree, SVM
* **NLP:** NLTK VADER sentiment analyzer
* **Imbalance Handling:** SMOTE, scale_pos_weight
* **Explainability:** SHAP
* **Environment:** Conda, Jupyter Notebooks

---

## 📁 Project structure

```
airbnb-predictive-modeling/
│
├── code/
│   └── code.py
│
├── data/
│   ├── listings.csv
│   └── listings_cleaned.csv
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
│   ├── model_helpers.py
│   └── preprocessing.py
│
├── visuals/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   └── 5/
│
├── main.py
├── requirements.txt
├── setup_structure.py
└── README.md
```

---

## 🚀 Setup & installation

```bash
git clone https://[your-repository-url]/airbnb-predictive-modeling.git
cd airbnb-predictive-modeling

conda create -n airbnb_env python=3.10 -y
conda activate airbnb_env

pip install -r requirements.txt
```

---

## 🏃 Usage

### ▶️ Run main pipeline (Tasks 1–4)

Runs preprocessing, regression and classification.

```bash
python main.py
```

#### What this does

* Loads `data/listings.csv`
* Cleans & engineers features → saves `data/listings_cleaned.csv`
* Trains & compares **Regression Models:** Linear, Ridge, Lasso, ElasticNet
* Trains & compares **Classification Models:** Logistic Regression, Decision Tree, SVM, GaussianNB (with SMOTE + tuning)

---

### ▶️ Run advanced XGBoost analysis (Task 5)

```bash
python code/code.py
```

#### What this does

* Loads `data/listings_cleaned.csv`
* Fixes `5E-1` string corruption in numeric fields
* Trains tuned XGBoost probability model using `scale_pos_weight`
* Generates advanced probability scores
* Identifies **non-Superhosts with >80% likelihood** of becoming Superhosts

---

## 🔬 Notebook summaries

### 📘 001_data_preprocessing.ipynb

* Read ~79 columns, dropped 30+ irrelevant
* Engineered: `host_duration_days`, `amenities_count`, `description_sentiment`
* Handled missing values & outliers
* Fixed persistent **5E-1 string error**
* Identified severe class imbalance in `host_is_superhost`

### 📘 002_regression_model.ipynb

* Used ColumnTransformer (Scaling + OHE)
* Models: Linear, Ridge, Lasso, ElasticNet
* **Best:** Ridge (R² ≈ 0.70, RMSE ≈ $113)

### 📘 003_classification_model.ipynb

* Baseline model: Logistic Regression
* SMOTE applied via ImbPipeline
* Stratified splitting
* **Results:** Recall ≈ 74%, Precision ≈ 54%

### 📘 004_model_comparison.ipynb

* Compared Decision Tree, SVM, Logistic Regression, GaussianNB
* Hyperparameter tuning with GridSearchCV
* **Best:** Decision Tree (F1 ≈ 0.64)
* **Worst:** GaussianNB (feature correlation issue)

### 📘 005_advanced_predictive_probability.ipynb

* Tuned XGBoost using scale_pos_weight
* Performance: F1 ≈ 0.75, AUC ≈ 0.90
* Found non-Superhosts with perfect review scores but lower response rate

---

## 🛠 Key techniques & methods

| Area                | Techniques                                                        |
| ------------------- | ----------------------------------------------------------------- |
| Preprocessing       | ColumnTransformer, OneHotEncoder, StandardScaler, Pipeline        |
| Feature Engineering | VADER Sentiment, Date parsing, Amenities counts, Outlier handling |
| Regression          | Linear, Ridge, Lasso, ElasticNet                                  |
| Classification      | LogisticRegression, DecisionTree, SVM, GaussianNB, XGBoost        |
| Imbalance Handling  | SMOTE, scale_pos_weight, Stratified split                         |
| Tuning              | GridSearchCV                                                      |
| Explainability      | SHAP                                                              |

---

## ✅ Outputs / expected files

* `data/listings_cleaned.csv` — cleaned dataset
* `visuals/` — plots generated during notebook analysis
* `notebooks/` — full experimentation history

---

## 🧭 Tips & GitHub Markdown hints

> **TIP:** If README appears collapsed, ensure you are editing the root `README.md` file.
>
> **TIP:** Use relative links such as:
> `[Preprocessing Notebook](notebooks/001_data_preprocessing.ipynb)`

**Common Markdown elements used here:**

* Headings → `#`, `##`, `###`
* Code blocks → triple backticks
* Inline code → `like this`
* Task lists → `- [ ] item`
* Links → `[text](url)`
* Images → `![alt](path/img.png)`

---

## 🤝 Contributing

Pull requests are welcome! Please follow standard GitHub workflow:

* Fork the repo
* Create a new branch
* Commit changes
* Open a PR

---

## 📝 License

This project is licensed under the **MIT License**.

---

