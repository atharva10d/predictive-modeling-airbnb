import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# --- Sklearn ---
# Preprocessing and pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# Metrics
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve, 
    precision_recall_curve, calibration_curve
)

# --- Imbalanced-learn ---
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# --- FEATURE/TARGET DEFINITION ---

def get_regression_features():
    """Returns the feature lists for the regression task."""
    numeric_features = [
        'latitude', 'longitude', 'accommodates', 'bedrooms', 'bathrooms_numeric', 
        'host_duration_days', 'host_response_rate', 'review_scores_rating', 
        'review_scores_cleanliness', 'review_scores_location', 'amenities_count', 
        'keyword_count', 'description_sentiment', 'neigh_overview_sentiment', 
        'description_length'
    ]
    categorical_features = [
        'property_type', 'room_type', 'is_shared_bath', 'neighbourhood_cleansed', 
        'host_response_time'
    ]
    return numeric_features, categorical_features

def get_classification_features():
    """Returns the feature lists for the classification task."""
    # For this project, they are the same
    return get_regression_features()

def create_preprocessor(numeric_features, categorical_features, sparse_output=False):
    """
    Creates the ColumnTransformer.
    Set sparse_output=False for models like GaussianNB or XGBoost (for SHAP).
    """
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=sparse_output))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

# --- TASK 2: REGRESSION ---

def run_regression_comparison(X_train, y_train, X_test, y_test, preprocessor):
    """Trains and evaluates all 4 regression models."""
    print("Running Regression Model Comparison...")
    results = {}
    
    # Define models
    models = [
        ('Linear Regression', LinearRegression()),
        ('Ridge', Ridge(alpha=1.0)),
        ('Lasso', Lasso(alpha=0.01)),
        ('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=0.5))
    ]
    
    for name, model in models:
        start_time = time.time()
        print(f"Training {name}...")
        
        # Create a standard sklearn pipeline
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        
        # Get log predictions
        y_pred_log = pipe.predict(X_test)
        
        # Inverse transform to actual prices for RMSE/MAE
        y_pred_price = np.expm1(y_pred_log)
        y_test_price = np.expm1(y_test)
        
        # Calculate Metrics
        r2 = r2_score(y_test, y_pred_log) 
        rmse = np.sqrt(mean_squared_error(y_test_price, y_pred_price))
        mae = mean_absolute_error(y_test_price, y_pred_price)
        
        results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
        print(f"...done in {time.time() - start_time:.2f}s")

    # Format and return results
    results_df = pd.DataFrame(results).T.sort_values(by='RMSE')
    print("Regression comparison complete.")
    return results_df

# --- TASK 3 & 4: CLASSIFICATION ---

def run_classification_comparison(X_train, y_train, X_test, y_test, preprocessor):
    """Trains, tunes, and compares the 4 classification models."""
    print("Running Classification Model Comparison...")
    model_results = {}
    
    # 1. Logistic Regression (Baseline)
    print("Training Logistic Regression...")
    pipe_logreg = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', LogisticRegression(random_state=42, solver='liblinear'))
    ])
    pipe_logreg.fit(X_train, y_train)
    model_results['Logistic Regression'] = {'model': pipe_logreg}
    
    # 2. Decision Tree (Tuned)
    print("Tuning Decision Tree...")
    pipe_dt = ImbPipeline(steps=[
        ('preprocessor', preprocessor), ('smote', SMOTE(random_state=42)),
        ('model', DecisionTreeClassifier(random_state=42))
    ])
    # Smaller grid for a faster script
    param_grid_dt = {'model__max_depth': [10, 20], 'model__min_samples_leaf': [5, 10]}
    grid_dt = GridSearchCV(pipe_dt, param_grid_dt, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
    grid_dt.fit(X_train, y_train)
    model_results['Decision Tree'] = {'model': grid_dt.best_estimator_}
    print(f"Best DT params: {grid_dt.best_params_}")

    # 3. SVM (Tuned)
    print("Tuning SVM...")
    pipe_svm = ImbPipeline(steps=[
        ('preprocessor', preprocessor), ('smote', SMOTE(random_state=42)),
        ('model', SVC(probability=True, random_state=42)) # probability=True for ROC
    ])
    # Very small grid for speed. 'linear' kernel is fast.
    param_grid_svm = {'model__kernel': ['linear'], 'model__C': [0.1]}
    grid_svm = GridSearchCV(pipe_svm, param_grid_svm, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
    grid_svm.fit(X_train, y_train)
    model_results['SVM (Linear)'] = {'model': grid_svm.best_estimator_}
    print(f"Best SVM params: {grid_svm.best_params_}")

    # 4. Naïve Bayes
    print("Training Naïve Bayes...")
    pipe_nb = ImbPipeline(steps=[
        ('preprocessor', preprocessor), ('smote', SMOTE(random_state=42)),
        ('model', GaussianNB())
    ])
    pipe_nb.fit(X_train, y_train)
    model_results['Naïve Bayes'] = {'model': pipe_nb}
    
    # --- Get predictions and reports ---
    print("\n--- Classification Reports ---")
    reports = {}
    for name, results in model_results.items():
        y_pred = results['model'].predict(X_test)
        # Get the full report as a dictionary
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        # Save metrics for the 'Superhost' class (which is '1')
        reports[name] = report_dict['1'] 
        
        # Print the string report
        print(f"\n--- Report for: {name} ---")
        print(classification_report(y_test, y_pred, target_names=['Not Superhost', 'Superhost']))
        
    # Format and return results
    reports_df = pd.DataFrame(reports).T.sort_values(by='f1-score', ascending=False)
    return reports_df, model_results

# --- TASK 5: ADVANCED MODEL ---

def run_advanced_xgb(X_train, y_train, preprocessor):
    """Trains and tunes the advanced XGBoost model."""
    print("Training Advanced XGBoost Model...")
    
    # 1. Calculate Imbalance Ratio (replaces SMOTE)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Calculated 'scale_pos_weight' for imbalance: {scale_pos_weight:.2f}")

    # 2. Build Standard Pipeline (no ImbPipeline)
    pipeline_xgb = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,  # Handle imbalance
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            use_label_encoder=False # Suppress warning
        ))
    ])

    # 3. Hyperparameter Tuning
    print("Starting Hyperparameter Tuning for XGBoost...")
    # This is the grid from our notebook
    param_grid = {
        'model__n_estimators': [150, 250],
        'model__max_depth': [5, 8],
        'model__learning_rate': [0.05, 0.1]
    }
    
    # We score on 'roc_auc' as it's best for imbalanced data
    grid_xgb = GridSearchCV(
        pipeline_xgb, param_grid, cv=3, 
        scoring='roc_auc', n_jobs=-1, verbose=1
    )

    # 4. Fit Model
    grid_xgb.fit(X_train, y_train)
    best_model = grid_xgb.best_estimator_
    
    print(f"Best XGBoost params: {grid_xgb.best_params_}")
    print(f"Best CV ROC AUC: {grid_grid_xgb.best_score_:.4f}")
    
    return best_model