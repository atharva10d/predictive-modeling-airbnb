import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Import our custom utility functions
try:
    from utils import preprocessing
    from utils import model_helpers
except ImportError:
    print("Error: Could not import from 'utils' directory.")
    print("Please ensure 'utils/__init__.py', 'utils/preprocessing.py', and 'utils/model_helpers.py' exist.")
    exit()

# --- Define Global Paths ---
# Use relative paths so the script is portable
CLEANED_DATA_PATH = os.path.join('data', 'listings_cleaned.csv')
VISUALS_PATH = os.path.join('visuals', '5')

def run_advanced_analysis():
    """Main script for the advanced XGBoost task."""
    
    print("--- STARTING ADVANCED ANALYSIS (TASK 5) ---")
    
    # --- 1. Load Data ---
    if not os.path.exists(CLEANED_DATA_PATH):
        print(f"ERROR: Cleaned data not found at {CLEANED_DATA_PATH}")
        print("Please run 'main.py' or Notebook 001 first to generate the cleaned data.")
        return
        
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"Loaded {CLEANED_DATA_PATH}")

    # --- 2. Define Features & Target ---
    num_features, cat_features = model_helpers.get_classification_features()
    
    # Filter to only features that exist in the loaded data
    num_features = [col for col in num_features if col in df.columns]
    cat_features = [col for col in cat_features if col in df.columns]

    # --- 3. Apply ROBUST CLEANING (Hotfix) ---
    # This is critical for XGBoost to run without the '[5E-1]' error
    # This function returns the cleaned df and the medians dict
    df_clean, train_medians = preprocessing.robust_clean_numeric(
        df, num_features, medians_dict=None
    )
    
    X = df_clean[num_features + cat_features]
    y = df_clean['host_is_superhost'].astype(int)
    
    # --- 4. Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # --- 5. Train Advanced Model ---
    # Create preprocessor (must set sparse_output=False for XGBoost)
    preprocessor_xgb = model_helpers.create_preprocessor(
        num_features, cat_features, sparse_output=False
    )
    
    best_xgb_model = model_helpers.run_advanced_xgb(
        X_train, y_train, preprocessor_xgb
    )
    
    print("\n--- Advanced Model Evaluation (on Test Set) ---")
    y_pred = best_xgb_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Not Superhost', 'Superhost']))
    
    # --- 6. Identify Potential Hosts ---
    print("\n--- Identifying High-Potential Hosts ---")
    
    # Get probabilities for the *entire* (cleaned) dataset
    # The best_xgb_model is a full pipeline, so it will preprocess 'X'
    full_dataset_proba = best_xgb_model.predict_proba(X)[:, 1]
    df_clean['probability'] = full_dataset_proba
    
    # Filter for high-potential hosts
    potential_hosts = df_clean[
        (df_clean['host_is_superhost'] == 0) & 
        (df_clean['probability'] > 0.80)
    ]

    print(f"Found {len(potential_hosts)} non-Superhosts with >80% probability.")
    
    key_analysis_features = [
        'review_scores_rating', 'host_response_rate', 
        'reviews_per_month', 'host_duration_days', 'probability'
    ]
    key_analysis_features = [col for col in key_analysis_features if col in df_clean.columns]

    if not potential_hosts.empty:
        print("\n--- Analysis of 'High-Potential' Hosts (Prob > 80%) ---")
        print(potential_hosts[key_analysis_features].describe())
    else:
        print("\nNo hosts found with >80% potential.")

    print("\n--- Analysis of *All* Non-Superhosts (for comparison) ---")
    all_non_superhosts = df_clean[df_clean['host_is_superhost'] == 0]
    print(all_non_superhosts[key_analysis_features].describe())
    
    print("\n--- ADVANCED ANALYSIS COMPLETE ---")

if __name__ == "__main__":
    # Ensure NLTK data is downloaded before running
    preprocessing.download_nltk_data()
    run_advanced_analysis()