import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os

def load_data(path):
    """Loads the raw CSV data."""
    if not os.path.exists(path):
        print(f"ERROR: File not found at {path}")
        return None
        
    try:
        df = pd.read_csv(path)
        print(f"Data loaded successfully from {path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def download_nltk_data():
    """Downloads the VADER lexicon if not already present."""
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("Downloading NLTK VADER lexicon (one-time setup)...")
        nltk.download('vader_lexicon')

def drop_irrelevant_cols(df):
    """Drops all irrelevant, redundant, or high-null columns."""
    print("Dropping irrelevant columns...")
    
    # Calculate columns with > 90% missing
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    cols_to_drop_high_null = missing_percentage[missing_percentage > 90].index
    
    cols_to_drop_irrelevant = [
        'id', 'listing_url', 'scrape_id', 'picture_url', 'host_id', 'host_url', 
        'host_thumbnail_url', 'host_picture_url', 'source', 'last_scraped', 
        'calendar_last_scraped', 'host_name', 'host_location', 'host_neighbourhood', 
        'neighbourhood', 'minimum_minimum_nights', 'maximum_minimum_nights', 
        'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 
        'maximum_nights_avg_ntm', 'availability_30', 'availability_60', 
        'availability_90', 'availability_eoy', 'number_of_reviews_ltm', 
        'number_of_reviews_l30d', 'number_of_reviews_ly', 
        'estimated_occupancy_l365d', 'estimated_revenue_l365d'
    ]
    
    all_cols_to_drop = list(cols_to_drop_high_null) + cols_to_drop_irrelevant
    # Get unique columns that exist in the dataframe
    existing_cols_to_drop = [col for col in set(all_cols_to_drop) if col in df.columns]
    
    df_cleaned = df.drop(columns=existing_cols_to_drop)
    print(f"Dropped {len(existing_cols_to_drop)} columns.")
    return df_cleaned

def engineer_features(df):
    """Runs all feature engineering steps."""
    print("Engineering features...")
    df_eng = df.copy()

    # 1. Price
    if 'price' in df_eng.columns:
        df_eng['price'] = df_eng['price'].astype(str).str.replace(r'[$,]', '', regex=True).astype(float)

    # 2. Booleans
    bool_cols = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
    for col in bool_cols:
        if col in df_eng.columns:
            df_eng[col] = df_eng[col].map({'t': 1, 'f': 0})

    # 3. Percentages
    perc_cols = ['host_response_rate', 'host_acceptance_rate']
    for col in perc_cols:
        if col in df_eng.columns:
            df_eng[col] = df_eng[col].astype(str).str.replace('%', '', regex=False).astype(float) / 100.0

    # 4. Date Feature
    if 'host_since' in df_eng.columns:
        df_eng['host_since'] = pd.to_datetime(df_eng['host_since'], errors='coerce')
        df_eng['host_duration_days'] = (pd.Timestamp.now() - df_eng['host_since']).dt.days
        df_eng = df_eng.drop(columns=['host_since', 'first_review', 'last_review'], errors='ignore')

    # 5. Bathroom Parsing
    if 'bathrooms_text' in df_eng.columns:
        df_eng['bathrooms_numeric'] = df_eng['bathrooms_text'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
        df_eng['is_shared_bath'] = df_eng['bathrooms_text'].astype(str).str.contains('shared').astype(int)
        df_eng = df_eng.drop(columns=['bathrooms_text', 'bathrooms'], errors='ignore')

    # 6. Amenities Parsing
    if 'amenities' in df_eng.columns:
        df_eng['amenities_count'] = df_eng['amenities'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
        df_eng = df_eng.drop(columns=['amenities'])

    # 7. Text/NLP Features
    download_nltk_data() # Ensure VADER is ready
    sia = SentimentIntensityAnalyzer()
    df_eng['description_sentiment'] = df_eng['description'].fillna('').apply(lambda x: sia.polarity_scores(x)['compound'])
    df_eng['neigh_overview_sentiment'] = df_eng['neighborhood_overview'].fillna('').apply(lambda x: sia.polarity_scores(x)['compound'])
    keywords = ['fast wifi', 'newly renovated', 'ocean view', 'luxury', 'quiet', 'central', 'spacious']
    df_eng['keyword_count'] = df_eng['description'].astype(str).str.lower().apply(lambda x: sum(keyword in x for keyword in keywords))
    df_eng['description_length'] = df_eng['description'].str.len()
    
    text_cols_to_drop = ['name', 'description', 'neighborhood_overview', 'host_about']
    df_eng = df_eng.drop(columns=[col for col in text_cols_to_drop if col in df_eng.columns])
    
    print("Feature engineering complete.")
    return df_eng

def handle_missing_values(df):
    """Imputes all missing values using median, mode, or zero."""
    print("Handling missing values...")
    df_filled = df.copy()

    # 1. Drop rows where targets are unusable
    df_filled = df_filled.dropna(subset=['price', 'host_is_superhost'])
    df_filled = df_filled[df_filled['price'] > 0] 

    # 2. Numerical Imputation (Median)
    review_cols = [col for col in df_filled.columns if 'review_scores_' in col]
    median_impute_cols = ['host_response_rate', 'host_acceptance_rate', 'bedrooms', 
                          'beds', 'bathrooms_numeric', 'host_duration_days']
    median_impute_cols = [col for col in median_impute_cols if col in df_filled.columns] + review_cols
    
    for col in median_impute_cols:
        median_val = df_filled[col].median()
        df_filled[col] = df_filled[col].fillna(median_val)

    # 3. Numerical Imputation (Zero)
    zero_impute_cols = ['reviews_per_month', 'description_length', 
                        'neigh_overview_sentiment', 'description_sentiment']
    zero_impute_cols = [col for col in zero_impute_cols if col in df_filled.columns]
    df_filled[zero_impute_cols] = df_filled[zero_impute_cols].fillna(0)

    # 4. Categorical Imputation (Mode)
    mode_impute_cols = ['host_response_time', 'host_has_profile_pic', 'host_identity_verified']
    mode_impute_cols = [col for col in mode_impute_cols if col in df_filled.columns]

    for col in mode_impute_cols:
        mode_val = df_filled[col].mode()[0]
        df_filled[col] = df_filled[col].fillna(mode_val)
        
    print(f"Missing values handled. Rows remaining: {len(df_filled)}")
    return df_filled

def handle_outliers(df):
    """Removes extreme outliers for price and minimum_nights."""
    print("Handling outliers...")
    df_out = df.copy()
    
    # Price
    price_quantile_99 = df_out['price'].quantile(0.99)
    df_out = df_out[df_out['price'] < price_quantile_99]
    
    # Minimum Nights
    df_out = df_out[df_out['minimum_nights'] <= 365]
    
    print(f"Outliers removed. Rows remaining: {len(df_out)}")
    return df_out

def robust_clean_numeric(df, numeric_cols, medians_dict=None):
    """
    The critical hotfix function to clean string/bracket errors
    from numeric columns.
    
    If medians_dict is None, it calculates them.
    If medians_dict is provided, it uses them.
    """
    print("Applying robust hotfix for numeric string errors...")
    df_cleaned = df.copy()
    calculating_medians = False
    
    # If medians are not provided, we must calculate them
    if medians_dict is None:
        medians_dict = {}
        calculating_medians = True
        print("Calculating clean medians from the provided DataFrame...")
        # Pass 1: Calculate clean medians
        for col in numeric_cols:
            col_str = df_cleaned[col].astype(str).str.replace(r'[\[\]]', '', regex=True)
            clean_col_numeric = pd.to_numeric(col_str, errors='coerce')
            median_val = clean_col_numeric.median()
            medians_dict[col] = 0.0 if pd.isna(median_val) else median_val # Use 0.0 for float consistency

    # Pass 2: Apply cleaning and filling
    for col in numeric_cols:
        col_str = df_cleaned[col].astype(str).str.replace(r'[\[\]]', '', regex=True)
        df_cleaned[col] = pd.to_numeric(col_str, errors='coerce')
        df_cleaned[col] = df_cleaned[col].fillna(medians_dict[col])
        
    print("Robust cleaning complete.")
    
    if calculating_medians:
        return df_cleaned, medians_dict
    else:
        return df_cleaned

def run_full_preprocessing(raw_data_path, cleaned_data_save_path=None):
    """
    Runs the entire preprocessing pipeline from start to finish
    and saves the cleaned file.
    """
    df = load_data(raw_data_path)
    if df is None:
        return None
        
    df = drop_irrelevant_cols(df)
    df = engineer_features(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    
    # Add log_price
    df['log_price'] = np.log1p(df['price'])
    
    # --- Final Robust Clean ---
    # Define numeric features *after* engineering
    numeric_features = [
        'latitude', 'longitude', 'accommodates', 'bedrooms', 'bathrooms_numeric', 
        'host_duration_days', 'host_response_rate', 'review_scores_rating', 
        'review_scores_cleanliness', 'review_scores_location', 'amenities_count', 
        'keyword_count', 'description_sentiment', 'neigh_overview_sentiment', 
        'description_length'
    ]
    # Filter for any that might have been dropped if they didn't exist
    numeric_features = [col for col in numeric_features if col in df.columns]
    
    # Clean the final dataset and get the medians
    df, medians = robust_clean_numeric(df, numeric_features, medians_dict=None)
    
    if cleaned_data_save_path:
        df.to_csv(cleaned_data_save_path, index=False)
        print(f"Full preprocessing complete. Cleaned data saved to {cleaned_data_save_path}")
    else:
        print("Full preprocessing complete.")
        
    return df, medians