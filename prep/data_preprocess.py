import pandas as pd
import numpy as np
import ast
import os

# Define the relative path to the movies metadata CSV
# Ensure your CSV file is inside a folder named 'data'
CSV_PATH = os.path.join("data", "movies_metadata.csv")

def load_data(path: str) -> pd.DataFrame:
    """
    Loads the movie metadata CSV file with specific columns to save memory.
    """
    print(f"Loading data from: {path}...")
    try:
        # Columns we actually need for the analysis
        cols_to_use = [
            'id', 'title', 'budget', 'revenue', 'release_date', 
            'genres', 'production_companies', 'runtime', 
            'vote_average', 'vote_count', 'popularity'
        ]
        
        # low_memory=False helps with mixed types in the raw Kaggle dataset
        df = pd.read_csv(path, usecols=cols_to_use, low_memory=False)
        print(f"Data loaded successfully: {len(df)} rows.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {path}.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def safe_parse_json(x):
    """
    Safely evaluates a string containing a Python literal (like a list of dicts).
    Returns an empty list if parsing fails.
    """
    try:
        # If it's already a list (rare but possible depending on pandas version/loading), return it
        if isinstance(x, list):
            return x
        # If it's NaN or empty string
        if pd.isna(x) or x == '':
            return []
        # Safely evaluate the string
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw DataFrame:
    1. Parses JSON columns (genres, companies).
    2. Converts types (dates, numbers).
    3. Calculates Profit and ROI.
    4. Filters out invalid rows.
    """
    print("Starting data preprocessing...")
    
    # --- 1. Parse JSON Columns ---
    # Extract genre names from the list of dictionaries
    # e.g., "[{'id': 1, 'name': 'Comedy'}]" -> "Comedy"
    print("Parsing JSON columns...")
    df['genres_list'] = df['genres'].apply(safe_parse_json)
    df['primary_genre'] = df['genres_list'].apply(lambda x: x[0]['name'] if len(x) > 0 else np.nan)
    
    # --- 2. Numeric Conversions ---
    # Force numeric types, turning errors (like string '0') into numbers or NaN
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
    
    # Handle 0 values in budget/revenue (common in this dataset)
    # We replace 0 with NaN so they don't skew averages, then drop them later for financial analysis
    df['budget'] = df['budget'].replace(0, np.nan)
    df['revenue'] = df['revenue'].replace(0, np.nan)

    # --- 3. Date Handling ---
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year

    # --- 4. Feature Engineering ---
    # Profit = Revenue - Budget
    df['profit'] = df['revenue'] - df['budget']
    
    # ROI = Profit / Budget
    df['roi'] = df['profit'] / df['budget']

    # --- 5. Filtering ---
    # Keep only rows that have valid financial data and a genre for our main analysis
    df_clean = df.dropna(subset=['budget', 'revenue', 'primary_genre', 'year'])
    
    print(f"Preprocessing complete. {len(df_clean)} entries remain after cleaning.")
    
    return df_clean