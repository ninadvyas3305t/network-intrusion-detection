"""Data Preprocessing Module for Network Intrusion Detection
Handles data acquisition validation, cleaning, standardizing categorical strings,
leakage prevention (removing identifier columns), and reproducible saving.
"""

import os
import logging
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
RAW_TRAIN_FILE = "data/raw/UNSW_NB15_training-set.csv"
RAW_TEST_FILE = "data/raw/UNSW_NB15_testing-set.csv"
PROCESSED_TRAIN_FILE = "data/processed/train_cleaned.csv"
PROCESSED_TEST_FILE = "data/processed/test_cleaned.csv"

# Columns to drop for leakage prevention
DROP_COLUMNS = ["id"]

# Categorical column definitions
CATEGORICAL_COLUMNS = ["proto", "service", "state"]
TARGET_BINARY = "label"
TARGET_MULTICLASS = "attack_cat"

def load_raw_data(train_path: str = RAW_TRAIN_FILE, test_path: str = RAW_TEST_FILE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw training and testing datasets."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Raw train file not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Raw test file not found at {test_path}")

    logger.info(f"Loading raw train dataset from {train_path}...")
    train_df = pd.read_csv(train_path)
    logger.info(f"Loaded train data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")

    logger.info(f"Loading raw test dataset from {test_path}...")
    test_df = pd.read_csv(test_path)
    logger.info(f"Loaded test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

    return train_df, test_df

def clean_dataframe(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Clean and standardize a single dataframe:
    1. Drop ID column to avoid data leakage
    2. Strip whitespace and standardize casing on string columns
    3. Replace '-' in service column with 'none'
    4. Ensure correct target labels
    """
    df_clean = df.copy()

    # 1. Drop leakage columns
    for col in DROP_COLUMNS:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])
            logger.info(f"Dropped leakage column: '{col}'")

    # 2. Clean categorical strings
    for col in CATEGORICAL_COLUMNS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
            if col == "service":
                # Replace '-' placeholder with 'none'
                df_clean[col] = df_clean[col].replace({"-": "none"})

    # 3. Clean multiclass target if present
    if TARGET_MULTICLASS in df_clean.columns:
        df_clean[TARGET_MULTICLASS] = df_clean[TARGET_MULTICLASS].astype(str).str.strip()
        # Standardize attack category spelling / casing if any
        df_clean[TARGET_MULTICLASS] = df_clean[TARGET_MULTICLASS].replace({
            "Backdoors": "Backdoor"
        })

    # 4. Check for infinite values in numeric columns and clip/replace
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_BINARY in num_cols:
        num_cols.remove(TARGET_BINARY)
    
    inf_count = np.isinf(df_clean[num_cols].values).sum()
    if inf_count > 0:
        logger.warning(f"Found {inf_count} infinite values; replacing with NaN")
        df_clean[num_cols] = df_clean[num_cols].replace([np.inf, -np.inf], np.nan)

    return df_clean

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate and return key statistics about the dataset."""
    summary = {
        "total_records": len(df),
        "total_features": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "columns": list(df.columns),
    }

    if TARGET_BINARY in df.columns:
        summary["binary_distribution"] = df[TARGET_BINARY].value_counts().to_dict()
    if TARGET_MULTICLASS in df.columns:
        summary["multiclass_distribution"] = df[TARGET_MULTICLASS].value_counts().to_dict()

    return summary

def run_data_cleaning_pipeline(
    raw_train_path: str = RAW_TRAIN_FILE,
    raw_test_path: str = RAW_TEST_FILE,
    processed_dir: str = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete data cleaning pipeline and save processed files."""
    os.makedirs(processed_dir, exist_ok=True)
    train_df, test_df = load_raw_data(raw_train_path, raw_test_path)

    logger.info("Cleaning training dataset...")
    train_clean = clean_dataframe(train_df, is_train=True)

    logger.info("Cleaning testing dataset...")
    test_clean = clean_dataframe(test_df, is_train=False)

    train_out = os.path.join(processed_dir, "train_cleaned.csv")
    test_out = os.path.join(processed_dir, "test_cleaned.csv")

    logger.info(f"Saving cleaned training set to {train_out}...")
    train_clean.to_csv(train_out, index=False)

    logger.info(f"Saving cleaned testing set to {test_out}...")
    test_clean.to_csv(test_out, index=False)

    train_summary = get_data_summary(train_clean)
    test_summary = get_data_summary(test_clean)

    logger.info(f"Pipeline complete. Cleaned Train: {train_summary['total_records']} rows, {train_summary['total_features']} cols")
    logger.info(f"Cleaned Test: {test_summary['total_records']} rows, {test_summary['total_features']} cols")

    return train_clean, test_clean

if __name__ == "__main__":
    run_data_cleaning_pipeline()
