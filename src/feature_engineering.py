"""Feature Engineering & Preprocessing Pipeline
Defines and executes the leak-free preprocessing transformer:
- Standardizes numerical flow metrics using StandardScaler
- One-hot encodes categorical attributes ('proto', 'service', 'state') with handle_unknown='ignore'
- Encodes multiclass target ('attack_cat') with LabelEncoder
- Preserves official benchmark split strictly
- Serializes pipeline artifacts and metadata for inference
"""

import os
import json
import logging
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TRAIN_FILE = "data/processed/train_cleaned.csv"
DEFAULT_TEST_FILE = "data/processed/test_cleaned.csv"
MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"

CATEGORICAL_FEATURES = ["proto", "service", "state"]
TARGET_BINARY = "label"
TARGET_MULTICLASS = "attack_cat"

def get_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Extract ordered numerical and categorical feature names excluding targets."""
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    targets = [TARGET_BINARY, TARGET_MULTICLASS]
    num_cols = [c for c in df.columns if c not in cat_cols and c not in targets]
    return num_cols, cat_cols

def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Construct ColumnTransformer for numerical scaling and categorical one-hot encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), cat_cols),
        ],
        remainder="drop"
    )
    return preprocessor

def run_feature_pipeline(
    train_path: str = DEFAULT_TRAIN_FILE,
    test_path: str = DEFAULT_TEST_FILE,
    models_dir: str = MODELS_DIR,
    output_dir: str = PROCESSED_DIR,
) -> Dict[str, Any]:
    """Execute feature engineering pipeline:
    1. Load cleaned benchmark splits
    2. Fit preprocessor strictly on train features
    3. Transform train and test features
    4. Fit and transform multiclass LabelEncoder on train
    5. Save serialized preprocessors, feature metadata, and processed arrays
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading cleaned datasets for feature transformation...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    num_cols, cat_cols = get_feature_lists(train_df)
    logger.info(f"Identified {len(num_cols)} numerical features and {len(cat_cols)} categorical features.")

    # Separate features and targets
    X_train_raw = train_df[num_cols + cat_cols]
    y_train_binary = train_df[TARGET_BINARY].to_numpy(dtype=np.int32)
    y_train_multi_raw = train_df[TARGET_MULTICLASS].to_numpy()

    X_test_raw = test_df[num_cols + cat_cols]
    y_test_binary = test_df[TARGET_BINARY].to_numpy(dtype=np.int32)
    y_test_multi_raw = test_df[TARGET_MULTICLASS].to_numpy()

    # Fit preprocessor strictly on X_train_raw to prevent leakage
    logger.info("Fitting ColumnTransformer on training features...")
    preprocessor = build_preprocessor(num_cols, cat_cols)
    preprocessor.fit(X_train_raw)

    logger.info("Transforming train and test feature matrices...")
    X_train_trans = preprocessor.transform(X_train_raw).astype(np.float32)
    X_test_trans = preprocessor.transform(X_test_raw).astype(np.float32)

    # Encode multiclass targets
    logger.info("Fitting LabelEncoder on multiclass target...")
    label_encoder = LabelEncoder()
    y_train_multi = label_encoder.fit_transform(y_train_multi_raw)
    y_test_multi = label_encoder.transform(y_test_multi_raw)

    # Extract all output feature names
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + cat_feature_names

    logger.info(f"Transformed train shape: {X_train_trans.shape}")
    logger.info(f"Transformed test shape:  {X_test_trans.shape}")
    logger.info(f"Total encoded features:  {len(all_feature_names)}")

    # Save artifacts
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    label_encoder_path = os.path.join(models_dir, "label_encoder.joblib")
    metadata_path = os.path.join(models_dir, "feature_metadata.json")

    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(label_encoder, label_encoder_path)
    logger.info(f"Saved preprocessor to {preprocessor_path}")
    logger.info(f"Saved label encoder to {label_encoder_path}")

    # Build and save feature metadata
    metadata = {
        "raw_numerical_features": num_cols,
        "raw_categorical_features": cat_cols,
        "encoded_feature_names": all_feature_names,
        "total_encoded_features": len(all_feature_names),
        "target_binary_classes": [0, 1],
        "target_multiclass_classes": label_encoder.classes_.tolist(),
        "class_to_index": {cls: int(idx) for idx, cls in enumerate(label_encoder.classes_)},
        "index_to_class": {int(idx): cls for idx, cls in enumerate(label_encoder.classes_)},
        "train_samples": int(X_train_trans.shape[0]),
        "test_samples": int(X_test_trans.shape[0]),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved feature metadata to {metadata_path}")

    # Save compressed arrays for rapid model training
    train_arrays_path = os.path.join(output_dir, "train_features.npz")
    test_arrays_path = os.path.join(output_dir, "test_features.npz")

    np.savez_compressed(
        train_arrays_path,
        X=X_train_trans,
        y_binary=y_train_binary,
        y_multi=y_train_multi,
    )
    np.savez_compressed(
        test_arrays_path,
        X=X_test_trans,
        y_binary=y_test_binary,
        y_multi=y_test_multi,
    )
    logger.info(f"Saved compressed feature arrays to {train_arrays_path} and {test_arrays_path}")

    return metadata

if __name__ == "__main__":
    run_feature_pipeline()
