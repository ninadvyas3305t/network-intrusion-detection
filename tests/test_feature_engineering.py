"""Unit tests for feature engineering pipeline and transformer artifacts."""

import os
import joblib
import pytest
import numpy as np
import pandas as pd
from src.feature_engineering import (
    build_preprocessor,
    get_feature_lists,
    CATEGORICAL_FEATURES,
    TARGET_BINARY,
    TARGET_MULTICLASS,
)

@pytest.fixture
def sample_feature_dataframe():
    """Create a synthetic dataset matching the schema of cleaned UNSW-NB15."""
    data = {
        "dur": [0.1, 1.5, 0.02],
        "spkts": [10, 50, 4],
        "dpkts": [8, 45, 0],
        "sbytes": [500, 4000, 200],
        "dbytes": [800, 6000, 0],
        "rate": [100.0, 50.0, 200.0],
        "sttl": [64, 32, 64],
        "dttl": [60, 0, 0],
        "proto": ["tcp", "udp", "tcp"],
        "service": ["http", "none", "dns"],
        "state": ["fin", "con", "int"],
        "attack_cat": ["Normal", "DoS", "Exploits"],
        "label": [0, 1, 1],
    }
    return pd.DataFrame(data)

def test_get_feature_lists(sample_feature_dataframe):
    """Verify numeric and categorical features are extracted accurately."""
    num_cols, cat_cols = get_feature_lists(sample_feature_dataframe)
    assert set(cat_cols) == {"proto", "service", "state"}
    assert "label" not in num_cols
    assert "attack_cat" not in num_cols
    assert "dur" in num_cols

def test_preprocessor_transformation(sample_feature_dataframe):
    """Verify ColumnTransformer scales and one-hot encodes correctly."""
    num_cols, cat_cols = get_feature_lists(sample_feature_dataframe)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    X_raw = sample_feature_dataframe[num_cols + cat_cols]
    preprocessor.fit(X_raw)
    X_trans = preprocessor.transform(X_raw)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == 3
    # No NaNs or Infs
    assert not np.isnan(X_trans).any()
    assert not np.isinf(X_trans).any()

def test_unseen_category_handling(sample_feature_dataframe):
    """Verify handle_unknown='ignore' avoids crashing on novel categories."""
    num_cols, cat_cols = get_feature_lists(sample_feature_dataframe)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    X_train = sample_feature_dataframe[num_cols + cat_cols]
    preprocessor.fit(X_train)

    # Create unseen data instance
    unseen_data = X_train.copy()
    unseen_data.loc[0, "proto"] = "exotic_protocol_999"
    unseen_data.loc[0, "service"] = "unknown_service_xyz"

    # Should not raise exception
    X_unseen_trans = preprocessor.transform(unseen_data)
    assert X_unseen_trans.shape == (3, preprocessor.transform(X_train).shape[1])
    assert not np.isnan(X_unseen_trans).any()

def test_saved_artifacts_load_and_run():
    """Verify saved joblib artifacts exist and can perform inference."""
    preprocessor_path = "models/preprocessor.joblib"
    label_encoder_path = "models/label_encoder.joblib"

    assert os.path.exists(preprocessor_path)
    assert os.path.exists(label_encoder_path)

    preprocessor = joblib.load(preprocessor_path)
    label_encoder = joblib.load(label_encoder_path)

    assert hasattr(preprocessor, "transform")
    assert hasattr(label_encoder, "inverse_transform")
    assert len(label_encoder.classes_) == 10
    assert "Normal" in label_encoder.classes_
