"""Unit tests for trained model loading, inference pipelines, and output formats."""

import os
import json
import joblib
import pytest
import numpy as np

MODELS_DIR = "models"
RESULTS_FILE = os.path.join(MODELS_DIR, "training_results.json")

def test_artifacts_exist():
    """Verify all serialized model artifacts and metadata exist."""
    required_files = [
        os.path.join(MODELS_DIR, "best_model_binary.joblib"),
        os.path.join(MODELS_DIR, "best_model_multiclass.joblib"),
        os.path.join(MODELS_DIR, "preprocessor.joblib"),
        os.path.join(MODELS_DIR, "label_encoder.joblib"),
        os.path.join(MODELS_DIR, "feature_metadata.json"),
        RESULTS_FILE,
    ]
    for filepath in required_files:
        assert os.path.exists(filepath), f"Missing artifact: {filepath}"

def test_training_results_content():
    """Verify training results contains valid measured metrics."""
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    assert "binary_experiments" in data
    assert "multiclass_experiments" in data
    assert "Logistic Regression (Baseline)" in data["binary_experiments"]
    assert "Random Forest" in data["binary_experiments"]
    assert "XGBoost" in data["binary_experiments"]

    # Verify real metrics are present and non-zero
    xgb_binary = data["binary_experiments"]["XGBoost"]["metrics"]
    assert 0.80 <= xgb_binary["accuracy"] <= 1.0
    assert 0.80 <= xgb_binary["f1"] <= 1.0
    assert data["binary_experiments"]["XGBoost"]["training_time_seconds"] > 0
    assert data["binary_experiments"]["XGBoost"]["peak_memory_mb"] > 0

def test_binary_model_inference():
    """Verify best binary model predicts accurately on transformed synthetic input."""
    model = joblib.load(os.path.join(MODELS_DIR, "best_model_binary.joblib"))
    
    # 194 features input
    dummy_input = np.zeros((3, 194), dtype=np.float32)
    preds = model.predict(dummy_input)
    probs = model.predict_proba(dummy_input)

    assert len(preds) == 3
    assert set(preds).issubset({0, 1})
    assert probs.shape == (3, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

def test_multiclass_model_inference():
    """Verify best multiclass model produces valid class indices and probabilities."""
    model = joblib.load(os.path.join(MODELS_DIR, "best_model_multiclass.joblib"))
    encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))

    dummy_input = np.zeros((3, 194), dtype=np.float32)
    preds = model.predict(dummy_input)
    probs = model.predict_proba(dummy_input)

    assert len(preds) == 3
    assert probs.shape == (3, 10)
    assert np.allclose(probs.sum(axis=1), 1.0)

    # Invert predictions to string names
    class_names = encoder.inverse_transform(preds)
    assert len(class_names) == 3
    for name in class_names:
        assert isinstance(name, str)
