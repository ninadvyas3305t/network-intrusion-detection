"""Unit tests for inference engine, custom thresholding, and SHAP explainability."""

import os
import pytest
import pandas as pd
import numpy as np

from src.predict import IntrusionPredictor
from src.explainability import IntrusionExplainer

@pytest.fixture(scope="module")
def predictor():
    """Shared IntrusionPredictor instance for testing."""
    return IntrusionPredictor()

@pytest.fixture(scope="module")
def sample_test_record():
    """Load a single real flow sample from the raw test partition."""
    df = pd.read_csv("data/raw/UNSW_NB15_testing-set.csv", nrows=1)
    return df.iloc[0].to_dict()

def test_predictor_initialization(predictor):
    """Verify that predictor loads all transformers and models properly."""
    assert predictor.preprocessor is not None
    assert predictor.binary_model is not None
    assert predictor.multi_model is not None
    assert predictor.label_encoder is not None
    assert len(predictor.required_input_features) == 42
    assert len(predictor.classes) == 10

def test_validate_inputs(predictor):
    """Verify input validation detects missing features."""
    valid_df = pd.DataFrame([{"dur": 0.1, "proto": "tcp", "service": "http", "state": "fin"}])
    # Should flag missing columns since full 42 features are required
    is_valid, missing = predictor.validate_inputs(valid_df)
    assert not is_valid
    assert len(missing) > 0

def test_predict_batch_structure(predictor):
    """Verify predict_batch produces expected output columns and probability bounds."""
    df = pd.read_csv("data/raw/UNSW_NB15_testing-set.csv", nrows=10)
    res = predictor.predict_batch(df)

    assert "prediction" in res.columns
    assert "attack_probability" in res.columns
    assert "normal_probability" in res.columns
    assert "predicted_category" in res.columns
    assert "category_confidence" in res.columns

    assert set(res["prediction"]).issubset({"Normal", "Attack"})
    assert (res["attack_probability"] >= 0.0).all() and (res["attack_probability"] <= 1.0).all()
    assert (res["normal_probability"] >= 0.0).all() and (res["normal_probability"] <= 1.0).all()
    assert np.allclose(res["attack_probability"] + res["normal_probability"], 1.0, atol=1e-3)

def test_predict_single_with_explanation(predictor, sample_test_record):
    """Verify single instance prediction produces valid prediction and SHAP explanation."""
    res = predictor.predict_single(sample_test_record, with_explanation=True)

    assert res["prediction"] in ["Normal", "Attack"]
    assert 0.0 <= res["attack_probability"] <= 1.0
    assert res["predicted_category"] in predictor.classes
    assert "explanation" in res

    expl = res["explanation"]
    assert "top_contributing_factors" in expl
    assert len(expl["top_contributing_factors"]) > 0
    for factor in expl["top_contributing_factors"]:
        assert "feature" in factor
        assert "shap_value" in factor
        assert "description" in factor

def test_custom_threshold_effect(predictor, sample_test_record):
    """Verify that shifting decision threshold influences binary decision boundary."""
    # Run with very low threshold (sensitive to attack)
    res_low = predictor.predict_single(sample_test_record, threshold=0.01)
    # Run with very high threshold (conservative)
    res_high = predictor.predict_single(sample_test_record, threshold=0.99)

    prob = res_low["attack_probability"]
    if 0.01 <= prob < 0.99:
        assert res_low["prediction"] == "Attack"
        assert res_high["prediction"] == "Normal"

def test_global_shap_artifacts_exist():
    """Verify precomputed global SHAP summary file and plot exist."""
    summary_path = "models/shap_summary.json"
    plot_path = "data/processed/model_figures/shap_summary_bar.png"

    assert os.path.exists(summary_path)
    assert os.path.exists(plot_path)

def test_threshold_analysis_artifacts_exist():
    """Verify threshold analysis JSON and trade-off plot exist."""
    analysis_path = "models/threshold_analysis.json"
    plot_path = "data/processed/model_figures/threshold_tradeoff.png"

    assert os.path.exists(analysis_path)
    assert os.path.exists(plot_path)
