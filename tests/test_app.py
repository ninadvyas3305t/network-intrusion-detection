"""Unit tests for Streamlit application components and integration."""

import os
import json
import pytest

from app.app import (
    get_training_results,
    get_threshold_results,
    get_eda_summary,
    get_shap_summary,
)

def test_app_resources_load_correctly():
    """Verify that cached data loading functions in app.py retrieve valid JSON structures."""
    train_res = get_training_results()
    assert "binary_experiments" in train_res
    assert "multiclass_experiments" in train_res
    assert "XGBoost" in train_res["binary_experiments"]

    thresh_res = get_threshold_results()
    assert "test_threshold_curve" in thresh_res
    assert thresh_res["production_threshold_retained"] == 0.50

    eda_res = get_eda_summary()
    assert eda_res["total_records"] == 257673
    assert len(eda_res["high_correlations"]) > 0

    shap_res = get_shap_summary()
    assert "top_features" in shap_res
    assert len(shap_res["top_features"]) == 20

def test_app_required_figures_exist():
    """Verify all visual figures referenced by app.py are present on disk."""
    required_figures = [
        "data/processed/eda_figures/binary_class_distribution.png",
        "data/processed/eda_figures/attack_categories_distribution.png",
        "data/processed/eda_figures/correlation_heatmap.png",
        "data/processed/eda_figures/feature_distributions_by_class.png",
        "data/processed/model_figures/cm_binary_xgboost.png",
        "data/processed/model_figures/cm_binary_random_forest.png",
        "data/processed/model_figures/cm_binary_logistic_regression_baseline.png",
        "data/processed/model_figures/cm_multi_xgboost_multiclass.png",
        "data/processed/model_figures/threshold_tradeoff.png",
        "data/processed/model_figures/shap_summary_bar.png",
    ]
    for fig_path in required_figures:
        assert os.path.exists(fig_path), f"Missing figure artifact: {fig_path}"
