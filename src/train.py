"""Model Training & Experimentation Pipeline
Trains Baseline (Logistic Regression), Random Forest, and XGBoost models
for both Binary Classification (Primary) and Multiclass Classification (Secondary).
Strictly measures wall-clock training duration and peak memory usage.
"""

import os
import json
import time
import tracemalloc
import logging
from typing import Dict, Any, Tuple
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluate import (
    evaluate_predictions,
    measure_inference_speed,
    plot_confusion_matrix,
)

# Avoid Matplotlib cache issues
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
DATA_DIR = "data/processed"
RESULTS_FILE = os.path.join(MODELS_DIR, "training_results.json")
FIGURES_DIR = os.path.join(DATA_DIR, "model_figures")

def load_processed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Load benchmark feature matrices, binary labels, multiclass labels, and class names."""
    train_data = np.load(os.path.join(DATA_DIR, "train_features.npz"))
    test_data = np.load(os.path.join(DATA_DIR, "test_features.npz"))

    with open(os.path.join(MODELS_DIR, "feature_metadata.json"), "r") as f:
        meta = json.load(f)

    target_classes = meta["target_multiclass_classes"]

    X_train = train_data["X"]
    y_train_binary = train_data["y_binary"]
    y_train_multi = train_data["y_multi"]

    X_test = test_data["X"]
    y_test_binary = test_data["y_binary"]
    y_test_multi = test_data["y_multi"]

    logger.info(f"Loaded benchmark datasets: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, y_train_binary, y_train_multi, X_test, y_test_binary, y_test_multi, target_classes

def train_and_evaluate_model(
    model_name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    is_binary: bool = True,
    target_names: list = None,
) -> Dict[str, Any]:
    """Train a single model, measure exact training time and memory, and evaluate."""
    logger.info(f"Starting training for {model_name} (Binary: {is_binary})...")

    # Measure exact training time and memory
    tracemalloc.start()
    t_start = time.perf_counter()

    model.fit(X_train, y_train)

    train_time_sec = time.perf_counter() - t_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_memory_mb = round(peak_bytes / (1024 * 1024), 2)
    train_time_sec = round(train_time_sec, 2)
    logger.info(f"Completed {model_name} in {train_time_sec}s (Peak RAM: {peak_memory_mb} MB)")

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)

    # Multi-metric evaluation
    metrics = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        target_names=target_names,
        is_binary=is_binary,
    )

    # Latency evaluation
    latency = measure_inference_speed(model, X_test[:5000])

    result = {
        "model_name": model_name,
        "is_binary": is_binary,
        "training_time_seconds": train_time_sec,
        "peak_memory_mb": peak_memory_mb,
        "inference_latency": latency,
        "metrics": metrics,
    }

    return result

def run_all_experiments() -> Dict[str, Any]:
    """Execute complete benchmarking across models for binary and multiclass tasks."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    X_train, y_train_binary, y_train_multi, X_test, y_test_binary, y_test_multi, target_classes = load_processed_data()

    results = {
        "binary_experiments": {},
        "multiclass_experiments": {},
        "system_meta": {
            "samples_train": int(X_train.shape[0]),
            "samples_test": int(X_test.shape[0]),
            "features_count": int(X_train.shape[1]),
        }
    }

    # ==========================================
    # 1. PRIMARY TASK: Binary Classification
    # ==========================================
    logger.info("==================================================")
    logger.info("1. PRIMARY TASK: Binary Classification (Normal vs Attack)")
    logger.info("==================================================")

    binary_models = {
        "Logistic Regression (Baseline)": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=20, min_samples_split=5, n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, eval_metric="logloss", random_state=42
        ),
    }

    best_binary_name = None
    best_binary_f1 = -1.0
    best_binary_model = None

    for name, model in binary_models.items():
        res = train_and_evaluate_model(
            model_name=name,
            model=model,
            X_train=X_train,
            y_train=y_train_binary,
            X_test=X_test,
            y_test=y_test_binary,
            is_binary=True,
        )
        results["binary_experiments"][name] = res

        # Save model
        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_save_path = os.path.join(MODELS_DIR, f"{clean_name}_binary.joblib")
        joblib.dump(model, model_save_path)

        # Plot confusion matrix
        cm_path = os.path.join(FIGURES_DIR, f"cm_binary_{clean_name}.png")
        plot_confusion_matrix(
            cm=res["metrics"]["confusion_matrix"],
            labels=["Normal", "Attack"],
            title=f"Confusion Matrix: {name} (Binary)",
            output_path=cm_path,
        )

        f1 = res["metrics"]["f1"]
        if f1 > best_binary_f1:
            best_binary_f1 = f1
            best_binary_name = name
            best_binary_model = model

    logger.info(f"Best Binary Model: {best_binary_name} with F1 = {best_binary_f1:.4f}")
    joblib.dump(best_binary_model, os.path.join(MODELS_DIR, "best_model_binary.joblib"))
    results["best_binary_model"] = best_binary_name

    # ==========================================
    # 2. SECONDARY TASK: Multiclass Classification
    # ==========================================
    logger.info("==================================================")
    logger.info("2. SECONDARY TASK: Multiclass Classification (10 Classes)")
    logger.info("==================================================")

    multi_models = {
        "Logistic Regression (Multiclass)": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", class_weight="balanced", random_state=42
        ),
        "Random Forest (Multiclass)": RandomForestClassifier(
            n_estimators=100, max_depth=20, min_samples_split=5, class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "XGBoost (Multiclass)": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1,
            eval_metric="mlogloss", random_state=42
        ),
    }

    best_multi_name = None
    best_multi_macro_f1 = -1.0
    best_multi_model = None

    for name, model in multi_models.items():
        res = train_and_evaluate_model(
            model_name=name,
            model=model,
            X_train=X_train,
            y_train=y_train_multi,
            X_test=X_test,
            y_test=y_test_multi,
            is_binary=False,
            target_names=target_classes,
        )
        results["multiclass_experiments"][name] = res

        # Save model
        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_save_path = os.path.join(MODELS_DIR, f"{clean_name}.joblib")
        joblib.dump(model, model_save_path)

        # Plot confusion matrix
        cm_path = os.path.join(FIGURES_DIR, f"cm_multi_{clean_name}.png")
        plot_confusion_matrix(
            cm=res["metrics"]["confusion_matrix"],
            labels=target_classes,
            title=f"Confusion Matrix: {name} (10 Classes)",
            output_path=cm_path,
            figsize=(10, 8),
        )

        macro_f1 = res["metrics"]["f1_macro"]
        if macro_f1 > best_multi_macro_f1:
            best_multi_macro_f1 = macro_f1
            best_multi_name = name
            best_multi_model = model

    logger.info(f"Best Multiclass Model: {best_multi_name} with Macro-F1 = {best_multi_macro_f1:.4f}")
    joblib.dump(best_multi_model, os.path.join(MODELS_DIR, "best_model_multiclass.joblib"))
    results["best_multiclass_model"] = best_multi_name

    # Save complete JSON results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved complete experiment results to {RESULTS_FILE}")

    return results

if __name__ == "__main__":
    run_all_experiments()
