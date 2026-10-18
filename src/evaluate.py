"""Model Evaluation Module
Provides comprehensive multi-metric evaluations:
- Accuracy, Precision, Recall, F1-Score (Weighted and Macro)
- Per-class metrics highlighting rare classes (Worms, Shellcode)
- ROC-AUC calculations (Binary and Multiclass OvR)
- Confusion matrix generation and visualization
- Wall-clock inference latency measurement
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Avoid Matplotlib cache permission warnings
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EVAL_FIG_DIR = "data/processed/model_figures"

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
    is_binary: bool = True,
) -> Dict[str, Any]:
    """Compute comprehensive multi-metric evaluation dictionary."""
    acc = float(accuracy_score(y_true, y_pred))

    if is_binary:
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        roc_auc = None
        if y_prob is not None:
            try:
                # If 2D probability matrix, use probability of positive class
                prob_pos = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] > 1 else y_prob
                roc_auc = float(roc_auc_score(y_true, prob_pos))
            except Exception as e:
                logger.warning(f"Could not compute binary ROC-AUC: {e}")

        cm = confusion_matrix(y_true, y_pred).tolist()
        class_report = classification_report(
            y_true, y_pred, target_names=["Normal", "Attack"], output_dict=True, zero_division=0
        )

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "confusion_matrix": cm,
            "classification_report": class_report,
        }

    else:
        # Multiclass evaluation
        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        roc_auc_macro = None
        if y_prob is not None:
            try:
                roc_auc_macro = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            except Exception as e:
                logger.warning(f"Could not compute multiclass ROC-AUC: {e}")

        cm = confusion_matrix(y_true, y_pred).tolist()
        class_report = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
        )

        return {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_weighted": round(f1_weighted, 4),
            "roc_auc_macro": round(roc_auc_macro, 4) if roc_auc_macro is not None else None,
            "confusion_matrix": cm,
            "classification_report": class_report,
        }

def measure_inference_speed(model, X_sample: np.ndarray, num_iterations: int = 5) -> Dict[str, float]:
    """Measure inference latency per sample and total batch runtime."""
    total_samples = len(X_sample)
    latencies = []

    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = model.predict(X_sample)
        latencies.append(time.perf_counter() - t0)

    avg_batch_time_s = float(np.mean(latencies))
    avg_per_sample_ms = float((avg_batch_time_s / total_samples) * 1000)

    return {
        "batch_inference_time_seconds": round(avg_batch_time_s, 4),
        "per_sample_latency_ms": round(avg_per_sample_ms, 5),
    }

def plot_confusion_matrix(
    cm: List[List[int]],
    labels: List[str],
    title: str,
    output_path: str,
    figsize: tuple = (7, 6)
) -> str:
    """Plot and save confusion matrix heatmap."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm_arr = np.array(cm)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=10, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
