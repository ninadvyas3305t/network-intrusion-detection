"""Explainability Module for Network Intrusion Detection
Provides both Global and Local (Instance-Level) explanations using SHAP TreeExplainer
and XGBoost native feature importances.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Avoid Matplotlib cache issues
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
FIGURES_DIR = "data/processed/model_figures"
DATA_DIR = "data/processed"

FEATURE_DESCRIPTIONS = {
    "sttl": "Source to destination Time-To-Live (TTL value)",
    "dttl": "Destination to source Time-To-Live (TTL value)",
    "smean": "Mean packet size transmitted by source",
    "dmean": "Mean packet size transmitted by destination",
    "sbytes": "Total source to destination transaction bytes",
    "dbytes": "Total destination to source transaction bytes",
    "rate": "Total packets per second transmission rate",
    "dur": "Total flow connection duration in seconds",
    "sloss": "Source packet drop / retransmission count",
    "dloss": "Destination packet drop / retransmission count",
    "synack": "TCP SYN-ACK handshake response latency",
    "ackdat": "TCP ACK-to-DATA final handshake delay",
    "tcprtt": "TCP round-trip latency (synack + ackdat)",
    "ct_srv_src": "Number of connections from same source IP to same service in last 100 flows",
    "ct_srv_dst": "Number of connections to same destination IP and service in last 100 flows",
    "ct_dst_ltm": "Connections to destination IP address within past 100 flows",
    "ct_src_dport_ltm": "Connections from source to same destination port",
    "ct_dst_sport_ltm": "Connections to destination from same source port",
    "ct_dst_src_ltm": "Connections having same source and destination IP",
    "ct_state_ttl": "State and TTL combination identifier count",
}

class IntrusionExplainer:
    """Provides SHAP TreeExplainer and feature attribution for binary and multiclass models."""

    def __init__(
        self,
        model_path: str = os.path.join(MODELS_DIR, "best_model_binary.joblib"),
        metadata_path: str = os.path.join(MODELS_DIR, "feature_metadata.json"),
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        self.model = joblib.load(model_path)
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata["encoded_feature_names"]
        self.explainer = shap.TreeExplainer(self.model)

    def explain_instance(self, sample_1d: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """Compute local SHAP explanation for a single network flow instance.
        sample_1d: 1D numpy array of length 194 (preprocessed features).
        """
        if sample_1d.ndim == 1:
            X = sample_1d.reshape(1, -1)
        else:
            X = sample_1d

        # Model prediction and probability
        prob = float(self.model.predict_proba(X)[0, 1])
        pred_label = "Attack" if prob >= 0.50 else "Normal"

        # Compute SHAP values
        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list):
            # If binary returned as list of arrays for [class0, class1]
            s_val = shap_vals[1][0]
        elif shap_vals.ndim == 2:
            s_val = shap_vals[0]
        else:
            s_val = shap_vals[0]

        base_val = float(self.explainer.expected_value) if hasattr(self.explainer, "expected_value") else 0.0
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[-1])

        # Separate positive contributors (pushing towards Attack) and negative (pushing towards Normal)
        contributions = []
        for i, val in enumerate(s_val):
            fname = self.feature_names[i]
            base_feat = fname.split("__")[-1]
            desc = FEATURE_DESCRIPTIONS.get(base_feat, f"Encoded network feature: {fname}")
            contributions.append({
                "feature": fname,
                "base_feature": base_feat,
                "description": desc,
                "shap_value": round(float(val), 4),
                "feature_value": round(float(X[0, i]), 4),
            })

        # Sort by absolute impact
        contributions_sorted = sorted(contributions, key=lambda x: abs(x["shap_value"]), reverse=True)
        top_factors = contributions_sorted[:top_k]

        pushing_attack = [c for c in contributions_sorted if c["shap_value"] > 0][:top_k]
        pushing_normal = [c for c in contributions_sorted if c["shap_value"] < 0][:top_k]

        return {
            "prediction": pred_label,
            "attack_probability": round(prob, 4),
            "normal_probability": round(1.0 - prob, 4),
            "base_value": round(base_val, 4),
            "top_contributing_factors": top_factors,
            "top_pushing_attack": pushing_attack,
            "top_pushing_normal": pushing_normal,
        }

    def compute_and_save_global_summary(
        self,
        test_features_path: str = os.path.join(DATA_DIR, "test_features.npz"),
        sample_size: int = 1000,
        output_json: str = os.path.join(MODELS_DIR, "shap_summary.json"),
        output_fig: str = os.path.join(FIGURES_DIR, "shap_summary_bar.png"),
    ) -> Dict[str, Any]:
        """Compute global SHAP importance over a representative test sample and save artifacts."""
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        os.makedirs(os.path.dirname(output_fig), exist_ok=True)

        data = np.load(test_features_path)
        X_sample = data["X"][:sample_size]

        logger.info(f"Computing global SHAP values over {sample_size} test samples...")
        shap_vals = self.explainer.shap_values(X_sample)
        if isinstance(shap_vals, list):
            s_matrix = shap_vals[1]
        else:
            s_matrix = shap_vals

        mean_abs_shap = np.abs(s_matrix).mean(axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1]

        top_20 = []
        for rank, idx in enumerate(sorted_indices[:20], 1):
            fname = self.feature_names[idx]
            base_feat = fname.split("__")[-1]
            desc = FEATURE_DESCRIPTIONS.get(base_feat, f"Feature: {fname}")
            top_20.append({
                "rank": rank,
                "feature": fname,
                "base_feature": base_feat,
                "mean_abs_shap": round(float(mean_abs_shap[idx]), 4),
                "description": desc,
            })

        # Generate plot
        top_15 = top_20[:15]
        names = [f["feature"] for f in top_15][::-1]
        vals = [f["mean_abs_shap"] for f in top_15][::-1]

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(names, vals, color="#2b5c8f", edgecolor="black")
        ax.set_title("Top 15 Most Influential Features (SHAP Global Importance)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean |SHAP Value| (Average Impact on Model Output)")
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", ha="left", va="center", fontsize=9)

        plt.tight_layout()
        fig.savefig(output_fig, dpi=300)
        plt.close(fig)
        logger.info(f"Saved global SHAP bar plot to {output_fig}")

        summary_data = {
            "sample_size": sample_size,
            "total_features": len(self.feature_names),
            "top_features": top_20,
        }

        with open(output_json, "w") as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"Saved global SHAP summary JSON to {output_json}")

        return summary_data

def generate_shap_artifacts():
    """CLI / programmatic generator for SHAP artifacts."""
    explainer = IntrusionExplainer()
    explainer.compute_and_save_global_summary()

if __name__ == "__main__":
    generate_shap_artifacts()
