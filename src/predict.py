"""Inference Pipeline for Network Intrusion Detection
Provides end-to-end prediction engine for:
1. Batch CSV inference
2. Single-flow / manual feature inference
3. Validation of input columns
4. Preprocessing using identical saved transformers
5. Binary & Multiclass predictions with probability scores
6. Local SHAP instance explanation integration
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import sys
import joblib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.explainability import IntrusionExplainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"

class IntrusionPredictor:
    """End-to-end inference engine loading saved preprocessor and model artifacts."""

    def __init__(
        self,
        models_dir: str = MODELS_DIR,
        binary_model_file: str = "best_model_binary.joblib",
        multi_model_file: str = "best_model_multiclass.joblib",
    ):
        self.models_dir = models_dir

        preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
        label_encoder_path = os.path.join(models_dir, "label_encoder.joblib")
        metadata_path = os.path.join(models_dir, "feature_metadata.json")
        binary_path = os.path.join(models_dir, binary_model_file)
        multi_path = os.path.join(models_dir, multi_model_file)

        for p in [preprocessor_path, label_encoder_path, metadata_path, binary_path, multi_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required artifact not found: {p}")

        logger.info(f"Loading inference pipeline artifacts from {models_dir}...")
        self.preprocessor = joblib.load(preprocessor_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.binary_model = joblib.load(binary_path)
        self.multi_model = joblib.load(multi_path)

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.num_cols = self.metadata["raw_numerical_features"]
        self.cat_cols = self.metadata["raw_categorical_features"]
        self.required_input_features = self.num_cols + self.cat_cols
        self.classes = self.label_encoder.classes_.tolist()

        # Lazy initialized explainer
        self._explainer: Optional[IntrusionExplainer] = None

    @property
    def explainer(self) -> IntrusionExplainer:
        """Lazy loader for SHAP explainer."""
        if self._explainer is None:
            self._explainer = IntrusionExplainer(
                model_path=os.path.join(self.models_dir, "best_model_binary.joblib"),
                metadata_path=os.path.join(self.models_dir, "feature_metadata.json"),
            )
        return self._explainer

    def validate_inputs(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate whether incoming dataframe has required features.
        Returns (is_valid, missing_columns).
        """
        missing = [col for col in self.required_input_features if col not in df.columns]
        return len(missing) == 0, missing

    def clean_incoming_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize incoming dataframe strings and formats."""
        df_clean = df.copy()

        # Drop ID / targets if present in uploaded test files
        for col in ["id", "label", "attack_cat"]:
            if col in df_clean.columns:
                df_clean = df_clean.drop(columns=[col])

        # Clean categorical columns
        for col in self.cat_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
                if col == "service":
                    df_clean[col] = df_clean[col].replace({"-": "none"})
            else:
                # Default fill if missing
                df_clean[col] = "none" if col == "service" else ("tcp" if col == "proto" else "fin")

        # Fill any missing numerical features with 0.0
        for col in self.num_cols:
            if col not in df_clean.columns:
                df_clean[col] = 0.0
            else:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0.0)

        # Enforce column order
        return df_clean[self.required_input_features]

    def predict_batch(
        self,
        df: pd.DataFrame,
        threshold: float = 0.50,
    ) -> pd.DataFrame:
        """Run batch inference on a DataFrame of network flows.
        Returns input dataframe augmented with predictions and probabilities.
        """
        df_clean = self.clean_incoming_data(df)
        X_trans = self.preprocessor.transform(df_clean).astype(np.float32)

        # 1. Binary Predictions
        binary_probs = self.binary_model.predict_proba(X_trans)
        attack_probs = binary_probs[:, 1]
        binary_preds = np.where(attack_probs >= threshold, "Attack", "Normal")

        # 2. Multiclass Predictions
        multi_probs = self.multi_model.predict_proba(X_trans)
        multi_pred_indices = np.argmax(multi_probs, axis=1)
        multi_confidences = np.max(multi_probs, axis=1)
        multi_preds = self.label_encoder.inverse_transform(multi_pred_indices)

        # Build output dataframe
        df_result = df.copy()
        df_result["prediction"] = binary_preds
        df_result["attack_probability"] = np.round(attack_probs, 4)
        df_result["normal_probability"] = np.round(1.0 - attack_probs, 4)
        df_result["predicted_category"] = multi_preds
        df_result["category_confidence"] = np.round(multi_confidences, 4)

        return df_result

    def predict_single(
        self,
        record: Union[Dict[str, Any], pd.Series],
        threshold: float = 0.50,
        with_explanation: bool = True,
    ) -> Dict[str, Any]:
        """Run inference on a single network flow record with optional SHAP explanation."""
        if isinstance(record, dict):
            df_single = pd.DataFrame([record])
        else:
            df_single = pd.DataFrame([record.to_dict()])

        df_clean = self.clean_incoming_data(df_single)
        X_trans = self.preprocessor.transform(df_clean).astype(np.float32)

        # Binary
        prob_attack = float(self.binary_model.predict_proba(X_trans)[0, 1])
        binary_pred = "Attack" if prob_attack >= threshold else "Normal"

        # Multiclass
        multi_probs = self.multi_model.predict_proba(X_trans)[0]
        multi_idx = int(np.argmax(multi_probs))
        multi_conf = float(multi_probs[multi_idx])
        multi_pred = str(self.label_encoder.inverse_transform([multi_idx])[0])

        result = {
            "prediction": binary_pred,
            "attack_probability": round(prob_attack, 4),
            "normal_probability": round(1.0 - prob_attack, 4),
            "threshold_used": threshold,
            "predicted_category": multi_pred,
            "category_confidence": round(multi_conf, 4),
            "all_category_probabilities": {
                cls: round(float(multi_probs[i]), 4)
                for i, cls in enumerate(self.classes)
            },
        }

        if with_explanation:
            explanation = self.explainer.explain_instance(X_trans[0], top_k=5)
            result["explanation"] = explanation

        return result

if __name__ == "__main__":
    predictor = IntrusionPredictor()
    test_raw = pd.read_csv("data/raw/UNSW_NB15_testing-set.csv", nrows=5)
    sample_preds = predictor.predict_batch(test_raw)
    print("Sample Batch Predictions:")
    print(sample_preds[["prediction", "attack_probability", "predicted_category", "category_confidence"]])

    print("\nSingle Prediction with Explanation:")
    single_res = predictor.predict_single(test_raw.iloc[0].to_dict())
    print("Prediction:", single_res["prediction"], "| Conf:", single_res["attack_probability"])
    print("Top factors:", [f["feature"] for f in single_res["explanation"]["top_contributing_factors"]])
