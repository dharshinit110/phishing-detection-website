"""
Prediction utilities for phishing website detection.

This module exposes a simple `predict_url` function that:
    * Loads the trained model from disk
    * Extracts URL-based features
    * Returns the predicted class and optional confidence score
"""

from __future__ import annotations

import os
from typing import Tuple

import joblib
import numpy as np

from .feature_extraction import extract_url_features, extract_uci_url_subset_features


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.pkl")


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model file cannot be found."""


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load the trained model from disk.
    """
    if not os.path.exists(model_path):
        raise ModelNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Train the model first by running 'python -m src.train_model'."
        )
    return joblib.load(model_path)


def preprocess_url(url: str) -> np.ndarray:
    """
    Convert a raw URL string into a feature vector suitable for prediction.
    """
    features = extract_url_features(url)
    return np.asarray(features, dtype=float).reshape(1, -1)


def _preprocess_url_for_bundle(url: str, feature_mode: str) -> np.ndarray:
    """
    Convert a raw URL into the feature vector expected by the trained model bundle.
    """
    if feature_mode == "custom_url_8":
        feats = extract_url_features(url)
        return np.asarray(feats, dtype=float).reshape(1, -1)

    if feature_mode == "uci_url_subset_8":
        feats = extract_uci_url_subset_features(url)
        return np.asarray(feats, dtype=float).reshape(1, -1)

    raise ValueError(f"Unsupported feature_mode in model: {feature_mode}")


def predict_url(url: str, model_path: str = DEFAULT_MODEL_PATH) -> Tuple[int, float]:
    """
    Predict whether the given URL is phishing or legitimate.

    Returns:
        label: 0 for legitimate, 1 for phishing
        confidence: model's estimated probability for the predicted class
    """
    if not isinstance(url, str) or not url.strip():
        # Invalid URL input: treat as legitimate with low confidence
        return 0, 0.0

    loaded = load_model(model_path)

    # Backward compatibility: accept either raw estimator or a bundle dict
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        meta = loaded.get("meta", {})
        feature_mode = meta.get("feature_mode", "custom_url_8")
        X = _preprocess_url_for_bundle(url, feature_mode)
    else:
        model = loaded
        X = preprocess_url(url)

    # Some models (e.g. RandomForest, LogisticRegression) implement predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        label = int(np.argmax(proba))
        confidence = float(np.max(proba))
    else:
        label = int(model.predict(X)[0])
        confidence = 0.0

    return label, confidence


__all__ = [
    "predict_url",
    "load_model",
    "ModelNotFoundError",
]

