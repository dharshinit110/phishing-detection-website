"""
Train a machine learning model for phishing website detection.

Steps:
    * Load dataset from CSV
    * Extract URL-based features
    * Split into train/test sets
    * Train an XGBClassifier (default), RandomForest, or LogisticRegression
    * Evaluate accuracy on the test set
    * Persist the trained model to disk via joblib
"""

from __future__ import annotations

import os
import argparse
from typing import Tuple, List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .feature_extraction import build_feature_matrix


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET_PATH = os.path.join(
    PROJECT_ROOT, "dataset", "uci_phishing_websites", "Training Dataset.arff"
)
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.pkl")


def _read_arff_to_dataframe(arff_path: str) -> pd.DataFrame:
    """
    Minimal ARFF reader for the UCI 'Phishing Websites' dataset.

    Supports:
      - @attribute declarations
      - @data section with comma-separated values
    """
    if not os.path.exists(arff_path):
        raise FileNotFoundError(f"Dataset not found at '{arff_path}'")

    with open(arff_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("%")]

    attributes: List[str] = []
    data_started = False
    rows: List[List[str]] = []

    for ln in lines:
        lower = ln.lower()
        if lower.startswith("@attribute"):
            # Format: @attribute name { ... }  OR  @attribute name numeric
            parts = ln.split()
            if len(parts) < 2:
                continue
            attr_name = parts[1].strip()
            # Remove quotes if present
            if (attr_name.startswith("'") and attr_name.endswith("'")) or (
                attr_name.startswith('"') and attr_name.endswith('"')
            ):
                attr_name = attr_name[1:-1]
            attributes.append(attr_name)
        elif lower.startswith("@data"):
            data_started = True
        elif data_started and not lower.startswith("@"):
            rows.append([x.strip() for x in ln.split(",")])

    if not attributes or not rows:
        raise ValueError(
            "ARFF parsing failed (no attributes or data rows found). "
            "Please verify the dataset file."
        )

    df = pd.DataFrame(rows, columns=attributes)
    # Convert everything to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].replace("?", np.nan), errors="coerce")
    df = df.dropna()
    return df


def _load_url_csv_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV dataset must contain 'url' and 'label' columns.")
    X = build_feature_matrix(df["url"].tolist())
    y = df["label"].astype(int).values
    return X, y


def _load_uci_arff_url_subset(arff_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load the UCI ARFF dataset and keep only URL-derivable subset features.

    We train on a subset so the Streamlit app can still accept a raw URL.
    """
    df = _read_arff_to_dataframe(arff_path)
    required_cols = [
        "having_IP_Address",
        "URL_Length",
        "Shortining_Service",
        "having_At_Symbol",
        "Prefix_Suffix",
        "having_Sub_Domain",
        "SSLfinal_State",
        "HTTPS_token",
        "Result",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"ARFF dataset missing columns: {missing}")

    feature_cols = required_cols[:-1]
    X = df[feature_cols].astype(int).to_numpy()

    # UCI encoding: Result {-1, 1} where -1 = phishing, 1 = legitimate
    result = df["Result"].astype(int).to_numpy()
    y = np.where(result == -1, 1, 0).astype(int)

    return X, y, feature_cols


def load_dataset(dataset_path: str = DEFAULT_DATASET_PATH) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the phishing dataset.

    Supported formats:
      - CSV with columns: url,label (label: 0 legit, 1 phishing)
      - ARFF (UCI Phishing Websites dataset)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at '{dataset_path}'")

    ext = os.path.splitext(dataset_path)[1].lower()
    if ext == ".csv":
        X, y = _load_url_csv_dataset(dataset_path)
        meta = {"feature_mode": "custom_url_8", "feature_names": [
            "url_length",
            "num_dots",
            "has_at_symbol",
            "ip_in_domain",
            "uses_https",
            "subdomain_count",
            "hyphen_in_domain",
            "suspicious_keyword",
        ]}
        return X, y, meta

    if ext == ".arff":
        X, y, feature_names = _load_uci_arff_url_subset(dataset_path)
        meta = {"feature_mode": "uci_url_subset_8", "feature_names": feature_names}
        return X, y, meta

    raise ValueError(f"Unsupported dataset format: {ext}. Use .csv or .arff")


def build_model(model_type: str = "xgboost"):
    """
    Build and return a model instance for phishing detection.

    Supported model_type values:
        - 'xgboost' (default)
        - 'random_forest'
        - 'logistic_regression'
    """
    model_type = model_type.lower()
    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def train_and_evaluate(
    model_type: str = "xgboost",
    test_size: float = 0.2,
    random_state: int = 42,
    dataset_path: str = DEFAULT_DATASET_PATH,
) -> Tuple[object, float]:
    """
    Train the given model on the dataset and evaluate on a test split.

    Returns:
        model: trained model instance
        accuracy: accuracy score on the test set
    """
    X, y, meta = load_dataset(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    estimator = build_model(model_type=model_type)
    estimator.fit(X_train, y_train)

    y_pred = estimator.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    model_bundle = {
        "model": estimator,
        "meta": meta,
    }

    return model_bundle, acc


def save_model(model, model_path: str = DEFAULT_MODEL_PATH) -> None:
    """
    Persist a trained model to disk.
    """
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


def parse_args() -> argparse.Namespace:
    """
    CLI argument parser to control training behavior.
    """
    parser = argparse.ArgumentParser(
        description="Train a phishing website detection model."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to CSV dataset (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Where to save the trained model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["xgboost", "random_forest", "logistic_regression"],
        default="xgboost",
        help="Type of model to train.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of dataset to use as test set.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Train a model using CLI arguments and save it to disk.
    """
    args = parse_args()
    model, acc = train_and_evaluate(
        model_type=args.model_type,
        test_size=args.test_size,
        dataset_path=args.dataset,
    )
    save_model(model, args.model_path)
    print(f"Training completed. Final accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()

