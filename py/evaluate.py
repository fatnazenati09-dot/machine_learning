from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score


def get_project_dir() -> Path:
    """
    Return the root project directory.
    """
    return Path(__file__).resolve().parent.parent


def get_outputs_dir() -> Path:
    """
    Return outputs directory and create it if needed.
    """
    path = get_project_dir() / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate a fitted binary classification model on the test set.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
    return metrics


def evaluate_with_cv(model, X_train, y_train, cv, scoring: str = "f1") -> tuple[float, float]:
    """
    Evaluate a model with cross-validation on the training data only.
    """
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
    return float(np.mean(scores)), float(np.std(scores))


def print_evaluation(name: str, metrics: dict, cv_mean: float | None = None, cv_std: float | None = None) -> None:
    """
    Print evaluation metrics in a readable way.
    """
    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")

    if cv_mean is not None:
        print(f"CV Mean F1: {cv_mean:.4f}")
    if cv_std is not None:
        print(f"CV Std F1: {cv_std:.4f}")

    print("\nClassification report:")
    print(metrics["classification_report"])
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])


def build_results_row(name: str, metrics: dict, cv_mean: float | None = None, cv_std: float | None = None) -> dict:
    """
    Build one row for a comparison table.
    """
    row = {
        "Model": name,
        "Test Accuracy": round(metrics["accuracy"], 4),
        "Test ROC-AUC": round(metrics["roc_auc"], 4),
        "Test Precision": round(metrics["precision"], 4),
        "Test Recall": round(metrics["recall"], 4),
        "Test F1": round(metrics["f1"], 4),
    }

    if cv_mean is not None:
        row["CV Mean F1"] = round(cv_mean, 4)
    if cv_std is not None:
        row["CV Std F1"] = round(cv_std, 4)

    return row


def save_results_csv(df: pd.DataFrame, filename: str) -> Path:
    """
    Save a dataframe to outputs/metrics.
    """
    metrics_dir = get_outputs_dir() / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    filepath = metrics_dir / filename
    df.to_csv(filepath, index=False)
    return filepath


def save_model(model, filename: str) -> Path:
    """
    Save a trained model to outputs/models.
    """
    models_dir = get_outputs_dir() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    filepath = models_dir / filename
    joblib.dump(model, filepath)
    return filepath


def load_model(filename: str):
    """
    Load a trained model from outputs/models.
    """
    filepath = get_outputs_dir() / "models" / filename
    return joblib.load(filepath)