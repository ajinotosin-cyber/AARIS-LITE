"""
evaluation.py
--------------
Metrics for the classifier and regressor, computed on a real held-out
test split -- used by model_training.py so retraining reports honest
numbers instead of none at all (this file was an empty stub before this
pass).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    r2_score, mean_squared_error, mean_absolute_error,
)


def evaluate_classifier(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "report": classification_report(y_test, pred, zero_division=0),
    }


def evaluate_regressor(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    return {
        "r2": r2_score(y_test, pred),
        "rmse": rmse,
        "mae": mean_absolute_error(y_test, pred),
    }


def print_classifier_report(name: str, metrics: dict) -> None:
    print(f"\n===== {name}: classifier evaluation =====")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(metrics["report"])


def print_regressor_report(name: str, metrics: dict) -> None:
    print(f"\n===== {name}: regressor evaluation =====")
    print(f"R2  : {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE : {metrics['mae']:.4f}")
