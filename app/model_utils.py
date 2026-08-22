"""
model_utils.py
---------------
Model loading and inference for AARIS-LITE. No Streamlit import here --
kept independent so it can be unit tested directly.

Every possible failure mode gets its own explicit status; nothing is
ever silently presented as a normal "GOOD standing" or otherwise
favourable result if analysis genuinely failed.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Optional

import pandas as pd

import config

STATUS_OK = "OK"
STATUS_MODELS_UNAVAILABLE = "MODELS_UNAVAILABLE"
STATUS_PREDICTION_FAILED = "PREDICTION_FAILED"

STANDING_GOOD = "GOOD"
STANDING_AT_RISK = "AT RISK"


@dataclass
class ModelBundle:
    classifier: Optional[object] = None
    regressor: Optional[object] = None
    anomaly_model: Optional[object] = None
    load_error: Optional[str] = None

    @property
    def classifier_available(self) -> bool:
        return self.classifier is not None

    @property
    def regressor_available(self) -> bool:
        return self.regressor is not None

    @property
    def anomaly_model_available(self) -> bool:
        return self.anomaly_model is not None

    @property
    def any_available(self) -> bool:
        return self.classifier_available or self.regressor_available or self.anomaly_model_available


def _load_one(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_models() -> ModelBundle:
    """Never raises. Loads each of the three models independently, so a
    problem with one (e.g. a corrupted anomaly_model.pkl) doesn't take
    down the classifier or regressor too. Any failures are recorded in
    load_error for the UI to display honestly."""
    bundle = ModelBundle()
    errors = []

    for attr, path, label in (
        ("classifier", config.CLASSIFIER_MODEL_PATH, "classifier"),
        ("regressor", config.REGRESSION_MODEL_PATH, "regression"),
        ("anomaly_model", config.ANOMALY_MODEL_PATH, "anomaly-detection"),
    ):
        try:
            setattr(bundle, attr, _load_one(path))
        except FileNotFoundError:
            errors.append(f"{label} model file not found ({path}).")
        except (pickle.UnpicklingError, EOFError, ValueError) as exc:
            errors.append(f"{label} model file is corrupted or incompatible: {exc}")
        except Exception as exc:  # last resort -- never let a load failure crash the app
            errors.append(f"Unexpected error loading {label} model: {exc}")

    if errors:
        bundle.load_error = " ".join(errors)
    return bundle


@dataclass
class PredictionResult:
    status: str
    standing: Optional[str] = None
    predicted_cgpa: Optional[float] = None
    anomalous_subjects: Optional[list] = None
    error_message: str = ""


def predict_standing(features: list[float], models: ModelBundle) -> PredictionResult:
    if not models.classifier_available:
        return PredictionResult(
            status=STATUS_MODELS_UNAVAILABLE,
            error_message=models.load_error or "Classifier model is unavailable.",
        )
    try:
        X = pd.DataFrame([features], columns=["Score", "CourseCount"])
        prediction = models.classifier.predict(X)[0]
    except Exception as exc:
        return PredictionResult(status=STATUS_PREDICTION_FAILED, error_message=f"Prediction failed: {exc}")

    standing = STANDING_GOOD if int(prediction) == 1 else STANDING_AT_RISK
    return PredictionResult(status=STATUS_OK, standing=standing)


def predict_cgpa(features: list[float], models: ModelBundle) -> PredictionResult:
    if not models.regressor_available:
        return PredictionResult(
            status=STATUS_MODELS_UNAVAILABLE,
            error_message=models.load_error or "Regression model is unavailable.",
        )
    try:
        X = pd.DataFrame([features], columns=["Score", "CourseCount"])
        predicted = float(models.regressor.predict(X)[0])
    except Exception as exc:
        return PredictionResult(status=STATUS_PREDICTION_FAILED, error_message=f"Prediction failed: {exc}")

    return PredictionResult(status=STATUS_OK, predicted_cgpa=predicted)


def detect_anomalous_subjects(subjects, models: ModelBundle) -> PredictionResult:
    """subjects: list of feature_engineering.SubjectResult (or anything
    with .name and .score). Flags individual subject scores that look
    like outliers relative to the training distribution of real course
    scores -- this genuinely uses anomaly_model.pkl, which the previous
    app loaded but never actually used."""
    if not models.anomaly_model_available:
        return PredictionResult(
            status=STATUS_MODELS_UNAVAILABLE,
            error_message=models.load_error or "Anomaly-detection model is unavailable.",
        )
    if not subjects:
        return PredictionResult(status=STATUS_OK, anomalous_subjects=[])

    try:
        X = pd.DataFrame([[s.score] for s in subjects], columns=["Score"])
        flags = models.anomaly_model.predict(X)
    except Exception as exc:
        return PredictionResult(status=STATUS_PREDICTION_FAILED, error_message=f"Anomaly detection failed: {exc}")

    anomalous = [s.name for s, flag in zip(subjects, flags) if flag == -1]
    return PredictionResult(status=STATUS_OK, anomalous_subjects=anomalous)
