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
import statistics
from dataclasses import dataclass
from typing import Optional

import pandas as pd

import config
import feature_engineering

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


def compute_cgpa(summary, previous_record=None) -> PredictionResult:
    """CGPA is a deterministic, credit-unit-weighted calculation -- not
    an ML prediction. This replaces an earlier version (predict_cgpa)
    that used regression_model.pkl -- a model trained on only
    [Score, CourseCount], with no knowledge of credit units at all, and
    therefore structurally incapable of producing a genuinely
    credit-weighted CGPA.

    previous_record: an optional validation.PreviousRecord (cgpa +
    total_units from prior semesters). When provided, CGPA is the real
    university formula for a running cumulative average -- this
    semester's GPA blended with the prior cumulative record via
    feature_engineering.combine_cgpa(). When omitted (the common case
    for a first-time user with no prior semester to combine), CGPA
    equals this semester's GPA exactly -- correct, not a fallback
    approximation, since with only one semester's worth of courses
    there is nothing to distinguish "this semester" from "cumulative".

    Kept as a PredictionResult (status + predicted_cgpa) rather than a
    bare float so every existing call site (app.py, report_generator.py)
    that already handles STATUS_OK/predicted_cgpa needed no restructuring
    -- only the status can now never be anything but STATUS_OK, since a
    deterministic calculation over already-validated input has no
    "model unavailable" failure mode."""
    if previous_record is not None:
        combined = feature_engineering.combine_cgpa(
            current_gpa=summary.gpa, current_units=summary.total_units,
            previous_cgpa=previous_record.cgpa, previous_units=previous_record.total_units,
        )
        return PredictionResult(status=STATUS_OK, predicted_cgpa=combined)
    return PredictionResult(status=STATUS_OK, predicted_cgpa=summary.cgpa)


def detect_anomalous_subjects(subjects, models: ModelBundle) -> PredictionResult:
    """Flags individual subject scores that are statistically inconsistent
    with the SAME student's own other submitted scores -- a per-student
    contextual check, not a comparison against the training population.

    This replaces an earlier version that used anomaly_model.pkl (an
    Isolation Forest trained on the global historical score distribution)
    to flag any score in roughly the top percentile of that population --
    which meant ordinary, believable scores like 80 were flagged purely
    for being statistically rare in the training data, with zero
    knowledge of what was normal for the specific student being
    evaluated. That model is no longer used for this check (models
    parameter is accepted for call-site compatibility but intentionally
    unused): a trained model has nothing to offer here that plain
    descriptive statistics on the student's own scores doesn't already
    do better, since the whole point is per-student context, not
    population membership.

    Logic (see config.py for the named thresholds):
      1. A score is only even a CANDIDATE if it's >= HIGH_SCORE_CONFIRMATION_FLOOR.
      2. The check only runs at all when there are >= MIN_SUBJECTS_FOR_SCORE_CONFIRMATION
         total scores -- below that, there isn't enough of the student's own
         data to say what's statistically normal for them, so nothing is flagged.
      3. Otherwise: compare the candidate score against the mean/stdev of the
         student's OTHER scores (excluding itself). If it's >=
         SCORE_CONFIRMATION_Z_THRESHOLD standard deviations above that mean,
         confirmation is recommended. If the other scores have zero variance
         (e.g. all identical), fall back to a conservative absolute-gap check
         instead of dividing by zero.
    """
    if not subjects:
        return PredictionResult(status=STATUS_OK, anomalous_subjects=[])

    try:
        scores = [float(s.score) for s in subjects]
        n = len(scores)
        flagged: list[str] = []

        if n >= config.MIN_SUBJECTS_FOR_SCORE_CONFIRMATION:
            for i, subject in enumerate(subjects):
                if scores[i] < config.HIGH_SCORE_CONFIRMATION_FLOOR:
                    continue

                others = scores[:i] + scores[i + 1:]
                other_mean = statistics.mean(others)

                if len(others) >= 2:
                    other_stdev = statistics.stdev(others)
                else:
                    other_stdev = 0.0

                if other_stdev > 0:
                    z_score = (scores[i] - other_mean) / other_stdev
                    is_inconsistent = z_score >= config.SCORE_CONFIRMATION_Z_THRESHOLD
                else:
                    # Zero variance among the other scores (e.g. every
                    # other score is identical) -- no meaningful Z-score
                    # possible, so fall back to a conservative absolute gap.
                    is_inconsistent = (
                        scores[i] - other_mean
                    ) >= config.SCORE_CONFIRMATION_ZERO_VARIANCE_GAP

                if is_inconsistent:
                    flagged.append(subject.name)

        return PredictionResult(status=STATUS_OK, anomalous_subjects=flagged)
    except Exception as exc:
        return PredictionResult(status=STATUS_PREDICTION_FAILED, error_message=f"Score confirmation check failed: {exc}")
