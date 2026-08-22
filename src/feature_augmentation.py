"""
feature_augmentation.py
------------------------
This module resolves the core ML-integrity problem found during the audit:

The shipped classifier/regressor (RandomForestClassifier / LinearRegression)
were trained on each student's FULL multi-year record: Score = mean score
across ~57-66 courses, CourseCount = ~57-66. But the app asks a student to
enter a handful of CURRENT subjects (previously hardcoded to a fake
CourseCount=8; now a genuine 1-12 entry system). Feeding CourseCount=8 or
CourseCount=12 into a model trained exclusively on CourseCount~60 is a
severe out-of-distribution extrapolation — the previous app was already
doing this (with a fabricated, hardcoded CourseCount=8 that didn't even
come from real input), and expanding the hardcoded literal to 12 would not
have fixed that; it would have been the same fabrication with a different
number.

The honest fix implemented here: retrain on a *subsampled* dataset that is
actually representative of "a student's performance across a handful of
their real courses" — built entirely from real, already-collected course
scores in data/Grade_CS_Students.xlsx, with no invented students or
scores. For each real student, we repeatedly draw random subsets of size
k (for k in 1..MAX_SUBJECTS) from that student's own real course scores,
compute Score=mean(subset) and CourseCount=k, and label the subset with
that same student's TRUE overall GoodStanding (derived from their full
record). This is a standard subsampling/augmentation technique: the labels
are real (each one traces back to an actual student's true recorded
standing), only the *feature aggregation window* is varied.

Documented limitation (see README): a k-course subsample's label is an
approximation, not literal ground truth for "how would this student have
been classified after only these k courses" — early-program performance
can differ from eventual full-program standing. This is disclosed
explicitly in the UI and README; predictions are presented as indicative,
not authoritative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_MIN_SUBJECTS = 1
DEFAULT_MAX_SUBJECTS = 12
DEFAULT_SAMPLES_PER_STUDENT_PER_K = 3
DEFAULT_RANDOM_STATE = 42


def build_subsampled_training_table(
    long_df: pd.DataFrame,
    good_standing_by_id: pd.Series,
    min_subjects: int = DEFAULT_MIN_SUBJECTS,
    max_subjects: int = DEFAULT_MAX_SUBJECTS,
    samples_per_student_per_k: int = DEFAULT_SAMPLES_PER_STUDENT_PER_K,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """
    long_df: per-student-per-course rows with a 'Score' column and 'ID'.
    good_standing_by_id: Series indexed by student ID -> 0/1 true overall
        GoodStanding label (from the FULL record, per
        data_preprocessing.build_per_student_table).

    Returns a DataFrame with columns [Score, CourseCount, GoodStanding],
    suitable for training the classifier, with CourseCount spanning
    min_subjects..max_subjects (in-distribution for the app's real
    12-subject entry form).
    """
    rng = np.random.default_rng(random_state)
    rows = []

    for student_id, group in long_df.groupby("ID"):
        scores = group["Score"].to_numpy()
        if len(scores) == 0 or student_id not in good_standing_by_id.index:
            continue
        label = good_standing_by_id.loc[student_id]

        for k in range(min_subjects, max_subjects + 1):
            if k > len(scores):
                break  # this student doesn't have enough real courses for this k
            for _ in range(samples_per_student_per_k):
                sample = rng.choice(scores, size=k, replace=False)
                rows.append({
                    "ID": student_id,
                    "Score": float(np.mean(sample)),
                    "CourseCount": k,
                    "GoodStanding": int(label),
                })

    return pd.DataFrame(rows)


def build_subsampled_regression_table(
    long_df: pd.DataFrame,
    cgpa_by_id: pd.Series,
    min_subjects: int = DEFAULT_MIN_SUBJECTS,
    max_subjects: int = DEFAULT_MAX_SUBJECTS,
    samples_per_student_per_k: int = DEFAULT_SAMPLES_PER_STUDENT_PER_K,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Same subsampling scheme, targeting the student's true overall CGPA
    (here, CGPA == GPA computed from their full record — see
    data_preprocessing.build_per_student_table) instead of the binary
    GoodStanding label, for the regression model."""
    rng = np.random.default_rng(random_state)
    rows = []

    for student_id, group in long_df.groupby("ID"):
        scores = group["Score"].to_numpy()
        if len(scores) == 0 or student_id not in cgpa_by_id.index:
            continue
        target = cgpa_by_id.loc[student_id]

        for k in range(min_subjects, max_subjects + 1):
            if k > len(scores):
                break
            for _ in range(samples_per_student_per_k):
                sample = rng.choice(scores, size=k, replace=False)
                rows.append({
                    "ID": student_id,
                    "Score": float(np.mean(sample)),
                    "CourseCount": k,
                    "CGPA": float(target),
                })

    return pd.DataFrame(rows)
