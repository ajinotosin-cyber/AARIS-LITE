"""
feature_engineering.py
------------------------
Turns a list of validated subject entries into every derived value the
app needs: per-subject letter grades, GPA, and the Score/CourseCount
feature pair the ML models expect.

This is the one place that computation happens -- previously "average
score" and "the ML model's CourseCount" were computed in two completely
disconnected ways (a real per-course GPA calculator that never touched
the model, and a model-feeding code path that used a hardcoded
course_count = 8 unrelated to anything the student entered). Now both the
GPA the student sees and the features the model sees come from the exact
same entered subjects.
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from validation import SubjectEntry


@dataclass
class SubjectResult:
    name: str
    score: float
    grade: str
    grade_point: int
    units: int


@dataclass
class AcademicSummary:
    subjects: list[SubjectResult]
    subject_count: int
    total_units: int
    average_score: float
    gpa: float
    cgpa: float
    degree_class: str


def assign_grade(score: float) -> str:
    for threshold, grade in config.GRADE_CUTOFFS:
        if score >= threshold:
            return grade
    return "F"


def classify_degree(cgpa: float) -> str:
    for threshold, label in config.DEGREE_CLASSIFICATION_CUTOFFS:
        if cgpa >= threshold:
            return label
    return config.DEGREE_CLASSIFICATION_DEFAULT


def build_academic_summary(subjects: list[SubjectEntry]) -> AcademicSummary:
    """The single function both the prediction flow and the report
    generator call -- guarantees the average score, GPA, and subject
    count they display are always consistent with each other.

    GPA is credit-unit-weighted (grade_point x units, summed, divided by
    total units) -- the standard university formula, and the actual
    reason a "Units" field was added: a 2-unit elective and a 6-unit
    core course should not count equally toward GPA. Average Score is
    deliberately left as a plain (unweighted) mean of the raw scores --
    it feeds classifier_model.pkl / regression_model.pkl / anomaly_model.pkl
    via model_features() below, which were trained on a plain average and
    were NOT retrained with a units-aware feature, so this must stay
    exactly as those models expect it regardless of the GPA change."""
    results = []
    for s in subjects:
        grade = assign_grade(s.score)
        results.append(SubjectResult(
            name=s.name, score=s.score, grade=grade, grade_point=config.GRADE_POINTS[grade],
            units=s.units,
        ))

    subject_count = len(results)
    total_units = sum(r.units for r in results)
    average_score = sum(r.score for r in results) / subject_count if subject_count else 0.0
    gpa = sum(r.grade_point * r.units for r in results) / total_units if total_units else 0.0

    return AcademicSummary(
        subjects=results,
        subject_count=subject_count,
        total_units=total_units,
        average_score=average_score,
        gpa=gpa,
        # CGPA uses the identical credit-weighted formula as GPA -- this
        # app only ever evaluates one set of entered courses at a time
        # (no persisted multi-semester history), so there is no
        # separate "cumulative across other semesters" data to combine
        # with; they are numerically equal here by design, not by bug.
        cgpa=gpa,
        degree_class=classify_degree(gpa),
    )


def combine_cgpa(current_gpa: float, current_units: int, previous_cgpa: float, previous_units: int) -> float:
    """The standard, real formula universities use to roll a new
    semester's GPA into a running cumulative CGPA: unweight the previous
    CGPA back into total quality points earned so far (previous_cgpa *
    previous_units), add this semester's quality points (current_gpa *
    current_units), then re-weight by the combined units.

    combined_CGPA = (previous_cgpa * previous_units + current_gpa * current_units)
                    / (previous_units + current_units)

    Both "GPA" (this semester alone) and "CGPA" (this blended, running
    total) are meaningfully different once previous-semester data is
    provided -- unlike the single-semester case, where they're
    numerically identical by definition (see build_academic_summary)."""
    total_units = previous_units + current_units
    if total_units <= 0:
        return current_gpa
    previous_quality_points = previous_cgpa * previous_units
    current_quality_points = current_gpa * current_units
    return (previous_quality_points + current_quality_points) / total_units


def model_features(summary: AcademicSummary) -> list[float]:
    """The exact [Score, CourseCount] feature vector the classifier and
    regressor expect, derived from the SAME summary shown to the user."""
    return [summary.average_score, summary.subject_count]
