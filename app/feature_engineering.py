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


@dataclass
class AcademicSummary:
    subjects: list[SubjectResult]
    subject_count: int
    average_score: float
    gpa: float
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
    count they display are always consistent with each other."""
    results = []
    for s in subjects:
        grade = assign_grade(s.score)
        results.append(SubjectResult(
            name=s.name, score=s.score, grade=grade, grade_point=config.GRADE_POINTS[grade],
        ))

    subject_count = len(results)
    average_score = sum(r.score for r in results) / subject_count if subject_count else 0.0
    gpa = sum(r.grade_point for r in results) / subject_count if subject_count else 0.0

    return AcademicSummary(
        subjects=results,
        subject_count=subject_count,
        average_score=average_score,
        gpa=gpa,
        degree_class=classify_degree(gpa),
    )


def model_features(summary: AcademicSummary) -> list[float]:
    """The exact [Score, CourseCount] feature vector the classifier and
    regressor expect, derived from the SAME summary shown to the user."""
    return [summary.average_score, summary.subject_count]
