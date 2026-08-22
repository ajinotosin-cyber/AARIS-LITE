"""
data_preprocessing.py
----------------------
Turns the raw wide-format transcript spreadsheet (one row per student, one
column per course, ~66 course columns) into:

  1. long_df -- one row per (student, course) with a real, non-null Score.
  2. A per-student full-record summary: Score (mean across ALL their real
     courses), CourseCount (how many real courses they have), GPA (mean
     grade point across those courses), CGPA (== GPA here, since this
     dataset has one snapshot per student, not per semester), and
     GoodStanding (CGPA >= the app's own threshold).

This file was an empty stub before this pass -- feature_augmentation.py
already documented exactly the long_df/label-series contract it needed,
but nothing produced them. This is that missing piece.

Grade cutoffs, grade points, and the good-standing threshold are imported
from app/config.py rather than redefined here, so retraining can never
silently drift from the thresholds the live app actually uses. This
mirrors and formalizes the methodology already used once, informally, in
notebook/AARIS_Notebook.ipynb (verified against output/AARIS_results.csv,
which this function's output reproduces).
"""

from __future__ import annotations

import os
import sys

import pandas as pd

# app/ is a sibling directory to src/, and is not a Python package (no
# __init__.py, by design -- it's the Streamlit app's own root, which
# Streamlit puts on sys.path when it runs app.py directly). Adding it to
# sys.path here is what lets training reuse the exact same config
# constants and grade-assignment logic the live app uses, instead of a
# second, potentially-drifting copy.
_APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import config  # noqa: E402  (app/config.py, via the sys.path shim above)
from feature_engineering import assign_grade  # noqa: E402  (app/feature_engineering.py)

ID_COLUMNS = ("Year of enrolment", "ID")


def load_raw_wide_table(xlsx_path: str) -> pd.DataFrame:
    """Loads the raw spreadsheet and forward-fills 'Year of enrolment',
    which is only populated on the first row of each enrolment cohort in
    the source file (as seen in the notebook's own preprocessing)."""
    df = pd.read_excel(xlsx_path)
    if "Year of enrolment" in df.columns:
        df["Year of enrolment"] = df["Year of enrolment"].ffill()
    return df


def to_long_format(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Melts the wide per-course table into one row per (student, course),
    dropping the many NaN cells for courses a given student never took."""
    id_vars = [c for c in ID_COLUMNS if c in wide_df.columns]
    long_df = wide_df.melt(id_vars=id_vars, var_name="Course", value_name="Score")
    long_df = long_df.dropna(subset=["Score"])
    long_df["Score"] = long_df["Score"].astype(float)
    return long_df


def build_per_student_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per student with columns:
        ID, Score (mean), MaxScore, MinScore, ScoreVar, CourseCount,
        GPA, CGPA, GoodStanding

    This is the FULL-RECORD summary (every real course the student has a
    score for) -- NOT the in-distribution 1-12 subject representation the
    app's models are actually trained on. That subsampled version is built
    by feature_augmentation.py from this table's long_df input and the
    GoodStanding/CGPA columns produced here.
    """
    long_df = long_df.copy()
    long_df["Grade"] = long_df["Score"].apply(assign_grade)
    long_df["GradePoint"] = long_df["Grade"].map(config.GRADE_POINTS)

    stats = long_df.groupby("ID")["Score"].agg(
        Score="mean", MaxScore="max", MinScore="min", ScoreVar="var",
    ).reset_index()
    course_count = long_df.groupby("ID").size().reset_index(name="CourseCount")
    gpa = long_df.groupby("ID")["GradePoint"].mean().reset_index().rename(columns={"GradePoint": "GPA"})

    table = stats.merge(course_count, on="ID").merge(gpa, on="ID")
    table["CGPA"] = table["GPA"]  # one snapshot per student in this dataset -- CGPA == GPA
    table["GoodStanding"] = (table["CGPA"] >= config.GOOD_STANDING_CGPA_THRESHOLD).astype(int)
    return table


def load_and_build(xlsx_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience entry point: raw xlsx path -> (long_df, per_student_table)."""
    wide = load_raw_wide_table(xlsx_path)
    long_df = to_long_format(wide)
    per_student = build_per_student_table(long_df)
    return long_df, per_student
