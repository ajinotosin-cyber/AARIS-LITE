"""
config.py
---------
Shared constants for the AARIS-LITE Streamlit app. Single source of truth
for values that were previously scattered as magic numbers throughout
app.py (MAX_COURSES=8 in one place, a hardcoded course_count=8 in
another, grade cutoffs duplicated conceptually between the app and the
training notebook).
"""

import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))

REGRESSION_MODEL_PATH = os.path.join(APP_DIR, "regression_model.pkl")
CLASSIFIER_MODEL_PATH = os.path.join(APP_DIR, "classifier_model.pkl")
ANOMALY_MODEL_PATH = os.path.join(APP_DIR, "anomaly_model.pkl")

# Subject-entry capacity. This used to be two disconnected constants
# (MAX_COURSES = 8 for the GPA calculator's dynamic rows, and a
# hardcoded course_count = 8 fed straight into the classifier,
# unrelated to any real input) -- now a single constant governing the
# one, unified subject-entry component every feature in the app uses.
MIN_SUBJECTS = 1
MAX_SUBJECTS = 12

# Score domain the models were trained on (raw 0-100 course scores).
MIN_SCORE = 0.0
MAX_SCORE = 100.0

GRADE_CUTOFFS = (
    (70, "A"),
    (60, "B"),
    (50, "C"),
    (45, "D"),
    (40, "E"),
)
GRADE_POINTS = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}

GOOD_STANDING_CGPA_THRESHOLD = 2.5

DEGREE_CLASSIFICATION_CUTOFFS = (
    (4.50, "First Class Honours"),
    (3.50, "Second Class Upper"),
    (2.40, "Second Class Lower"),
    (1.50, "Third Class"),
)
DEGREE_CLASSIFICATION_DEFAULT = "Academic Probation"

MAX_SEMESTERS = 10
