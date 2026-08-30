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

# Course/credit units -- standard range for a single university course
# (a typical course is 1-6 units; the upper bound is generous enough to
# also cover unusually large courses like a final-year project without
# being unbounded).
MIN_COURSE_UNITS = 1
MAX_COURSE_UNITS = 10
DEFAULT_COURSE_UNITS = 3

# Course unit (credit hour) domain, matching how real universities weight
# a course's contribution to GPA -- most institutions use 1-6 units per
# course, with 3 being the most common single-course load.
MIN_UNIT = 1
MAX_UNIT = 6
DEFAULT_UNIT = 3

GRADE_CUTOFFS = (
    (70, "A"),
    (60, "B"),
    (50, "C"),
    (45, "D"),
    (40, "E"),
)
GRADE_POINTS = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}

# Previous academic record (optional) -- lets a returning student combine
# a prior cumulative CGPA + total units earned so far with this
# semester's newly-entered courses, producing a genuine running CGPA
# rather than one that resets to just the current semester every time.
MIN_PREVIOUS_CGPA = 0.0
MAX_PREVIOUS_CGPA = float(max(GRADE_POINTS.values()))
MIN_PREVIOUS_UNITS = 1
MAX_PREVIOUS_UNITS = 500

GOOD_STANDING_CGPA_THRESHOLD = 2.5

DEGREE_CLASSIFICATION_CUTOFFS = (
    (4.50, "First Class Honours"),
    (3.50, "Second Class Upper"),
    (2.40, "Second Class Lower"),
    (1.50, "Third Class"),
)
DEGREE_CLASSIFICATION_DEFAULT = "Academic Probation"

MAX_SEMESTERS = 10

# ---------------------------------------------------------------------------
# High-score confirmation logic (see model_utils.detect_anomalous_subjects)
# ---------------------------------------------------------------------------
# An ordinary high score (75, 80, 85...) is a normal, believable academic
# result and must never be flagged just for being high. A score only
# becomes a CANDIDATE for confirmation once it's at least this value --
# and even then, only if it's also statistically inconsistent with the
# specific student's OWN other scores (see Z-score threshold below), not
# merely because it falls in this range. Originally 95.0; lowered to 90.0
# after real testing showed 90-100 is the range that should realistically
# be considered "exceptionally high" for this purpose.
HIGH_SCORE_CONFIRMATION_FLOOR = 90.0

# Below this many total course scores, there isn't enough of the
# student's own data to say anything statistically meaningful about what's
# "consistent" for them -- a 2-3 point sample doesn't support a reliable
# mean/spread estimate. Below this count, NO confirmation check is run,
# regardless of how high any individual score is (three scores of 95, 96,
# 97 must not be flagged just because there isn't enough context to say
# they're unusual -- there also isn't enough to say they're normal).
# Originally 5; lowered to 4 to match a realistic minimum semester course
# load (a common real submission size) while still excluding the 3-course
# case above, which genuinely is too small a sample.
MIN_SUBJECTS_FOR_SCORE_CONFIRMATION = 4

# How many standard deviations above the student's OWN other-score mean a
# high score must be before it's considered statistically inconsistent
# with their established performance. 2.0 (roughly the 97.5th percentile
# of a normal distribution) -- verified directly (not assumed) against a
# 5-score submission of [92, 95, 90, 98, 100]: a lower value (1.5) that
# looked reasonable on paper actually flagged this genuinely consistent
# cluster as inconsistent, which contradicts the whole point of this
# check. 2.0 correctly leaves it unflagged while still catching a
# genuine outlier (e.g. 98 among four scores of 55-70) by a wide margin,
# confirmed the same way.
SCORE_CONFIRMATION_Z_THRESHOLD = 2.0

# Fallback absolute gap (score points) used only when the student's other
# scores have zero variance (e.g. three identical scores) and a Z-score
# can't be computed at all -- avoids a division by zero while still
# applying a conservative, defensible check rather than skipping entirely.
SCORE_CONFIRMATION_ZERO_VARIANCE_GAP = 10.0
