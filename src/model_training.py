"""
model_training.py
-------------------
Retrains all three AARIS-LITE models: the RandomForestClassifier (good
standing), the LinearRegression regressor (predicted CGPA), and the
IsolationForest anomaly detector (unusual individual subject scores).

WHY THIS SCRIPT CHANGED
------------------------
The version of this script previously in the repo would not even run
against data/Grade_CS_Students.xlsx -- it read a "Score"/"CourseCoount"/
"GPA" columns that don't exist in that file (the raw file is wide-format:
one column per course, not a pre-aggregated "Score" column). It also
didn't match the models actually shipped in app/: the shipped classifier
was trained on 2 features (['Score', 'CourseCount']), but this script
trained one on a single feature (['Score']) -- confirmed by inspecting
the real classifier_model.pkl's .feature_names_in_ and cross-checking
against notebook/AARIS_Notebook.ipynb, which is what the shipped models
were actually produced from.

More importantly, the notebook's own methodology -- and the previously
shipped models -- computed Score/CourseCount from a student's FULL
academic record (all ~57-66 real courses they have scores for). The app
only ever asks a student to enter a handful of CURRENT subjects (1-12).
Feeding CourseCount=8 or 12 into a model trained exclusively on
CourseCount~60 is a severe out-of-distribution extrapolation.

This script now trains on data built by feature_augmentation.py's
subsampling approach: for each real student, repeatedly draw random
k-sized subsets (k = 1..12) of that same student's own real course
scores, average them, and label the subset with that student's TRUE
overall standing/CGPA (from their full record, via
data_preprocessing.py). Every label traces back to a real, recorded
student outcome; only the size of the aggregation window is varied, so
the trained models are genuinely in-distribution for the app's 1-12
subject entry form. See feature_augmentation.py's module docstring for
the full reasoning and its documented limitation.

USAGE
-----
    cd src
    python model_training.py

Or from anywhere:
    python /path/to/AARIS-LITE/src/model_training.py \
        --dataset /path/to/Grade_CS_Students.xlsx \
        --output-dir /path/to/app
"""

from __future__ import annotations

import argparse
import os
import pickle

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import data_preprocessing as dp
import evaluation as ev
import feature_augmentation as fa

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "data", "Grade_CS_Students.xlsx")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "app")

FEATURE_COLUMNS = ["Score", "CourseCount"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def train_classifier(subsampled_df):
    X = subsampled_df[FEATURE_COLUMNS]
    y = subsampled_df["GoodStanding"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    metrics = ev.evaluate_classifier(model, X_test, y_test)
    ev.print_classifier_report("Good-standing classifier", metrics)
    return model, metrics


def train_regressor(subsampled_df):
    X = subsampled_df[FEATURE_COLUMNS]
    y = subsampled_df["CGPA"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    metrics = ev.evaluate_regressor(model, X_test, y_test)
    ev.print_regressor_report("CGPA regressor", metrics)
    return model, metrics


def train_anomaly_model(long_df):
    """Fit on the FULL population of real individual course scores (not
    subsampled) -- this model answers a different question ("is this one
    subject score unusual?"), for which the full real distribution is the
    right training set, not a per-student aggregation window."""
    X = long_df[["Score"]]
    model = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
    model.fit(X)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH,
                         help="Path to Grade_CS_Students.xlsx")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                         help="Directory to write the three .pkl files into")
    parser.add_argument("--min-subjects", type=int, default=fa.DEFAULT_MIN_SUBJECTS)
    parser.add_argument("--max-subjects", type=int, default=fa.DEFAULT_MAX_SUBJECTS)
    parser.add_argument("--samples-per-student-per-k", type=int,
                         default=fa.DEFAULT_SAMPLES_PER_STUDENT_PER_K)
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset} ...")
    long_df, per_student = dp.load_and_build(args.dataset)
    print(f"Long-format rows: {len(long_df)} | Students: {len(per_student)}")

    good_standing_by_id = per_student.set_index("ID")["GoodStanding"]
    cgpa_by_id = per_student.set_index("ID")["CGPA"]

    print("\nBuilding in-distribution (1-{}-subject) subsampled training tables..."
          .format(args.max_subjects))
    cls_table = fa.build_subsampled_training_table(
        long_df, good_standing_by_id,
        min_subjects=args.min_subjects, max_subjects=args.max_subjects,
        samples_per_student_per_k=args.samples_per_student_per_k,
    )
    reg_table = fa.build_subsampled_regression_table(
        long_df, cgpa_by_id,
        min_subjects=args.min_subjects, max_subjects=args.max_subjects,
        samples_per_student_per_k=args.samples_per_student_per_k,
    )
    print(f"Classifier training rows: {len(cls_table)} (CourseCount range: "
          f"{cls_table['CourseCount'].min()}-{cls_table['CourseCount'].max()})")
    print(f"Regressor training rows : {len(reg_table)} (CourseCount range: "
          f"{reg_table['CourseCount'].min()}-{reg_table['CourseCount'].max()})")

    clf_model, clf_metrics = train_classifier(cls_table)
    reg_model, reg_metrics = train_regressor(reg_table)
    anomaly_model = train_anomaly_model(long_df)

    os.makedirs(args.output_dir, exist_ok=True)
    paths = {
        "classifier_model.pkl": clf_model,
        "regression_model.pkl": reg_model,
        "anomaly_model.pkl": anomaly_model,
    }
    for filename, model in paths.items():
        out_path = os.path.join(args.output_dir, filename)
        with open(out_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved {out_path}")

    print("\nRetraining complete. All three models now expect features "
          f"{FEATURE_COLUMNS} (classifier/regressor) and ['Score'] (anomaly), "
          f"and are trained in-distribution for {args.min_subjects}-"
          f"{args.max_subjects} subject entries.")


if __name__ == "__main__":
    main()
