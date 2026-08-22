"""
validation.py
--------------
Input validation for the AARIS-LITE app. Previously the app trusted all
input implicitly (e.g. a Course Grade selectbox has no way to be
invalid, but course scores, matric numbers, and subject names were never
checked at all in the old prediction flow, which didn't even collect
per-subject input).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import config


@dataclass
class SubjectEntry:
    name: str
    score: float


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_score(raw_value, field_label: str = "Score") -> tuple[bool, str]:
    """Returns (is_valid, error_message). Handles non-numeric and
    out-of-range input explicitly rather than letting Streamlit's
    number_input silently clamp or the app crash downstream."""
    if raw_value is None:
        return False, f"{field_label} is required."
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return False, f"{field_label} must be a number."
    if value < config.MIN_SCORE or value > config.MAX_SCORE:
        return False, f"{field_label} must be between {config.MIN_SCORE:.0f} and {config.MAX_SCORE:.0f}."
    return True, ""


def validate_subject_name(name: str) -> tuple[bool, str]:
    name = (name or "").strip()
    if not name:
        return False, "Course name is required."
    if len(name) > 60:
        return False, "Course name is too long."
    return True, ""


def validate_subject_entries(raw_entries: list[tuple[str, object]]) -> ValidationResult:
    """
    raw_entries: list of (subject_name, raw_score) pairs, exactly as
    collected from the UI (including blank/incomplete rows).

    Rules:
      - Completely blank rows (no name AND no score) are silently
        ignored -- they're just unused capacity in the up-to-12 grid,
        not an error.
      - A row with a name but an invalid/missing score, or a score but
        no name, is a genuine validation error.
      - Duplicate subject names (case-insensitive) are rejected -- two
        entries for the same subject aren't meaningful for the average.
      - At least one complete, valid subject is required overall.
    """
    result = ValidationResult(valid=True)
    seen_names = set()
    complete_count = 0

    for idx, (raw_name, raw_score) in enumerate(raw_entries, start=1):
        name = (raw_name or "").strip()
        has_name = bool(name)
        has_score = raw_score is not None and str(raw_score).strip() != ""

        if not has_name and not has_score:
            continue  # unused row, not an error

        if has_name and not has_score:
            result.errors.append(f"Course {idx} ('{name}'): score is missing.")
            result.valid = False
            continue

        if has_score and not has_name:
            result.errors.append(f"Course {idx}: name is missing.")
            result.valid = False
            continue

        name_ok, name_err = validate_subject_name(name)
        if not name_ok:
            result.errors.append(f"Course {idx}: {name_err}")
            result.valid = False
            continue

        score_ok, score_err = validate_score(raw_score, field_label=f"Course {idx} ('{name}') score")
        if not score_ok:
            result.errors.append(score_err)
            result.valid = False
            continue

        key = name.lower()
        if key in seen_names:
            result.errors.append(f"Duplicate course: '{name}' is entered more than once.")
            result.valid = False
            continue
        seen_names.add(key)
        complete_count += 1

    if complete_count == 0 and result.valid:
        result.errors.append("Enter at least one course with a name and a score.")
        result.valid = False

    return result


def collect_valid_subjects(raw_entries: list[tuple[str, object]]) -> list[SubjectEntry]:
    """Returns only the complete, valid subject entries -- call
    validate_subject_entries() first and check .valid before relying on
    this for anything user-facing."""
    subjects = []
    seen = set()
    for raw_name, raw_score in raw_entries:
        name = (raw_name or "").strip()
        if not name or raw_score is None or str(raw_score).strip() == "":
            continue
        ok, _ = validate_score(raw_score)
        if not ok:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        subjects.append(SubjectEntry(name=name, score=float(raw_score)))
    return subjects


def validate_matric_number(matric: str) -> tuple[bool, str]:
    matric = (matric or "").strip()
    if not matric:
        return False, "Matric number is required."
    if len(matric) > 30:
        return False, "Matric number is too long."
    return True, ""
