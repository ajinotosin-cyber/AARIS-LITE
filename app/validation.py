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
    units: int


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


def validate_units(raw_value, field_label: str = "Units") -> tuple[bool, str]:
    """Returns (is_valid, error_message). Units must be a whole number
    within a realistic course-credit range -- not just any positive
    number, since e.g. a 0-unit or 200-unit "course" is a data-entry
    error, not a real academic scenario."""
    if raw_value is None:
        return False, f"{field_label} is required."
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return False, f"{field_label} must be a number."
    if value != int(value):
        return False, f"{field_label} must be a whole number."
    value = int(value)
    if value < config.MIN_COURSE_UNITS or value > config.MAX_COURSE_UNITS:
        return False, f"{field_label} must be between {config.MIN_COURSE_UNITS} and {config.MAX_COURSE_UNITS}."
    return True, ""


def validate_subject_name(name: str) -> tuple[bool, str]:
    name = (name or "").strip()
    if not name:
        return False, "Course name is required."
    if len(name) > 60:
        return False, "Course name is too long."
    return True, ""


def validate_subject_entries(raw_entries: list[tuple[str, object, object]]) -> ValidationResult:
    """
    raw_entries: list of (subject_name, raw_score, raw_units) triples,
    exactly as collected from the UI (including blank/incomplete rows).

    Rules:
      - Completely blank rows (no name AND no score) are silently
        ignored -- they're just unused capacity in the up-to-12 grid,
        not an error.
      - A row with a name but an invalid/missing score, or a score but
        no name, is a genuine validation error.
      - Units must be a whole number in a realistic course-credit range
        whenever a row is otherwise complete (name + score present).
      - Duplicate subject names (case-insensitive) are rejected -- two
        entries for the same subject aren't meaningful for the average.
      - At least one complete, valid subject is required overall.
    """
    result = ValidationResult(valid=True)
    seen_names = set()
    complete_count = 0

    for idx, (raw_name, raw_score, raw_units) in enumerate(raw_entries, start=1):
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

        units_ok, units_err = validate_units(raw_units, field_label=f"Course {idx} ('{name}') units")
        if not units_ok:
            result.errors.append(units_err)
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


def collect_valid_subjects(raw_entries: list[tuple[str, object, object]]) -> list[SubjectEntry]:
    """Returns only the complete, valid subject entries -- call
    validate_subject_entries() first and check .valid before relying on
    this for anything user-facing."""
    subjects = []
    seen = set()
    for raw_name, raw_score, raw_units in raw_entries:
        name = (raw_name or "").strip()
        if not name or raw_score is None or str(raw_score).strip() == "":
            continue
        score_ok, _ = validate_score(raw_score)
        if not score_ok:
            continue
        units_ok, _ = validate_units(raw_units)
        if not units_ok:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        subjects.append(SubjectEntry(name=name, score=float(raw_score), units=int(raw_units)))
    return subjects


def validate_matric_number(matric: str) -> tuple[bool, str]:
    matric = (matric or "").strip()
    if not matric:
        return False, "Matric number is required."
    if len(matric) > 30:
        return False, "Matric number is too long."
    return True, ""


@dataclass
class PreviousRecord:
    cgpa: float
    total_units: int


def validate_previous_record(raw_cgpa, raw_units) -> tuple[bool, str, "PreviousRecord | None"]:
    """Previous CGPA + previous total units are optional as a PAIR --
    if the student leaves both blank, there's no prior record to combine
    with (this is the common case for a first-time user just trying the
    tool). But if EITHER is filled in, both are required and validated:
    a CGPA without units (or vice versa) isn't a usable prior record --
    you can't blend into a running total without knowing how many units
    that CGPA was actually earned over.

    Returns (is_valid, error_message, PreviousRecord-or-None). The third
    element is None both when validation fails AND when both fields were
    legitimately left blank -- callers should treat None as "no prior
    record to combine", not assume it means an error occurred."""
    has_cgpa = raw_cgpa is not None and str(raw_cgpa).strip() != ""
    has_units = raw_units is not None and str(raw_units).strip() != ""

    if not has_cgpa and not has_units:
        return True, "", None

    if has_cgpa and not has_units:
        return False, "Previous total units is required when previous CGPA is entered.", None
    if has_units and not has_cgpa:
        return False, "Previous CGPA is required when previous total units is entered.", None

    try:
        cgpa_value = float(raw_cgpa)
    except (TypeError, ValueError):
        return False, "Previous CGPA must be a number.", None
    if cgpa_value < config.MIN_PREVIOUS_CGPA or cgpa_value > config.MAX_PREVIOUS_CGPA:
        return False, (f"Previous CGPA must be between {config.MIN_PREVIOUS_CGPA:.2f} "
                        f"and {config.MAX_PREVIOUS_CGPA:.2f}."), None

    try:
        units_value = float(raw_units)
    except (TypeError, ValueError):
        return False, "Previous total units must be a number.", None
    if units_value != int(units_value):
        return False, "Previous total units must be a whole number.", None
    units_value = int(units_value)
    if units_value < config.MIN_PREVIOUS_UNITS or units_value > config.MAX_PREVIOUS_UNITS:
        return False, (f"Previous total units must be between {config.MIN_PREVIOUS_UNITS} "
                        f"and {config.MAX_PREVIOUS_UNITS}."), None

    return True, "", PreviousRecord(cgpa=cgpa_value, total_units=units_value)
