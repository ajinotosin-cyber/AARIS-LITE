import unittest

import validation as v


class TestValidateScore(unittest.TestCase):
    def test_none_invalid(self):
        ok, _ = v.validate_score(None)
        self.assertFalse(ok)

    def test_non_numeric_invalid(self):
        ok, _ = v.validate_score("abc")
        self.assertFalse(ok)

    def test_below_range_invalid(self):
        ok, _ = v.validate_score(-5)
        self.assertFalse(ok)

    def test_above_range_invalid(self):
        ok, _ = v.validate_score(150)
        self.assertFalse(ok)

    def test_boundary_min_valid(self):
        ok, _ = v.validate_score(0)
        self.assertTrue(ok)

    def test_boundary_max_valid(self):
        ok, _ = v.validate_score(100)
        self.assertTrue(ok)

    def test_typical_valid(self):
        ok, _ = v.validate_score(72.5)
        self.assertTrue(ok)


class TestValidateSubjectName(unittest.TestCase):
    def test_empty_invalid(self):
        ok, _ = v.validate_subject_name("")
        self.assertFalse(ok)

    def test_whitespace_only_invalid(self):
        ok, _ = v.validate_subject_name("   ")
        self.assertFalse(ok)

    def test_too_long_invalid(self):
        ok, _ = v.validate_subject_name("A" * 100)
        self.assertFalse(ok)

    def test_normal_valid(self):
        ok, _ = v.validate_subject_name("CS101")
        self.assertTrue(ok)


class TestValidateUnits(unittest.TestCase):
    def test_typical_valid(self):
        ok, _ = v.validate_units(3)
        self.assertTrue(ok)

    def test_boundary_min_valid(self):
        ok, _ = v.validate_units(1)
        self.assertTrue(ok)

    def test_boundary_max_valid(self):
        ok, _ = v.validate_units(10)
        self.assertTrue(ok)

    def test_zero_invalid(self):
        ok, _ = v.validate_units(0)
        self.assertFalse(ok)

    def test_above_range_invalid(self):
        ok, _ = v.validate_units(11)
        self.assertFalse(ok)

    def test_negative_invalid(self):
        ok, _ = v.validate_units(-2)
        self.assertFalse(ok)

    def test_non_whole_number_invalid(self):
        """Units are a count, not a continuous measurement -- 2.5 units
        isn't a real course credit value."""
        ok, err = v.validate_units(2.5)
        self.assertFalse(ok)
        self.assertIn("whole number", err)

    def test_none_invalid(self):
        ok, _ = v.validate_units(None)
        self.assertFalse(ok)

    def test_non_numeric_invalid(self):
        ok, _ = v.validate_units("abc")
        self.assertFalse(ok)


class TestValidateSubjectEntries(unittest.TestCase):
    def test_all_blank_rows_invalid_overall(self):
        entries = [("", None, 3)] * 4
        result = v.validate_subject_entries(entries)
        self.assertFalse(result.valid)
        self.assertIn("at least one course", result.errors[0].lower())

    def test_single_valid_entry(self):
        entries = [("CS101", 75, 3), ("", None, 3), ("", None, 3)]
        result = v.validate_subject_entries(entries)
        self.assertTrue(result.valid)

    def test_name_without_score_is_error(self):
        entries = [("CS101", None, 3)]
        result = v.validate_subject_entries(entries)
        self.assertFalse(result.valid)
        self.assertTrue(any("score is missing" in e.lower() for e in result.errors))

    def test_score_without_name_is_error(self):
        entries = [("", 75, 3)]
        result = v.validate_subject_entries(entries)
        self.assertFalse(result.valid)
        self.assertTrue(any("name is missing" in e.lower() for e in result.errors))

    def test_out_of_range_score_is_error(self):
        entries = [("CS101", 150, 3)]
        result = v.validate_subject_entries(entries)
        self.assertFalse(result.valid)

    def test_out_of_range_units_is_error(self):
        entries = [("CS101", 75, 0)]
        result = v.validate_subject_entries(entries)
        self.assertFalse(result.valid)
        self.assertTrue(any("units" in e.lower() for e in result.errors))

    def test_non_whole_number_units_is_error(self):
        entries = [("CS101", 75, 2.5)]
        result = v.validate_subject_entries(entries)
        self.assertFalse(result.valid)

    def test_duplicate_subject_names_rejected(self):
        entries = [("CS101", 70, 3), ("cs101", 80, 3)]
        result = v.validate_subject_entries(entries)
        self.assertFalse(result.valid)
        self.assertTrue(any("duplicate" in e.lower() for e in result.errors))

    def test_all_twelve_slots_filled_valid(self):
        entries = [(f"SUBJ{i}", 60 + i, 3) for i in range(12)]
        result = v.validate_subject_entries(entries)
        self.assertTrue(result.valid)

    def test_more_than_twelve_entries_still_validates_each_independently(self):
        """validate_subject_entries doesn't enforce the UI's 12-row cap
        itself (that's app.py's job via config.MAX_SUBJECTS) -- it must
        still behave sanely (no crash, correct per-row validation) if
        ever called with more rows than the UI currently allows."""
        entries = [(f"SUBJ{i}", 60, 3) for i in range(15)]
        result = v.validate_subject_entries(entries)
        self.assertTrue(result.valid)
        subjects = v.collect_valid_subjects(entries)
        self.assertEqual(len(subjects), 15)


class TestCollectValidSubjects(unittest.TestCase):
    def test_skips_incomplete_rows(self):
        entries = [("CS101", 75, 3), ("", None, 3), ("MA101", None, 3), ("", 80, 3)]
        subjects = v.collect_valid_subjects(entries)
        self.assertEqual(len(subjects), 1)
        self.assertEqual(subjects[0].name, "CS101")

    def test_units_correctly_carried_through(self):
        entries = [("CS101", 75, 4), ("MA101", 60, 2)]
        subjects = v.collect_valid_subjects(entries)
        self.assertEqual(subjects[0].units, 4)
        self.assertEqual(subjects[1].units, 2)

    def test_row_with_invalid_units_skipped_entirely(self):
        """Matches how an invalid score is handled (skip the whole row,
        not silently substitute a default) -- units follow the same rule."""
        entries = [("CS101", 75, 0), ("MA101", 60, 3)]
        subjects = v.collect_valid_subjects(entries)
        self.assertEqual(len(subjects), 1)
        self.assertEqual(subjects[0].name, "MA101")

    def test_skips_duplicates_keeping_first(self):
        entries = [("CS101", 70, 3), ("CS101", 90, 3)]
        subjects = v.collect_valid_subjects(entries)
        self.assertEqual(len(subjects), 1)
        self.assertEqual(subjects[0].score, 70)


class TestValidateMatricNumber(unittest.TestCase):
    def test_empty_invalid(self):
        ok, _ = v.validate_matric_number("")
        self.assertFalse(ok)

    def test_valid(self):
        ok, _ = v.validate_matric_number("ST101")
        self.assertTrue(ok)


class TestValidatePreviousRecord(unittest.TestCase):
    def test_both_blank_is_valid_with_no_record(self):
        ok, err, record = v.validate_previous_record(None, None)
        self.assertTrue(ok)
        self.assertEqual(err, "")
        self.assertIsNone(record)

    def test_both_empty_string_is_valid_with_no_record(self):
        ok, err, record = v.validate_previous_record("", "")
        self.assertTrue(ok)
        self.assertIsNone(record)

    def test_cgpa_without_units_is_error(self):
        ok, err, record = v.validate_previous_record(3.5, None)
        self.assertFalse(ok)
        self.assertIn("units", err.lower())
        self.assertIsNone(record)

    def test_units_without_cgpa_is_error(self):
        ok, err, record = v.validate_previous_record(None, 90)
        self.assertFalse(ok)
        self.assertIn("cgpa", err.lower())
        self.assertIsNone(record)

    def test_both_provided_and_valid(self):
        ok, err, record = v.validate_previous_record(3.75, 90)
        self.assertTrue(ok)
        self.assertEqual(err, "")
        self.assertEqual(record.cgpa, 3.75)
        self.assertEqual(record.total_units, 90)

    def test_cgpa_above_max_scale_is_error(self):
        ok, err, record = v.validate_previous_record(6.0, 90)
        self.assertFalse(ok)
        self.assertIsNone(record)

    def test_cgpa_below_zero_is_error(self):
        ok, err, record = v.validate_previous_record(-1.0, 90)
        self.assertFalse(ok)
        self.assertIsNone(record)

    def test_cgpa_at_boundary_values_valid(self):
        ok_low, _, record_low = v.validate_previous_record(0.0, 90)
        self.assertTrue(ok_low)
        ok_high, _, record_high = v.validate_previous_record(5.0, 90)
        self.assertTrue(ok_high)

    def test_non_whole_units_is_error(self):
        ok, err, record = v.validate_previous_record(3.5, 90.5)
        self.assertFalse(ok)
        self.assertIn("whole number", err.lower())
        self.assertIsNone(record)

    def test_units_below_minimum_is_error(self):
        ok, err, record = v.validate_previous_record(3.5, 0)
        self.assertFalse(ok)
        self.assertIsNone(record)

    def test_units_above_maximum_is_error(self):
        ok, err, record = v.validate_previous_record(3.5, 1000)
        self.assertFalse(ok)
        self.assertIsNone(record)

    def test_non_numeric_cgpa_is_error(self):
        ok, err, record = v.validate_previous_record("abc", 90)
        self.assertFalse(ok)
        self.assertIsNone(record)

    def test_non_numeric_units_is_error(self):
        ok, err, record = v.validate_previous_record(3.5, "abc")
        self.assertFalse(ok)
        self.assertIsNone(record)


if __name__ == "__main__":
    unittest.main()
