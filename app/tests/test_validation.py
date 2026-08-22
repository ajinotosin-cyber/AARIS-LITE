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


class TestValidateSubjectEntries(unittest.TestCase):
    def test_all_blank_rows_invalid_overall(self):
        entries = [("", None)] * 4
        result = v.validate_subject_entries(entries)
        self.assertFalse(result.valid)
        self.assertIn("at least one course", result.errors[0].lower())

    def test_single_valid_entry(self):
        entries = [("CS101", 75), ("", None), ("", None)]
        result = v.validate_subject_entries(entries)
        self.assertTrue(result.valid)

    def test_name_without_score_is_error(self):
        entries = [("CS101", None)]
        result = v.validate_subject_entries(entries)
        self.assertFalse(result.valid)
        self.assertTrue(any("score is missing" in e.lower() for e in result.errors))

    def test_score_without_name_is_error(self):
        entries = [("", 75)]
        result = v.validate_subject_entries(entries)
        self.assertFalse(result.valid)
        self.assertTrue(any("name is missing" in e.lower() for e in result.errors))

    def test_out_of_range_score_is_error(self):
        entries = [("CS101", 150)]
        result = v.validate_subject_entries(entries)
        self.assertFalse(result.valid)

    def test_duplicate_subject_names_rejected(self):
        entries = [("CS101", 70), ("cs101", 80)]
        result = v.validate_subject_entries(entries)
        self.assertFalse(result.valid)
        self.assertTrue(any("duplicate" in e.lower() for e in result.errors))

    def test_all_twelve_slots_filled_valid(self):
        entries = [(f"SUBJ{i}", 60 + i) for i in range(12)]
        result = v.validate_subject_entries(entries)
        self.assertTrue(result.valid)

    def test_more_than_twelve_entries_still_validates_each_independently(self):
        """validate_subject_entries doesn't enforce the UI's 12-row cap
        itself (that's app.py's job via config.MAX_SUBJECTS) -- it must
        still behave sanely (no crash, correct per-row validation) if
        ever called with more rows than the UI currently allows."""
        entries = [(f"SUBJ{i}", 60) for i in range(15)]
        result = v.validate_subject_entries(entries)
        self.assertTrue(result.valid)
        subjects = v.collect_valid_subjects(entries)
        self.assertEqual(len(subjects), 15)


class TestCollectValidSubjects(unittest.TestCase):
    def test_skips_incomplete_rows(self):
        entries = [("CS101", 75), ("", None), ("MA101", None), ("", 80)]
        subjects = v.collect_valid_subjects(entries)
        self.assertEqual(len(subjects), 1)
        self.assertEqual(subjects[0].name, "CS101")

    def test_skips_duplicates_keeping_first(self):
        entries = [("CS101", 70), ("CS101", 90)]
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


if __name__ == "__main__":
    unittest.main()
