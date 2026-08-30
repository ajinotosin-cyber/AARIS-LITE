import unittest

from feature_engineering import assign_grade, classify_degree, build_academic_summary, model_features, combine_cgpa
from validation import SubjectEntry


class TestAssignGrade(unittest.TestCase):
    def test_a_boundary(self):
        self.assertEqual(assign_grade(70), "A")

    def test_b_boundary(self):
        self.assertEqual(assign_grade(60), "B")

    def test_below_e_is_f(self):
        self.assertEqual(assign_grade(39.9), "F")

    def test_e_boundary(self):
        self.assertEqual(assign_grade(40), "E")


class TestClassifyDegree(unittest.TestCase):
    def test_first_class(self):
        self.assertEqual(classify_degree(4.5), "First Class Honours")

    def test_probation(self):
        self.assertEqual(classify_degree(1.0), "Academic Probation")

    def test_second_upper_boundary(self):
        self.assertEqual(classify_degree(3.5), "Second Class Upper")


class TestBuildAcademicSummary(unittest.TestCase):
    def test_empty_list_does_not_crash(self):
        summary = build_academic_summary([])
        self.assertEqual(summary.subject_count, 0)
        self.assertEqual(summary.average_score, 0.0)
        self.assertEqual(summary.gpa, 0.0)

    def test_single_subject(self):
        subjects = [SubjectEntry(name="CS101", score=75, units=3)]
        summary = build_academic_summary(subjects)
        self.assertEqual(summary.subject_count, 1)
        self.assertEqual(summary.average_score, 75)
        self.assertEqual(summary.subjects[0].grade, "A")
        self.assertEqual(summary.gpa, 5)

    def test_average_and_gpa_consistent_across_twelve_subjects(self):
        subjects = [SubjectEntry(name=f"SUBJ{i}", score=60 + i, units=3) for i in range(12)]
        summary = build_academic_summary(subjects)
        self.assertEqual(summary.subject_count, 12)
        expected_avg = sum(60 + i for i in range(12)) / 12
        self.assertAlmostEqual(summary.average_score, expected_avg)
        # Every displayed grade must correspond to that subject's own score.
        for result in summary.subjects:
            self.assertEqual(result.grade, assign_grade(result.score))

    def test_model_features_uses_same_summary_values(self):
        subjects = [SubjectEntry(name="CS101", score=80, units=3), SubjectEntry(name="MA101", score=60, units=3)]
        summary = build_academic_summary(subjects)
        features = model_features(summary)
        self.assertEqual(features, [summary.average_score, summary.subject_count])
        self.assertEqual(features, [70.0, 2])


class TestCombineCgpa(unittest.TestCase):
    """The standard university formula for rolling a new semester's GPA
    into a running cumulative CGPA -- verified against hand calculations,
    not just plausible-looking output."""

    def test_hand_calculated_example(self):
        """Previous: 3.60 CGPA over 90 units. Current: 4.20 GPA over 18 units.
        (3.60*90 + 4.20*18) / (90+18) = (324 + 75.6) / 108 = 3.7"""
        result = combine_cgpa(current_gpa=4.20, current_units=18, previous_cgpa=3.60, previous_units=90)
        self.assertAlmostEqual(result, 3.7, places=9)

    def test_second_hand_calculated_example(self):
        """Previous: 3.00 CGPA over 60 units. Current: 5.00 GPA over 12 units.
        (3.00*60 + 5.00*12) / (60+12) = (180 + 60) / 72 = 3.333..."""
        result = combine_cgpa(current_gpa=5.00, current_units=12, previous_cgpa=3.00, previous_units=60)
        self.assertAlmostEqual(result, 240 / 72, places=9)

    def test_zero_previous_units_reduces_to_current_gpa(self):
        """Defensive edge case: zero previous units means the previous
        CGPA contributes nothing to the weighted average regardless of
        its value (0 weight), so the result is exactly the current GPA."""
        result = combine_cgpa(current_gpa=4.20, current_units=18, previous_cgpa=3.60, previous_units=0)
        self.assertAlmostEqual(result, 4.20, places=9)

    def test_identical_previous_and_current_gpa_stays_the_same(self):
        """If the previous CGPA and this semester's GPA are identical,
        the blended result must equal that same value regardless of the
        unit split -- a basic sanity/consistency check on the formula."""
        result = combine_cgpa(current_gpa=4.0, current_units=15, previous_cgpa=4.0, previous_units=75)
        self.assertAlmostEqual(result, 4.0, places=9)

    def test_result_always_between_the_two_inputs(self):
        """A weighted average of two values must always fall between
        them (inclusive) -- a genuine mathematical property worth
        directly verifying, not just checking one specific number."""
        result = combine_cgpa(current_gpa=4.8, current_units=12, previous_cgpa=2.5, previous_units=100)
        self.assertGreaterEqual(result, 2.5)
        self.assertLessEqual(result, 4.8)

    def test_large_previous_history_dominates_small_current_semester(self):
        """A large previous record (e.g. 100 units) should barely move
        for one small current semester (e.g. 15 units) -- the result
        should stay close to the previous CGPA, not the current GPA."""
        result = combine_cgpa(current_gpa=5.0, current_units=15, previous_cgpa=3.0, previous_units=100)
        # Result should be much closer to 3.0 (previous) than to 5.0 (current)
        self.assertLess(abs(result - 3.0), abs(result - 5.0))


if __name__ == "__main__":
    unittest.main()
