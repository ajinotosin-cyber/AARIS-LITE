import unittest

from feature_engineering import assign_grade, classify_degree, build_academic_summary, model_features
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
        subjects = [SubjectEntry(name="CS101", score=75)]
        summary = build_academic_summary(subjects)
        self.assertEqual(summary.subject_count, 1)
        self.assertEqual(summary.average_score, 75)
        self.assertEqual(summary.subjects[0].grade, "A")
        self.assertEqual(summary.gpa, 5)

    def test_average_and_gpa_consistent_across_twelve_subjects(self):
        subjects = [SubjectEntry(name=f"SUBJ{i}", score=60 + i) for i in range(12)]
        summary = build_academic_summary(subjects)
        self.assertEqual(summary.subject_count, 12)
        expected_avg = sum(60 + i for i in range(12)) / 12
        self.assertAlmostEqual(summary.average_score, expected_avg)
        # Every displayed grade must correspond to that subject's own score.
        for result in summary.subjects:
            self.assertEqual(result.grade, assign_grade(result.score))

    def test_model_features_uses_same_summary_values(self):
        subjects = [SubjectEntry(name="CS101", score=80), SubjectEntry(name="MA101", score=60)]
        summary = build_academic_summary(subjects)
        features = model_features(summary)
        self.assertEqual(features, [summary.average_score, summary.subject_count])
        self.assertEqual(features, [70.0, 2])


if __name__ == "__main__":
    unittest.main()
