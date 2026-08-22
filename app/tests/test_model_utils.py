import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import model_utils as mu
from feature_engineering import SubjectResult


class TestModelLoading(unittest.TestCase):
    def test_real_models_load_successfully(self):
        models = mu.load_models()
        self.assertTrue(models.classifier_available, models.load_error)
        self.assertTrue(models.regressor_available, models.load_error)
        self.assertTrue(models.anomaly_model_available, models.load_error)
        self.assertIsNone(models.load_error)
        self.assertTrue(models.any_available)

    def test_missing_classifier_reported_honestly_others_still_load(self):
        with patch("config.CLASSIFIER_MODEL_PATH", "/nonexistent/classifier_model.pkl"):
            models = mu.load_models()
        self.assertFalse(models.classifier_available)
        self.assertTrue(models.regressor_available)
        self.assertTrue(models.anomaly_model_available)
        self.assertIn("not found", models.load_error.lower())
        self.assertTrue(models.any_available)

    def test_all_models_missing(self):
        with patch("config.CLASSIFIER_MODEL_PATH", "/nope/a.pkl"), \
             patch("config.REGRESSION_MODEL_PATH", "/nope/b.pkl"), \
             patch("config.ANOMALY_MODEL_PATH", "/nope/c.pkl"):
            models = mu.load_models()
        self.assertFalse(models.any_available)

    def test_corrupted_model_file_reported_honestly(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"not a real pickle")
            path = f.name
        try:
            with patch("config.CLASSIFIER_MODEL_PATH", path):
                models = mu.load_models()
            self.assertFalse(models.classifier_available)
            self.assertIn("corrupted", models.load_error.lower())
        finally:
            os.unlink(path)


class TestPredictStanding(unittest.TestCase):
    def setUp(self):
        self.models = mu.load_models()
        self.assertTrue(self.models.classifier_available)

    def test_returns_ok_status_with_real_model(self):
        result = mu.predict_standing([70.0, 5], self.models)
        self.assertEqual(result.status, mu.STATUS_OK)
        self.assertIn(result.standing, (mu.STANDING_GOOD, mu.STANDING_AT_RISK))

    def test_works_across_full_1_to_12_subject_range(self):
        """The core fix this pass made: the model must genuinely accept
        CourseCount anywhere in 1-12 without error (previously it was
        trained on CourseCount~60, wildly out of distribution)."""
        for count in range(1, 13):
            result = mu.predict_standing([65.0, count], self.models)
            self.assertEqual(result.status, mu.STATUS_OK, f"failed at count={count}")
            self.assertIn(result.standing, (mu.STANDING_GOOD, mu.STANDING_AT_RISK))

    def test_models_unavailable_status(self):
        empty = mu.ModelBundle(load_error="simulated failure")
        result = mu.predict_standing([70.0, 5], empty)
        self.assertEqual(result.status, mu.STATUS_MODELS_UNAVAILABLE)
        self.assertIsNone(result.standing)

    def test_prediction_exception_is_failed_not_good(self):
        broken_clf = MagicMock()
        broken_clf.predict.side_effect = RuntimeError("simulated crash")
        broken_models = mu.ModelBundle(classifier=broken_clf, regressor=self.models.regressor,
                                        anomaly_model=self.models.anomaly_model)
        result = mu.predict_standing([70.0, 5], broken_models)
        self.assertEqual(result.status, mu.STATUS_PREDICTION_FAILED)
        self.assertIsNone(result.standing)
        self.assertNotEqual(result.standing, mu.STANDING_GOOD)


class TestPredictCgpa(unittest.TestCase):
    def setUp(self):
        self.models = mu.load_models()
        self.assertTrue(self.models.regressor_available)

    def test_returns_ok_with_real_model(self):
        result = mu.predict_cgpa([70.0, 5], self.models)
        self.assertEqual(result.status, mu.STATUS_OK)
        self.assertIsInstance(result.predicted_cgpa, float)

    def test_works_across_full_1_to_12_subject_range(self):
        for count in range(1, 13):
            result = mu.predict_cgpa([65.0, count], self.models)
            self.assertEqual(result.status, mu.STATUS_OK, f"failed at count={count}")

    def test_models_unavailable_status(self):
        empty = mu.ModelBundle(load_error="simulated failure")
        result = mu.predict_cgpa([70.0, 5], empty)
        self.assertEqual(result.status, mu.STATUS_MODELS_UNAVAILABLE)
        self.assertIsNone(result.predicted_cgpa)


class TestDetectAnomalousSubjects(unittest.TestCase):
    def setUp(self):
        self.models = mu.load_models()
        self.assertTrue(self.models.anomaly_model_available)

    def test_empty_list_ok(self):
        result = mu.detect_anomalous_subjects([], self.models)
        self.assertEqual(result.status, mu.STATUS_OK)
        self.assertEqual(result.anomalous_subjects, [])

    def test_flags_extreme_high_score(self):
        """Verified against the real anomaly_model.pkl's decision_function:
        the training data's real scores have a floor of 40 (the
        institution's grading minimum), so the model's normal region
        extends down to that floor -- it flags unusually HIGH outliers,
        not low ones near/below 40. This test reflects that actual,
        verified behavior rather than an assumption."""
        subjects = [
            SubjectResult(name="Normal1", score=65, grade="B", grade_point=4),
            SubjectResult(name="Normal2", score=70, grade="A", grade_point=5),
            SubjectResult(name="ExtremeHigh", score=100, grade="A", grade_point=5),
        ]
        result = mu.detect_anomalous_subjects(subjects, self.models)
        self.assertEqual(result.status, mu.STATUS_OK)
        self.assertIn("ExtremeHigh", result.anomalous_subjects)

    def test_models_unavailable_status(self):
        empty = mu.ModelBundle(load_error="simulated failure")
        subjects = [SubjectResult(name="X", score=70, grade="A", grade_point=5)]
        result = mu.detect_anomalous_subjects(subjects, empty)
        self.assertEqual(result.status, mu.STATUS_MODELS_UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
