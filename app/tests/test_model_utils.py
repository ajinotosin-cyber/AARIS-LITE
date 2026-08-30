import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import model_utils as mu
from feature_engineering import SubjectResult, build_academic_summary
from validation import SubjectEntry, PreviousRecord


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


class TestComputeCgpa(unittest.TestCase):
    """Covers the deterministic, credit-unit-weighted CGPA calculation
    (see model_utils.compute_cgpa's docstring for the full rationale).
    This replaced an earlier version (predict_cgpa) that used
    regression_model.pkl -- a model trained on only [Score, CourseCount],
    with no knowledge of credit units at all, and therefore structurally
    incapable of a genuinely credit-weighted CGPA."""

    def _summary_for(self, subjects):
        return build_academic_summary(subjects)

    def test_returns_ok_status(self):
        summary = self._summary_for([SubjectEntry(name="CS101", score=70, units=3)])
        result = mu.compute_cgpa(summary)
        self.assertEqual(result.status, mu.STATUS_OK)
        self.assertIsInstance(result.predicted_cgpa, float)

    def test_matches_the_weighted_gpa_exactly(self):
        """CGPA and GPA are computed via the identical formula in this
        single-submission app (see build_academic_summary's docstring) --
        they must be numerically equal, not independently derived."""
        summary = self._summary_for([
            SubjectEntry(name="CS101", score=85, units=3),
            SubjectEntry(name="MA101", score=55, units=2),
        ])
        result = mu.compute_cgpa(summary)
        self.assertAlmostEqual(result.predicted_cgpa, summary.gpa, places=9)

    def test_works_across_full_1_to_12_subject_range(self):
        for count in range(1, 13):
            subjects = [SubjectEntry(name=f"C{i}", score=65.0, units=3) for i in range(count)]
            summary = self._summary_for(subjects)
            result = mu.compute_cgpa(summary)
            self.assertEqual(result.status, mu.STATUS_OK, f"failed at count={count}")

    def test_never_unavailable_regardless_of_model_state(self):
        """Regression guard: unlike the old ML-based predict_cgpa(), this
        is pure arithmetic over already-validated input -- it must never
        report STATUS_MODELS_UNAVAILABLE, since it doesn't depend on any
        model being loaded at all."""
        summary = self._summary_for([SubjectEntry(name="CS101", score=70, units=3)])
        result = mu.compute_cgpa(summary)
        self.assertNotEqual(result.status, mu.STATUS_MODELS_UNAVAILABLE)

    def test_credit_weighting_verified_against_hand_calculation(self):
        """(5*3 + 3*2) / (3+2) = 4.2 -- an A-grade 3-unit course and a
        C-grade 2-unit course, weighted by their actual credit units."""
        summary = self._summary_for([
            SubjectEntry(name="Heavy", score=85, units=3),  # A = 5 points
            SubjectEntry(name="Light", score=55, units=2),  # C = 3 points
        ])
        result = mu.compute_cgpa(summary)
        self.assertAlmostEqual(result.predicted_cgpa, 4.2, places=9)

    def test_with_previous_record_produces_genuinely_blended_value(self):
        """This is the whole point of the previous_record parameter: with
        it, CGPA is no longer identical to this semester's GPA -- it's a
        real, different, blended figure."""
        summary = self._summary_for([SubjectEntry(name="CS101", score=90, units=6)])  # GPA = 5.0
        previous = PreviousRecord(cgpa=3.0, total_units=90)
        result = mu.compute_cgpa(summary, previous_record=previous)
        self.assertNotAlmostEqual(result.predicted_cgpa, summary.gpa, places=2)
        expected = (3.0 * 90 + summary.gpa * summary.total_units) / (90 + summary.total_units)
        self.assertAlmostEqual(result.predicted_cgpa, expected, places=9)

    def test_without_previous_record_still_matches_current_gpa(self):
        """Regression guard: passing previous_record=None (the default)
        must behave exactly as before this feature was added."""
        summary = self._summary_for([SubjectEntry(name="CS101", score=70, units=3)])
        result = mu.compute_cgpa(summary, previous_record=None)
        self.assertAlmostEqual(result.predicted_cgpa, summary.gpa, places=9)

    def test_previous_record_status_always_ok(self):
        summary = self._summary_for([SubjectEntry(name="CS101", score=70, units=3)])
        previous = PreviousRecord(cgpa=3.5, total_units=60)
        result = mu.compute_cgpa(summary, previous_record=previous)
        self.assertEqual(result.status, mu.STATUS_OK)


class TestDetectAnomalousSubjects(unittest.TestCase):
    """Covers the per-student contextual score-confirmation logic (see
    config.py's HIGH_SCORE_CONFIRMATION_FLOOR / MIN_SUBJECTS_FOR_SCORE_CONFIRMATION
    / SCORE_CONFIRMATION_Z_THRESHOLD and model_utils.detect_anomalous_subjects'
    docstring for the full rationale). This replaced an earlier version
    that used anomaly_model.pkl (a single-feature Isolation Forest
    trained on the global historical score distribution) to flag any
    score in roughly the top percentile of that population -- which
    meant an ordinary, believable score like 80 was flagged purely for
    being globally rare, with no knowledge of what was normal for the
    specific student being evaluated."""

    def setUp(self):
        self.models = mu.load_models()

    def make(self, scores):
        return [SubjectResult(name=f"Course{i + 1}", score=s, grade="A", grade_point=5, units=3)
                for i, s in enumerate(scores)]

    def test_empty_list_ok(self):
        result = mu.detect_anomalous_subjects([], self.models)
        self.assertEqual(result.status, mu.STATUS_OK)
        self.assertEqual(result.anomalous_subjects, [])

    def test_ordinary_high_scores_not_flagged(self):
        """The exact scenario reported as a bug: 80, 75, 80 must not
        produce a confirmation warning -- these are normal, believable
        academic results, not statistically inconsistent with anything."""
        result = mu.detect_anomalous_subjects(self.make([80, 75, 80]), self.models)
        self.assertEqual(result.status, mu.STATUS_OK)
        self.assertEqual(result.anomalous_subjects, [])

    def test_various_ordinary_high_scores_never_flagged_regardless_of_sample_size(self):
        for scores in ([70, 75, 80, 85], [85, 88, 90, 87, 89]):
            with self.subTest(scores=scores):
                result = mu.detect_anomalous_subjects(self.make(scores), self.models)
                self.assertEqual(result.anomalous_subjects, [])

    def test_consistently_high_performer_not_flagged_with_sufficient_data(self):
        """A student consistently scoring 90-98 with a score of 96 among
        them -- 96 is normal FOR THIS STUDENT and must not be flagged,
        even though it clears the high-score floor, because it isn't
        inconsistent with their own established performance."""
        result = mu.detect_anomalous_subjects(self.make([90, 92, 94, 98, 96]), self.models)
        self.assertEqual(result.status, mu.STATUS_OK)
        self.assertEqual(result.anomalous_subjects, [])

    def test_genuinely_consistent_90_to_100_cluster_not_flagged(self):
        """Regression test for a real miscalibration bug found via direct
        testing: after the confirmation floor was lowered from 95 to 90,
        the Z-score threshold was also lowered (2.0 -> 1.5) without
        re-verifying the combination -- and 1.5 was too sensitive,
        incorrectly flagging a genuinely consistent cluster of scores all
        in the 90-100 range ([92, 95, 90, 98, 100]) as if the highest one
        were inconsistent with the others. Restored to 2.0, which
        correctly leaves this case unflagged while still catching a
        genuine outlier (see the sibling test immediately below)."""
        result = mu.detect_anomalous_subjects(self.make([92, 95, 90, 98, 100]), self.models)
        self.assertEqual(result.anomalous_subjects, [])

    def test_sudden_high_score_flagged_when_inconsistent_with_sufficient_history(self):
        """A student consistently scoring 55-70, with a sudden 98 among
        five total scores -- this IS statistically inconsistent with
        their established performance and should be recommended for
        confirmation."""
        result = mu.detect_anomalous_subjects(self.make([55, 60, 65, 70, 98]), self.models)
        self.assertEqual(result.status, mu.STATUS_OK)
        self.assertIn("Course5", result.anomalous_subjects)

    def test_small_sample_never_flagged_even_with_all_high_scores(self):
        """Three scores of 95, 96, 97 together must not be automatically
        assumed suspicious -- there isn't enough data (below
        MIN_SUBJECTS_FOR_SCORE_CONFIRMATION) to say anything statistically
        meaningful about what's consistent for this student."""
        result = mu.detect_anomalous_subjects(self.make([95, 96, 97]), self.models)
        self.assertEqual(result.status, mu.STATUS_OK)
        self.assertEqual(result.anomalous_subjects, [])

    def test_small_sample_not_flagged_even_with_a_stark_outlier(self):
        """Regression guard for the 'unreliable statistical method with a
        small sample' requirement specifically: even a score that WOULD
        be flagged with enough data (60, 65, 99) must not be flagged
        with only 3 total scores."""
        result = mu.detect_anomalous_subjects(self.make([60, 65, 99]), self.models)
        self.assertEqual(result.anomalous_subjects, [])

    def test_score_below_confirmation_floor_never_flagged_regardless_of_context(self):
        """A score of 89 -- just under the 90 floor -- must never be a
        candidate for confirmation, even if wildly inconsistent with a
        very tight, very low history, since only exceptionally high
        scores are candidates at all."""
        result = mu.detect_anomalous_subjects(self.make([40, 41, 40, 42, 89]), self.models)
        self.assertEqual(result.anomalous_subjects, [])

    def test_zero_variance_other_scores_with_large_gap_flagged(self):
        """When the student's other scores are all identical (no
        variance, so no Z-score can be computed), a sufficiently large
        absolute gap still triggers the conservative fallback check."""
        result = mu.detect_anomalous_subjects(self.make([80, 80, 80, 80, 100]), self.models)
        self.assertIn("Course5", result.anomalous_subjects)

    def test_zero_variance_other_scores_with_small_gap_not_flagged(self):
        result = mu.detect_anomalous_subjects(self.make([80, 80, 80, 80, 85]), self.models)
        self.assertEqual(result.anomalous_subjects, [])

    def test_all_identical_high_scores_not_flagged(self):
        result = mu.detect_anomalous_subjects(self.make([96, 96, 96, 96, 96]), self.models)
        self.assertEqual(result.anomalous_subjects, [])

    def test_works_correctly_even_when_models_are_entirely_unavailable(self):
        """The new logic is pure statistics on the submitted scores --
        it no longer needs any trained model (unlike the Isolation
        Forest it replaced), so it must keep working even when model
        loading has completely failed."""
        empty_models = mu.ModelBundle(load_error="simulated failure")
        result = mu.detect_anomalous_subjects(self.make([55, 60, 65, 70, 98]), empty_models)
        self.assertEqual(result.status, mu.STATUS_OK)
        self.assertIn("Course5", result.anomalous_subjects)

    def test_malformed_subject_reported_as_prediction_failed_not_crash(self):
        class BadSubject:
            name = "Broken"
            score = "not-a-number"
        result = mu.detect_anomalous_subjects([BadSubject()] * 5, self.models)
        self.assertEqual(result.status, mu.STATUS_PREDICTION_FAILED)


if __name__ == "__main__":
    unittest.main()
