"""
End-to-end smoke test using Streamlit's official AppTest harness, driving
the actual app.py.
"""
import os
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import model_utils as mu

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

FORBIDDEN_PHRASES = [
    "predictions are indicative",
    "not authoritative",
    "underlying model",
    "trained model",
    "limited feature set",
    "average score and subject count",
    "model limitations",
    "anomaly model",
    "classifier",
    "regression model",
    "feature engineering",
]


class TestAppSmoke(unittest.TestCase):
    def test_app_boots(self):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        self.assertFalse(at.exception, f"App raised on default load: {at.exception}")

    def test_add_course_button_grows_grid_up_to_twelve(self):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        # Starts at 4 rows; click "+ Add Course" repeatedly to reach 12.
        for _ in range(8):
            add_buttons = [b for b in at.button if b.label.startswith("+ Add Course")]
            self.assertTrue(add_buttons, "Add Course button disappeared before reaching 12")
            add_buttons[0].click().run(timeout=30)
            self.assertFalse(at.exception)
        # At 12, the add button should be gone (replaced by the "maximum reached" caption).
        add_buttons = [b for b in at.button if b.label.startswith("+ Add Course")]
        self.assertEqual(len(add_buttons), 0)
        captions = " ".join(c.value for c in at.caption)
        self.assertIn("Maximum of 12", captions)

    def test_valid_single_course_submission(self):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        at.text_input(key="subj_name_0").set_value("CS101")
        at.number_input(key="subj_score_0").set_value(75)
        analyze = [b for b in at.button if b.label == "Analyze Academic Profile"][0]
        analyze.click().run(timeout=30)
        self.assertFalse(at.exception)
        body = " ".join(m.value for m in at.markdown)
        self.assertIn("Academic Standing", body)
        self.assertIn("GPA", body)

    def test_empty_submission_shows_validation_error(self):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        analyze = [b for b in at.button if b.label == "Analyze Academic Profile"][0]
        analyze.click().run(timeout=30)
        self.assertFalse(at.exception)
        self.assertTrue(len(at.error) >= 1)
        errors = " ".join(e.value for e in at.error)
        self.assertIn("at least one course", errors.lower())

    def test_twelve_course_submission_end_to_end(self):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        for _ in range(8):
            add_buttons = [b for b in at.button if b.label.startswith("+ Add Course")]
            add_buttons[0].click().run(timeout=30)

        for i in range(12):
            at.text_input(key=f"subj_name_{i}").set_value(f"SUBJ{i}")
            at.number_input(key=f"subj_score_{i}").set_value(60 + i % 20)

        analyze = [b for b in at.button if b.label == "Analyze Academic Profile"][0]
        analyze.click().run(timeout=30)
        self.assertFalse(at.exception, f"12-course submission crashed: {at.exception}")
        body = " ".join(m.value for m in at.markdown)
        self.assertIn("Academic Standing", body)

    def test_invalid_score_shows_error_not_crash(self):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        at.text_input(key="subj_name_0").set_value("CS101")
        # number_input itself clamps to [0,100] in the widget, so to
        # exercise validation.py's own out-of-range check we simulate the
        # duplicate-name path instead, which the widget can't prevent.
        at.text_input(key="subj_name_1").set_value("CS101")
        at.number_input(key="subj_score_0").set_value(70)
        at.number_input(key="subj_score_1").set_value(80)
        analyze = [b for b in at.button if b.label == "Analyze Academic Profile"][0]
        analyze.click().run(timeout=30)
        self.assertFalse(at.exception)
        errors = " ".join(e.value for e in at.error)
        self.assertIn("duplicate", errors.lower())

    def test_models_unavailable_does_not_crash_app(self):
        broken = mu.ModelBundle(load_error="simulated: all models missing")
        with patch("model_utils.load_models", return_value=broken):
            at = AppTest.from_file(APP_PATH)
            at.run(timeout=30)
            self.assertFalse(at.exception, f"App crashed with no models: {at.exception}")
            infos = " ".join(i.value for i in at.info)
            self.assertIn("temporarily unavailable", infos.lower())
            # The degraded-state banner must not expose backend terminology.
            for phrase in FORBIDDEN_PHRASES:
                self.assertNotIn(phrase, infos.lower())

            at.text_input(key="subj_name_0").set_value("CS101")
            at.number_input(key="subj_score_0").set_value(75)
            analyze = [b for b in at.button if b.label == "Analyze Academic Profile"][0]
            analyze.click().run(timeout=30)
            self.assertFalse(at.exception)
            body = " ".join(m.value for m in at.markdown)
            self.assertIn("Unavailable", body)
            self.assertNotIn("GOOD", body)  # never a fake favourable result

    def test_no_model_disclaimer_text_on_interface(self):
        """No prediction/model disclaimer or backend terminology should
        appear anywhere on the page after a normal successful analysis."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        at.text_input(key="subj_name_0").set_value("CS101")
        at.number_input(key="subj_score_0").set_value(75)
        analyze = [b for b in at.button if b.label == "Analyze Academic Profile"][0]
        analyze.click().run(timeout=30)
        self.assertFalse(at.exception)
        body = " ".join(m.value for m in at.markdown).lower()
        for phrase in FORBIDDEN_PHRASES:
            self.assertNotIn(phrase, body, f"Forbidden phrase leaked onto the interface: {phrase!r}")

    def test_high_score_note_shown_on_interface_when_flagged(self):
        """A submission with a genuine score outlier (statistically
        inconsistent with the student's own other scores, with a
        sufficient sample size) gets the professional 'Score Confirmation
        Recommended' note -- never accusatory 'anomaly'/'suspicious' wording."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        # Default view shows 4 rows; add a 5th to reach the minimum
        # sample size the confirmation check requires.
        add_button = [b for b in at.button if b.label.startswith("+ Add Course")][0]
        add_button.click().run(timeout=30)

        names = ["Normal1", "Normal2", "Normal3", "Normal4", "Extreme"]
        scores = [55, 60, 58, 62, 98]
        for i, (name, score) in enumerate(zip(names, scores)):
            at.text_input(key=f"subj_name_{i}").set_value(name)
            at.number_input(key=f"subj_score_{i}").set_value(score)
        analyze = [b for b in at.button if b.label == "Analyze Academic Profile"][0]
        analyze.click().run(timeout=30)
        self.assertFalse(at.exception)
        infos = " ".join(i.value for i in at.info)
        self.assertIn("Score Confirmation Recommended", infos)
        self.assertIn("Extreme", infos)
        self.assertIn("statistically inconsistent", infos)
        body = " ".join(m.value for m in at.markdown).lower()
        warnings = " ".join(w.value for w in at.warning).lower()
        self.assertNotIn("anomaly", body)
        self.assertNotIn("anomaly", warnings)
        self.assertNotIn("suspicious", infos.lower())


    def test_high_score_note_absent_on_interface_when_nothing_flagged(self):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        at.text_input(key="subj_name_0").set_value("CS101")
        at.number_input(key="subj_score_0").set_value(65)
        analyze = [b for b in at.button if b.label == "Analyze Academic Profile"][0]
        analyze.click().run(timeout=30)
        self.assertFalse(at.exception)
        infos = " ".join(i.value for i in at.info)
        self.assertNotIn("Please confirm", infos)

    def test_no_recommendations_section_on_interface(self):
        """Recommendations must still be generated (for the PDF) but must
        not be rendered anywhere on the Streamlit interface itself."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        at.text_input(key="subj_name_0").set_value("CS101")
        at.number_input(key="subj_score_0").set_value(30)  # low score -> non-trivial recommendation text
        analyze = [b for b in at.button if b.label == "Analyze Academic Profile"][0]
        analyze.click().run(timeout=30)
        self.assertFalse(at.exception)
        body = " ".join(m.value for m in at.markdown)
        self.assertNotIn("Recommendations", body)
        self.assertNotIn("Recommended Actions", body)
        self.assertNotIn("academic advising", body.lower())  # a real risk_scoring.py recommendation string

    def test_cgpa_label_used_for_predicted_value(self):
        """The hero-row predicted value is labeled 'CGPA' (matching the
        PDF), distinct from the directly-computed 'GPA' KPI card shown in
        the supporting-metrics row below it."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        at.text_input(key="subj_name_0").set_value("CS101")
        at.number_input(key="subj_score_0").set_value(75)
        analyze = [b for b in at.button if b.label == "Analyze Academic Profile"][0]
        analyze.click().run(timeout=30)
        self.assertFalse(at.exception)
        body = " ".join(m.value for m in at.markdown)
        self.assertIn(">CGPA<", body)
        self.assertIn(">GPA<", body)  # the separate, directly-computed GPA card
        self.assertNotIn("Predicted GPA", body)
        self.assertNotIn("Predicted CGPA", body)

    def test_course_terminology_used_not_subject(self):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        at.text_input(key="subj_name_0").set_value("CS101")
        at.number_input(key="subj_score_0").set_value(75)
        analyze = [b for b in at.button if b.label == "Analyze Academic Profile"][0]
        analyze.click().run(timeout=30)
        self.assertFalse(at.exception)
        body = " ".join(m.value for m in at.markdown)
        self.assertIn("Courses Entered", body)
        self.assertNotIn("Subjects Entered", body)
        expander_labels = [e.label for e in at.get("expander")]
        self.assertIn("Course Performance Breakdown", expander_labels)
        self.assertNotIn("Subject Performance Breakdown", expander_labels)
        self.assertNotIn("Subject breakdown", expander_labels)


class TestAppSmokePdfIntegration(unittest.TestCase):
    def test_pdf_report_download_button_appears_without_crashing(self):
        """End-to-end: app.py passes standing_result/cgpa_result/
        anomaly_result into generate_report_pdf() (previously it only
        passed matric/summary/recommendations, so the PDF never had
        access to the prediction results at all). This confirms that
        wiring doesn't crash and produces a real download button.
        Byte-level verification of the generated PDF itself (page count,
        content, every scenario) lives in test_report_generator.py."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        at.text_input(key="subj_name_0").set_value("CS101")
        at.number_input(key="subj_score_0").set_value(75)
        matric_inputs = [w for w in at.text_input if w.label == "Matric Number"]
        matric_inputs[0].set_value("ST101")
        analyze = [b for b in at.button if b.label == "Analyze Academic Profile"][0]
        analyze.click().run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"App raised: {list(at.exception)}")
        download_buttons = at.get("download_button")
        self.assertEqual(len(download_buttons), 1)
        self.assertEqual(download_buttons[0].label, "Download Student Report (PDF)")

    def test_previous_record_end_to_end_produces_blended_cgpa(self):
        """Real end-to-end test of the multi-semester CGPA feature,
        driving the actual widgets a user interacts with -- not just
        calling the underlying functions directly."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        at.number_input(key="previous_cgpa_input").set_value(3.00)
        at.number_input(key="previous_units_input").set_value(90)
        at.text_input(key="subj_name_0").set_value("CS101")
        at.number_input(key="subj_score_0").set_value(90)  # A = 5.0 grade points
        at.number_input(key="subj_units_0").set_value(6)
        analyze = [b for b in at.button if b.label == "Analyze Academic Profile"][0]
        analyze.click().run(timeout=30)
        self.assertFalse(at.exception)
        body = " ".join(m.value for m in at.markdown)
        # GPA (this semester) = 5.00, CGPA (blended with 3.00 over 90 prior
        # units) must show a genuinely different, lower number -- not 5.00.
        self.assertIn("5.00", body)
        self.assertNotIn("CGPA</div><div class=\"aaris-value-good\">5.00", body)
        captions = " ".join(c.value for c in at.caption)
        self.assertIn("90", captions)
        self.assertIn("3.00", captions)

    def test_previous_cgpa_without_units_shows_validation_error(self):
        """Entering only one half of the optional pair must be rejected,
        not silently ignored or silently defaulted."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        at.number_input(key="previous_cgpa_input").set_value(3.5)
        at.text_input(key="subj_name_0").set_value("CS101")
        at.number_input(key="subj_score_0").set_value(75)
        analyze = [b for b in at.button if b.label == "Analyze Academic Profile"][0]
        analyze.click().run(timeout=30)
        self.assertFalse(at.exception)
        errors = " ".join(e.value for e in at.error)
        self.assertIn("units", errors.lower())

    def test_no_previous_record_entered_works_exactly_as_before(self):
        """Regression guard: leaving the optional section blank entirely
        must not affect the normal, existing single-semester flow."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        at.text_input(key="subj_name_0").set_value("CS101")
        at.number_input(key="subj_score_0").set_value(75)
        analyze = [b for b in at.button if b.label == "Analyze Academic Profile"][0]
        analyze.click().run(timeout=30)
        self.assertFalse(at.exception)
        errors = " ".join(e.value for e in at.error)
        self.assertEqual(errors, "")


if __name__ == "__main__":
    unittest.main()
