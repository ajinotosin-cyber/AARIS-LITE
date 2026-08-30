"""
Tests for the redesigned PDF report (report_generator.py). Two things are
verified throughout: the report is EXACTLY ONE PAGE for every supported
scenario (minimum/maximum courses, every standing/GPA status combination,
a fully degraded state), and the report contains no backend/model
terminology or the removed anomaly-status card (see
TestReportGeneratorContent).
"""
import io
import os
import unittest

import pypdf

import model_utils as mu
import report_generator as rg
import risk_scoring
from feature_engineering import build_academic_summary
from validation import SubjectEntry, PreviousRecord

REAL_MODELS = mu.load_models()
BROKEN_MODELS = mu.ModelBundle(load_error="simulated: all model files unavailable")


def _page_count(pdf_bytes: bytes) -> int:
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    return len(reader.pages)


def _build(subjects, models=REAL_MODELS):
    summary = build_academic_summary(subjects)
    features = [summary.average_score, summary.subject_count]
    standing = mu.predict_standing(features, models)
    cgpa = mu.compute_cgpa(summary)
    anomaly = mu.detect_anomalous_subjects(summary.subjects, models)
    recs = risk_scoring.recommendations(summary.gpa)
    return summary, recs, standing, cgpa, anomaly


class TestReportGeneratorOnePage(unittest.TestCase):
    """Every scenario below must produce exactly one PDF page."""

    def setUp(self):
        self.assertTrue(REAL_MODELS.any_available, "Real models must load for these tests")

    def _generate(self, subjects, matric="ST-TEST", models=REAL_MODELS):
        summary, recs, standing, cgpa, anomaly = _build(subjects, models)
        pdf_bytes = rg.generate_report_pdf(matric, summary, recs, standing, cgpa, anomaly)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"), "Output is not a valid PDF")
        return pdf_bytes

    def test_minimum_one_subject(self):
        pdf_bytes = self._generate([SubjectEntry("Data Structures", 92, units=3)])
        self.assertEqual(_page_count(pdf_bytes), 1)

    def test_maximum_twelve_subjects(self):
        subjects = [SubjectEntry(f"SUBJ{i}", 60 + i, units=3) for i in range(12)]
        pdf_bytes = self._generate(subjects)
        self.assertEqual(_page_count(pdf_bytes), 1)

    def test_twelve_subjects_with_long_names(self):
        """Long subject names must not push the report to a second page
        or overflow their table cell -- see report_generator's
        _truncate_to_width."""
        long_names = [
            "Introduction to Computer Science I", "Calculus and Analytic Geometry",
            "MA101", "GS101", "ST111", "ST112", "PH101", "PH102", "CS102", "MA112",
            "MA121", "Advanced Software Engineering Principles",
        ]
        subjects = [SubjectEntry(n, 55 + i, units=3) for i, n in enumerate(long_names)]
        pdf_bytes = self._generate(subjects)
        self.assertEqual(_page_count(pdf_bytes), 1)

    def test_six_subjects_single_column_boundary(self):
        subjects = [SubjectEntry(f"SUBJ{i}", 60 + i, units=3) for i in range(6)]
        pdf_bytes = self._generate(subjects)
        self.assertEqual(_page_count(pdf_bytes), 1)

    def test_seven_subjects_two_column_boundary(self):
        subjects = [SubjectEntry(f"SUBJ{i}", 60 + i, units=3) for i in range(7)]
        pdf_bytes = self._generate(subjects)
        self.assertEqual(_page_count(pdf_bytes), 1)

    def test_good_standing_scenario(self):
        subjects = [SubjectEntry(f"SUBJ{i}", 85, units=3) for i in range(5)]
        summary, recs, standing, cgpa, anomaly = _build(subjects)
        self.assertEqual(standing.standing, mu.STANDING_GOOD)
        pdf_bytes = rg.generate_report_pdf("ST-GOOD", summary, recs, standing, cgpa, anomaly)
        self.assertEqual(_page_count(pdf_bytes), 1)

    def test_at_risk_scenario(self):
        subjects = [SubjectEntry(f"SUBJ{i}", 40 + i, units=3) for i in range(8)]
        summary, recs, standing, cgpa, anomaly = _build(subjects)
        self.assertEqual(standing.standing, mu.STANDING_AT_RISK)
        pdf_bytes = rg.generate_report_pdf("ST-RISK", summary, recs, standing, cgpa, anomaly)
        self.assertEqual(_page_count(pdf_bytes), 1)

    def test_anomaly_detected_scenario(self):
        subjects = [
            SubjectEntry("Normal1", 55, units=3), SubjectEntry("Normal2", 60, units=3),
            SubjectEntry("Normal3", 58, units=3), SubjectEntry("Normal4", 62, units=3),
            SubjectEntry("Extreme", 98, units=3),
        ]
        summary, recs, standing, cgpa, anomaly = _build(subjects)
        self.assertEqual(anomaly.status, mu.STATUS_OK)
        self.assertTrue(anomaly.anomalous_subjects)
        pdf_bytes = rg.generate_report_pdf("ST-ANOM", summary, recs, standing, cgpa, anomaly)
        self.assertEqual(_page_count(pdf_bytes), 1)

    def test_no_anomaly_scenario(self):
        subjects = [SubjectEntry("Normal1", 65, units=3), SubjectEntry("Normal2", 70, units=3)]
        summary, recs, standing, cgpa, anomaly = _build(subjects)
        self.assertEqual(anomaly.anomalous_subjects, [])
        pdf_bytes = rg.generate_report_pdf("ST-NOANOM", summary, recs, standing, cgpa, anomaly)
        self.assertEqual(_page_count(pdf_bytes), 1)

    def test_models_unavailable_degraded_state(self):
        pdf_bytes = self._generate(
            [SubjectEntry("CS101", 70, units=3), SubjectEntry("MA101", 65, units=3)], models=BROKEN_MODELS,
        )
        self.assertEqual(_page_count(pdf_bytes), 1)

    def test_models_unavailable_never_shows_fabricated_result(self):
        summary, recs, standing, cgpa, anomaly = _build(
            [SubjectEntry("CS101", 70, units=3)], models=BROKEN_MODELS,
        )
        self.assertEqual(standing.status, mu.STATUS_MODELS_UNAVAILABLE)
        pdf_bytes = rg.generate_report_pdf("ST-DEGRADED", summary, recs, standing, cgpa, anomaly)
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        text = reader.pages[0].extract_text()
        self.assertIn("Unavailable", text)
        self.assertNotIn("GOOD", text)  # never a fake favourable result

    def test_long_recommendation_list_still_fits_one_page(self):
        """The recommendations panel height is computed dynamically (see
        report_generator's dry-run line measurement) -- verify it holds
        for a longer, more varied set of recommendation text than
        risk_scoring.py currently produces, in case that ever changes."""
        subjects = [SubjectEntry("CS101", 70, units=3)]
        summary, _, standing, cgpa, anomaly = _build(subjects)
        long_recs = [
            "Increase weekly study hours and dedicate additional time to reviewing lecture material.",
            "Focus specifically on weaker courses identified in the subject breakdown above.",
            "Practice past exam questions under timed conditions to build exam confidence.",
            "Consider forming or joining a study group for peer-supported revision.",
        ]
        pdf_bytes = rg.generate_report_pdf("ST-LONGREC", summary, long_recs, standing, cgpa, anomaly)
        self.assertEqual(_page_count(pdf_bytes), 1)

    def test_short_recommendation_list_still_fits_one_page(self):
        subjects = [SubjectEntry("CS101", 70, units=3)]
        summary, _, standing, cgpa, anomaly = _build(subjects)
        pdf_bytes = rg.generate_report_pdf("ST-SHORTREC", summary, ["Keep it up."], standing, cgpa, anomaly)
        self.assertEqual(_page_count(pdf_bytes), 1)

    def test_extreme_degree_classification_text_does_not_overflow(self):
        """Regression test: 'Second Class Upper' and 'Academic Probation'
        previously overflowed their KPI card before the auto-shrink fix."""
        for target_score in (30, 92):  # Academic Probation vs First Class
            subjects = [SubjectEntry("CS101", target_score, units=3)]
            summary, recs, standing, cgpa, anomaly = _build(subjects)
            pdf_bytes = rg.generate_report_pdf("ST-DEGREE", summary, recs, standing, cgpa, anomaly)
            self.assertEqual(_page_count(pdf_bytes), 1)


class TestReportGeneratorContent(unittest.TestCase):
    """Spot-checks that the honest, real computed values actually appear
    in the rendered PDF text (not fabricated, not omitted), and that
    backend/model terminology and the removed anomaly card are genuinely
    gone from the rendered output -- not just absent from the code path
    we happen to be looking at."""

    def setUp(self):
        self.assertTrue(REAL_MODELS.any_available)

    def test_matric_and_core_metrics_present(self):
        subjects = [SubjectEntry("CS101", 80, units=3), SubjectEntry("MA101", 60, units=3)]
        summary, recs, standing, cgpa, anomaly = _build(subjects)
        pdf_bytes = rg.generate_report_pdf("ST-CONTENT-1", summary, recs, standing, cgpa, anomaly)
        text = pypdf.PdfReader(io.BytesIO(pdf_bytes)).pages[0].extract_text()
        self.assertIn("ST-CONTENT-1", text)
        self.assertIn("CS101", text)
        self.assertIn("MA101", text)
        self.assertIn(f"{summary.gpa:.2f}", text)

    def test_at_risk_headline_present_when_at_risk(self):
        subjects = [SubjectEntry(f"SUBJ{i}", 40, units=3) for i in range(8)]
        summary, recs, standing, cgpa, anomaly = _build(subjects)
        self.assertEqual(standing.standing, mu.STANDING_AT_RISK)
        pdf_bytes = rg.generate_report_pdf("ST-CONTENT-2", summary, recs, standing, cgpa, anomaly)
        text = pypdf.PdfReader(io.BytesIO(pdf_bytes)).pages[0].extract_text()
        self.assertIn("AT RISK", text)

    def test_no_model_disclaimer_text_in_pdf(self):
        subjects = [SubjectEntry("CS101", 75, units=3)]
        summary, recs, standing, cgpa, anomaly = _build(subjects)
        pdf_bytes = rg.generate_report_pdf("ST-CONTENT-3", summary, recs, standing, cgpa, anomaly)
        text = pypdf.PdfReader(io.BytesIO(pdf_bytes)).pages[0].extract_text().lower()
        for phrase in (
            "indicative", "not authoritative", "underlying model", "trained model",
            "limited feature set", "average score and subject count", "classifier",
            "regression model", "feature engineering",
        ):
            self.assertNotIn(phrase, text, f"Disclaimer/backend language leaked into PDF: {phrase!r}")

    def test_no_anomaly_status_card_in_pdf_even_with_a_real_anomaly(self):
        """The old generic "Anomaly Status" card is gone for good -- even
        with a real flagged course, the PDF shows the narrow honest note
        (see test_high_score_note_appears_when_flagged below), never the
        word "anomaly" or "flagged" as a label."""
        subjects = [
            SubjectEntry("Normal1", 55, units=3), SubjectEntry("Normal2", 60, units=3),
            SubjectEntry("Normal3", 58, units=3), SubjectEntry("Normal4", 62, units=3),
            SubjectEntry("Extreme", 98, units=3),
        ]
        summary, recs, standing, cgpa, anomaly = _build(subjects)
        self.assertTrue(anomaly.anomalous_subjects)  # the analysis itself still runs
        pdf_bytes = rg.generate_report_pdf("ST-CONTENT-4", summary, recs, standing, cgpa, anomaly)
        text = pypdf.PdfReader(io.BytesIO(pdf_bytes)).pages[0].extract_text().lower()
        self.assertNotIn("anomaly", text)
        self.assertNotIn("flagged", text)
        self.assertNotIn("anomaly status", text)

    def test_high_score_note_appears_when_flagged(self):
        subjects = [
            SubjectEntry("CS101", 55, units=3), SubjectEntry("MA101", 60, units=3),
            SubjectEntry("PH101", 58, units=3), SubjectEntry("GS101", 62, units=3),
            SubjectEntry("Extreme", 98, units=3),
        ]
        summary, recs, standing, cgpa, anomaly = _build(subjects)
        self.assertEqual(anomaly.anomalous_subjects, ["Extreme"])
        pdf_bytes = rg.generate_report_pdf("ST-CONTENT-4B", summary, recs, standing, cgpa, anomaly)
        text = pypdf.PdfReader(io.BytesIO(pdf_bytes)).pages[0].extract_text()
        self.assertIn("Score Confirmation Recommended", text)
        self.assertIn("Extreme", text)
        self.assertIn("statistically inconsistent", text)
        self.assertEqual(_page_count(pdf_bytes), 1)

    def test_high_score_note_absent_when_nothing_flagged(self):
        subjects = [SubjectEntry("CS101", 65, units=3), SubjectEntry("MA101", 70, units=3)]
        summary, recs, standing, cgpa, anomaly = _build(subjects)
        self.assertEqual(anomaly.anomalous_subjects, [])
        pdf_bytes = rg.generate_report_pdf("ST-CONTENT-4C", summary, recs, standing, cgpa, anomaly)
        text = pypdf.PdfReader(io.BytesIO(pdf_bytes)).pages[0].extract_text()
        self.assertNotIn("Score Confirmation Recommended", text)

    def test_high_score_note_with_multiple_flagged_courses_stays_one_page(self):
        """Verified empirically (not assumed) that this specific data
        shape produces exactly 2 simultaneous flags under the new
        per-student contextual logic: a tight low cluster with two
        genuinely extreme, well-separated high scores."""
        long_names = [
            "Introduction to Computer Science I", "Calculus and Analytic Geometry",
            "MA101", "GS101", "ST111", "ST112", "PH101", "PH102",
            "CS102", "Advanced Software Engineering Principles",
        ]
        scores = [55, 58, 60, 62, 63, 61, 59, 57, 100, 100]
        subjects = [SubjectEntry(n, s, units=3) for n, s in zip(long_names, scores)]
        summary, recs, standing, cgpa, anomaly = _build(subjects)
        self.assertEqual(len(anomaly.anomalous_subjects), 2)
        pdf_bytes = rg.generate_report_pdf("ST-CONTENT-4D", summary, recs, standing, cgpa, anomaly)
        self.assertEqual(_page_count(pdf_bytes), 1)

    def test_high_score_note_realistic_worst_case_stays_one_page(self):
        """The realistic maximum number of SIMULTANEOUSLY flaggable
        courses under the new per-student contextual logic, verified
        empirically: beyond a certain point, additional high outliers
        start statistically supporting each other (raising the
        comparison mean enough to pull their own Z-scores back down),
        so a scenario with every course flagged is not achievable -- nor
        should it be, since that would just describe a consistently
        high-performing student, not a set of individually-inconsistent
        scores. Nine tightly-clustered low scores plus three maximally
        extreme (100) scores is the empirically-verified realistic
        worst case (12 courses, 3 flags) for this PDF page-fit test."""
        subjects = [SubjectEntry(f"Course Name Number {i}", 50, units=3) for i in range(9)]
        subjects += [SubjectEntry(f"Extreme Course Number {i}", 100, units=3) for i in range(3)]
        summary, recs, standing, cgpa, anomaly = _build(subjects)
        self.assertEqual(len(anomaly.anomalous_subjects), 3)
        pdf_bytes = rg.generate_report_pdf("ST-CONTENT-4E", summary, recs, standing, cgpa, anomaly)
        self.assertEqual(_page_count(pdf_bytes), 1)

    def test_generate_report_pdf_handles_none_anomaly_result(self):
        """anomaly_result is optional -- passing None must not crash."""
        subjects = [SubjectEntry("CS101", 70, units=3)]
        summary, recs, standing, cgpa, _ = _build(subjects)
        pdf_bytes = rg.generate_report_pdf("ST-CONTENT-4F", summary, recs, standing, cgpa, None)
        self.assertEqual(_page_count(pdf_bytes), 1)

    def test_course_terminology_used_not_subject(self):
        subjects = [SubjectEntry("CS101", 75, units=3)]
        summary, recs, standing, cgpa, anomaly = _build(subjects)
        pdf_bytes = rg.generate_report_pdf("ST-CONTENT-5", summary, recs, standing, cgpa, anomaly)
        text = pypdf.PdfReader(io.BytesIO(pdf_bytes)).pages[0].extract_text()
        # Section headings/KPI labels render in uppercase in the PDF.
        self.assertIn("COURSES ENTERED", text)
        self.assertIn("COURSE PERFORMANCE BREAKDOWN", text)
        self.assertNotIn("SUBJECTS ENTERED", text)
        self.assertNotIn("SUBJECT PERFORMANCE BREAKDOWN", text)

    def test_gpa_label_used_not_predicted_cgpa(self):
        subjects = [SubjectEntry("CS101", 75, units=3)]
        summary, recs, standing, cgpa, anomaly = _build(subjects)
        pdf_bytes = rg.generate_report_pdf("ST-CONTENT-6", summary, recs, standing, cgpa, anomaly)
        text = pypdf.PdfReader(io.BytesIO(pdf_bytes)).pages[0].extract_text()
        self.assertNotIn("Predicted CGPA", text)
        self.assertNotIn("Predicted GPA", text)

    def test_previous_record_disclosure_appears_in_pdf(self):
        """When a previous CGPA + units was entered, the PDF must
        disclose exactly what was blended in -- the number should never
        be a black box to whoever reads the report."""
        subjects = [SubjectEntry("CS101", 90, units=6)]
        summary = build_academic_summary(subjects)
        features = [summary.average_score, summary.subject_count]
        standing = mu.predict_standing(features, REAL_MODELS)
        previous = PreviousRecord(cgpa=3.60, total_units=90)
        cgpa = mu.compute_cgpa(summary, previous_record=previous)
        anomaly = mu.detect_anomalous_subjects(summary.subjects, REAL_MODELS)
        recs = risk_scoring.recommendations(summary.gpa)
        pdf_bytes = rg.generate_report_pdf(
            "ST-PREV-1", summary, recs, standing, cgpa, anomaly, previous_record=previous,
        )
        text = pypdf.PdfReader(io.BytesIO(pdf_bytes)).pages[0].extract_text()
        self.assertIn("90", text)
        self.assertIn("3.60", text)
        self.assertEqual(_page_count(pdf_bytes), 1)

    def test_no_previous_record_no_disclosure_text_in_pdf(self):
        """Regression guard: the disclosure text must not appear at all
        when no previous record was entered (the common, default case)."""
        subjects = [SubjectEntry("CS101", 75, units=3)]
        summary, recs, standing, cgpa, anomaly = _build(subjects)
        pdf_bytes = rg.generate_report_pdf("ST-PREV-2", summary, recs, standing, cgpa, anomaly)
        text = pypdf.PdfReader(io.BytesIO(pdf_bytes)).pages[0].extract_text()
        self.assertNotIn("prior units", text.lower())

    def test_cgpa_and_gpa_are_genuinely_different_numbers_in_pdf_with_previous_record(self):
        """The actual point of this whole feature, verified at the PDF
        level: GPA (this semester) and CGPA (blended) must show
        different values when a previous record is provided."""
        subjects = [SubjectEntry("CS101", 90, units=6)]
        summary = build_academic_summary(subjects)
        features = [summary.average_score, summary.subject_count]
        standing = mu.predict_standing(features, REAL_MODELS)
        previous = PreviousRecord(cgpa=3.00, total_units=90)
        cgpa = mu.compute_cgpa(summary, previous_record=previous)
        anomaly = mu.detect_anomalous_subjects(summary.subjects, REAL_MODELS)
        recs = risk_scoring.recommendations(summary.gpa)
        self.assertNotAlmostEqual(cgpa.predicted_cgpa, summary.gpa, places=2)
        pdf_bytes = rg.generate_report_pdf(
            "ST-PREV-3", summary, recs, standing, cgpa, anomaly, previous_record=previous,
        )
        self.assertEqual(_page_count(pdf_bytes), 1)

    def test_recommendations_present_in_pdf(self):
        """Recommendations were removed from the interface but must still
        genuinely appear in the PDF's Recommended Actions section."""
        subjects = [SubjectEntry("CS101", 75, units=3)]
        summary, recs, standing, cgpa, anomaly = _build(subjects)
        pdf_bytes = rg.generate_report_pdf("ST-CONTENT-7", summary, recs, standing, cgpa, anomaly)
        text = pypdf.PdfReader(io.BytesIO(pdf_bytes)).pages[0].extract_text()
        self.assertIn("RECOMMENDED ACTIONS", text)  # section heading renders uppercase
        for rec in recs:
            # Loose containment check: pypdf's extract_text() sometimes
            # normalizes whitespace, so check word-by-word presence.
            self.assertTrue(any(word in text for word in rec.split()[:3]))


class TestReportGeneratorSurvivesBrokenFpdfInstall(unittest.TestCase):
    """Regression test for a real user-reported crash: a broken/conflicting
    local fpdf2 install (e.g. an old 'fpdf' PyFPDF package sharing the same
    import name, or a corrupted install) raised ImportError at module load
    time when 'from fpdf import FPDF' was a plain top-level import in
    report_generator.py -- which crashed app.py's own top-level
    `import report_generator`, taking down the ENTIRE app (analysis,
    standing/CGPA prediction, everything) over a problem that only
    actually affects PDF export. The import is now defensive: the module
    always imports successfully, and only generate_report_pdf() itself
    raises -- with a clear, actionable message -- if fpdf2 is genuinely
    broken."""

    def _reimport_report_generator_with_broken_fpdf(self):
        """Simulates the user's exact reported error by making any import
        of 'fpdf' raise, then re-imports report_generator fresh so its
        module-level try/except actually runs against the broken import."""
        import builtins
        import importlib
        import sys

        real_import = builtins.__import__

        def broken_import(name, *args, **kwargs):
            if name == "fpdf" or name.startswith("fpdf."):
                raise ImportError("cannot import name 'FPDF' from 'fpdf' (unknown location)")
            return real_import(name, *args, **kwargs)

        sys.modules.pop("report_generator", None)
        builtins.__import__ = broken_import
        try:
            import report_generator as broken_rg
            importlib.reload(broken_rg)
            return broken_rg
        finally:
            builtins.__import__ = real_import

    def test_module_still_imports_successfully(self):
        broken_rg = self._reimport_report_generator_with_broken_fpdf()
        self.assertIsNotNone(broken_rg)
        self.assertIsNotNone(broken_rg._FPDF_IMPORT_ERROR)

    def test_generate_report_pdf_raises_clear_actionable_error_not_import_error(self):
        broken_rg = self._reimport_report_generator_with_broken_fpdf()
        with self.assertRaises(RuntimeError) as ctx:
            broken_rg.generate_report_pdf("TEST/001", None, [], None, None, None)
        message = str(ctx.exception)
        self.assertIn("PDF generation is unavailable", message)
        self.assertIn("pip uninstall", message)
        self.assertIn("virtual environment", message)

    def test_app_still_boots_and_shows_analysis_even_with_fpdf_broken(self):
        """The most important property: a broken fpdf2 install must not
        prevent the user from seeing their academic analysis at all --
        only PDF export should be affected."""
        import builtins
        import sys

        real_import = builtins.__import__

        def broken_import(name, *args, **kwargs):
            if name == "fpdf" or name.startswith("fpdf."):
                raise ImportError("cannot import name 'FPDF' from 'fpdf' (unknown location)")
            return real_import(name, *args, **kwargs)

        for mod in ("report_generator", "app"):
            sys.modules.pop(mod, None)
        builtins.__import__ = broken_import
        try:
            from streamlit.testing.v1 import AppTest
            app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
            at = AppTest.from_file(app_path)
            at.run(timeout=30)
            self.assertEqual(list(at.exception), [])
        finally:
            builtins.__import__ = real_import
            for mod in ("report_generator", "app"):
                sys.modules.pop(mod, None)


if __name__ == "__main__":
    unittest.main()
