"""
report_generator.py
---------------------
Builds AARIS-LITE's one-page "Academic Performance Report" PDF. Writes to
a real temp file (never the project directory) and cleans up afterward --
see the module history note at the bottom of this docstring.

DESIGN
------
A4, single page, always. Six bands, top to bottom:
  1. Header: AARIS-LITE wordmark, report title, matric number, timestamp,
     divider.
  2. Academic summary: 4 KPI cards (courses entered, average score, GPA,
     degree class) -- all read directly from the SAME AcademicSummary the
     Streamlit UI displays, never recomputed.
  3. Performance assessment: two matching cards, Academic Standing and
     CGPA, each rendered from the actual PredictionResult the app already
     computed. An unavailable result is rendered in a calm, neutral
     color with plain academic language -- never as a favourable-looking
     result, and never with backend/model terminology. ("CGPA" here is
     the model's cumulative-GPA estimate -- deliberately a different
     label from the directly-computed "GPA" KPI card above, since they
     are two different numbers.)
  4. High-score confirmation note (conditional): a single plain-language
     line, shown ONLY when detect_anomalous_subjects() actually flagged
     something, asking the student to confirm an unusually high score.
     Height is measured with a dry-run pass, same as the recommendations
     panel, so it can never push the report to a second page. See
     "ANOMALY DETECTION" below for exactly what this does and doesn't
     mean.
  5. Course breakdown: a compact table, 1 or 2 columns depending on
     course count (>6 courses splits into 2 columns) so 12 courses never
     forces a second page. Long course names are truncated for display
     only (the underlying data is untouched).
  6. Recommended actions: the exact list from risk_scoring.recommendations(),
     in a bordered panel, height measured with a dry-run pass so the
     layout adapts if recommendation text length ever changes.
  7. Footer: subtle divider + attribution + timestamp + "Page 1 of 1".

Every color is taken directly from app.py's own CSS (the same hex values,
converted to RGB) so the PDF and the live app share one visual identity.

ANOMALY DETECTION -- WHAT THE HIGH-SCORE NOTE ACTUALLY MEANS
----------------------------------------------------------------
anomaly_model.pkl (an Isolation Forest) is trained on the real historical
distribution of individual course scores. Verified directly against its
decision_function: it is asymmetric -- it reliably flags unusually HIGH
scores (empirically, roughly 80+ against this dataset, where real scores
mostly cluster in the 40s-70s) but never flags low scores, because the
training data has a hard floor around 40 and nothing below that looks
"unusual" to the model. So this is framed as a narrow, honest "please
confirm this score" data-entry check -- not a general "anomaly" or
"performance consistency" claim, which the model doesn't actually
support. See app.py for the identical framing on the interface side.

WHAT THIS REPORT DELIBERATELY DOES NOT SHOW
---------------------------------------------
No model/prediction disclaimer, no "indicative, not authoritative"
language, no mention of classifiers, regressors, training data, or
feature sets, and no generic "Anomaly Status" card. This is a
presentation decision, not a change to what's actually computed -- every
number shown is the same real, unmodified calculation as before; only
the label and surrounding language changed. See app.py for the
equivalent interface-side decisions.

HISTORY
-------
Before this pass, the PDF was Arial-12pt cell() calls dumped top to
bottom with no visual structure, and it never received the ML prediction
results at all (generate_report_pdf only took matric_number, summary,
recommendations -- standing/CGPA/anomaly were computed in app.py but
never passed through). A later pass extended the signature to accept the
PredictionResult objects and added a 3-card Performance Assessment
(Academic Standing / Predicted CGPA / Anomaly Status) plus a model
disclaimer paragraph. The next pass removed the disclaimer and the
anomaly card entirely, relabeled "Predicted CGPA" to plain "GPA", and
renamed "Subject" to "Course" throughout -- but that briefly left the
model's predicted value and the directly-computed GPA sharing the same
"GPA" label on the interface with no way to tell them apart. This pass
relabels the predicted-value card to "CGPA" (distinct from the
directly-computed "GPA" KPI card) and reintroduces anomaly detection in
a narrow, honest form: a single conditional sentence confirming an
unusually high score, never a generic "anomaly" label.

Previously app.py wrote the PDF directly into its own directory as
"student_report.pdf" (a relative path resolved against the process's
current working directory) -- every use of the "Generate Student Report"
button left a stray file behind, and one such file had been accidentally
committed to the repo. This module writes to a proper temp file instead,
reads the bytes back for the Streamlit download button, and cleans up
afterward. Nothing is left behind in the project directory.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime

from feature_engineering import AcademicSummary
import model_utils as mu

# ---------------------------------------------------------------------------
# fpdf2 is imported defensively, not at plain module level. On a machine
# with a broken/corrupted fpdf2 install or a Python version fpdf2 doesn't
# yet fully support, letting this import fail at module load time would
# crash app.py's own top-level `import report_generator` -- taking down
# the ENTIRE app (analysis, standing/CGPA prediction, everything) over a
# problem that only actually affects PDF export. Instead, the failure is
# captured here and only surfaced -- with a clear, actionable message --
# if/when the user actually tries to generate a PDF.
# ---------------------------------------------------------------------------
try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    _FPDF_IMPORT_ERROR: str | None = None
except ImportError as exc:
    FPDF = object  # placeholder base class so _ReportPDF below can still be defined
    XPos = YPos = None
    _FPDF_IMPORT_ERROR = (
        f"PDF generation is unavailable: the 'fpdf2' package could not be imported "
        f"({exc}). This is almost always a local Python-environment problem, not a "
        f"bug in this app -- most commonly an old 'fpdf' (PyFPDF) package installed "
        f"alongside fpdf2 (they share the same import name), or a corrupted install. "
        f"Try, in order: (1) pip uninstall --yes fpdf, then pip install --upgrade "
        f"fpdf2 -- (2) if that doesn't help, create a fresh virtual environment "
        f"(python -m venv venv) and reinstall requirements.txt into it, which "
        f"sidesteps any conflicting global package state entirely."
    )

# ---------------------------------------------------------------------------
# PALETTE -- lifted directly from app.py's CSS (same hex values) so the PDF
# and the live app are visually the same product.
# ---------------------------------------------------------------------------

GREEN_PRIMARY = (31, 122, 76)      # #1f7a4c -- brand accent, headings, GOOD status
GREEN_SECONDARY = (47, 158, 109)   # #2f9e6d -- subtitle
GREEN_FILL = (232, 245, 238)       # #e8f5ee -- light card fill
GREEN_FILL_SOFT = (244, 251, 247)  # #f4fbf7 -- alternate light fill
GREEN_BORDER = (207, 232, 217)     # #cfe8d9 -- card borders
GREEN_TEXT_DARK = (20, 67, 42)     # #14432a -- KPI values
GREEN_TEXT_MUTED = (76, 138, 104)  # #4c8a68 -- KPI labels
NOTE_TEXT = (90, 122, 104)         # #5a7a68 -- disclaimer/footnote text
FOOTER_TEXT = (109, 140, 122)      # #6d8c7a -- footer

RISK_BORDER = (195, 74, 74)        # #c34a4a -- AT RISK accent (the only "alert" color used)
RISK_FILL = (250, 235, 235)        # light red fill
MUTED_BORDER = (140, 163, 151)     # neutral gray-green -- "unavailable" state (calm, not an alarm)
MUTED_FILL = (245, 248, 246)       # light neutral fill

WHITE = (255, 255, 255)
BODY_TEXT = (40, 50, 45)
TABLE_HEADER_FILL = (31, 122, 76)
TABLE_ROW_ALT = (244, 251, 247)
TABLE_BORDER = (222, 234, 227)

PAGE_MARGIN = 15  # mm, all sides
CONTENT_WIDTH = 210 - (2 * PAGE_MARGIN)  # A4 width 210mm

# Internal render statuses for assessment cards -- distinct from
# model_utils.STATUS_* (see _status_style's docstring for why).
RENDER_GOOD = "RENDER_GOOD"
RENDER_RISK = "RENDER_RISK"
RENDER_MUTED = "RENDER_MUTED"


def _fit_text_to_width(pdf: _ReportPDF, text: str, max_width: float,
                        family: str, style: str, start_size: float, min_size: float) -> float:
    """Shrinks font size (in steps of 0.5pt) until `text` fits within
    max_width at the given font, down to min_size; returns the size used.
    Guarantees KPI/headline values (which can be arbitrary-length strings
    like a degree classification) never overflow their card."""
    size = start_size
    pdf.set_font(family, style, size)
    while size > min_size and pdf.get_string_width(text) > max_width:
        size -= 0.5
        pdf.set_font(family, style, size)
    return size


def _truncate_to_width(pdf: _ReportPDF, text: str, max_width: float,
                        family: str, style: str, size: float) -> str:
    """Truncates `text` with a trailing '...' until it fits max_width at a
    FIXED font size (used where shrinking the font isn't appropriate,
    e.g. table cells and card subtext, so type size stays consistent)."""
    pdf.set_font(family, style, size)
    if pdf.get_string_width(text) <= max_width:
        return text
    while text and pdf.get_string_width(text + "...") > max_width:
        text = text[:-1]
    return text.rstrip() + "..." if text else "..."


class _ReportPDF(FPDF):
    """Thin subclass so section-heading styling is defined once."""

    def section_heading(self, text: str, y: float | None = None) -> None:
        if y is not None:
            self.set_y(y)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*GREEN_PRIMARY)
        self.cell(0, 7, text.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*GREEN_BORDER)
        self.set_line_width(0.3)
        y2 = self.get_y() + 0.5
        self.line(PAGE_MARGIN, y2, PAGE_MARGIN + CONTENT_WIDTH, y2)
        self.set_y(y2 + 2.5)


def _status_style(render_status: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Returns (border_color, fill_color) for an internal render status
    (RENDER_GOOD / RENDER_RISK / RENDER_MUTED) -- deliberately NOT the raw
    model_utils.STATUS_* value, because STATUS_OK only means "the
    prediction succeeded", not "the outcome is favourable" (an AT RISK
    standing is still STATUS_OK). Conflating the two previously rendered
    an AT RISK card in the same green as a GOOD one. RENDER_MUTED covers
    every "unavailable" case with one calm, neutral treatment rather than
    an alarming red/amber -- an unavailable result isn't the student's
    fault and shouldn't look like an error."""
    if render_status == RENDER_GOOD:
        return GREEN_PRIMARY, GREEN_FILL
    if render_status == RENDER_MUTED:
        return MUTED_BORDER, MUTED_FILL
    return RISK_BORDER, RISK_FILL  # RENDER_RISK


def _draw_kpi_card(pdf: _ReportPDF, x: float, y: float, w: float, h: float,
                    label: str, value: str) -> None:
    pdf.set_draw_color(*GREEN_BORDER)
    pdf.set_fill_color(*GREEN_FILL_SOFT)
    pdf.set_line_width(0.25)
    pdf.rect(x, y, w, h, style="DF", round_corners=True, corner_radius=2)

    # Label wraps onto a 2nd line if it doesn't fit at a fixed 8pt --
    # this keeps every KPI label the same visual size (unlike shrinking
    # the font, which would make "DEGREE CLASSIFICATION" look smaller
    # than "GPA"), and guarantees it can never overflow the card no
    # matter how narrow the card or how long a future label gets.
    # Regression test: "DEGREE CLASSIFICATION" previously overflowed
    # past the card's right edge (and off the page) because only the
    # VALUE below was ever fit-checked, not the label itself.
    max_w = w - 8
    pdf.set_xy(x + 4, y + 3.2)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*GREEN_TEXT_MUTED)
    pdf.multi_cell(max_w, 3.6, label.upper(), align="L")

    # Value font auto-shrinks so it can never overflow the card, however
    # long the string (e.g. "Second Class Upper", "Academic Probation").
    size = _fit_text_to_width(pdf, value, max_w, "Helvetica", "B", start_size=18, min_size=10)
    pdf.set_xy(x + 4, y + h - 12)
    pdf.set_font("Helvetica", "B", size)
    pdf.set_text_color(*GREEN_TEXT_DARK)
    pdf.cell(max_w, 9, value, new_x=XPos.LEFT, new_y=YPos.NEXT)


def _draw_assessment_card(pdf: _ReportPDF, x: float, y: float, w: float, h: float,
                           label: str, headline: str, subtext: str, status: str) -> None:
    border, fill = _status_style(status)

    pdf.set_draw_color(*border)
    pdf.set_fill_color(*fill)
    pdf.set_line_width(0.3)
    pdf.rect(x, y, w, h, style="DF", round_corners=True, corner_radius=2)
    # Left accent bar
    pdf.set_fill_color(*border)
    pdf.rect(x, y, 1.8, h, style="F")

    inner_w = w - 10

    pdf.set_xy(x + 6, y + 3.2)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*GREEN_TEXT_MUTED)
    pdf.multi_cell(inner_w, 3.6, label.upper(), align="L")  # wraps rather than overflows -- see _draw_kpi_card

    headline_size = _fit_text_to_width(pdf, headline, inner_w, "Helvetica", "B",
                                        start_size=14, min_size=10)
    pdf.set_xy(x + 6, y + 9)
    pdf.set_font("Helvetica", "B", headline_size)
    r, g, b = border
    pdf.set_text_color(r, g, b)
    pdf.cell(inner_w, 6.5, headline, new_x=XPos.LEFT, new_y=YPos.NEXT)

    # Subtext: fixed small size, wrapped to a guaranteed-fitting max 2
    # lines (truncated with an ellipsis if it would need a 3rd), so it
    # can never clip against the bottom of the card.
    sub_size = 7.3
    line_h = 3.4
    max_lines = 2
    pdf.set_font("Helvetica", "", sub_size)
    lines = pdf.multi_cell(inner_w, line_h, subtext, dry_run=True, output="LINES")
    if len(lines) > max_lines:
        # Truncate the raw text progressively until it wraps to <= max_lines.
        candidate = subtext
        while candidate and len(pdf.multi_cell(inner_w, line_h, candidate + "...",
                                                 dry_run=True, output="LINES")) > max_lines:
            candidate = candidate[:-1]
        subtext = (candidate.rstrip() + "...") if candidate else subtext[:1]

    pdf.set_xy(x + 6, y + 17)
    pdf.set_font("Helvetica", "", sub_size)
    pdf.set_text_color(*BODY_TEXT)
    pdf.multi_cell(inner_w, line_h, subtext, align="L")


def _standing_content(standing_result) -> tuple[str, str, str]:
    """Plain, professional academic language only -- no mention of
    classifiers, models, or predictions. A failure of any kind (model
    unavailable or a genuine prediction error) is presented identically
    as "Unavailable" -- the distinction between those two backend causes
    is an implementation detail, not something the student needs to see,
    and neither is ever shown as a favourable result."""
    if standing_result.status == mu.STATUS_OK:
        good = standing_result.standing == mu.STANDING_GOOD
        return (
            standing_result.standing,
            "Meets good-standing threshold" if good else "Below good-standing threshold",
            RENDER_GOOD if good else RENDER_RISK,
        )
    return "Unavailable", "Academic standing unavailable for this report", RENDER_MUTED


def _cgpa_content(cgpa_result, previous_record=None) -> tuple[str, str, str]:
    """Labeled 'CGPA' (not 'GPA') in the Performance Assessment section.
    When no previous record was entered, this is the same credit-unit-
    weighted calculation as the KPI row's 'GPA' card (summary.gpa; see
    feature_engineering.build_academic_summary's docstring for why
    they're computed identically in that case) -- shown under its own
    label since the two cards previously both said "GPA" with different
    numbers, which was genuinely ambiguous. When a previous CGPA + units
    WAS entered, the subtitle discloses exactly what was blended in, so
    the number is never a black box. status is always STATUS_OK in
    practice now that this is a deterministic calculation rather than a
    model prediction -- the unavailable branch is kept only as a
    defensive fallback."""
    if cgpa_result.status == mu.STATUS_OK:
        if previous_record is not None:
            sub = f"Includes {previous_record.total_units} prior units at {previous_record.cgpa:.2f} CGPA"
        else:
            sub = ""
        return f"{cgpa_result.predicted_cgpa:.2f}", sub, RENDER_GOOD
    return "Unavailable", "CGPA unavailable for this report", RENDER_MUTED


def generate_report_pdf(
    matric_number: str,
    summary: AcademicSummary,
    recommendations: list[str],
    standing_result,
    cgpa_result,
    anomaly_result=None,
    previous_record=None,
) -> bytes:
    """
    standing_result / cgpa_result: the exact model_utils.PredictionResult
    objects app.py already computed for this submission (from
    predict_standing / predict_cgpa) -- reused here, never recalculated,
    so the PDF can never show a different result than the one the student
    saw on screen.

    anomaly_result: model_utils.PredictionResult from
    detect_anomalous_subjects(). When it flags one or more courses, a
    single honest "please confirm this score" sentence is shown (see
    module docstring, ANOMALY DETECTION) -- never a generic "Anomaly
    Status" card or backend terminology. Pass None to omit the check
    entirely (e.g. if the caller never ran it).
    """
    if _FPDF_IMPORT_ERROR is not None:
        raise RuntimeError(_FPDF_IMPORT_ERROR)

    pdf = _ReportPDF(format="A4")
    pdf.set_auto_page_break(False)
    pdf.set_margins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)
    pdf.add_page()
    pdf.set_text_color(*BODY_TEXT)

    generated_at = datetime.now().strftime("%b %d, %Y %H:%M")

    # ---------------- 1. HEADER ----------------
    pdf.set_xy(PAGE_MARGIN, PAGE_MARGIN)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(*GREEN_PRIMARY)
    pdf.cell(130, 11, "AARIS-LITE", new_x=XPos.LEFT, new_y=YPos.NEXT)

    pdf.set_x(PAGE_MARGIN)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(*GREEN_SECONDARY)
    pdf.cell(130, 7, "Academic Performance Report", new_x=XPos.LEFT, new_y=YPos.NEXT)

    pdf.set_xy(PAGE_MARGIN + 120, PAGE_MARGIN + 2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*BODY_TEXT)
    pdf.cell(CONTENT_WIDTH - 120, 5.5, f"Matric: {matric_number}", align="R", new_x=XPos.LEFT, new_y=YPos.NEXT)
    pdf.set_x(PAGE_MARGIN + 120)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*NOTE_TEXT)
    pdf.cell(CONTENT_WIDTH - 120, 5.5, f"Generated: {generated_at}", align="R", new_x=XPos.LEFT, new_y=YPos.NEXT)

    divider_y = PAGE_MARGIN + 22
    pdf.set_draw_color(*GREEN_PRIMARY)
    pdf.set_line_width(0.7)
    pdf.line(PAGE_MARGIN, divider_y, PAGE_MARGIN + CONTENT_WIDTH, divider_y)

    # ---------------- 2. ACADEMIC SUMMARY (KPI cards) ----------------
    kpi_y = divider_y + 7
    kpi_h = 27
    gap = 6
    kpi_w = (CONTENT_WIDTH - 3 * gap) / 4
    kpis = [
        ("Courses Entered", str(summary.subject_count)),
        ("Average Score", f"{summary.average_score:.1f}"),
        ("GPA", f"{summary.gpa:.2f}"),
        ("Degree Classification", summary.degree_class),
    ]
    for i, (label, value) in enumerate(kpis):
        x = PAGE_MARGIN + i * (kpi_w + gap)
        _draw_kpi_card(pdf, x, kpi_y, kpi_w, kpi_h, label, value)

    # ---------------- 3. PERFORMANCE ASSESSMENT ----------------
    assess_y = kpi_y + kpi_h + 10
    pdf.section_heading("Performance Assessment", y=assess_y)
    cards_y = pdf.get_y()
    card_h = 32
    card_w = (CONTENT_WIDTH - gap) / 2

    standing_headline, standing_sub, standing_status = _standing_content(standing_result)
    cgpa_headline, cgpa_sub, cgpa_status = _cgpa_content(cgpa_result, previous_record=previous_record)

    _draw_assessment_card(pdf, PAGE_MARGIN, cards_y, card_w, card_h,
                           "Academic Standing", standing_headline, standing_sub, standing_status)
    _draw_assessment_card(pdf, PAGE_MARGIN + card_w + gap, cards_y, card_w, card_h,
                           "CGPA", cgpa_headline, cgpa_sub, cgpa_status)

    # ---------------- 4. HIGH-SCORE CONFIRMATION NOTE (conditional) ----------------
    # Only flags a score that is both exceptionally high AND statistically
    # inconsistent with the specific student's own other scores (see
    # model_utils.detect_anomalous_subjects -- see app.py for the same
    # note, kept in sync). Only rendered when something was actually
    # flagged; height is measured with a dry-run pass so it can never push
    # the report to a second page regardless of how many courses trigger it.
    note_top = cards_y + card_h + 6
    note_bottom = note_top
    if anomaly_result is not None and anomaly_result.status == mu.STATUS_OK and anomaly_result.anomalous_subjects:
        flagged = anomaly_result.anomalous_subjects
        if len(flagged) == 1:
            subject_phrase = f"the {flagged[0]} score"
        else:
            subject_phrase = f"the following scores: {', '.join(flagged)}"
        note_text = (
            f"Score Confirmation Recommended: one or more exceptionally high scores "
            f"({subject_phrase}) may warrant review because they appear statistically "
            f"inconsistent with the student's available academic performance data. Please "
            f"verify the score entry against the original academic record."
        )

        pdf.set_font("Helvetica", "I", 8.5)
        lines = pdf.multi_cell(CONTENT_WIDTH, 4.2, note_text, dry_run=True, output="LINES")
        note_h = max(1, len(lines)) * 4.2

        pdf.set_xy(PAGE_MARGIN, note_top)
        pdf.set_text_color(*NOTE_TEXT)
        pdf.multi_cell(CONTENT_WIDTH, 4.2, note_text, align="L")
        note_bottom = note_top + note_h

    # ---------------- 5. COURSE PERFORMANCE BREAKDOWN ----------------
    table_y = note_bottom + 8
    pdf.section_heading("Course Performance Breakdown", y=table_y)
    table_top = pdf.get_y()

    subjects = summary.subjects
    two_col = len(subjects) > 6
    row_h = 8.2
    header_h = 9

    if two_col:
        col_w = (CONTENT_WIDTH - 8) / 2
        split = (len(subjects) + 1) // 2
        columns = [subjects[:split], subjects[split:]]
    else:
        col_w = CONTENT_WIDTH
        columns = [subjects]

    name_w, score_w, grade_w, units_w = col_w * 0.48, col_w * 0.20, col_w * 0.16, col_w * 0.16
    table_font = 9

    for col_idx, col_subjects in enumerate(columns):
        x0 = PAGE_MARGIN + col_idx * (col_w + 8)
        y = table_top

        pdf.set_xy(x0, y)
        pdf.set_fill_color(*TABLE_HEADER_FILL)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", table_font)
        pdf.cell(name_w, header_h, "Course", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(score_w, header_h, "Score", fill=True, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(grade_w, header_h, "Grade", fill=True, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(units_w, header_h, "Units", fill=True, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        y += header_h

        pdf.set_font("Helvetica", "", table_font)
        for i, subj in enumerate(col_subjects):
            pdf.set_xy(x0, y)
            if i % 2 == 1:
                pdf.set_fill_color(*TABLE_ROW_ALT)
                pdf.rect(x0, y, col_w, row_h, style="F")
            pdf.set_draw_color(*TABLE_BORDER)
            pdf.set_text_color(*BODY_TEXT)
            display_name = _truncate_to_width(pdf, subj.name, name_w - 4, "Helvetica", "", table_font)
            pdf.cell(name_w, row_h, display_name, new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.cell(score_w, row_h, f"{subj.score:g}", align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.set_font("Helvetica", "B", table_font)
            pdf.set_text_color(*GREEN_PRIMARY)
            pdf.cell(grade_w, row_h, subj.grade, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.set_font("Helvetica", "", table_font)
            pdf.set_text_color(*BODY_TEXT)
            pdf.cell(units_w, row_h, str(subj.units), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            y += row_h

        pdf.set_draw_color(*TABLE_BORDER)
        pdf.set_line_width(0.2)
        pdf.line(x0, y, x0 + col_w, y)

    rows_per_col = max(len(c) for c in columns) if columns else 0
    table_bottom = table_top + header_h + rows_per_col * row_h

    # ---------------- 5. RECOMMENDED ACTIONS ----------------
    rec_y = table_bottom + 9
    pdf.section_heading("Recommended Actions", y=rec_y)
    panel_top = pdf.get_y()

    bullet_w = CONTENT_WIDTH - 12
    rec_line_h = 5.5
    pdf.set_font("Helvetica", "", 10)
    line_heights = []
    for rec in recommendations:
        lines = pdf.multi_cell(bullet_w, rec_line_h, rec, dry_run=True, output="LINES")
        line_heights.append(max(1, len(lines)) * rec_line_h)
    item_gap = 6
    panel_h = sum(line_heights) + item_gap * len(recommendations) + 6

    pdf.set_draw_color(*GREEN_BORDER)
    pdf.set_fill_color(*GREEN_FILL_SOFT)
    pdf.set_line_width(0.25)
    pdf.rect(PAGE_MARGIN, panel_top, CONTENT_WIDTH, panel_h, style="DF", round_corners=True, corner_radius=2)

    cy = panel_top + 5
    for rec in recommendations:
        pdf.set_xy(PAGE_MARGIN + 6, cy)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*GREEN_PRIMARY)
        pdf.cell(5, rec_line_h, "-", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_xy(PAGE_MARGIN + 11, cy)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*BODY_TEXT)
        pdf.multi_cell(bullet_w - 6, rec_line_h, rec, align="L")
        cy = pdf.get_y() + item_gap

    # ---------------- 6. FOOTER ----------------
    footer_y = 297 - PAGE_MARGIN - 8
    pdf.set_draw_color(*GREEN_BORDER)
    pdf.set_line_width(0.25)
    pdf.line(PAGE_MARGIN, footer_y, PAGE_MARGIN + CONTENT_WIDTH, footer_y)

    pdf.set_xy(PAGE_MARGIN, footer_y + 2)
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(*FOOTER_TEXT)
    pdf.cell(CONTENT_WIDTH / 2, 5, "Generated by AARIS-LITE Academic Intelligence System")
    pdf.set_xy(PAGE_MARGIN + CONTENT_WIDTH / 2, footer_y + 2)
    pdf.cell(CONTENT_WIDTH / 2, 5, f"{generated_at}  |  Page 1 of 1", align="R")

    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        pdf.output(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.remove(tmp_path)
