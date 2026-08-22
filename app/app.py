import streamlit as st

import config
import model_utils as mu
import report_generator
import risk_scoring
from feature_engineering import build_academic_summary, model_features
from validation import validate_matric_number, validate_subject_entries, collect_valid_subjects

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AARIS-Lite", page_icon="🎓", layout="wide")

# ---------------------------------------------------------------------------
# THEME — preserves AARIS-Lite's existing green academic identity
# (background gradient #e8f5ee/#f4fbf7, accent #1f7a4c, secondary #2f9e6d).
# One shared card template (.aaris-card) is used for every result on the
# page -- Academic Standing and GPA now render through the exact same
# template as Courses Entered / Average Score / Degree Classification, so
# GPA is never visually "secondary" to Academic Standing (see the value
# color modifiers below, which are the only thing that differs).
# ---------------------------------------------------------------------------

st.markdown("""
<style>
.stApp{ background: linear-gradient(135deg,#e8f5ee,#f4fbf7); }

.main-title{ font-size:42px; font-weight:700; color:#1f7a4c; margin-bottom:0; }
.sub-title{ font-size:17px; color:#2f9e6d; margin-top:2px; margin-bottom:22px; }
h2, h3{ color:#1f7a4c; }

div.stButton > button{
    background-color:#1f7a4c; color:white; border-radius:8px;
    height:2.8em; font-weight:600; border:none;
}
div.stButton > button:hover{ background-color:#2c9660; }

.stTextInput input, .stNumberInput input, .stSelectbox select{ border-radius:6px; }

.aaris-card{
    background:white; border-radius:10px; padding:18px 20px; margin-bottom:14px;
    border:1px solid #cfe8d9; box-shadow:0 1px 3px rgba(31,122,76,.08);
    min-height:84px;
}
.aaris-metric-label{ font-size:12px; font-weight:700; letter-spacing:.04em; text-transform:uppercase; color:#4c8a68; margin-bottom:4px; }
.aaris-metric-value{ font-size:26px; font-weight:700; color:#14432a; }
.aaris-metric-sub{ font-size:12.5px; color:#5a7a68; margin-top:4px; }

.aaris-value-good{ color:#1f7a4c; }
.aaris-value-risk{ color:#c34a4a; }
.aaris-value-muted{ color:#8ba39a; }

.aaris-footer{ margin-top:40px; text-align:center; color:#6d8c7a; font-size:12.5px;
    border-top:1px solid #cfe8d9; padding-top:14px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">AARIS-Lite™</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI Academic Records &amp; Intelligence System</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MODEL LOADING — never crashes; degrades quietly. Deliberately does not
# expose which internal component is affected or why (no model/classifier/
# file-path language on the interface) -- see result_card()'s "Unavailable"
# state below for how an outage is actually communicated to the student.
# ---------------------------------------------------------------------------

if "models" not in st.session_state:
    st.session_state.models = mu.load_models()
models: mu.ModelBundle = st.session_state.models

if not models.any_available:
    st.info("Academic standing and GPA results are temporarily unavailable. "
            "You can still enter and review course information below.")

# ---------------------------------------------------------------------------
# COURSE ENTRY — one unified, dynamic 1-12 course grid. Every downstream
# feature (GPA, degree classification, academic standing, PDF report) is
# computed from these SAME entries.
# ---------------------------------------------------------------------------

st.header("Student Academic Profile")

matric = st.text_input("Matric Number", placeholder="e.g. ST101")

if "subject_row_count" not in st.session_state:
    st.session_state.subject_row_count = min(4, config.MAX_SUBJECTS)

st.markdown(
    f'<div class="aaris-metric-sub">Enter between {config.MIN_SUBJECTS} and '
    f'{config.MAX_SUBJECTS} courses. Only the course name is required to '
    f'start — leave a row completely blank to skip it.</div>',
    unsafe_allow_html=True,
)
st.write("")

raw_entries = []
for i in range(st.session_state.subject_row_count):
    c1, c2 = st.columns([2, 1])
    with c1:
        name = st.text_input(f"Course {i + 1} name", key=f"subj_name_{i}", label_visibility="collapsed",
                              placeholder=f"Course {i + 1} name")
    with c2:
        score = st.number_input(f"Course {i + 1} score", key=f"subj_score_{i}", label_visibility="collapsed",
                                 min_value=0.0, max_value=100.0, value=None, step=1.0,
                                 placeholder="Score (0-100)")
    raw_entries.append((name, score))

add_col, remove_col, _ = st.columns([1, 1, 3])
with add_col:
    if st.session_state.subject_row_count < config.MAX_SUBJECTS:
        if st.button(f"+ Add Course ({st.session_state.subject_row_count}/{config.MAX_SUBJECTS})"):
            st.session_state.subject_row_count += 1
            st.rerun()
    else:
        st.caption(f"Maximum of {config.MAX_SUBJECTS} courses reached.")
with remove_col:
    if st.session_state.subject_row_count > config.MIN_SUBJECTS:
        if st.button("- Remove Last Course"):
            st.session_state.subject_row_count -= 1
            st.rerun()

analyze_clicked = st.button("Analyze Academic Profile", type="primary")


def result_card(label: str, value: str, value_class: str, subtext: str = "") -> str:
    """Renders one card through the SAME template every metric on this
    page uses -- Academic Standing and GPA are visually identical to
    Courses Entered / Average Score / Degree Classification; only the
    value's color (good/risk/muted) differs."""
    sub_html = f'<div class="aaris-metric-sub">{subtext}</div>' if subtext else ""
    return (
        f'<div class="aaris-card"><div class="aaris-metric-label">{label}</div>'
        f'<div class="aaris-metric-value {value_class}">{value}</div>{sub_html}</div>'
    )


if analyze_clicked:
    validation_result = validate_subject_entries(raw_entries)

    if not validation_result.valid:
        for err in validation_result.errors:
            st.error(err)
    else:
        subjects = collect_valid_subjects(raw_entries)
        summary = build_academic_summary(subjects)
        features = model_features(summary)

        standing_result = mu.predict_standing(features, models)
        cgpa_result = mu.predict_cgpa(features, models)
        anomaly_result = mu.detect_anomalous_subjects(summary.subjects, models)

        # ---------------- ACADEMIC STANDING + CGPA (primary results) ----------------
        st.subheader("Academic Summary")

        if standing_result.status == mu.STATUS_OK:
            standing_value = standing_result.standing
            standing_class = "aaris-value-good" if standing_result.standing == mu.STANDING_GOOD else "aaris-value-risk"
        else:
            standing_value, standing_class = "Unavailable", "aaris-value-muted"

        if cgpa_result.status == mu.STATUS_OK:
            cgpa_value, cgpa_class = f"{cgpa_result.predicted_cgpa:.2f}", "aaris-value-good"
        else:
            cgpa_value, cgpa_class = "Unavailable", "aaris-value-muted"

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown(result_card("Academic Standing", standing_value, standing_class), unsafe_allow_html=True)
        with r1c2:
            # Labeled "CGPA" (not "GPA") to distinguish it from the directly
            # computed GPA card in the row below -- these are two different
            # numbers (this one is the model's cumulative-GPA estimate), and
            # this label must stay in sync with report_generator.py's PDF,
            # which shows the exact same value under the same "CGPA" label.
            st.markdown(result_card("CGPA", cgpa_value, cgpa_class), unsafe_allow_html=True)

        # ---------------- SUPPORTING METRICS ----------------
        # Mirrors the PDF's own KPI row exactly (Courses Entered, Average
        # Score, GPA, Degree Classification) so the interface and the
        # downloaded report always show the same four supporting numbers.
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            st.markdown(result_card("Courses Entered", str(summary.subject_count), ""), unsafe_allow_html=True)
        with r2c2:
            st.markdown(result_card("Average Score", f"{summary.average_score:.1f}", ""), unsafe_allow_html=True)
        with r2c3:
            st.markdown(result_card("GPA", f"{summary.gpa:.2f}", ""), unsafe_allow_html=True)
        with r2c4:
            st.markdown(result_card("Degree Classification", summary.degree_class, ""), unsafe_allow_html=True)

        # ---------------- HIGH-SCORE CONFIRMATION NOTE ----------------
        # Narrow and honest: the anomaly model only ever flags scores that
        # are unusually HIGH relative to the real historical course-score
        # distribution it was trained on (verified directly against its
        # decision_function -- it does not flag low scores). So this is
        # framed as a plain "please confirm" data-entry check, not a vague
        # "anomaly detected" alert, and it only appears when something was
        # actually flagged.
        if anomaly_result.status == mu.STATUS_OK and anomaly_result.anomalous_subjects:
            flagged = anomaly_result.anomalous_subjects
            if len(flagged) == 1:
                note = f"Please confirm: {flagged[0]} score appears unusually high."
            else:
                note = f"Please confirm: the following scores appear unusually high: {', '.join(flagged)}."
            st.info(note)

        # ---------------- COURSE PERFORMANCE BREAKDOWN ----------------
        with st.expander("Course Performance Breakdown", expanded=True):
            st.dataframe(
                [{"Course": s.name, "Score": s.score, "Grade": s.grade} for s in summary.subjects],
                width="stretch", hide_index=True,
            )

        # ---------------- PDF REPORT ----------------
        st.subheader("Download Report")
        matric_ok, matric_err = validate_matric_number(matric)
        if not matric_ok:
            st.info(f"Enter a matric number above to generate a downloadable PDF report. ({matric_err})")
        else:
            recs = risk_scoring.recommendations(summary.gpa)
            pdf_bytes = report_generator.generate_report_pdf(
                matric, summary, recs, standing_result, cgpa_result, anomaly_result,
            )
            st.download_button(
                label="Download Student Report (PDF)",
                data=pdf_bytes,
                file_name=f"AARIS_report_{matric}.pdf",
                mime="application/pdf",
            )

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------

st.markdown('<div class="aaris-footer">© 2026 AARIS-Lite Academic Intelligence System</div>',
            unsafe_allow_html=True)
