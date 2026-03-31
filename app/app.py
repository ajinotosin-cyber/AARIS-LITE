import streamlit as st

import numpy as np

import pickle

import matplotlib.pyplot as plt

import os
from fpdf import FPDF
import tempfile
import sklearn


# ------------------------------

# PAGE CONFIG

# ------------------------------

st.set_page_config(

    page_title="AARIS-Lite",

    page_icon="🎓",

    layout="wide"

)



# ------------------------------

# THEME / STYLES (GREEN)

# ------------------------------

st.markdown("""

<style>



/* Background */

.stApp{

    background: linear-gradient(135deg,#e8f5ee,#f4fbf7);

}



/* Title */

.main-title{

    font-size:48px;

    font-weight:700;

    color:#1f7a4c;

    margin-bottom:0px;

}



/* Subtitle */

.sub-title{

    font-size:20px;

    color:#2f9e6d;

    margin-top:-10px;

    margin-bottom:20px;

}



/* Section headers */

h2, h3 {

    color:#1f7a4c;

}



/* Buttons */

div.stButton > button{

    background-color:#1f7a4c;

    color:white;

    border-radius:8px;

    height:3em;

    width:100%;

    font-weight:600;

}



/* Inputs */

.stTextInput input,

.stNumberInput input{

    border-radius:6px;

}



</style>

""", unsafe_allow_html=True)



# ------------------------------

# HEADER

# ------------------------------

st.markdown('<p class="main-title">AARIS-Lite™</p>', unsafe_allow_html=True)

st.markdown('<p class="sub-title">AI Academic Records & Intelligence System</p>', unsafe_allow_html=True)



# Optional banner image (place aaris_banner.png in the same folder)

if os.path.exists("aaris_banner.png"):

    st.image("aaris_banner.png", use_container_width=True)



# ------------------------------

# LOAD MODELS

# ------------------------------

BASE_DIR = os.path.dirname(__file__)
regression_path = os.path.join(BASE_DIR, "regression_model.pkl")
classifier_path = os.path.join(BASE_DIR, "classifier_model.pkl")

reg_model = pickle.load(open(regression_path, "rb"))
clf_model = pickle.load(open(classifier_path, "rb"))

# ------------------------------

# DEGREE CLASSIFICATION

# ------------------------------

def classify_degree(cgpa):



    if cgpa >= 4.50:

        return "First Class Honours"



    elif cgpa >= 3.50:

        return "Second Class Upper"



    elif cgpa >= 2.40:

        return "Second Class Lower"



    elif cgpa >= 1.50:

        return "Third Class"



    else:

        return "Academic Probation"



# ------------------------------

# RECOMMENDATIONS

# ------------------------------

def recommendations(cgpa):



    if cgpa >= 4.0:

        return [

            "Maintain strong academic performance",

            "Participate in research opportunities",

            "Mentor junior students"

        ]



    elif cgpa >= 2.5:

        return [

            "Increase weekly study hours",

            "Focus on weaker courses",

            "Practice past exam questions"

        ]



    else:

        return [

            "Seek academic advising immediately",

            "Attend tutorial sessions",

            "Follow a structured study schedule"

        ]



# =====================================================

# STUDENT PERFORMANCE PREDICTION

# =====================================================

st.header("Student Performance Prediction")



col1, col2, col3 = st.columns(3)



with col1:

    matric = st.text_input("Matric Number")



with col2:

    gpa = st.number_input("Current GPA", 0.0, 5.0)



with col3:

    cgpa = st.number_input("Current CGPA", 0.0, 5.0)





if st.button("Predict Performance"):



    score = cgpa * 20
    course_count = 8



    
    features = np.array([[score, course_count]])



    standing = clf_model.predict(features)[0]



    if standing == 1:

        st.success("Academic Standing: GOOD")



    else:

        st.error("Academic Standing: AT RISK")


# =====================================================

# GPA CALCULATOR (DYNAMIC COURSES)

# =====================================================

st.header("🎓 GPA Calculator (5.0 Nigerian System)")



grade_map = {

"A":5,

"B":4,

"C":3,

"D":2,

"E":1,

"F":0

}



MAX_COURSES = 8



if "course_count_dynamic" not in st.session_state:

    st.session_state.course_count_dynamic = 1



courses_data = []



for i in range(st.session_state.course_count_dynamic):



    st.subheader(f"Course {i+1}")



    col1, col2, col3 = st.columns(3)



    with col1:

        code = st.text_input("Course Code", key=f"code{i}")



    with col2:

        unit = st.number_input("Course Unit", min_value=1, max_value=6, key=f"unit{i}")



    with col3:

        grade = st.selectbox("Course Grade", ["A","B","C","D","E","F"], key=f"grade{i}")



    courses_data.append((code, unit, grade))



    if code != "" and st.session_state.course_count_dynamic < MAX_COURSES:

        if i == st.session_state.course_count_dynamic - 1:

            st.session_state.course_count_dynamic += 1

            st.rerun()



if st.button("Calculate GPA"):



    total_points = 0

    total_units = 0



    for code, unit, grade in courses_data:



        if code != "":

            total_points += grade_map[grade] * unit

            total_units += unit



    if total_units > 0:



        gpa_result = total_points / total_units

        st.success(f"GPA = {round(gpa_result,2)}")



    else:

        st.error("Enter at least one course")



# =====================================================

# CGPA CALCULATOR

# =====================================================

st.header("CGPA Calculator")



semesters = st.number_input("Number of Semesters",1,10)



gpas = []



for i in range(int(semesters)):



    sem_gpa = st.number_input(f"Semester {i+1} GPA",0.0,5.0,key=f"sem{i}")

    gpas.append(sem_gpa)



if st.button("Calculate CGPA"):



    cgpa_result = sum(gpas) / len(gpas)

    st.success(f"CGPA = {round(cgpa_result,2)}")



# =====================================================

# STUDENT ACADEMIC REPORT

# =====================================================

st.header("Student Academic Report")



matric_report = st.text_input("Student Matric Number")



semesters_report = st.number_input("Number of Semesters", 1, 10, key="semesters_report") 



report_gpas = []

course_codes = []



for i in range(int(semesters_report)):



    col1,col2 = st.columns(2)



    with col1:

        code = st.text_input(f"Course Code Semester {i+1}", key=f"course_code_{i}")



    with col2:

        gpa_val = st.number_input(f"Semester {i+1} GPA",0.0,5.0,key=f"semester_gpa{i}")



    course_codes.append(code)

    report_gpas.append(gpa_val)



if st.button("Generate Student Report"):



    cgpa = sum(report_gpas)/len(report_gpas)



    recs = recommendations(cgpa)



    # create PDF

    pdf = FPDF()

    pdf.add_page()



    pdf.set_font("Arial","B",16)

    pdf.cell(0,10,"AARIS-Lite Academic Report",ln=True)



    pdf.set_font("Arial","",12)



    pdf.cell(0,10,f"Matric Number: {matric_report}",ln=True)

    pdf.cell(0,10,f"Final CGPA: {round(cgpa,2)}",ln=True)



    pdf.ln(5)



    pdf.cell(0,10,"Courses:",ln=True)



    for c in course_codes:

        if c != "":

            pdf.cell(0,8,c,ln=True)



    pdf.ln(5)



    pdf.cell(0,10,"Recommendations:",ln=True)



    for r in recs:

        pdf.cell(0,8,"- "+r,ln=True)



    pdf_file = "student_report.pdf"

    pdf.output(pdf_file)



    with open(pdf_file,"rb") as f:

        st.download_button(

            label="Download Student Report",

            data=f,

            file_name="AARIS_student_report.pdf",

            mime="application/pdf"

        )

# =====================================================

# FOOTER

# =====================================================

st.caption("© 2026 AARIS-Lite Academic Intelligence System")