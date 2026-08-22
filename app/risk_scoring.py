"""
risk_scoring.py
-----------------
Advice/recommendation logic, keyed off a student's GPA. Previously
defined inline inside app.py; moved here as part of separating
presentation from logic. (This module's name was previously used for an
empty, unused stub under src/ -- that file is gone; this is the real,
used module, living under app/ because it's production logic, not a
training-time utility.)
"""

from __future__ import annotations


def recommendations(gpa: float) -> list[str]:
    if gpa >= 4.0:
        return [
            "Maintain strong academic performance",
            "Participate in research opportunities",
            "Mentor junior students",
        ]
    elif gpa >= 2.5:
        return [
            "Increase weekly study hours",
            "Focus on weaker courses",
            "Practice past exam questions",
        ]
    else:
        return [
            "Seek academic advising immediately",
            "Attend tutorial sessions",
            "Follow a structured study schedule",
        ]
