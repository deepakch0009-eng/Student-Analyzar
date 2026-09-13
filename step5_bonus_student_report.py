"""
step5_bonus_student_report.py
--------------------------------
BONUS / STRETCH GOAL: Personalized Student Report Lookup
=============================================================

This is optional extra practice for students who finish the main
4 phases and want to go further. Type in a student's name and get
their own personal mini-report -- this is a great way to practice
COMBINING everything learned in Steps 1-4 into one small feature.

Run it with:
    python step5_bonus_student_report.py
Then type a name when prompted (e.g. "Isha", "Aarav", "Priya").

Concepts combined here:
    - list.index() to find a student's row position
    - NumPy indexing (arr[idx]) to pull their row of scores
    - np.argsort() to rank them against the whole class
    - np.where() to compute their grade
    - Matplotlib bar chart, built just for this one student
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from step1_data import SUBJECTS, STUDENT_NAMES


def load_data():
    scores = np.load("data/subject_scores.npy")
    return scores


def student_report(name, scores):
    if name not in STUDENT_NAMES:
        print(f"'{name}' not found. Try one of: {STUDENT_NAMES[:5]}...")
        return

    idx = STUDENT_NAMES.index(name)
    student_scores = scores[idx]           # this student's row
    student_avg = student_scores.mean()

    # Rank this student against the whole class
    all_averages = scores.mean(axis=1)
    rank = int(np.sum(all_averages > student_avg)) + 1  # how many beat them, +1

    grade = "A" if student_avg >= 85 else "B" if student_avg >= 70 else \
            "C" if student_avg >= 55 else "D"

    best_subject = SUBJECTS[np.argmax(student_scores)]
    worst_subject = SUBJECTS[np.argmin(student_scores)]

    print(f"\n{'=' * 50}")
    print(f"  REPORT CARD: {name}")
    print(f"{'=' * 50}")
    for subject, mark in zip(SUBJECTS, student_scores):
        print(f"  {subject:20s}: {mark:5.1f}")
    print(f"  {'-' * 40}")
    print(f"  Average:            {student_avg:.1f}")
    print(f"  Grade:              {grade}")
    print(f"  Class Rank:         {rank} out of {len(STUDENT_NAMES)}")
    print(f"  Strongest subject:  {best_subject}")
    print(f"  Weakest subject:    {worst_subject}")

    # A small personal bar chart
    os.makedirs("output/charts", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(SUBJECTS, student_scores, color="#4c72b0")
    bars[np.argmax(student_scores)].set_color("#55a868")
    bars[np.argmin(student_scores)].set_color("#c44e52")
    ax.axhline(student_avg, color="gray", linestyle="--",
               label=f"Their average = {student_avg:.1f}")
    ax.set_title(f"{name}'s Subject-wise Performance", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    filename = f"output/charts/report_{name.lower()}.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Chart saved -> {filename}")


if __name__ == "__main__":
    scores = load_data()
    print("Available students:", ", ".join(STUDENT_NAMES))
    name = input("\nEnter a student's name to see their report: ").strip()
    student_report(name, scores)
