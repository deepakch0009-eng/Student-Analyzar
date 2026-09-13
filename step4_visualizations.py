"""
step4_visualizations.py
--------------------------
PHASE 4: MATPLOTLIB -- Turning Numbers Into Insight
========================================================

Numbers in a terminal are hard to feel. Charts make patterns obvious
at a glance. This file builds every common chart type a beginner
should know, each answering a specific real question.

Matplotlib methods covered in this file:
    plt.plot()          -> line chart      (a student's progress over time)
    plt.bar()            -> bar chart       (average score per subject)
    plt.hist()            -> histogram       (distribution of all scores)
    plt.scatter()        -> scatter plot    (does Math score relate to Science score?)
    plt.pie()            -> pie chart       (grade distribution)
    plt.subplots()       -> multi-panel figure (one-page report card)
    plt.xlabel/ylabel/title/legend/grid -> labeling & polish
    plt.savefig()        -> exporting the chart as an image file
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from step1_data import SUBJECTS, STUDENT_NAMES

OUT_DIR = "output/charts"


def load_data():
    scores = np.load("data/subject_scores.npy")
    trend = np.load("data/test_trend.npy")
    return scores, trend


def chart_bar_subject_averages(scores):
    """BAR CHART: which subject does the class score highest/lowest in?"""
    subject_mean = scores.mean(axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(SUBJECTS, subject_mean, color="#4c72b0", edgecolor="black")

    # Highlight the highest bar in green, lowest in red -- a nice
    # beginner trick using np.argmax/argmin
    bars[np.argmax(subject_mean)].set_color("#55a868")
    bars[np.argmin(subject_mean)].set_color("#c44e52")

    ax.set_title("Average Score per Subject", fontsize=14, fontweight="bold")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Average Score")
    ax.set_ylim(0, 100)
    for i, v in enumerate(subject_mean):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/01_bar_subject_averages.png", dpi=150)
    plt.close(fig)


def chart_histogram_score_distribution(scores):
    """HISTOGRAM: how are ALL scores (across every subject) distributed?"""
    all_scores = scores.flatten()  # turn the 2D array into 1D for the histogram

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(all_scores, bins=15, color="#8172b2", edgecolor="white", alpha=0.85)
    ax.axvline(all_scores.mean(), color="red", linestyle="--", linewidth=2,
               label=f"Mean = {all_scores.mean():.1f}")
    ax.axvline(np.median(all_scores), color="green", linestyle="--", linewidth=2,
               label=f"Median = {np.median(all_scores):.1f}")

    ax.set_title("Distribution of All Scores (Every Student, Every Subject)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Score")
    ax.set_ylabel("Number of Scores")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/02_histogram_distribution.png", dpi=150)
    plt.close(fig)


def chart_scatter_math_vs_science(scores):
    """SCATTER PLOT: is there a relationship between Math and Science scores?"""
    math_scores = scores[:, 0]
    science_scores = scores[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(math_scores, science_scores, s=80, color="#4c72b0",
               edgecolor="black", alpha=0.75)

    # Simple trend line using np.polyfit (a very common companion to
    # scatter plots -- fits a straight line through the points)
    slope, intercept = np.polyfit(math_scores, science_scores, 1)
    x_line = np.linspace(math_scores.min(), math_scores.max(), 50)
    ax.plot(x_line, slope * x_line + intercept, color="red", linewidth=2,
            label=f"Trend line (slope={slope:.2f})")

    ax.set_title("Math Score vs Science Score", fontsize=14, fontweight="bold")
    ax.set_xlabel("Math Score")
    ax.set_ylabel("Science Score")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/03_scatter_math_vs_science.png", dpi=150)
    plt.close(fig)


def chart_pie_grade_distribution(scores):
    """PIE CHART: what fraction of the class falls into each grade band?"""
    student_avg = scores.mean(axis=1)
    grades = np.where(student_avg >= 85, "A",
              np.where(student_avg >= 70, "B",
              np.where(student_avg >= 55, "C", "D")))

    unique, counts = np.unique(grades, return_counts=True)
    colors_map = {"A": "#55a868", "B": "#4c72b0", "C": "#dd8452", "D": "#c44e52"}
    colors = [colors_map[g] for g in unique]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(counts, labels=unique, autopct="%1.0f%%", colors=colors,
           startangle=90, wedgeprops={"edgecolor": "white"})
    ax.set_title("Class Grade Distribution", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/04_pie_grade_distribution.png", dpi=150)
    plt.close(fig)


def chart_line_student_progress(trend, student_name="Isha"):
    """LINE CHART: how did one student's score change across 4 tests?"""
    idx = STUDENT_NAMES.index(student_name)
    student_scores = trend[idx]
    tests = ["Test 1", "Test 2", "Test 3", "Test 4"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tests, student_scores, marker="o", markersize=8,
            color="#4c72b0", linewidth=2.5)
    for i, v in enumerate(student_scores):
        ax.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=9)

    ax.set_title(f"{student_name}'s Progress Across the Year", fontsize=14, fontweight="bold")
    ax.set_xlabel("Test")
    ax.set_ylabel("Average Score")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/05_line_student_progress.png", dpi=150)
    plt.close(fig)


def chart_report_dashboard(scores, trend):
    """
    MULTI-PANEL FIGURE (plt.subplots(2, 2)): combine 4 charts into one
    'report card' image -- the kind of one-page summary a teacher or
    school could actually print out.
    """
    subject_mean = scores.mean(axis=0)
    student_avg = scores.mean(axis=1)
    all_scores = scores.flatten()
    grades = np.where(student_avg >= 85, "A",
              np.where(student_avg >= 70, "B",
              np.where(student_avg >= 55, "C", "D")))
    unique, counts = np.unique(grades, return_counts=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top-left: subject averages (bar)
    axes[0, 0].bar(SUBJECTS, subject_mean, color="#4c72b0")
    axes[0, 0].set_title("Average Score per Subject")
    axes[0, 0].tick_params(axis="x", rotation=30)

    # Top-right: overall score distribution (histogram)
    axes[0, 1].hist(all_scores, bins=15, color="#8172b2", alpha=0.85)
    axes[0, 1].set_title("Overall Score Distribution")

    # Bottom-left: grade pie chart
    axes[1, 0].pie(counts, labels=unique, autopct="%1.0f%%", startangle=90)
    axes[1, 0].set_title("Grade Distribution")

    # Bottom-right: class average trend across all 4 tests (line)
    class_avg_per_test = trend.mean(axis=0)
    axes[1, 1].plot(["Test 1", "Test 2", "Test 3", "Test 4"],
                     class_avg_per_test, marker="o", color="#55a868", linewidth=2.5)
    axes[1, 1].set_title("Class Average Across Tests")
    axes[1, 1].set_ylim(0, 100)

    fig.suptitle("Student Performance -- Class Report Dashboard",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/06_report_dashboard.png", dpi=150)
    plt.close(fig)


def generate_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    scores, trend = load_data()

    chart_bar_subject_averages(scores)
    chart_histogram_score_distribution(scores)
    chart_scatter_math_vs_science(scores)
    chart_pie_grade_distribution(scores)
    chart_line_student_progress(trend, student_name="Isha")
    chart_report_dashboard(scores, trend)

    print(f"[viz] Saved 6 charts -> {OUT_DIR}/")


if __name__ == "__main__":
    print("=" * 65)
    print("STEP 4: VISUALIZATIONS")
    print("=" * 65)
    generate_all()
