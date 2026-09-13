"""
main.py
--------
Runs the ENTIRE Student Exam Performance Analyzer, step by step,
exactly in the order a beginner should learn it:

    STEP 1: NumPy fundamentals   -> create & explore the data
    STEP 2: NumPy statistics     -> mean, std, argmax, percentile...
    STEP 3: Boolean filtering    -> masks, np.where, count_nonzero
    STEP 4: Matplotlib charts    -> bar, hist, scatter, pie, line, dashboard

Run this file to see the whole project work end-to-end:
    python main.py

Or run each step file individually to study it in isolation:
    python step1_data.py
    python step2_statistics.py
    python step3_filtering.py
    python step4_visualizations.py
"""

import step1_data
import step2_statistics
import step3_filtering
import step4_visualizations
import numpy as np
import os


def banner(text):
    print("\n" + "#" * 65)
    print("#  " + text)
    print("#" * 65)


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("output/charts", exist_ok=True)

    banner("STEP 1: CREATE & EXPLORE THE DATASET (NumPy fundamentals)")
    scores = step1_data.generate_subject_scores()
    trend = step1_data.generate_test_trend(scores)
    step1_data.explore_array(scores)
    np.save("data/subject_scores.npy", scores)
    np.save("data/test_trend.npy", trend)

    # Also save a human-readable CSV version for reference
    header = "Name," + ",".join(step1_data.SUBJECTS)
    with open("data/subject_scores.csv", "w") as f:
        f.write(header + "\n")
        for name, row in zip(step1_data.STUDENT_NAMES, scores):
            f.write(name + "," + ",".join(map(str, row)) + "\n")

    banner("STEP 2: STATISTICAL ANALYSIS (NumPy stats methods)")
    step2_statistics.run_all()

    banner("STEP 3: BOOLEAN FILTERING (masks, np.where, count_nonzero)")
    step3_filtering.run_all()

    banner("STEP 4: VISUALIZATIONS (Matplotlib chart types)")
    step4_visualizations.generate_all()

    banner("PROJECT COMPLETE")
    print("Check the output/charts/ folder for all 6 generated charts.")
    print("Check data/subject_scores.csv to see the raw dataset.")


if __name__ == "__main__":
    main()
