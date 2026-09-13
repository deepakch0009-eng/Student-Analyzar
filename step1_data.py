"""
step1_data.py
--------------
PHASE 1: NUMPY FUNDAMENTALS -- Creating & Exploring Data
==========================================================

This is the very first step of the project. Before we can analyze
anything, we need data. Here we learn the NumPy methods used to CREATE
and INSPECT arrays.

NumPy methods covered in this file:
    np.array()              -> turn a list of lists into a 2D array
    np.random.seed()        -> make random data reproducible
    np.random.randint()     -> generate random test marks
    .shape / .ndim / .size  -> inspect array dimensions
    arr[row, col]           -> basic indexing
    arr[:, col]             -> column slicing
    arr[row, :]             -> row slicing
    np.arange()             -> generate a sequence (student IDs)

Real-life problem:
    A teacher has 30 students and 5 subjects. Instead of a messy
    spreadsheet, we store everything as ONE NumPy array so we can do
    fast, vectorized math on it later.
"""

import numpy as np

# ---------------------------------------------------------------------
# 1. np.random.seed() -- makes "random" numbers reproducible.
#    Without this, you'd get different marks every time you run the
#    script, which makes it hard to learn from / debug.
# ---------------------------------------------------------------------
np.random.seed(42)

SUBJECTS = ["Math", "Science", "English", "History", "Computer Science"]
NUM_STUDENTS = 30

STUDENT_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan",
    "Krishna", "Ishaan", "Ananya", "Diya", "Saanvi", "Aadhya", "Kiara",
    "Myra", "Anika", "Navya", "Riya", "Aarohi", "Kabir", "Dhruv", "Rohan",
    "Yash", "Advait", "Priya", "Sneha", "Meera", "Tanya", "Isha",
]


def generate_subject_scores():
    """
    Generate a 2D array of shape (30 students, 5 subjects).

    np.random.randint(low, high, size) generates random integers.
    We give Science and Math slightly wider ranges to make the data
    feel more realistic (some subjects naturally have more spread).
    """
    scores = np.column_stack([
        np.random.randint(35, 100, NUM_STUDENTS),   # Math
        np.random.randint(30, 100, NUM_STUDENTS),   # Science
        np.random.randint(45, 98, NUM_STUDENTS),    # English
        np.random.randint(40, 95, NUM_STUDENTS),    # History
        np.random.randint(50, 100, NUM_STUDENTS),   # Computer Science
    ])
    return scores


def generate_test_trend(base_scores):
    """
    Generate a (30 students, 4 tests) array showing how EACH student's
    overall performance changed across 4 tests during the year.

    We start near their average subject score and add small random
    "improvement" or "dip" noise per test -- this is what lets us draw
    a realistic progress line chart later (Phase 4).
    """
    student_avg = base_scores.mean(axis=1)  # average score per student
    trend = np.zeros((NUM_STUDENTS, 4))
    for test_num in range(4):
        # Students generally improve slightly over the year (+2 per test)
        # plus some random noise (+/- 8 marks)
        noise = np.random.randint(-8, 9, NUM_STUDENTS)
        trend[:, test_num] = np.clip(student_avg + test_num * 2 + noise, 0, 100)
    return trend.round(1)


def explore_array(scores):
    """Demonstrates the basic array-inspection methods every NumPy
    beginner needs: .shape, .ndim, .size, and indexing/slicing."""
    print("Array shape (rows, cols):", scores.shape)   # (30, 5)
    print("Array dimensions (ndim):", scores.ndim)      # 2
    print("Total number of values (size):", scores.size)  # 150

    print("\nFirst student's row  scores[0, :] ->", scores[0, :])
    print("Math column          scores[:, 0]  ->", scores[:5, 0], "... (first 5 shown)")
    print("Single value         scores[0, 0]  ->", scores[0, 0])

    # np.arange() generates a sequence -- perfect for student IDs
    student_ids = np.arange(1, NUM_STUDENTS + 1)
    print("\nStudent IDs (np.arange):", student_ids[:10], "...")


if __name__ == "__main__":
    print("=" * 65)
    print("STEP 1: GENERATING THE STUDENT MARKS DATASET")
    print("=" * 65)

    scores = generate_subject_scores()
    trend = generate_test_trend(scores)

    explore_array(scores)

    # Save both arrays to disk so later steps can reuse them without
    # regenerating (np.save keeps them as efficient binary .npy files)
    import os
    os.makedirs("data", exist_ok=True)
    np.save("data/subject_scores.npy", scores)
    np.save("data/test_trend.npy", trend)

    # Also save a human-readable CSV version for reference
    header = "Name," + ",".join(SUBJECTS)
    with open("data/subject_scores.csv", "w") as f:
        f.write(header + "\n")
        for name, row in zip(STUDENT_NAMES, scores):
            f.write(name + "," + ",".join(map(str, row)) + "\n")

    print(f"\nSaved: data/subject_scores.npy  (shape {scores.shape})")
    print(f"Saved: data/test_trend.npy      (shape {trend.shape})")
    print("Saved: data/subject_scores.csv  (human-readable)")
