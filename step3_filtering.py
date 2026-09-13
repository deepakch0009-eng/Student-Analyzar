"""
step3_filtering.py
--------------------
PHASE 3: BOOLEAN LOGIC & FILTERING -- Where NumPy Really Clicks
===================================================================

This is usually the "aha!" moment for beginners: instead of writing a
for-loop with an if-statement, NumPy lets you filter an entire array
in ONE line using a boolean condition.

NumPy methods covered in this file:
    arr > value              -> boolean mask (array of True/False)
    arr[mask]                -> fancy indexing: keep only True positions
    np.where(cond, a, b)     -> conditional replacement (like an if/else
                                 applied to every element at once)
    np.count_nonzero()       -> count how many True values in a mask

Real-life problem:
    "Give me the list of students who failed", "How many students got
    a distinction in Science?", "Label every score as Pass/Fail" --
    these are exactly the kind of questions a teacher asks, and boolean
    masking answers them instantly without writing a single loop.
"""

import numpy as np
from step1_data import SUBJECTS, STUDENT_NAMES


PASS_MARK = 40
DISTINCTION_MARK = 90


def load_data():
    return np.load("data/subject_scores.npy")


def find_failures(scores):
    print("\n--- BOOLEAN MASK: Students who failed (any subject < 40) ---")
    # (scores < PASS_MARK) is a boolean mask of shape (30, 5): True
    # wherever a score is below the pass mark.
    fail_mask = scores < PASS_MARK

    # .any(axis=1) collapses each row: True if ANY subject in that
    # row failed. This gives us one True/False PER STUDENT.
    failed_any_subject = fail_mask.any(axis=1)

    # Fancy indexing: pass the boolean array directly as an index to
    # pull out only the names where the mask is True.
    failed_names = np.array(STUDENT_NAMES)[failed_any_subject]

    print(f"Number of students who failed at least one subject: "
          f"{np.count_nonzero(failed_any_subject)}")
    print("Names:", failed_names.tolist())

    return fail_mask, failed_any_subject


def find_distinctions(scores):
    print("\n--- BOOLEAN MASK: Distinctions (score >= 90) per subject ---")
    distinction_mask = scores >= DISTINCTION_MARK

    for i, subject in enumerate(SUBJECTS):
        count = np.count_nonzero(distinction_mask[:, i])
        print(f"{subject:20s} -> {count} student(s) scored 90+")

    return distinction_mask


def pass_fail_labels(scores):
    print("\n--- np.where(): Labeling every score as Pass/Fail ---")
    # np.where(condition, value_if_true, value_if_false)
    # Applied element-wise across the WHOLE 2D array at once.
    labels = np.where(scores >= PASS_MARK, "Pass", "Fail")

    # Show it for the first 3 students as a demonstration
    for i in range(3):
        row = dict(zip(SUBJECTS, labels[i].tolist()))
        print(f"{STUDENT_NAMES[i]:10s}: {row}")

    return labels


def grade_bands(student_avg):
    """
    Convert average scores into letter grades using nested np.where().
    This demonstrates chaining conditions together.
    """
    print("\n--- np.where() chained: Assigning Grade Bands ---")
    grades = np.where(student_avg >= 85, "A",
              np.where(student_avg >= 70, "B",
              np.where(student_avg >= 55, "C", "D")))

    unique, counts = np.unique(grades, return_counts=True)
    print("Grade distribution:", dict(zip(unique.tolist(), counts.tolist())))

    return grades


def run_all():
    print("=" * 65)
    print("STEP 3: BOOLEAN FILTERING")
    print("=" * 65)

    scores = load_data()
    fail_mask, failed_any_subject = find_failures(scores)
    distinction_mask = find_distinctions(scores)
    labels = pass_fail_labels(scores)

    student_avg = scores.mean(axis=1)
    grades = grade_bands(student_avg)

    return {
        "fail_mask": fail_mask,
        "failed_any_subject": failed_any_subject,
        "distinction_mask": distinction_mask,
        "labels": labels,
        "grades": grades,
    }


if __name__ == "__main__":
    run_all()
