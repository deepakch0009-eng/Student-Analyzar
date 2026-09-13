"""
step2_statistics.py
---------------------
PHASE 2: NUMPY STATISTICS -- The Heart of the Project
========================================================

Now that we have the data, we ask real questions a teacher would ask:
    - What's the class average in each subject?
    - Which subject has the most inconsistent (spread out) scores?
    - Who is the top-performing student overall?
    - Which subject is the class weakest in?
    - What score do you need to be in the "top 25%"?

NumPy methods covered in this file:
    np.mean()                axis=0 vs axis=1  -> averages
    np.median()                                -> middle value
    np.std() / np.var()                        -> spread/consistency
    np.min() / np.max()                        -> extremes
    np.argmin() / np.argmax()                  -> POSITION of extremes
    np.sort() / np.argsort()                   -> ranking
    np.percentile()                            -> distribution cut-offs
    np.sum()                 axis=0 vs axis=1  -> totals

KEY CONCEPT -- axis:
    axis=0 -> operate DOWN each column (i.e. across all students,
              for one subject at a time)  -> gives 5 results (one per subject)
    axis=1 -> operate ACROSS each row (i.e. across all subjects,
              for one student at a time)  -> gives 30 results (one per student)
"""

import numpy as np
from step1_data import SUBJECTS, STUDENT_NAMES


def load_data():
    scores = np.load("data/subject_scores.npy")
    trend = np.load("data/test_trend.npy")
    return scores, trend


def subject_level_stats(scores):
    print("\n--- SUBJECT-LEVEL STATS (axis=0 -> down each column) ---")
    subject_mean = scores.mean(axis=0)     # average per subject
    subject_median = np.median(scores, axis=0)
    subject_std = scores.std(axis=0)       # spread per subject
    subject_max = scores.max(axis=0)
    subject_min = scores.min(axis=0)

    for i, subject in enumerate(SUBJECTS):
        print(f"{subject:20s} mean={subject_mean[i]:5.1f}  "
              f"median={subject_median[i]:5.1f}  std={subject_std[i]:5.1f}  "
              f"min={subject_min[i]:3d}  max={subject_max[i]:3d}")

    # np.argmax()/np.argmin() give the POSITION (index), not the value.
    # Here we use them on the std array to find which SUBJECT (not
    # student) has the most/least consistent scores.
    most_inconsistent = SUBJECTS[np.argmax(subject_std)]
    most_consistent = SUBJECTS[np.argmin(subject_std)]
    weakest_subject = SUBJECTS[np.argmin(subject_mean)]
    strongest_subject = SUBJECTS[np.argmax(subject_mean)]

    print(f"\nMost inconsistent subject (highest std): {most_inconsistent}")
    print(f"Most consistent subject   (lowest std):  {most_consistent}")
    print(f"Class's weakest subject   (lowest mean): {weakest_subject}")
    print(f"Class's strongest subject (highest mean): {strongest_subject}")

    return subject_mean, subject_std


def student_level_stats(scores):
    print("\n--- STUDENT-LEVEL STATS (axis=1 -> across each row) ---")
    student_total = scores.sum(axis=1)     # total marks per student
    student_avg = scores.mean(axis=1)      # average marks per student

    # argmax on student_total gives the ROW INDEX of the top student
    top_idx = np.argmax(student_total)
    bottom_idx = np.argmin(student_total)

    print(f"Top student:    {STUDENT_NAMES[top_idx]:10s} "
          f"total={student_total[top_idx]:.0f}  avg={student_avg[top_idx]:.1f}")
    print(f"Lowest student: {STUDENT_NAMES[bottom_idx]:10s} "
          f"total={student_total[bottom_idx]:.0f}  avg={student_avg[bottom_idx]:.1f}")

    return student_total, student_avg


def leaderboard(student_avg):
    print("\n--- CLASS LEADERBOARD (np.argsort) ---")
    # np.argsort() returns the INDICES that would sort the array.
    # [::-1] reverses it so the highest average comes first.
    ranking = np.argsort(student_avg)[::-1]

    print(f"{'Rank':5s}{'Name':12s}{'Average':>8s}")
    for rank, idx in enumerate(ranking[:10], start=1):
        print(f"{rank:<5d}{STUDENT_NAMES[idx]:12s}{student_avg[idx]:8.1f}")
    print("...(showing top 10 of 30)")

    return ranking


def percentile_cutoffs(student_avg):
    print("\n--- PERCENTILE CUT-OFFS (np.percentile) ---")
    p25 = np.percentile(student_avg, 25)
    p50 = np.percentile(student_avg, 50)  # same as median
    p75 = np.percentile(student_avg, 75)
    p90 = np.percentile(student_avg, 90)

    print(f"25th percentile: {p25:.1f}  (bottom quarter scores below this)")
    print(f"50th percentile: {p50:.1f}  (the median)")
    print(f"75th percentile: {p75:.1f}  (top quarter scores above this)")
    print(f"90th percentile: {p90:.1f}  (top 10% scores above this)")

    return p25, p50, p75, p90


def run_all():
    print("=" * 65)
    print("STEP 2: STATISTICAL ANALYSIS")
    print("=" * 65)

    scores, trend = load_data()
    subject_mean, subject_std = subject_level_stats(scores)
    student_total, student_avg = student_level_stats(scores)
    ranking = leaderboard(student_avg)
    percentiles = percentile_cutoffs(student_avg)

    return {
        "scores": scores,
        "trend": trend,
        "subject_mean": subject_mean,
        "subject_std": subject_std,
        "student_total": student_total,
        "student_avg": student_avg,
        "ranking": ranking,
        "percentiles": percentiles,
    }


if __name__ == "__main__":
    run_all()
