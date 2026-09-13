# Student Exam Performance Analyzer
### A first real-world project for learning NumPy + Matplotlib

This project answers a question every student instantly understands:
**"Given my class's exam marks, who's doing well, where is the class
struggling, and how are scores changing over time?"**

It's built in 4 clear phases, each in its own file, so you can learn
one concept at a time instead of being overwhelmed by one giant script.

---

## Why this project (and not something more "impressive")?

Every method used here has an obvious reason to exist. You won't be
learning syntax in a vacuum — every line answers a real question a
teacher or student would actually ask. That's what makes concepts
stick.

---

## Project Structure

```
student_analyzer/
├── main.py                        # Runs all 4 steps in order
├── step1_data.py                  # PHASE 1: NumPy fundamentals
├── step2_statistics.py            # PHASE 2: NumPy statistics
├── step3_filtering.py             # PHASE 3: Boolean masking
├── step4_visualizations.py        # PHASE 4: Matplotlib charts
├── step5_bonus_student_report.py  # BONUS: interactive lookup
├── requirements.txt
├── README.md
├── data/
│   ├── subject_scores.npy         # (generated) raw NumPy array
│   ├── test_trend.npy             # (generated) 4-test progress array
│   └── subject_scores.csv         # (generated) human-readable version
└── output/
    └── charts/
        ├── 01_bar_subject_averages.png
        ├── 02_histogram_distribution.png
        ├── 03_scatter_math_vs_science.png
        ├── 04_pie_grade_distribution.png
        ├── 05_line_student_progress.png
        └── 06_report_dashboard.png
```

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

This runs everything in order and prints a full explanation to your
terminal, then saves 6 charts into `output/charts/`.

**Better yet — run each step ONE AT A TIME** so you can actually read
and understand what's happening before moving to the next:

```bash
python step1_data.py            # see the raw dataset get created
python step2_statistics.py      # see mean/std/percentile/argmax in action
python step3_filtering.py       # see boolean masks & np.where
python step4_visualizations.py  # generate all the charts
python step5_bonus_student_report.py   # type a name, get their report
```

---

## The Dataset

A synthetic class of **30 students × 5 subjects** (Math, Science,
English, History, Computer Science), generated with
`np.random.randint()` so it runs instantly with no downloads. There's
also a second array of **4 tests across the year** per student, used
to teach trend/line charts.

Feel free to replace the generator with **real marks from your own
class** — just build a NumPy array of the same shape and everything
downstream works unchanged.

---

## Phase-by-Phase Breakdown

### Phase 1 — `step1_data.py`: NumPy Fundamentals
| Method | What it does here |
|---|---|
| `np.array()` / `np.column_stack()` | Build the 2D marks grid |
| `np.random.seed()` | Same "random" data every run (reproducible) |
| `np.random.randint()` | Generate realistic marks |
| `.shape`, `.ndim`, `.size` | Inspect the array: 30×5, 2D, 150 values |
| `arr[i, j]`, `arr[:, j]`, `arr[i, :]` | Get a single value, a column, a row |
| `np.arange()` | Generate student ID numbers |

### Phase 2 — `step2_statistics.py`: Statistics
| Method | Real question it answers |
|---|---|
| `.mean(axis=0)` | "What's the class average in each subject?" |
| `.mean(axis=1)` | "What's each student's average across subjects?" |
| `.std(axis=0)` | "Which subject has the most inconsistent scores?" |
| `np.argmax()` / `np.argmin()` | "WHO is the top student? WHICH subject is weakest?" |
| `np.argsort()` | "Give me the full class ranking" |
| `np.percentile()` | "What score puts you in the top 25%?" |

> **Key concept: `axis`.** `axis=0` collapses down each *column*
> (one result per subject). `axis=1` collapses across each *row*
> (one result per student). This trips up almost every beginner —
> the code has comments explaining it inline.

### Phase 3 — `step3_filtering.py`: Boolean Logic
| Method | Real question it answers |
|---|---|
| `scores < 40` | Creates a True/False grid — "which scores are fails?" |
| `arr[mask]` | "Give me just the NAMES of students who failed" |
| `np.where(cond, a, b)` | "Label every single score as Pass/Fail, all at once" |
| `np.count_nonzero()` | "How many students got a distinction?" |
| chained `np.where()` | "Assign A/B/C/D grade bands based on average" |

This is usually where it "clicks" for students — instead of writing a
for-loop with an if-statement, one line of NumPy filters the whole
dataset.

### Phase 4 — `step4_visualizations.py`: Matplotlib
| Chart | Method | Real question it answers |
|---|---|---|
| Bar chart | `plt.bar()` | Which subject is the class strongest/weakest in? |
| Histogram | `plt.hist()` | How are all 150 scores spread out? |
| Scatter + trend line | `plt.scatter()` + `np.polyfit()` | Do Math and Science scores relate to each other? |
| Pie chart | `plt.pie()` | What % of the class got each grade? |
| Line chart | `plt.plot()` | How did one student progress across 4 tests? |
| 2×2 dashboard | `plt.subplots(2, 2)` | Combine 4 charts into one printable report |

Every chart uses `xlabel`, `ylabel`, `title`, and `grid` so it's
actually readable — not just a naked plot.

### Bonus — `step5_bonus_student_report.py`
Type in any student's name and get a personalized mini report: their
scores, average, grade, class rank, strongest/weakest subject, and
their own bar chart. This combines everything from Phases 1–4 into one
small feature — great practice once the basics feel comfortable.

---

## Sample Results (from a real run — yours will differ slightly)

- Class-wide average marks range from **59.2 (Science)** to
  **74.3 (Math)** — Science is both the *weakest* and *most
  inconsistent* subject (highest standard deviation).
- **12 out of 30 students** failed at least one subject.
- The top student (**Isha**) averaged **81.4%**, ranked #1 overall.
- Grade distribution: **53% B, 40% C, 7% D** — nobody reached the "A"
  band (85+) in this particular random dataset, which itself is a
  useful class-wide insight!

---

## Ideas to Extend This (once you're comfortable)

- Swap the random data generator for **your own real class's marks**
- Add a subject you're weak in and see if `np.argmin` correctly spots it
- Try `np.corrcoef()` to formally measure how strongly two subjects relate
- Add a 6th subject and update everything — see what stays the same
  and what needs to change (this teaches you what's *hard-coded* vs
  *general* in the code)
- Move on to the companion project, **Retail Sales & Customer
  Analytics**, which adds Pandas and SciPy on top of these same ideas
