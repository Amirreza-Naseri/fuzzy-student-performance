# Fuzzy Student Performance Evaluation

A from-scratch **Mamdani fuzzy inference system** for analysing student performance from GPA and learning-outcome assessment scores. The project is an independent Python reproduction study based on a 2021 higher-education fuzzy-logic paper, with explicit validation, automated tests, and visual analysis of the rule system.

> **Portfolio focus:** fuzzy logic, rule-based AI, interpretable systems, research reproduction, numerical validation, testing, and visualization.

## Project overview

Conventional averages can hide differences between individual skills. This project explores a rule-based fuzzy alternative in which crisp student scores are converted to linguistic memberships (for example *Poor*, *Average*, *Good*), combined through expert rules, and defuzzified into interpretable performance values.

The implementation reproduces the paper-defined:

- GPA input membership functions (Table 2)
- examination/learning-outcome membership functions (Table 3)
- GPA -> Academic Value rule base (Table 4)
- Knowledge + Problem Solving -> Cognitive Value rule base (Table 5)
- Academic Value + Cognitive Value -> Academic Performance rule base (Table 6)
- 20-student validation samples from Tables 7 and 8

The Mamdani engine itself is implemented directly in Python/NumPy rather than relying on a fuzzy-logic package.

## Fuzzy inference pipeline

```text
Semester 1 GPA + Semester 2 GPA
                |
                v
         Academic Value
                |
                +--------------------+
                                     |
Knowledge + Problem Solving          |
                |                    |
                v                    |
         Cognitive Value ------------+
                |
                v
       Academic Performance
```

For each two-input fuzzy inference system the implementation uses:

1. trapezoidal fuzzification,
2. `min` for rule AND,
3. `max` for output aggregation,
4. centroid defuzzification.

## Results

Validation uses the 20-student values reported in the paper.

| Stage | MAE | RMSE |
|---|---:|---:|
| Academic Value | 0.0860 | 0.0910 |
| Cognitive Value | 0.0799 | 0.0849 |
| Academic Performance | **0.0318** | **0.0381** |
| Overall Performance* | 0.7728 | 0.8291 |

\*The final Overall Performance stage is shown as a **reconstructed assumption**, not an exact reproduction claim. The article provides its output membership functions graphically and does not tabulate a separate final-stage rule matrix numerically. The original project reused the Table 6 rule structure; this repository preserves that assumption explicitly so that the limitation is auditable.

This distinction is intentional: the project demonstrates not only implementation, but also how to identify where a research reproduction is well-supported by published parameters and where it depends on reconstruction choices.

## Visualizations

### Academic fuzzy surface

![Academic fuzzy surface](results/images/academic_surface.png)

### Paper values vs reproduced values

![Reference versus reproduction](results/images/table7_reference_vs_reproduction.png)

### GPA membership functions

![GPA membership functions](results/images/membership_gpa.png)

### Examination membership functions

![Exam membership functions](results/images/membership_exam.png)

### Rule-base visualization

![Academic rule heatmap](results/images/academic_rule_heatmap.png)

Additional figures, including cognitive rules and the reconstructed overall stage, are available in `results/images/`.

## Important implementation fix

The original prototype treated the endpoints of shoulder membership functions incorrectly. For example, the paper defines the *Excellent* GPA membership as `(3.0, 3.5, 4.0, 4.0)`. A GPA of exactly `4.0` must therefore have full membership in the right shoulder.

The corrected implementation handles ordinary trapezoids, triangular cases, left shoulders (`a == b`), and right shoulders (`c == d`) explicitly. Regression tests cover both boundary cases.

## Repository structure

```text
fuzzy-student-performance/
├── src/
│   ├── fuzzy_system.py
│   └── run_evaluation.py
├── tests/
│   └── test_fuzzy_system.py
├── data/
│   ├── table7_students.csv
│   ├── table8_students.csv
│   └── README.md
├── results/
│   ├── metrics.csv
│   ├── student_predictions.csv
│   └── images/
├── .github/workflows/tests.yml
├── CITATION.md
├── GITHUB_SETUP.md
├── requirements.txt
└── LICENSE
```

## Run locally

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/run_evaluation.py
pytest -q
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python src/run_evaluation.py
pytest -q
```

The evaluation script regenerates the numerical result CSV files and all visualizations.

## Tests

The test suite currently covers:

- left-shoulder membership boundaries,
- right-shoulder membership boundaries,
- zero membership outside support,
- paper rule-table sizes,
- clipping to the variable universe,
- output range checks,
- deterministic inference.

Current status:

```text
7 passed
```

## Research reference

Tengku Zatul Hidayah Tengku Petra and Mohd Juzaiddin Ab Aziz (2021), **Analysing Student Performance In Higher Education Using Fuzzy Logic Evaluation**, *International Journal of Scientific & Technology Research*, 10(01), 322-327.

See [`CITATION.md`](CITATION.md) for the article link and reproduction notes.

## Limitations

This is a small research-reproduction project, not a validated educational decision system. The sample contains only 20 students, the original study used MATLAB, and some output membership-function parameters are available only through published figures rather than tabulated numeric values. The repository therefore separates directly paper-supported parameters from reconstruction assumptions instead of presenting all outputs as exact reproductions.

## Skills demonstrated

`Python` · `NumPy` · `Matplotlib` · `Fuzzy Logic` · `Mamdani Inference` · `Expert Systems` · `Research Reproduction` · `Testing` · `Data Visualization`
