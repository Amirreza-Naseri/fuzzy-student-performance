"""Mamdani fuzzy inference system for student-performance evaluation.

Input membership functions and rule matrices follow Tengku Petra & Ab Aziz (2021).
Output membership-function breakpoints are reconstructed from the original project
implementation because the paper presents those output MFs graphically rather than
as machine-readable tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

MF = Tuple[float, float, float, float]
Rule = Tuple[str, str, str]


def trapmf(x: float | np.ndarray, a: float, b: float, c: float, d: float):
    """Trapezoidal membership function with correct left/right shoulder handling.

    Supports ordinary trapezoids, triangles (b == c), left shoulders (a == b),
    and right shoulders (c == d).
    """
    if not (a <= b <= c <= d):
        raise ValueError("Expected a <= b <= c <= d")

    xs = np.asarray(x, dtype=float)
    y = np.zeros_like(xs, dtype=float)

    # Plateau first; this makes x=a when a=b and x=d when c=d evaluate to 1.
    plateau = (xs >= b) & (xs <= c)
    y[plateau] = 1.0

    if b > a:
        rising = (xs > a) & (xs < b)
        y[rising] = (xs[rising] - a) / (b - a)

    if d > c:
        falling = (xs > c) & (xs < d)
        y[falling] = (d - xs[falling]) / (d - c)

    if np.isscalar(x):
        return float(y)
    return y


@dataclass(frozen=True)
class FuzzyVariable:
    name: str
    universe: Tuple[float, float]
    mfs: Mapping[str, MF]

    def fuzzify(self, x: float) -> Dict[str, float]:
        lo, hi = self.universe
        clipped = float(np.clip(x, lo, hi))
        return {term: trapmf(clipped, *params) for term, params in self.mfs.items()}


def rules_from_matrix(
    in1_terms: Sequence[str], in2_terms: Sequence[str], out_matrix: Sequence[Sequence[str]]
) -> List[Rule]:
    if len(out_matrix) != len(in1_terms) or any(len(row) != len(in2_terms) for row in out_matrix):
        raise ValueError("Rule-matrix dimensions do not match input terms")
    return [
        (in1_terms[i], in2_terms[j], out_matrix[i][j])
        for i in range(len(in1_terms))
        for j in range(len(in2_terms))
    ]


class MamdaniFIS2:
    """Two-input Mamdani FIS using min-AND, max aggregation, centroid defuzzification."""

    def __init__(
        self,
        in1: FuzzyVariable,
        in2: FuzzyVariable,
        out: FuzzyVariable,
        rules: Sequence[Rule],
        steps: int = 1001,
    ) -> None:
        self.in1 = in1
        self.in2 = in2
        self.out = out
        self.rules = list(rules)
        self.steps = max(int(steps), 101)

    def infer(self, x1: float, x2: float) -> float:
        mu1 = self.in1.fuzzify(x1)
        mu2 = self.in2.fuzzify(x2)
        activation = {term: 0.0 for term in self.out.mfs}

        for t1, t2, tout in self.rules:
            strength = min(mu1.get(t1, 0.0), mu2.get(t2, 0.0))
            activation[tout] = max(activation[tout], strength)

        lo, hi = self.out.universe
        xs = np.linspace(lo, hi, self.steps)
        aggregated = np.zeros_like(xs)
        for term, params in self.out.mfs.items():
            clipped = np.minimum(activation[term], trapmf(xs, *params))
            aggregated = np.maximum(aggregated, clipped)

        area = float(np.sum(aggregated))
        if area <= 1e-12:
            return (lo + hi) / 2.0
        return float(np.sum(xs * aggregated) / area)


# Paper Table 2: GPA input membership functions.
GPA = FuzzyVariable(
    "GPA",
    (0.0, 4.0),
    {
        "Poor": (0.0, 0.0, 1.5, 2.0),
        "Average": (1.0, 1.5, 2.5, 3.0),
        "Good": (2.0, 2.5, 3.17, 3.67),
        "Excellent": (3.0, 3.5, 4.0, 4.0),
    },
)

# Paper Table 3: learning-outcome examination input membership functions.
EXAM = FuzzyVariable(
    "Examination score",
    (0.0, 100.0),
    {
        "Very Poor": (0.0, 0.0, 30.0, 40.0),
        "Poor": (20.0, 30.0, 45.0, 55.0),
        "Average": (40.0, 50.0, 65.0, 75.0),
        "Good": (55.0, 65.0, 75.0, 85.0),
        "Very Good": (75.0, 85.0, 100.0, 100.0),
    },
)

# Output MF breakpoints reconstructed from the original implementation.
OUT1 = FuzzyVariable(
    "Output 1",
    (0.0, 1.0),
    {
        "Very Low": (0.0, 0.0, 0.2, 0.3),
        "Low": (0.3, 0.4, 0.4, 0.5),
        "Average": (0.6, 0.67, 0.67, 0.75),
        "High": (0.75, 0.85, 0.85, 0.95),
        "Very High": (0.85, 0.95, 1.0, 1.0),
    },
)

OUT2 = FuzzyVariable(
    "Output 2",
    (0.0, 1.0),
    {
        "Very Low": (0.0, 0.0, 0.3, 0.4),
        "Low": (0.3, 0.4, 0.4, 0.5),
        "Average": (0.4, 0.5, 0.5, 0.6),
        "High": (0.6, 0.7, 0.7, 0.8),
        "Very High": (0.8, 0.9, 1.0, 1.0),
    },
)

OVERALL = FuzzyVariable(
    "Overall performance",
    (0.0, 5.0),
    {
        "Very Low": (1.5, 2.0, 2.0, 2.5),
        "Low": (2.0, 2.5, 2.5, 3.0),
        "Average": (2.5, 3.0, 3.0, 3.5),
        "High": (3.0, 3.5, 3.5, 4.0),
        "Very High": (3.5, 4.0, 5.0, 5.0),
    },
)

GPA_TERMS = ["Poor", "Average", "Good", "Excellent"]
EXAM_TERMS = ["Very Poor", "Poor", "Average", "Good", "Very Good"]
LEVEL5 = ["Very Low", "Low", "Average", "High", "Very High"]

# Paper Table 4.
ACADEMIC_RULE_MATRIX = [
    ["Very Low", "Average", "High", "High"],
    ["Low", "Average", "High", "High"],
    ["Average", "High", "High", "Very High"],
    ["Average", "High", "High", "Very High"],
]

# Paper Table 5.
COGNITIVE_RULE_MATRIX = [
    ["Very Low", "Low", "Low", "Average", "High"],
    ["Very Low", "Low", "Average", "Average", "High"],
    ["Low", "Low", "Average", "High", "High"],
    ["Average", "Average", "Average", "High", "Very High"],
    ["Average", "Average", "High", "High", "Very High"],
]

# Paper Table 6: Academic Value x Cognitive Value -> Academic Performance.
PERFORMANCE_RULE_MATRIX = [
    ["Very Low", "Very Low", "Low", "Average", "High"],
    ["Very Low", "Low", "Average", "Average", "High"],
    ["Low", "Average", "Average", "High", "High"],
    ["Average", "Average", "High", "High", "Very High"],
    ["Average", "Average", "High", "Very High", "Very High"],
]

ACADEMIC_RULES = rules_from_matrix(GPA_TERMS, GPA_TERMS, ACADEMIC_RULE_MATRIX)
COGNITIVE_RULES = rules_from_matrix(EXAM_TERMS, EXAM_TERMS, COGNITIVE_RULE_MATRIX)
PERFORMANCE_RULES = rules_from_matrix(LEVEL5, LEVEL5, PERFORMANCE_RULE_MATRIX)

FIS_ACADEMIC = MamdaniFIS2(GPA, GPA, OUT1, ACADEMIC_RULES)
FIS_COGNITIVE = MamdaniFIS2(EXAM, EXAM, OUT1, COGNITIVE_RULES)
FIS_ACADEMIC_PERFORMANCE = MamdaniFIS2(OUT1, OUT1, OUT2, PERFORMANCE_RULES)

# The paper does not tabulate a separate final-output rule matrix. The original
# project reused Table 6 for Academic Performance x Personality Development.
# We preserve that assumption explicitly rather than presenting it as paper fact.
FIS_OVERALL_RECONSTRUCTION = MamdaniFIS2(OUT2, OUT2, OVERALL, PERFORMANCE_RULES)
