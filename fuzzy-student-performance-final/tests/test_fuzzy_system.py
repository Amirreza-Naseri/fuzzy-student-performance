import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fuzzy_system import (  # noqa: E402
    ACADEMIC_RULES,
    COGNITIVE_RULES,
    FIS_ACADEMIC,
    FIS_COGNITIVE,
    GPA,
    trapmf,
)


def test_left_shoulder_boundary():
    assert trapmf(0.0, 0.0, 0.0, 1.5, 2.0) == 1.0


def test_right_shoulder_boundary():
    assert trapmf(4.0, 3.0, 3.5, 4.0, 4.0) == 1.0


def test_outside_membership_is_zero():
    assert trapmf(-0.1, 0.0, 0.0, 1.5, 2.0) == 0.0
    assert trapmf(2.1, 0.0, 0.0, 1.5, 2.0) == 0.0


def test_rule_counts_match_paper_tables():
    assert len(ACADEMIC_RULES) == 16
    assert len(COGNITIVE_RULES) == 25


def test_fuzzify_clips_to_universe():
    assert GPA.fuzzify(-5.0) == GPA.fuzzify(0.0)
    assert GPA.fuzzify(9.0) == GPA.fuzzify(4.0)


def test_inference_ranges():
    assert 0.0 <= FIS_ACADEMIC.infer(3.58, 3.48) <= 1.0
    assert 0.0 <= FIS_COGNITIVE.infer(74, 84) <= 1.0


def test_deterministic_inference():
    first = FIS_ACADEMIC.infer(3.58, 3.48)
    second = FIS_ACADEMIC.infer(3.58, 3.48)
    assert np.isclose(first, second)
