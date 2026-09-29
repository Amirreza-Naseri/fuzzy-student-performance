from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from fuzzy_system import (
    ACADEMIC_RULE_MATRIX,
    COGNITIVE_RULE_MATRIX,
    EXAM,
    FIS_ACADEMIC,
    FIS_ACADEMIC_PERFORMANCE,
    FIS_COGNITIVE,
    FIS_OVERALL_RECONSTRUCTION,
    GPA,
    OVERALL,
    OUT1,
    OUT2,
    trapmf,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
IMAGES = RESULTS / "images"


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mae(actual, predicted):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(a - p)))


def rmse(actual, predicted):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def plot_memberships(variable, filename: str):
    lo, hi = variable.universe
    xs = np.linspace(lo, hi, 800)
    plt.figure(figsize=(8, 4.5))
    for term, params in variable.mfs.items():
        plt.plot(xs, trapmf(xs, *params), label=term)
    plt.xlabel(variable.name)
    plt.ylabel("Membership degree")
    plt.ylim(-0.02, 1.05)
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(IMAGES / filename, dpi=180)
    plt.close()


def plot_rule_matrix(matrix, row_labels, col_labels, filename, title):
    levels = {"Very Low": 0, "Low": 1, "Average": 2, "High": 3, "Very High": 4}
    arr = np.array([[levels[v] for v in row] for row in matrix], dtype=float)
    plt.figure(figsize=(7, 5))
    plt.imshow(arr, aspect="auto", vmin=0, vmax=4)
    plt.xticks(range(len(col_labels)), col_labels, rotation=35, ha="right")
    plt.yticks(range(len(row_labels)), row_labels)
    plt.xlabel("Input 2")
    plt.ylabel("Input 1")
    plt.title(title)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            plt.text(j, i, value.replace("Very ", "V. "), ha="center", va="center", fontsize=8)
    plt.colorbar(ticks=range(5), label="Output level (Very Low -> Very High)")
    plt.tight_layout()
    plt.savefig(IMAGES / filename, dpi=180)
    plt.close()


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    table7 = read_csv(DATA / "table7_students.csv")
    table8 = read_csv(DATA / "table8_students.csv")
    t8 = {int(r["student"]): r for r in table8}

    rows = []
    for r in table7:
        sid = int(r["student"])
        sem1 = float(r["sem1_gpa"])
        sem2 = float(r["sem2_gpa"])
        knowledge = float(r["knowledge"])
        problem_solving = float(r["problem_solving"])
        acad_ref = float(r["academic_value_reference"])
        cog_ref = float(r["cognitive_value_reference"])

        acad = FIS_ACADEMIC.infer(sem1, sem2)
        cog = FIS_COGNITIVE.infer(knowledge, problem_solving)
        academic_performance = FIS_ACADEMIC_PERFORMANCE.infer(acad, cog)

        r8 = t8[sid]
        ap_ref = float(r8["academic_performance"])
        pd_ref = float(r8["personality_development"])
        overall_paper = float(r8["overall_fuzzy_reference"])
        overall_reconstructed = FIS_OVERALL_RECONSTRUCTION.infer(ap_ref, pd_ref)

        rows.append({
            "student": sid,
            "academic_value_reference": acad_ref,
            "academic_value_reproduced": acad,
            "cognitive_value_reference": cog_ref,
            "cognitive_value_reproduced": cog,
            "academic_performance_reconstructed": academic_performance,
            "academic_performance_paper": ap_ref,
            "personality_development_paper": pd_ref,
            "overall_fuzzy_reference": overall_paper,
            "overall_reconstructed": overall_reconstructed,
        })

    out_csv = RESULTS / "student_predictions.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = [
        ("Academic Value", mae([r["academic_value_reference"] for r in rows], [r["academic_value_reproduced"] for r in rows]), rmse([r["academic_value_reference"] for r in rows], [r["academic_value_reproduced"] for r in rows])),
        ("Cognitive Value", mae([r["cognitive_value_reference"] for r in rows], [r["cognitive_value_reproduced"] for r in rows]), rmse([r["cognitive_value_reference"] for r in rows], [r["cognitive_value_reproduced"] for r in rows])),
        ("Academic Performance", mae([r["academic_performance_paper"] for r in rows], [r["academic_performance_reconstructed"] for r in rows]), rmse([r["academic_performance_paper"] for r in rows], [r["academic_performance_reconstructed"] for r in rows])),
        ("Overall Performance (reconstructed assumption)", mae([r["overall_fuzzy_reference"] for r in rows], [r["overall_reconstructed"] for r in rows]), rmse([r["overall_fuzzy_reference"] for r in rows], [r["overall_reconstructed"] for r in rows])),
    ]
    with (RESULTS / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "mae", "rmse"])
        writer.writerows(metrics)

    plot_memberships(GPA, "membership_gpa.png")
    plot_memberships(EXAM, "membership_exam.png")
    plot_memberships(OUT1, "membership_output1_reconstructed.png")
    plot_memberships(OUT2, "membership_output2_reconstructed.png")
    plot_memberships(OVERALL, "membership_overall_reconstructed.png")

    plot_rule_matrix(
        ACADEMIC_RULE_MATRIX,
        ["Poor", "Average", "Good", "Excellent"],
        ["Poor", "Average", "Good", "Excellent"],
        "academic_rule_heatmap.png",
        "Academic-value rule base (Paper Table 4)",
    )
    plot_rule_matrix(
        COGNITIVE_RULE_MATRIX,
        ["Very Poor", "Poor", "Average", "Good", "Very Good"],
        ["Very Poor", "Poor", "Average", "Good", "Very Good"],
        "cognitive_rule_heatmap.png",
        "Cognitive-value rule base (Paper Table 5)",
    )

    students = [r["student"] for r in rows]
    plt.figure(figsize=(9, 4.8))
    plt.plot(students, [r["academic_value_reference"] for r in rows], marker="o", label="Academic - paper")
    plt.plot(students, [r["academic_value_reproduced"] for r in rows], marker="o", label="Academic - reproduced")
    plt.plot(students, [r["cognitive_value_reference"] for r in rows], marker="s", label="Cognitive - paper")
    plt.plot(students, [r["cognitive_value_reproduced"] for r in rows], marker="s", label="Cognitive - reproduced")
    plt.xlabel("Student")
    plt.ylabel("Value")
    plt.xticks(students)
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(IMAGES / "table7_reference_vs_reproduction.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4.8))
    plt.plot(students, [r["overall_fuzzy_reference"] for r in rows], marker="o", label="Paper fuzzy result")
    plt.plot(students, [r["overall_reconstructed"] for r in rows], marker="o", label="Reconstructed final stage")
    plt.xlabel("Student")
    plt.ylabel("Overall performance")
    plt.xticks(students)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(IMAGES / "overall_reference_vs_reconstruction.png", dpi=180)
    plt.close()

    # Surface for the academic stage.
    grid = np.linspace(0, 4, 45)
    z = np.array([[FIS_ACADEMIC.infer(x, y) for y in grid] for x in grid])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    ax.plot_surface(X, Y, z, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("Semester 1 GPA")
    ax.set_ylabel("Semester 2 GPA")
    ax.set_zlabel("Academic Value")
    ax.set_title("Mamdani fuzzy surface")
    plt.tight_layout()
    plt.savefig(IMAGES / "academic_surface.png", dpi=180)
    plt.close()

    print("Evaluation complete")
    for stage, m, r in metrics:
        print(f"{stage}: MAE={m:.4f}, RMSE={r:.4f}")
    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    main()
