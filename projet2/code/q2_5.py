"""Question 2.5 : complexité optimale (k* du kNN) en fonction de N et de beta."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import make_centered_grid
from experiments import run_sweep

N_SAMPLES = 100
POINTS_PER_AXIS = 40
DEFAULT_N_TRAIN = 250
DEFAULT_BETA = 5.0
K_VALUES = [1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 90, 135, 200, 300, 450]   # grille géométrique, filtrée par k <= N
N_TRAIN_VALUES = [50, 100, 250, 500, 1000, 2000]
BETA_VALUES = [1.0, 2.0, 5.0, 10.0, 20.0]

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def analyse(rows, group_key, group_values, group_label, file_name):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    optimal_k, optimal_excess = [], []

    for group_value in group_values:
        group_rows = [row for row in rows if row[group_key] == group_value]
        k_values = [row["complexity"] for row in group_rows]
        excess_values = [row["squared_bias"] + row["variance"] for row in group_rows]
        axes[0].plot(k_values, excess_values, marker="o", markersize=3, label=f"{group_label} = {group_value}")
        best = int(np.argmin(excess_values))
        optimal_k.append(k_values[best])
        optimal_excess.append(excess_values[best])

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Erreur excédentaire (biais$^2$ + variance)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for axis, values, label in [(axes[1], optimal_k, "$k^*$"),
                                (axes[2], optimal_excess, "Erreur excédentaire en $k^*$")]:
        slope = np.polyfit(np.log(group_values), np.log(values), 1)[0]
        axis.loglog(group_values, values, "o-", label=f"pente (log-log) = {slope:.2f}")
        axis.set_xlabel(group_label)
        axis.set_ylabel(label)
        axis.legend()
        axis.grid(alpha=0.3, which="both")
        print(f"{label} en fonction de {group_label} : pente log-log = {slope:.3f}")

    print(f"{group_label} | k* = {optimal_k} | erreur excédentaire = {[round(v, 4) for v in optimal_excess]}")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / file_name)
    plt.close(figure)


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    X_test = make_centered_grid(POINTS_PER_AXIS)

    configurations_n = [(k, n, DEFAULT_BETA) for n in N_TRAIN_VALUES for k in K_VALUES if k <= n]
    rows_n = run_sweep(f"q2_5_knn_n_s{N_SAMPLES}", "knn", configurations_n, X_test, N_SAMPLES)
    analyse(rows_n, "n_train", N_TRAIN_VALUES, "N", "q2_5_n.pdf")

    configurations_beta = [(k, DEFAULT_N_TRAIN, b) for b in BETA_VALUES for k in K_VALUES if k <= DEFAULT_N_TRAIN]
    rows_beta = run_sweep(f"q2_5_knn_beta_s{N_SAMPLES}", "knn", configurations_beta, X_test, N_SAMPLES)
    analyse(rows_beta, "beta", BETA_VALUES, "$\\beta$", "q2_5_beta.pdf")


if __name__ == "__main__":
    main()
