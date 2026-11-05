"""Question 2.4 : erreurs moyennes en fonction de la complexité, de N et de beta (MLP, kNN, arbre)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import make_centered_grid
from experiments import run_sweep

N_SAMPLES = 100
POINTS_PER_AXIS = 40
DEFAULT_N_TRAIN = 250
DEFAULT_BETA = 5.0

COMPLEXITY_VALUES = {
    "mlp": [1, 2, 3, 5, 10, 20, 50],
    "knn": [1, 3, 5, 9, 15, 25, 45, 75, 125, 200],
    "tree": [1, 2, 3, 4, 5, 6, 8, 10, 12, 15],
}
FIXED_COMPLEXITY = {"mlp": 10, "knn": 15, "tree": 6}   # valeurs de 2.3, utilisées pour les balayages en N et beta
N_TRAIN_VALUES = [50, 100, 250, 500, 1000, 2000]
BETA_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

TITLES = {"mlp": "Réseau de neurones", "knn": "kNN", "tree": "Arbre de décision"}
COMPLEXITY_LABELS = {"mlp": "Nombre de neurones", "knn": "k", "tree": "Profondeur maximale"}
CURVES = [("expected_error", "Erreur attendue", "black", "-"),
          ("residual_error", "Erreur résiduelle", "gray", "--"),
          ("squared_bias", "Biais$^2$", "tab:blue", "-"),
          ("variance", "Variance", "tab:red", "-")]

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def plot_curves(axis, rows, x_key, x_label, title, log_x=True):
    x_values = [row[x_key] for row in rows]
    for key, label, color, linestyle in CURVES:
        axis.plot(x_values, [row[key] for row in rows], marker="o", markersize=3,
                  color=color, linestyle=linestyle, label=label)
    if log_x:
        axis.set_xscale("log")
    axis.set_xlabel(x_label)
    axis.set_title(title)
    axis.grid(alpha=0.3)


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    X_test = make_centered_grid(POINTS_PER_AXIS)

    for method in ["knn", "tree", "mlp"]:       # du plus rapide au plus lent
        print(f"=== {method} ===")
        fixed = FIXED_COMPLEXITY[method]
        label = COMPLEXITY_LABELS[method]

        def sweep(name, configurations):
            return run_sweep(f"q2_4_{method}_{name}_s{N_SAMPLES}", method, configurations, X_test, N_SAMPLES)

        rows_complexity = sweep("complexity", [(c, DEFAULT_N_TRAIN, DEFAULT_BETA) for c in COMPLEXITY_VALUES[method]])
        rows_n = sweep("n", [(fixed, n, DEFAULT_BETA) for n in N_TRAIN_VALUES])
        rows_beta = sweep("beta", [(fixed, DEFAULT_N_TRAIN, b) for b in BETA_VALUES])

        figure, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        plot_curves(axes[0], rows_complexity, "complexity", label,
                    f"Complexité (N = {DEFAULT_N_TRAIN}, $\\beta$ = {DEFAULT_BETA})", log_x=method != "tree")
        plot_curves(axes[1], rows_n, "n_train", "Taille N de l'échantillon",
                    f"Taille N ({label.lower()} = {fixed}, $\\beta$ = {DEFAULT_BETA})")
        plot_curves(axes[2], rows_beta, "beta", "Niveau de bruit $\\beta$",
                    f"Bruit $\\beta$ ({label.lower()} = {fixed}, N = {DEFAULT_N_TRAIN})")
        axes[0].set_ylabel("Valeur moyenne sur $[-1,1]^2$")
        axes[0].legend(fontsize=8)
        figure.suptitle(TITLES[method])
        figure.tight_layout()
        figure.savefig(FIGURES_DIR / f"q2_4_{method}.pdf")
        plt.close(figure)


if __name__ == "__main__":
    main()
