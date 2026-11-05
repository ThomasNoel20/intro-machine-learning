"""Question 2.3 : cartes de biais^2, variance et modèle moyen pour MLP, kNN et arbre."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bias_variance import estimate_bias_variance, mean_over_input_space
from data import make_grid
from models import get_factory

N_TRAIN = 250
BETA = 5.0
N_SAMPLES = 200
POINTS_PER_AXIS = 60

CONFIGURATIONS = {"mlp": 10, "knn": 15, "tree": 6}
TITLES = {"mlp": "Réseau de neurones", "knn": "kNN", "tree": "Arbre de décision"}
PARAMETER_LABELS = {"mlp": "neurones", "knn": "k", "tree": "profondeur max"}

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def draw_map(axis, values, title, vmax, cmap):
    image = axis.imshow(values, origin="lower", extent=[-1, 1, -1, 1], vmin=0, vmax=vmax, cmap=cmap)
    curve_x1 = np.linspace(-1, 1, 400)
    axis.plot(curve_x1, np.sin(np.pi * curve_x1), "k--", linewidth=0.8)  # frontière de Bayes (P=0.5)
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("$x_1$")
    axis.set_ylim(-1, 1)
    plt.colorbar(image, ax=axis, fraction=0.046)


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    X_grid, grid_shape = make_grid(POINTS_PER_AXIS)

    results = {}
    for method, complexity in CONFIGURATIONS.items():
        print(f"Estimation : {method} ({PARAMETER_LABELS[method]} = {complexity})")
        results[method] = estimate_bias_variance(get_factory(method, complexity), X_grid,
                                                 n_samples=N_SAMPLES, n_train=N_TRAIN, beta=BETA)

    vmax_bias = max(r["squared_bias"].max() for r in results.values())
    vmax_variance = max(r["variance"].max() for r in results.values())
    panels = [("bayes_model", "Modèle de Bayes", 1, "coolwarm"),
              ("mean_prediction", "Modèle moyen", 1, "coolwarm"),
              ("squared_bias", "Biais$^2$", vmax_bias, "viridis"),
              ("variance", "Variance", vmax_variance, "viridis")]

    for method, result in results.items():
        figure, axes = plt.subplots(1, 4, figsize=(17, 4))
        for axis, (key, title, vmax, cmap) in zip(axes, panels):
            draw_map(axis, result[key].reshape(grid_shape), title, vmax, cmap)
        axes[0].set_ylabel("$x_2$")
        figure.suptitle(f"{TITLES[method]} ({PARAMETER_LABELS[method]} = {CONFIGURATIONS[method]}, "
                        f"N = {N_TRAIN}, $\\beta$ = {BETA})")
        figure.tight_layout()
        figure.savefig(FIGURES_DIR / f"q2_3_{method}.pdf")
        plt.close(figure)

    print("\nMoyennes sur [-1, 1]^2")
    for method, result in results.items():
        m = mean_over_input_space(result)
        print(f"{method:5s} bruit={m['residual_error']:.4f} biais2={m['squared_bias']:.4f} "
              f"variance={m['variance']:.4f} erreur={m['expected_error']:.4f}")


if __name__ == "__main__":
    main()
