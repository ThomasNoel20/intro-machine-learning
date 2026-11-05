"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
from pathlib import Path

import numpy as np
from sklearn.utils import check_random_state


def _check_n_points(n_points):
    if isinstance(n_points, (bool, np.bool_)) or not isinstance(n_points, (int, np.integer)) or n_points < 1:
        raise ValueError("n_points must be a positive integer")


def make_dataset(n_points, random_state=None):
    """Return shuffled extended two moons, with X.shape == (n_points, 2).

    random_state accepts an integer, a numpy RandomState, or None. Each
    class follows a make_moons arc, extended from [0, pi] to
    [-delta, pi + delta], with delta = 0.40. Gaussian noise (std 0.12) is
    added before horizontal elongation and a pi/6 rotation. The angular
    extension, rather than the affine scaling, makes linear classification
    substantially harder.
    """
    _check_n_points(n_points)
    rng = check_random_state(random_state)
    delta = 0.40
    n_outer = n_points // 2
    n_inner = n_points - n_outer
    outer_angle = np.linspace(-delta, np.pi + delta, n_outer)
    inner_angle = np.linspace(-delta, np.pi + delta, n_inner)
    X = np.vstack((
        np.column_stack((np.cos(outer_angle), np.sin(outer_angle))),
        np.column_stack((1 - np.cos(inner_angle), 0.5 - np.sin(inner_angle))),
    ))
    y = np.concatenate((np.zeros(n_outer, dtype=int), np.ones(n_inner, dtype=int)))
    X += rng.normal(scale=0.12, size=X.shape)
    X *= np.array([2.0, 1.0])
    theta = np.pi / 6
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    X = np.dot(X, rotation_matrix)
    permutation = rng.permutation(n_points)
    return X[permutation], y[permutation]


def make_xor_dataset(n_points, random_state=None):
    """Uniform points in [-2, 2]^2; class 1 when the coordinate signs differ.

    Both features have identical class-conditional population marginals.
    Their joint dependence determines the label, violating the NB assumption.
    """
    _check_n_points(n_points)
    rng = check_random_state(random_state)
    X = rng.uniform(-2, 2, size=(n_points, 2))
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    permutation = rng.permutation(n_points)
    return X[permutation], y[permutation]


def save_dataset_figure(output_path=None, random_state=42):
    """Regenerate the vector figure next to statement.tex, from any cwd."""
    import matplotlib.pyplot as plt

    if output_path is None:
        output_path = Path(__file__).resolve().parent.parent / "two_moon.pdf"
    X, y = make_dataset(1500, random_state=random_state)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for label, color in [(0, "DodgerBlue"), (1, "orange")]:
        points = X[y == label]
        ax.scatter(points[:, 0], points[:, 1], color=color, alpha=.5,
                   s=12, label=f"Class {label}")
    ax.grid(True, alpha=.3)
    ax.set_xlabel("X_0")
    ax.set_ylabel("X_1")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Extended two-moon dataset")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return Path(output_path)


if __name__ == "__main__":
    print(save_dataset_figure())
