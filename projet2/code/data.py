"""Problème synthétique du projet 2 : P(y=1|x) = sigmoide(beta * (x2 - sin(pi*x1)))."""
import numpy as np
from sklearn.utils import check_random_state


def bayes_probability(X, beta):
    return 1.0 / (1.0 + np.exp(-beta * (X[:, 1] - np.sin(np.pi * X[:, 0]))))


def make_dataset(n_points, beta=5.0, random_state=None):
    rng = check_random_state(random_state)
    X = rng.uniform(-1.0, 1.0, size=(n_points, 2))
    y = (rng.uniform(size=n_points) < bayes_probability(X, beta)).astype(int)
    return X, y


def _grid(axis):
    grid_x1, grid_x2 = np.meshgrid(axis, axis)
    return np.column_stack((grid_x1.ravel(), grid_x2.ravel())), grid_x1.shape


def make_grid(points_per_axis=60):
    return _grid(np.linspace(-1.0, 1.0, points_per_axis))


def make_centered_grid(points_per_axis=40):
    edges = np.linspace(-1.0, 1.0, points_per_axis + 1)
    return _grid((edges[:-1] + edges[1:]) / 2.0)[0]
