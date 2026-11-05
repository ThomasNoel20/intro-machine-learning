"""Fabriques de modèles : get_factory(method, complexity)(seed) -> modèle non entraîné."""
from functools import partial

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def _make_model(method, complexity, seed):
    if method == "mlp":
        return MLPClassifier(hidden_layer_sizes=(complexity,), solver="lbfgs",
                             alpha=1e-5, max_iter=2000, random_state=seed)
    if method == "knn":
        return KNeighborsClassifier(n_neighbors=complexity)
    if method == "tree":
        return DecisionTreeClassifier(max_depth=complexity, random_state=seed)
    raise ValueError(method)


def get_factory(method, complexity):
    return partial(_make_model, method, complexity)
