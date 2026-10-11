"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from data import make_dataset, make_xor_dataset
from nb import NaiveBayesClassifier
from plot import plot_boundary
from utils import output_path, save_results

# Candidate values: the ones of the previous questions.
# With 5-fold CV, each model is trained on 960 points, so n_neighbors=901 is
# still valid (a 75/25 hold-out would leave 900 points: too few for k=901).
METHODS = {
    "Decision tree": (DecisionTreeClassifier(random_state=0),
                      {"max_depth": [1, 3, 6, 12, None]}),
    "KNN": (KNeighborsClassifier(),
            {"n_neighbors": [1, 9, 45, 225, 901]}),
    "KDE Naive Bayes": (NaiveBayesClassifier(),
                        {"bandwidth": [0.005, 0.03, 0.2, 0.8, 3.2]}),
}


def get_data(make, seed):
    # Generate 1500 points: 1200 for learning, 300 for testing.
    X, y = make(1500, random_state=seed)
    return X[:1200], y[:1200], X[1200:], y[1200:]


def tune(estimator, grid, X, y):
    """5-fold cross-validation on the learning set only; the best model is
    then refitted on the whole learning set (GridSearchCV default)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return GridSearchCV(estimator, grid, cv=cv, scoring="accuracy").fit(X, y)


def run(dataset_name, make, tag, lines):
    """Tune and evaluate the three methods over 5 generations of a dataset."""
    lines.append(f"{dataset_name}: mean +/- std of the test accuracy (seeds 0-4)")
    for name, (estimator, grid) in METHODS.items():
        accs, best = [], []
        for seed in range(5):
            X_tr, y_tr, X_te, y_te = get_data(make, seed)
            search = tune(estimator, grid, X_tr, y_tr)
            accs.append(search.score(X_te, y_te))
            best.append(list(search.best_params_.values())[0])
            if seed == 0:
                fname = name.lower().replace(" ", "_")
                plot_boundary(output_path("mc", f"{tag}_{fname}"),
                              search.best_estimator_, X_te, y_te,
                              title=f"{name} ({dataset_name})")
        lines.append(f"{name}: {np.mean(accs):.3f} +/- {np.std(accs):.3f}   "
                     f"selected values per seed: {best}")
    lines.append("")


if __name__ == "__main__":
    lines = []

    run("Two moons", make_dataset, "moons", lines)   # Q2
    run("XOR", make_xor_dataset, "xor", lines)       # Q4

    save_results("mc", lines)