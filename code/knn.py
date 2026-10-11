"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from data import make_dataset
from plot import plot_boundary
from utils import output_path, save_results

N_NEIGHBORS = [1, 9, 45, 225, 901]


def get_data(seed):
    # Generate 1500 points: 1200 for learning, 300 for testing.
    X, y = make_dataset(1500, random_state=seed)
    return X[:1200], y[:1200], X[1200:], y[1200:]


def train(X, y, k):
    return KNeighborsClassifier(n_neighbors=k).fit(X, y)


if __name__ == "__main__":
    lines = []

    # Q1: decision boundaries (seed 0), drawn on the test set
    X_tr, y_tr, X_te, y_te = get_data(0)
    lines.append("Q1 - seed 0")
    for k in N_NEIGHBORS:
        clf = train(X_tr, y_tr, k)
        plot_boundary(output_path("knn", f"knn_{k}"), clf, X_te, y_te,
                      title=f"n_neighbors={k}")
        lines.append(f"n_neighbors={k}: train={clf.score(X_tr, y_tr):.3f} "
                     f"test={clf.score(X_te, y_te):.3f}")

    # Q2: mean and std of the test accuracy over 5 generations
    lines.append("\nQ2 - mean +/- std of the test accuracy (seeds 0-4)")
    for k in N_NEIGHBORS:
        accs = []
        for seed in range(5):
            X_tr, y_tr, X_te, y_te = get_data(seed)
            accs.append(train(X_tr, y_tr, k).score(X_te, y_te))
        lines.append(f"n_neighbors={k}: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")

    save_results("knn", lines)