"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from data import make_dataset, make_xor_dataset
from plot import plot_boundary

import sys
from pathlib import Path


NB_SAMPLES = 1500
NB_LEARN = 1200
SEEDS = [0, 1, 2, 3, 4]

DEPTHS = [1, 3, 6, 12, None]
NEIGHBORS = [1, 9, 45, 225, 901]
BANDWIDTHS = [0.005, 0.03, 0.2, 0.8, 3.2]


class KDENaiveBayes:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.priors_ = np.array([np.mean(y == c) for c in self.classes_])
        self.X_by_class_ = [X[y == c] for c in self.classes_]
        return self

    def _log_density(self, Xc, x):
        h = self.bandwidth
        u = (x[:, None] - Xc[None, :]) / h
        a = -0.5 * u ** 2
        m = a.max(axis=1)
        lse = m + np.log(np.exp(a - m[:, None]).sum(axis=1))
        return lse - np.log(len(Xc) * h * np.sqrt(2 * np.pi))

    def _log_posterior(self, X):
        out = np.zeros((X.shape[0], len(self.classes_)))
        for k in range(len(self.classes_)):
            out[:, k] = np.log(self.priors_[k])
            for i in range(X.shape[1]):
                out[:, k] += self._log_density(self.X_by_class_[k][:, i], X[:, i])
        return out

    def predict_proba(self, X):
        lp = self._log_posterior(X)
        lp = lp - lp.max(axis=1, keepdims=True)
        p = np.exp(lp)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self._log_posterior(X), axis=1)]

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def save_output(filename):
    output_path = Path(filename).with_name(Path(filename).stem + "-answer.txt")

    return open(output_path, "w", encoding="utf-8")

def get_data(seed, xor=False):
    if xor:
        X, y = make_xor_dataset(NB_SAMPLES, random_state=seed)
    else:
        X, y = make_dataset(NB_SAMPLES, random_state=seed)

    return X[:NB_LEARN], y[:NB_LEARN], X[NB_LEARN:], y[NB_LEARN:]


def make_model(method, value):
    if method == "tree":
        return DecisionTreeClassifier(max_depth=value, random_state=0)
    
    if method == "knn":
        return KNeighborsClassifier(n_neighbors=value)
    
    return KDENaiveBayes(bandwidth=value)


def candidates(method):
    return {"tree": DEPTHS, "knn": NEIGHBORS, "nb": BANDWIDTHS}[method]


def tune(method, X, y, seed):
    rng = np.random.RandomState(seed)

    idx = rng.permutation(len(X))
    n_train = int(0.8 * len(X))

    tr, va = idx[:n_train], idx[n_train:]

    best_value, best_acc = None, -1.0

    for value in candidates(method):
        model = make_model(method, value).fit(X[tr], y[tr])
        acc = np.mean(model.predict(X[va]) == y[va])

        if acc > best_acc:
            best_value, best_acc = value, acc

    return best_value


def run(method, xor, plot_name):
    acces, chosen = [], []

    for seed in SEEDS:

        X_l, y_l, X_t, y_t = get_data(seed, xor)
        
        value = tune(method, X_l, y_l, seed)
        model = make_model(method, value).fit(X_l, y_l)

        acces.append(np.mean(model.predict(X_t) == y_t))
        chosen.append(value)

        if seed == SEEDS[0]:
            plot_boundary(plot_name, model, X_t, y_t,title="{} (tuned value = {})".format(method, value))

    return np.mean(acces), np.std(acces), chosen



if __name__ == "__main__":
    names = {
        "tree": "Decision tree",
        "knn": "k-NN",
        "nb": "KDE Naive Bayes"
    }

    original_stdout = sys.stdout

    with save_output(__file__) as output_file:
        sys.stdout = Tee(original_stdout, output_file)

        
        for xor, label in [(False, "two-moon"), (True, "xor")]:
            print("=== Dataset: {} ===".format(label))

            for method in ["tree", "knn", "nb"]:
                mean, std, chosen = run(method, xor, "mc_{}_{}".format(label, method))

                print("{:16s} acc = {:.3f} +/- {:.3f}   chosen values: {}".format(names[method], mean, std, chosen))

        sys.stdout = original_stdout
