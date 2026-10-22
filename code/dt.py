"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from data import make_dataset
from plot import plot_boundary


NB_SAMPLES = 1500
NB_LEARN = 1200
SEEDS = [0, 1, 2, 3, 4]
DEPTHS = [1, 3, 6, 12, None]


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


def get_data(seed):
    """1500 points: first 1200 for learning, last 300 for test."""
    X, y = make_dataset(NB_SAMPLES, random_state=seed)
    return X[:NB_LEARN], y[:NB_LEARN], X[NB_LEARN:], y[NB_LEARN:]


def main():
    X_l, y_l, X_t, y_t = get_data(SEEDS[0])
    print("Q1 - seed 0 (train / test accuracy)")
    for depth in DEPTHS:
        model = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_l, y_l)
        plot_boundary("dt_depth_{}".format(depth), model, X_t, y_t,
                      title="max_depth={}".format(depth))
        print("max_depth={}: train={:.3f} test={:.3f}".format(
            depth, model.score(X_l, y_l), model.score(X_t, y_t)))

    print("\nQ2 - test accuracy over seeds 0-4 (mean +/- std)")
    for depth in DEPTHS:
        acces = []
        for seed in SEEDS:
            X_l, y_l, X_t, y_t = get_data(seed)
            model = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_l, y_l)
            acces.append(model.score(X_t, y_t))
        print("max_depth={}: {:.3f} +/- {:.3f}".format(depth, np.mean(acces), np.std(acces)))


if __name__ == "__main__":
    with open("dt_results.txt", "w", encoding="utf-8") as f:
        sys.stdout = Tee(sys.__stdout__, f)
        main()
        sys.stdout = sys.__stdout__