"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin

from data import make_dataset
from plot import plot_boundary


class NaiveBayesClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, bandwidth=0.2):
        self.bandwidth = bandwidth

    def fit(self, X, y):
        """Fit a naive bayes classifier model using the training set
        (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        self.classes_ = np.unique(y)
        self.priors_ = np.array([np.mean(y == c) for c in self.classes_])
        self.X_by_class_ = [X[y == c] for c in self.classes_]
        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        lp = self.predict_log_proba(X)
        lp = lp - lp.max(axis=1, keepdims=True)
        p = np.exp(lp)
        return p / p.sum(axis=1, keepdims=True)

    def _log_density(self, Xc, x):
        h = self.bandwidth
        u = (x[:, None] - Xc[None, :]) / h
        a = -0.5 * u**2
        m = a.max(axis=1)
        lse = m + np.log(np.exp(a - m[:, None]).sum(axis=1))
        return lse - np.log(len(Xc) * h * np.sqrt(2 * np.pi))

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        out = np.zeros((X.shape[0], len(self.classes_)))
        for k in range(len(self.classes_)):
            out[:, k] = np.log(self.priors_[k])
            for i in range(X.shape[1]):
                out[:, k] += self._log_density(self.X_by_class_[k][:, i], X[:, i])
        return out


BANDWIDTHS = [0.005, 0.03, 0.2, 0.8, 3.2]


def get_data(seed):
    # Generate 1500 points: 1200 for learning, 300 for testing.
    X, y = make_dataset(1500, random_state=seed)
    return X[:1200], y[:1200], X[1200:], y[1200:]


if __name__ == "__main__":
    lines = []

    # Q3: decision boundaries (seed 0), drawn on the test set
    X_tr, y_tr, X_te, y_te = get_data(0)
    lines.append("Q3 - seed 0")
    for h in BANDWIDTHS:
        clf = NaiveBayesClassifier(bandwidth=h).fit(X_tr, y_tr)
        plot_boundary("nb_bandwitdh_" + str(h), clf, X_te, y_te, title=f"h={h}")
        lines.append(
            f"h={h}: train={clf.score(X_tr, y_tr):.3f} test={clf.score(X_te, y_te):.3f}"
        )

    # Q4: mean and std of the test accuracy over 5 generations
    lines.append("\nQ4 - mean +/- std of the test accuracy (seeds 0-4)")
    for h in BANDWIDTHS:
        accs = []
        for seed in range(5):
            X_tr, y_tr, X_te, y_te = get_data(seed)
            accs.append(
                NaiveBayesClassifier(bandwidth=h).fit(X_tr, y_tr).score(X_te, y_te)
            )
        lines.append(f"h={h}: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    with open("nb_results.txt", "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l)
