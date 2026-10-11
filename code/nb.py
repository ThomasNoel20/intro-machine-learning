"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import logsumexp

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from data import make_dataset
from plot import plot_boundary
from utils import output_path, save_results

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
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")

        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_prior_ = counts / counts.sum()
        self.n_features_in_ = X.shape[1]

        self.X_class_ = [X[y == c] for c in self.classes_]
        return self

    def _log_kde(self, x, train_values):
        # log p(x | y)
        h = self.bandwidth
        u = (x[:, None] - train_values[None, :]) / h
        log_kernel = -0.5 * u ** 2 - 0.5 * np.log(2 * np.pi)

        # log( 1/(n_y h) * sum_j K(u_j) )
        return logsumexp(log_kernel, axis=1) - np.log(len(train_values) * h)

    def _joint_log_likelihood(self, X):
        #log p(y) + sum_i log p(x_i | y)
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X must have shape (n_samples, %d)" % self.n_features_in_)

        jll = np.empty((X.shape[0], len(self.classes_)))
        for k in range(len(self.classes_)):
            jll[:, k] = np.log(self.class_prior_[k])
            for i in range(self.n_features_in_):
                jll[:, k] += self._log_kde(X[:, i], self.X_class_[k][:, i])
        return jll

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
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]

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
        return np.exp(self.predict_log_proba(X))

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
        jll = self._joint_log_likelihood(X)
        # log p(y|x) = jll - log sum_y' exp(jll_y')
        return jll - logsumexp(jll, axis=1, keepdims=True)


BANDWIDTHS = [0.005, 0.03, 0.2, 0.8, 3.2]


def get_data(seed):
    #Generate 1500 points: 1200 for learning, 300 for testing.
    X, y = make_dataset(1500, random_state=seed)
    return X[:1200], y[:1200], X[1200:], y[1200:]


if __name__ == "__main__":
    lines = []

    # Q3: decision boundaries (seed 0), drawn on the test set
    X_tr, y_tr, X_te, y_te = get_data(0)
    lines.append("Q3 - seed 0")
    for h in BANDWIDTHS:
        clf = NaiveBayesClassifier(bandwidth=h).fit(X_tr, y_tr)
        plot_boundary(output_path("nb", f"nb_h{h}"), clf, X_te, y_te,
                      title=f"h={h}")
        lines.append(f"h={h}: train={clf.score(X_tr, y_tr):.3f} "
                     f"test={clf.score(X_te, y_te):.3f}")

    # Q4: mean and std of the test accuracy over 5 generations
    lines.append("\nQ4 - mean +/- std of the test accuracy (seeds 0-4)")
    for h in BANDWIDTHS:
        accs = []
        for seed in range(5):
            X_tr, y_tr, X_te, y_te = get_data(seed)
            accs.append(NaiveBayesClassifier(bandwidth=h).fit(X_tr, y_tr).score(X_te, y_te))
        lines.append(f"h={h}: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")

    save_results("nb", lines)