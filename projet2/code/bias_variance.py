"""Protocole d'estimation du bruit résiduel, du biais^2, de la variance et de l'erreur attendue."""
import numpy as np
from joblib import Parallel, delayed

from data import bayes_probability, make_dataset

QUANTITIES = ["residual_error", "squared_bias", "variance", "expected_error"]


def _fit_and_predict(make_model, X_test, n_train, beta, seed):
    X_train, y_train = make_dataset(n_train, beta, random_state=seed)
    model = make_model(seed).fit(X_train, y_train)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X_test))
    return model.predict_proba(X_test)[:, classes.index(1)]


def estimate_bias_variance(make_model, X_test, n_samples=200, n_train=250,
                           beta=5.0, seed=0, n_jobs=-1):
    seeds = np.random.RandomState(seed).randint(0, 2**31 - 1, size=n_samples)
    predictions = np.array(Parallel(n_jobs=n_jobs)(
        delayed(_fit_and_predict)(make_model, X_test, n_train, beta, int(s))
        for s in seeds))                                  # forme (n_samples, n_test)

    bayes_model = bayes_probability(X_test, beta)
    mean_prediction = predictions.mean(axis=0)            # E_LS[f_LS(x)]
    residual_error = bayes_model * (1.0 - bayes_model)    # noise(x)
    squared_bias = (mean_prediction - bayes_model) ** 2   # bias^2(x)
    variance = predictions.var(axis=0)                    # var_LS(f_LS(x)), ddof=0

    return {"bayes_model": bayes_model, "mean_prediction": mean_prediction,
            "residual_error": residual_error, "squared_bias": squared_bias,
            "variance": variance,
            "expected_error": residual_error + squared_bias + variance}


def mean_over_input_space(result):
    return {name: float(result[name].mean()) for name in QUANTITIES}
