import numpy as np
from skfda.preprocessing.smoothing.validation import (
    LinearSmootherGeneralizedCVScorer, SmoothingParameterSearch,
    akaike_information_criterion)

from .utils import IgnoreWarnings


def normalize_grid(grid, low=0, high=1):
    g_min, g_max = np.min(grid), np.max(grid)
    return (grid - g_min)/(g_max - g_min)


def smooth_data(X, smoother, params, X_test=None):
    best_smoother = SmoothingParameterSearch(
        smoother,
        params,
        scoring=LinearSmootherGeneralizedCVScorer(
            akaike_information_criterion),
        n_jobs=-1,
    )

    with IgnoreWarnings():
        best_smoother.fit(X)

    X_tr = best_smoother.transform(X)

    if X_test is not None:
        X_test_tr = best_smoother.transform(X_test)
        return X_tr, best_smoother, X_test_tr

    return X_tr, best_smoother


def standardize_response(y, y_test):
    y_m = y.mean()
    y_sd = y.std()

    y_tr = (y - y_m)/y_sd
    y_test_tr = (y_test - y_m)/y_sd

    return y_tr, y_test_tr


def standardize_predictors(X, X_test, scale=False):
    X_m = X.mean(axis=0)

    if scale:
        X_sd = np.sqrt(X.var())
    else:
        X_sd = 1.0

    X_tr = (X - X_m)/X_sd
    X_test_tr = (X_test - X_m)/X_sd

    return X_tr, X_test_tr
