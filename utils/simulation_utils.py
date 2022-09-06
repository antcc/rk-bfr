# encoding: utf-8

import numpy as np
from rkbfr.bayesian_model import (ThetaSpace, apply_label_noise,
                                  generate_response_linear)
from rkbfr.utils import IgnoreWarnings
from scipy.integrate import trapz
from skfda.preprocessing.smoothing.validation import (
    LinearSmootherGeneralizedCVScorer, SmoothingParameterSearch,
    akaike_information_criterion)


def brownian_kernel(s, t, sigma=1.0):
    return sigma*np.minimum(s, t)


def fractional_brownian_kernel(s, t, H=0.8):
    return 0.5*(s**(2*H) + t**(2*H) - np.abs(s - t)**(2*H))


def ornstein_uhlenbeck_kernel(s, t, v=1.0):
    return np.exp(-np.abs(s - t)/v)


def squared_exponential_kernel(s, t, v=0.2):
    return np.exp(-(s - t)**2/(2*v**2))


def grollemund_smooth(t):
    return (5*np.exp(-20*(t-0.25)**2)
            - 2*np.exp(-20*(t-0.5)**2)
            + 2*np.exp(-20*(t-0.75)**2))


def cholaquidis_scenario3(t):
    return np.log(1 + 4*t)


def cov_matrix(kernel_fn, s, t):
    ss, tt = np.meshgrid(s, t, indexing='ij')

    # Evaluate the kernel over meshgrid (vectorized operation)
    K = kernel_fn(ss, tt)

    return K


def gp(grid, mean_vector, kernel_fn, n_samples, rng=None):
    grid = np.asarray(grid)

    if rng is None:
        rng = np.random.default_rng()
    if mean_vector is None:
        mean_vector = np.zeros(len(grid))

    kernel_matrix = cov_matrix(kernel_fn, grid, grid)

    X = rng.multivariate_normal(mean_vector, kernel_matrix, size=n_samples)

    return X


def generate_l2_dataset(
    X,
    grid,
    beta_coef,
    alpha0,
    sigma2,
    rng=None,
):
    """Generate dataset based on GP with a given kernel function."""
    if rng is None:
        rng = np.random.default_rng()

    beta = beta_coef(grid)
    y = alpha0 + trapz(y=X*beta, x=grid)

    if sigma2 > 0.0:
        y += np.sqrt(sigma2)*rng.standard_normal(size=len(y))

    return y


def generate_rkhs_dataset(
    X,
    grid,
    beta,
    tau,
    alpha0,
    sigma2,
    rng=None,
):
    """Generate dataset based on GP with a given kernel function."""
    if rng is None:
        rng = np.random.default_rng()

    X = X - X.mean(axis=0)  # The RKHS model assumes centered variables
    theta_true = np.concatenate((
        beta, tau,
        [alpha0], [sigma2]
    ))

    p = len(beta)
    theta_space = ThetaSpace(p, grid)
    y = generate_response_linear(
        X, theta_true, theta_space, noise=sigma2 > 0.0, rng=rng)

    return y


def generate_mixture_dataset(
        grid, mean_vector, mean_vector2,
        kernel_fn, kernel_fn2,
        n_samples, random_noise=None, rng=None,
):
    """Generate dataset based on a known distribution on X|Y."""
    if rng is None:
        rng = np.random.default_rng()
    if mean_vector is None:
        mean_vector = np.zeros(len(grid))
    if mean_vector2 is None:
        mean_vector2 = np.zeros(len(grid))

    # Generate samples with p=1/2
    prob = rng.binomial(1, 0.5, size=n_samples)

    n1 = np.count_nonzero(prob)
    n2 = n_samples - n1

    X1 = gp(grid, mean_vector, kernel_fn, n1, rng)
    X2 = gp(grid, mean_vector2, kernel_fn2, n2, rng)
    X = np.vstack((X1, X2))

    # Generate responses
    y = (n1 <= np.arange(n_samples)).astype(int)

    # Shuffle data
    idx = rng.permutation(np.arange(n_samples))
    X = X[idx, :]
    y = y[idx]

    if random_noise is not None:
        y = apply_label_noise(y, random_noise, rng)

    return X, y


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
