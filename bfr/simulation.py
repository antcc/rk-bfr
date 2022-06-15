# encoding: utf-8

import numpy as np
from scipy.integrate import trapz

from .bayesian_model import ThetaSpace, generate_response_linear


def brownian_kernel(s, t, sigma=1.0):
    return sigma*np.minimum(s, t)


def fractional_brownian_kernel(s, t, H=0.8):
    return 0.5*(s**(2*H) + t**(2*H) - np.abs(s - t)**(2*H))


def ornstein_uhlenbeck_kernel(s, t, v=1.0):
    return np.exp(-np.abs(s - t)/v)


def squared_exponential_kernel(s, t, v=0.2):
    return np.exp(-np.abs(s - t)**2/(2*v**2))


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

    kernel_matrix = cov_matrix(kernel_fn, grid, grid)

    X = rng.multivariate_normal(mean_vector, kernel_matrix, size=n_samples)
    X = X - X.mean(axis=0)  # Center data
    return X


def generate_gp_l2_dataset(
        grid, kernel_fn, n_samples,
        beta_coef, alpha0, sigma2, rng=None,
        mean_vector=None
):
    """Generate dataset based on GP with a given kernel function."""
    if rng is None:
        rng = np.random.default_rng()

    if mean_vector is None:
        mean_vector = np.zeros(len(grid))

    beta = beta_coef(grid)

    X = gp(grid, mean_vector, kernel_fn, n_samples, rng)
    y = alpha0 + trapz(y=X*beta, x=grid)

    if sigma2 > 0.0:
        y += np.sqrt(sigma2)*rng.standard_normal(size=n_samples)

    return X, y


def generate_gp_rkhs_dataset(
        grid, kernel_fn, n_samples,
        beta, tau, alpha0, sigma2, rng=None,
        mean_vector=None
):
    """Generate dataset based on GP with a given kernel function."""
    if mean_vector is None:
        mean_vector = np.zeros(len(grid))

    X = gp(grid, mean_vector, kernel_fn, n_samples, rng)

    theta_true = np.concatenate((
        beta, tau,
        [alpha0], [sigma2]
    ))

    p = len(beta)
    theta_space = ThetaSpace(p, grid)
    y = generate_response_linear(
        X, theta_true, theta_space, noise=sigma2 > 0.0, rng=rng)

    return X, y


def generate_classification_dataset(
        grid, kernel_fn, kernel_fn2,
        n_samples, rng=None, mean_vector=None,
        mean_vector2=None
):
    """Generate dataset based on a known distribution on X|Y."""
    if mean_vector is None:
        mean_vector = np.zeros(len(grid))
    if mean_vector2 is None:
        mean_vector2 = np.zeros(len(grid))

    X1 = gp(grid, mean_vector, kernel_fn, n_samples//2, rng)
    X2 = gp(grid, mean_vector2, kernel_fn2, n_samples - n_samples//2, rng)

    X = np.vstack((X1, X2))
    y = (n_samples//2 <= np.arange(n_samples)).astype(int)

    return X, y
