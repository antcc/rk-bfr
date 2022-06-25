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

    return X, y


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

    return X, y


def generate_mixture_dataset(
        grid, kernel_fn, kernel_fn2,
        n_samples, rng=None, mean_vector=None,
        mean_vector2=None
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

    return X, y
