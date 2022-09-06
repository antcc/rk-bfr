# encoding: utf-8

"""
Numerical approximation of the MLE in functional models.
"""

from __future__ import annotations

from multiprocessing import Pool
from typing import Callable

import numpy as np
import scipy

from .bayesian_model import ThetaSpace, neg_ll_linear, neg_ll_logistic
from .utils import fdata_to_numpy


def compute_mle(
    X: np.ndarray,
    y: np.ndarray,
    theta_space: ThetaSpace,
    *,
    kind: str = 'linear',
    method: str = 'Powell',
    strategy: str = 'global',
    n_reps: int = 4,
    n_jobs: int = 2,
    rng: np.random.Generator = None,
) -> [np.ndarray, float]:
    """Computational approximation of the MLE.

    Parameters
    ----------
    X : np.ndarray
        Regressor matrix.
    y : np.ndarray
        Response array.
    theta_space : ThetaSpace
        Parameter space.
    kind : str
        Specify 'linear' or 'logistic' regression.
    method : str
        Method for local optimization.
    strategy : str
        Whether to use a 'local' or 'global' strategy
    n_reps : int
        Number of independent repetitions.
    n_jobs : int
        Number of cores to use.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    [np.ndarray, float]
        MLE approximation and corresponding BIC.
    """

    if rng is None:
        rng = np.random.default_rng()

    p = theta_space.p_max
    n_dim = theta_space.n_dim
    X = fdata_to_numpy(X, theta_space.grid)

    theta_init = theta_space.forward(
        np.array([0.0]*p + [0.5]*p + [0.0] + [1.0])
    )

    tau_ttr_lb = theta_space.tau_ttr.forward(theta_space.tau_range[0])
    tau_ttr_ub = theta_space.tau_ttr.forward(theta_space.tau_range[1])
    sigma2_ttr_lb = theta_space.sigma2_ttr.forward(theta_space.eps)
    sigma2_ttr_ub = theta_space.sigma2_ttr.forward(theta_space.sigma2_ub)
    if np.isinf(sigma2_ttr_ub):
        sigma2_ttr_ub = None

    bounds = ([(None, None)]*p
              + [(tau_ttr_lb, tau_ttr_ub)]*p
              + [(None, None)]
              + [(sigma2_ttr_lb, sigma2_ttr_ub)])

    neg_ll = neg_ll_linear if kind == 'linear' else neg_ll_logistic

    if strategy == 'local':
        mle_theta_tr = scipy.optimize.minimize(
            neg_ll,
            x0=theta_init,
            bounds=bounds,
            method=method,
            args=(X, y, theta_space)
        ).x

        bic = compute_bic(
            theta_space, neg_ll, mle_theta_tr, X, y
        )

    elif strategy == 'global':
        mles = np.zeros((n_reps, n_dim))

        with Pool(n_jobs) as pool:
            random_states = [rng.integers(2**32) for i in range(n_reps)]

            args_optim = ((
                theta_space,
                X,
                y,
                theta_init,
                bounds,
                neg_ll,
                method,
                u) for u in random_states)

            mles = pool.starmap(
                _optimizer_global,
                args_optim
            )

            bics = compute_bic(
                theta_space, neg_ll, mles, X, y
            )

            # Retain only MLE with smaller BIC
            mle_theta_tr = mles[np.argmin(bics)]
            bic = bics[np.argmin(bics)]

    else:
        raise ValueError(
            "Parameter 'strategy' must be one of {'local', 'global'}.")

    # Transform back

    mle_theta = theta_space.backward(mle_theta_tr)

    return mle_theta, bic


def compute_bic(
    theta_space: ThetaSpace,
    neg_ll: Callable,
    mles: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Compute the BIC associated to MLE values.

    Parameters
    ----------
    theta_space : ThetaSpace
        Parameter space.
    neg_ll : Callable
        Negative log-likelihood function.
    mles : np.ndarray
        Values of MLE to compute the BIC.
    X : np.ndarray
        Regressor matrix.
    y : np.ndarray
        Response array.

    Returns
    -------
    np.ndarray
        BIC values.
    """
    n = X.shape[0]
    n_dim = theta_space.n_dim

    bics = np.array([n_dim*np.log(n)
                     + 2*neg_ll(mle_theta, X, y, theta_space)
                     for mle_theta in mles])

    return bics


def _optimizer_global(
    theta_space,
    X,
    y,
    theta_init,
    bounds,
    neg_ll,
    method,
    random_state,
):
    """Global optimizer for approximating the MLE."""
    mle = scipy.optimize.basinhopping(
        neg_ll,
        x0=theta_init,
        seed=random_state,
        minimizer_kwargs={"args": (X, y, theta_space),
                          "method": method,
                          "bounds": bounds}
    ).x

    return mle
