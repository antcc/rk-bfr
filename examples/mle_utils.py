import numpy as np
import scipy
from multiprocessing import Pool
from itertools import product

from bayesian_model import neg_ll_linear, neg_ll_logistic


def optimizer_global(random_state, args):
    theta_init, X, y, theta_space, is_classification, method, bounds = args
    neg_ll = neg_ll_logistic if is_classification else neg_ll_linear

    mle = scipy.optimize.basinhopping(
        neg_ll,
        x0=theta_init,
        seed=random_state,
        minimizer_kwargs={"args": (X, y, theta_space),
                          "method": method,
                          "bounds": bounds}
    ).x

    return mle


def compute_mle(
    theta_space,
    X,
    y,
    is_classification,
    method='Powell',
    strategy='global',
    n_jobs=2,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    p = theta_space.p
    theta_init = theta_space.forward(
        np.array([0.0]*p + [0.5]*p + [0.0] + [1.0]))

    neg_ll = neg_ll_logistic if is_classification else neg_ll_linear

    bounds = ([(None, None)]*p
              + [(theta_space.tau_lb, theta_space.tau_ub)]*p
              + [(None, None)]
              + [(None, None)])

    if strategy == 'local':
        mle_theta = scipy.optimize.minimize(
            neg_ll,
            x0=theta_init,
            bounds=bounds,
            method=method,
            args=(X, y, theta_space)
        ).x
        bic = compute_bic(
            theta_space, neg_ll, mle_theta, X, y
        )
    elif strategy == 'global':
        mles = np.zeros((n_jobs, theta_space.ndim))

        with Pool(n_jobs) as pool:
            random_states = [rng.integers(2**32) for i in range(n_jobs)]
            args_optim = [
                theta_init, X, y, theta_space,
                is_classification, method, bounds
            ]
            mles = pool.starmap(
                optimizer_global,
                product(random_states, [args_optim])
            )
            bics = compute_bic(
                theta_space, neg_ll, mles, X, y
            )
            mle_theta = mles[np.argmin(bics)]
            bic = bics[np.argmin(bics)]
    else:
        raise ValueError(
            "Parameter 'strategy' must be one of {'local', 'global'}.")

    return mle_theta, bic


def compute_bic(theta_space, neg_ll, mles, X, Y):
    n = X.shape[0]
    bics = np.array([theta_space.ndim*np.log(n)
                     + 2*neg_ll(mle_theta, X, Y, theta_space)
                     for mle_theta in mles])

    return bics[0] if len(mles) == 1 else bics
