# encoding: utf-8

from typing import Union, Optional, Dict, Any, Callable

import numpy as np

import utils


class ThetaSpace():
    eps = 1e-6
    N = 100

    def __init__(
        self,
        p=3,
        grid=None,
        *,
        names=[],
        names_ttr=[],
        labels=[],
        tau_ttr=utils.Identity(),
        sigma2_ttr=utils.LogSq()
    ):
        if grid is None:
            grid = np.linspace(1./self.N, 1, self.N)

        self.p = p
        self.grid = grid
        self.names = names
        self.names_ttr = names_ttr
        self.labels = labels

        self.tau_ttr = tau_ttr
        self.sigma2_ttr = sigma2_ttr

        self.tau_lb = tau_ttr.forward(0.0 + self.eps)
        self.tau_ub = tau_ttr.forward(1.0 - self.eps)
        self.ndim = 2*self.p + 2
        self.beta_idx = np.arange(0, self.p)
        self.tau_idx = np.arange(self.p, 2*self.p)
        self.alpha0_idx = -2
        self.sigma2_idx = -1

    def get_beta(self, theta):
        return theta[self.beta_idx]

    def get_tau(self, theta):
        return theta[self.tau_idx]

    def get_alpha0(self, theta):
        return theta[self.alpha0_idx]

    def get_sigma2(self, theta):
        return theta[self.sigma2_idx]

    def get_params(self, theta):
        beta = theta[self.beta_idx]
        tau = theta[self.tau_idx]
        alpha0 = theta[self.alpha0_idx]
        sigma2 = theta[self.sigma2_idx]

        return beta, tau, alpha0, sigma2

    def _perform_ttr(self, theta, ttrs):
        ndim = len(theta.shape)

        if ndim < 2:
            theta = theta[np.newaxis, :]

        tau_ttr = ttrs[0]
        sigma2_ttr = ttrs[1]

        theta_tr = np.hstack((
            theta[:, :self.p],
            tau_ttr(theta[:, self.p:2*self.p]),
            np.atleast_2d(theta[:, -2]).T,
            np.atleast_2d(sigma2_ttr(theta[:, -1])).T
        ))

        return theta_tr[0] if ndim == 1 else theta_tr

    def forward(self, theta):
        """Parameter is (β, τ, α0, σ2)."""
        theta_tr = self._perform_ttr(
            theta,
            [self.tau_ttr.forward, self.sigma2_ttr.forward])

        return theta_tr

    def backward(self, theta_tr):
        """Parameter 'theta_tr' is (β, logit τ, α0, log σ)."""
        theta = self._perform_ttr(
            theta_tr,
            [self.tau_ttr.backward, self.sigma2_ttr.backward])

        return theta

    def clip_bounds(self, theta):
        """Clip variables to their bounds."""

        theta_clp = np.copy(theta)

        # Restrict τ to [0, 1]
        theta_clp[:, self.tau_idx] = \
            np.clip(theta[:, self.tau_idx], self.tau_lb, self.tau_ub)

        # Restrict σ2 to the positive reals
        theta_clp[:, self.sigma2_idx] = \
            np.clip(theta_clp[:, self.sigma2_idx], 0.0, None)

        return theta_clp


RandomType = Union[
    int,
    np.random.RandomState,
    np.random.Generator,
    None
]

PriorType = Optional[
    Callable[
        [np.ndarray, np.ndarray, np.ndarray, ThetaSpace, Any],
        float
    ]
]


#
# Default log-prior model for linear regression
#

def log_prior_linear(
    theta_tr,
    X,
    y,
    theta_space,
    *,
    b0,
    g,
    eta,
):
    n, N = X.shape
    p = theta_space.p
    grid = theta_space.grid

    theta = theta_space.backward(theta_tr)
    beta, tau, alpha0, sigma2 = theta_space.get_params(theta)
    log_sigma = theta_space.get_sigma2(theta_tr)

    if (tau < theta_space.tau_lb).any() or (tau > theta_space.tau_ub).any():
        return -np.inf

    # Transform variables
    b = beta - b0

    # Compute and regularize G_tau
    idx = np.abs(grid - tau[:, np.newaxis]).argmin(1)
    X_tau = X[:, idx]
    G_tau = X_tau.T@X_tau
    G_tau = (G_tau + G_tau.T)/2.  # Enforce symmetry
    G_tau_reg = G_tau + eta * \
        np.max(np.linalg.eigvalsh(G_tau))*np.identity(p)

    # Compute log-prior
    log_prior = (0.5*utils.logdet(G_tau_reg)
                 - p*log_sigma
                 - b.T@G_tau_reg@b/(2*g*sigma2))

    return log_prior


#
# Log-likelihood and log-posterior for linear regression
#

def log_posterior_linear(
    theta_tr: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    theta_space: ThetaSpace,
    log_prior: PriorType = log_prior_linear,
    rng: RandomType = None,
    return_pp: bool = False,
    return_ll: bool = False,
    prior_kwargs: Optional[Dict] = None,
):
    if rng is None:
        rng = np.random.default_rng()

    # Compute log-prior
    lp = log_prior(theta_tr, X, y, theta_space, **prior_kwargs)

    if not np.isfinite(lp):
        if return_pp and return_ll:
            return -np.inf, np.full_like(y, -np.inf), np.full_like(y, -np.inf)
        elif return_pp:
            return -np.inf, np.full_like(y, -np.inf)
        elif return_ll:
            return -np.inf, np.full_like(y, -np.inf)
        else:
            return -np.inf

    # Compute log-likelihood (and possibly pointwise log-likelihood)
    if return_ll:
        ll, ll_pointwise = log_likelihood_linear(
            theta_tr, X, y, theta_space, return_ll)
    else:
        ll = log_likelihood_linear(theta_tr, X, y, theta_space, return_ll)

    # Compute log-posterior
    lpos = lp + ll

    # Compute posterior predictive samples
    if return_pp:
        theta = theta_space.backward(theta_tr)
        pp = utils.generate_response(X, theta, rng=rng)

    # Return information
    if return_pp and return_ll:
        return lpos, pp, ll_pointwise
    elif return_pp:
        return lpos, pp
    elif return_ll:
        return lpos, ll_pointwise
    else:
        return lpos


def log_likelihood_linear(
    theta_tr,
    X,
    y,
    theta_space,
    return_ll=False
):
    n, N = X.shape
    grid = theta_space.grid

    theta = theta_space.backward(theta_tr)
    beta, tau, alpha0, sigma2 = theta_space.get_params(theta)
    log_sigma = theta_space.get_sigma2(theta_tr)

    idx = np.abs(grid - tau[:, np.newaxis]).argmin(1)
    X_tau = X[:, idx]

    ll = (-n*log_sigma
          - np.linalg.norm(y - alpha0 - X_tau@beta)**2/(2*sigma2))

    if return_ll:
        # Add constant term so that it is the genuine log-probability
        ll_pointwise = (-log_sigma - 0.5*np.log(2*np.pi)
                        - (y - alpha0 - X_tau@beta)**2/(2*sigma2))
        return ll, ll_pointwise
    else:
        return ll


def neg_ll_linear(*args, **kwargs):
    return -log_likelihood_linear(*args, **kwargs)


#####

def neg_ll_logistic(*args, **kwargs):
    pass
