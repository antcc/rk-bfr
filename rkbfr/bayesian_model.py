# encoding: utf-8

"""
File with classes and functions useful to manipulate the Bayesian aspects
of our model (parameter space, posterior predictive generation, prediction, ...)
"""

from typing import Any, Callable, Dict, Optional, Union

import aesara.tensor as at
import numpy as np
import pymc as pm
from scipy.special import expit
from scipy.stats import beta as beta_dist

from .utils import apply_threshold, compute_mode_xarray


##
# TRANSFORMATIONS
##

class Identity():
    name = "Identity"

    def forward(self, x):
        return x

    def backward(self, y):
        return y


class Logit():
    eps = 1e-6
    name = "Logit"

    def forward(self, x):
        return np.log(x/(1 - x + self.eps) + self.eps)

    def backward(self, y):
        return expit(y)   # 1./(1 + np.exp(-y))


class LogSq():
    eps = 1e-6
    name = "LogSq"

    def forward(self, x):
        return np.log(np.sqrt(x) + self.eps)

    def backward(self, y):
        return np.exp(y)**2


##
# PARAMETER SPACE
##

class ThetaSpace():
    """Finite-dimensional parameter space.

    Parameters
    ----------
    p_max : type
        Maximum value of 'p', the dimension of the model.
    grid : type
        Discretization grid.
    include_p : type
        Whether to include 'p' in the Bayesian model (P_FREE)
    names : type
        Names of the parameters.
    labels : type
        Labels for the parameters.
    labeller : type
        Labeller for Arviz visualization.
    dim_name : type
        Dimension name for arviz.
    tau_range : type
        Range of the parameter tau.
    beta_range : type
        Range of the parameter theta.
    sigma2_ub : type
        Upper bound of the parameter sigma2.
    tau_ttr : type
        Transformation for the parameter tau.
    sigma2_ttr : type
        Transformation for the parameter sigma2.
    """

    eps = 1e-6
    N = 100

    def __init__(
        self,
        p_max=3,
        grid=None,
        *,
        include_p=False,
        names=[],
        labels=[],
        labeller=None,
        dim_name="theta",
        tau_range=(0, 1),
        beta_range=None,
        sigma2_ub=np.inf,
        tau_ttr=Identity(),
        sigma2_ttr=LogSq()
    ):

        # Save values

        if grid is None:
            grid = np.linspace(1./self.N, 1, self.N)

        self.p_max = p_max
        self.grid = grid
        self.include_p = include_p
        self.names = names
        self.labels = labels
        self.labeller = labeller
        self.dim_name = dim_name
        self.tau_range = tau_range
        self.beta_range = beta_range
        self.sigma2_ub = sigma2_ub
        self.tau_ttr = tau_ttr
        self.sigma2_ttr = sigma2_ttr

        # Set dimension and access indices

        self.n_dim = 2*self.p_max + 2
        self.n_params = 4
        self.beta_idx = np.arange(0, self.p_max)
        self.beta_idx_grouped = 0
        self.tau_idx = np.arange(self.p_max, 2*self.p_max)
        self.tau_idx_grouped = 1
        self.alpha0_idx = -2
        self.sigma2_idx = -1

        if self.include_p:
            self.n_dim += 1
            self.n_params += 1
            self.p_idx = 0
            self.beta_idx += 1
            self.beta_idx_grouped += 1
            self.tau_idx += 1
            self.tau_idx_grouped += 1

        self._check_dimensions()

        if len(self.names) > 0:
            self.names_ttr = self._get_names_ttr()

    def copy_p_fixed(self):
        """Create a copy of the object where the parameter 'p'
           is fixed."""
        if not self.include_p:
            return self

        return ThetaSpace(
            self.p_max,
            self.grid,
            include_p=False,
            names=self.names[1:],
            labels=self.labels[1:],
            dim_name=self.dim_name,
            tau_range=self.tau_range,
            beta_range=self.beta_range,
            sigma2_ub=self.sigma2_ub,
            tau_ttr=self.tau_ttr,
            sigma2_ttr=self.sigma2_ttr
        )

    def round_p(self, p):
        """Round p to the nearest integer."""
        return np.rint(p).astype(int)

    def slice_params(self, theta, transform=None, clip=True):
        """Get the individual parameters of the parameter vector."""
        self._check_theta(theta)

        if transform is not None:
            if transform == 'forward':
                theta = self._perform_ttr(theta, is_forward=True)
            elif transform == 'backward':
                theta = self._perform_ttr(theta, is_forward=False)
            else:
                raise ValueError(
                    "Incorrect value for 'transform'. Should be 'forward', "
                    "'backward' or None but got {}".format(transform))

        p = self.round_p(theta[..., self.p_idx]) \
            if self.include_p else self.p_max
        beta = theta[..., self.beta_idx]
        tau = theta[..., self.tau_idx]
        alpha0 = theta[..., self.alpha0_idx]
        sigma2 = theta[..., self.sigma2_idx]

        if self.include_p and clip:
            p_clp = np.clip(p, 1, self.p_max)
            beta = beta[..., :p_clp]
            tau = tau[..., :p_clp]

        return p, beta, tau, alpha0, sigma2

    def forward(self, theta):
        """Transform the parameters."""
        self._check_theta(theta)

        theta_tr = self._perform_ttr(
            theta,
            is_forward=True
        )

        return theta_tr

    def backward(self, theta_tr):
        """Inverse-transform the parameters."""
        self._check_theta(theta_tr)

        theta = self._perform_ttr(
            theta_tr,
            is_forward=False
        )

        return theta

    def set_unused_nan(self, theta, inplace=True):
        """Set unused values to NaN (only relevant when include_p=True)."""
        t = theta.copy() if not inplace else theta
        pp = self.round_p(t[self.p_idx])
        t[self.beta_idx[pp:]] = np.nan
        t[self.tau_idx[pp:]] = np.nan

        if not inplace:
            return t

    def clip_bounds(self, theta):
        """Clip the parameter 'theta' (in the original space)."""
        self._check_theta(theta)

        theta_clp = np.copy(theta)
        theta_clp = np.atleast_2d(theta_clp)

        # Restrict p
        if self.include_p:
            theta_clp[..., self.p_idx] = np.clip(
                theta_clp[..., self.p_idx],
                1,
                self.p_max
            )

        # Restrict β
        if self.beta_range is not None:
            theta_clp[..., self.beta_idx] = np.clip(
                theta_clp[..., self.beta_idx],
                self.beta_range[0],
                self.beta_range[1]
            )

        # Restrict τ
        theta_clp[..., self.tau_idx] = np.clip(
            theta_clp[..., self.tau_idx],
            self.tau_range[0],
            self.tau_range[1]
        )

        # Restrict σ2
        theta_clp[..., self.sigma2_idx] = np.clip(
            theta_clp[..., self.sigma2_idx],
            self.eps,
            self.sigma2_ub,
        )

        return theta_clp.squeeze()

    def _get_names_ttr(self):
        names = self.names.copy()
        tau_ttr_name = self.tau_ttr.name
        sigma2_ttr_name = self.sigma2_ttr.name

        if tau_ttr_name != "Identity":
            names[self.tau_idx_grouped] = (
                tau_ttr_name + " " + names[self.tau_idx_grouped])
        if sigma2_ttr_name != "Identity":
            names[self.sigma2_idx] = (
                sigma2_ttr_name + " " + names[self.sigma2_idx])

        return names

    def _check_theta(self, theta):
        if theta.shape[-1] != self.n_dim:
            raise ValueError(
                "Incorrect dimension for θ: should be {} but got {}".format(
                    self.n_dim,
                    theta.shape[-1])
            )

    def _check_dimensions(self):
        n_names = len(self.names)
        n_labels = len(self.labels)

        if n_names > 0 and n_names != self.n_params:
            raise ValueError(
                "Incorrect size for 'names': should be {} but got {}".format(
                    self.n_params,
                    n_names)
            )

        if n_labels > 0 and n_labels != self.n_dim:
            raise ValueError(
                "Incorrect size for 'labels': should be {} but got {}".format(
                    self.n_dim,
                    n_labels)
            )

    def _perform_ttr(self, theta, is_forward):
        theta_tr = np.copy(theta)
        theta_tr = np.atleast_2d(theta_tr)

        if is_forward:
            tau_ttr = self.tau_ttr.forward
            sigma2_ttr = self.sigma2_ttr.forward
        else:
            tau_ttr = self.tau_ttr.backward
            sigma2_ttr = self.sigma2_ttr.backward

        theta_tr[..., self.tau_idx] = tau_ttr(
            theta_tr[..., self.tau_idx])
        theta_tr[..., self.sigma2_idx] = sigma2_ttr(
            theta_tr[..., self.sigma2_idx])

        return theta_tr.squeeze()


RandomType = Union[
    int,
    np.random.RandomState,
    np.random.Generator,
]

PriorType = Callable[
    # theta, X, y, theta_space, **kwargs
    [np.ndarray, np.ndarray, np.ndarray, ThetaSpace, Any],
    float
]


def generate_response_linear(X, theta, theta_space, noise=True, rng=None):
    """Generate a linear RKHS response Y given X and θ"""
    theta = np.asarray(theta)
    ts = theta_space

    if theta.ndim == 1:
        theta_3d = theta[None, None, :]
    elif theta.ndim == 2:
        theta_3d = theta[None, ...]
    else:
        theta_3d = theta

    if ts.include_p:
        # Replace NaN with 0.0 to "turn off" the corresponding coefficients
        theta_3d = np.nan_to_num(theta_3d, nan=0.0)

    beta = theta_3d[..., ts.beta_idx, None]
    tau = theta_3d[..., ts.tau_idx, None]
    alpha0 = theta_3d[..., ts.alpha0_idx, None]

    idx = np.abs(ts.grid - tau).argmin(-1)
    X_idx = np.moveaxis(X[:, idx], 0, -2)
    y = alpha0 + (X_idx@beta).squeeze()

    if noise:
        if rng is None:
            rng = np.random.default_rng()

        sigma2 = theta_3d[..., ts.sigma2_idx, None]
        y += np.sqrt(sigma2)*rng.standard_normal(size=y.shape)

    return y.squeeze()


def generate_response_logistic(
    X,
    theta,
    theta_space,
    noise=True,
    return_prob=False,
    th=0.5,
    rng=None
):
    """Generate a logistic RKHS response Y given X and θ.

       Returns the response vector and (possibly) the probabilities associated.
    """
    y_lin = generate_response_linear(X, theta, theta_space, noise=False)

    if noise:
        y = probability_to_label(y_lin, rng=rng)
    else:
        if th == 0.5:
            # sigmoid(x) >= 0.5 iff x >= 0
            y = apply_threshold(y_lin, 0.0)
        else:
            y = apply_threshold(expit(y_lin), th)

    if return_prob:
        return expit(y_lin), y
    else:
        return y


def probability_to_label(y_lin, random_noise=None, rng=None):
    """Convert probabilities into class labels."""
    if rng is None:
        rng = np.random.default_rng()

    labels = rng.binomial(1, expit(y_lin))

    if random_noise is not None:
        labels = apply_label_noise(labels, random_noise, rng)

    return labels


def apply_label_noise(y, noise_frac=0.05, rng=None):
    """Apply a random noise to the labels."""
    if rng is None:
        rng = np.random.default_rng()

    y_noise = y.copy()
    n_noise = int(len(y)*noise_frac)

    idx_0 = rng.choice(np.where(y == 0)[0], size=n_noise)
    idx_1 = rng.choice(np.where(y == 1)[0], size=n_noise)

    y_noise[idx_0] = 1
    y_noise[idx_1] = 0

    return y_noise


def generate_pp(
        idata,
        X,
        theta_space,
        thin=1,
        kind='linear',
        rng=None,
        verbose=False
):
    """Generate posterior predictive distribution for the data X."""
    if rng is None:
        rng = np.random.default_rng()

    if verbose:
        print("Generating posterior predictive samples...")

    theta = idata.posterior[theta_space.names].to_stacked_array(
        "", sample_dims=("chain", "draw"))[:, ::thin, :]

    # Generate responses following the model
    if kind == 'logistic':
        p_star, y_star = generate_response_logistic(
            X, theta, theta_space, noise=True,
            return_prob=True, rng=rng
        )
    else:
        y_star = generate_response_linear(
            X, theta, theta_space, noise=True, rng=rng
        )

    if kind == 'logistic':
        return p_star, y_star
    else:
        return y_star


def point_estimate(idata, estimator_fn, names, skipna=False, bw='experimental'):
    """Summarize the posterior through a given estimator.

    If 'pe_fn' is a callable, it should have a specific signature, and also
    treat the (chain, draw) dimensions suitably."""
    posterior_trace = idata.posterior

    if callable(estimator_fn):
        theta_unstacked = estimator_fn(
            posterior_trace[names],
            dim=("chain", "draw"),
            skipna=skipna
        )
    elif estimator_fn == 'mean':
        theta_unstacked = posterior_trace[names].mean(
            dim=("chain", "draw"),
            skipna=skipna
        )
    elif estimator_fn == 'mode':
        theta_unstacked = compute_mode_xarray(
            posterior_trace[names],
            dim=("chain", "draw"),
            skipna=skipna,
            bw=bw
        )
    elif estimator_fn == 'median':
        theta_unstacked = posterior_trace[names].median(
            dim=("chain", "draw"),
            skipna=skipna
        )
    else:
        raise ValueError(
            "'estimator_fn' must be a callable or one of {mean, median, mode}.")

    theta = theta_unstacked.to_stacked_array("", sample_dims="").values

    return theta


def point_predict(
    X,
    idata,
    theta_space,
    estimator_fn='mode',
    kind='linear',
    skipna=False,
    th=0.5,
    bw='experimental'
):
    """Summarize-then-predict approach to prediction."""
    theta_hat = point_estimate(
        idata, estimator_fn, theta_space.names, skipna, bw)
    if kind == 'linear':
        y_pred = generate_response_linear(
            X, theta_hat, theta_space, noise=False)
    else:
        y_pred = generate_response_logistic(
            X, theta_hat, theta_space, noise=False, th=th)

    return y_pred, theta_hat


def bpv(pp_y, y, t_stat):
    """Compute bayesian p-values for a given statistic t_stat.
       - pp_y is an ndarray of shape (..., len(y)) representing the
         posterior predictive samples of the response variable.
       - t_stat is a vectorized function that accepts an 'axis' parameter."""
    pp_y_flat = pp_y.reshape(-1, len(y))
    t_stat_pp = t_stat(pp_y_flat, axis=-1)
    t_stat_observed = t_stat(y)

    return np.mean(t_stat_pp <= t_stat_observed)


#
# Default log-prior model for linear and logistic regression
#

def log_prior_linear_logistic(
    theta_tr,
    X,
    y,
    theta_space,
    *,
    b0,
    g,
    eta,
    prior_p=None,
    tau_params=None
):
    p, beta_full, tau_full, alpha0, sigma2 = \
        theta_space.slice_params(theta_tr, transform='backward', clip=False)

    # Check bounds on parameters
    if theta_space.include_p:
        if p < 1 or p > theta_space.p_max:
            return -np.inf

    if theta_space.beta_range is not None:
        if ((beta_full < theta_space.beta_range[0]).any()
                or (beta_full > theta_space.beta_range[1]).any()):
            return -np.inf

    if ((tau_full < theta_space.tau_range[0]).any()
            or (tau_full > theta_space.tau_range[1]).any()):
        return -np.inf

    if theta_space.sigma2_ub is not None:
        if sigma2 > theta_space.sigma2_ub:
            return -np.inf

    if sigma2 <= theta_space.eps:
        return -np.inf

    tau = tau_full[:p]
    beta = beta_full[:p]
    idx = np.abs(theta_space.grid - tau[:, np.newaxis]).argmin(1)
    X_tau = X[:, idx]
    log_sigma = LogSq().forward(sigma2)

    # Transform variables
    b = beta - b0[:p]

    # Compute and regularize G_tau
    idx = np.abs(theta_space.grid - tau[:, np.newaxis]).argmin(1)
    X_tau = X[:, idx]
    G_tau = X_tau.T@X_tau
    G_tau = (G_tau + G_tau.T)/2.  # Enforce symmetry
    G_tau_reg = G_tau + eta * \
        np.max(np.linalg.eigvalsh(G_tau))*np.identity(p)

    # Compute tau factor
    if tau_params is not None:
        tau_factor = np.sum(
            np.log(beta_dist.pdf(tau, tau_params[:p, 0], tau_params[:p, 1])))
    else:
        tau_factor = 0

    # Compute sigma2 factor
    # (currently we only allow for LogSq transformation)
    sigma2_factor = 2 if theta_space.sigma2_ttr.name == "Identity" else 0

    # Compute log-prior
    log_prior = (0.5*np.linalg.slogdet(G_tau_reg)[1]
                 + tau_factor
                 - (p + sigma2_factor)*log_sigma
                 - b.T@G_tau_reg@b/(2*g*sigma2))

    if theta_space.include_p:
        log_prior += np.log(prior_p[p])

    return log_prior


#
# Log-likelihood for linear and logistic regression
#


def log_likelihood_linear(
    theta_tr,
    X,
    y,
    theta_space,
    return_ll=False
):
    n = X.shape[0]
    p, beta, tau, alpha0, sigma2 = \
        theta_space.slice_params(theta_tr, transform='backward')

    idx = np.abs(theta_space.grid - tau[:, np.newaxis]).argmin(1)
    X_tau = X[:, idx]
    log_sigma = LogSq().forward(sigma2)

    ll = (-n*log_sigma
          - np.linalg.norm(y - alpha0 - X_tau@beta)**2/(2*sigma2))

    if return_ll:
        # Add constant term so that it is the genuine log-probability
        ll_pointwise = (-log_sigma - 0.5*np.log(2*np.pi)
                        - (y - alpha0 - X_tau@beta)**2/(2*sigma2))
        return ll, ll_pointwise
    else:
        return ll


def log_likelihood_logistic(
    theta_tr,
    X,
    y,
    theta_space,
    return_ll=False
):
    p, beta, tau, alpha0, sigma2 = \
        theta_space.slice_params(theta_tr, transform='backward')

    idx = np.abs(theta_space.grid - tau[:, np.newaxis]).argmin(1)
    X_tau = X[:, idx]

    linear_component = alpha0 + X_tau@beta
    ll_pointwise = linear_component*y - np.logaddexp(0, linear_component)
    ll = np.sum(ll_pointwise)

    if return_ll:
        return ll, ll_pointwise
    else:
        return ll


def neg_ll_linear(*args, **kwargs):
    return -log_likelihood_linear(*args, **kwargs)


def neg_ll_logistic(*args, **kwargs):
    return -log_likelihood_logistic(*args, **kwargs)


#
# Log-posterior for linear and logistic regression
#

def log_posterior_linear(
    theta_tr: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    theta_space: ThetaSpace,
    log_prior: PriorType = log_prior_linear_logistic,
    rng: Optional[RandomType] = None,
    return_pp: bool = False,
    return_ll: bool = False,
    prior_kwargs: Dict = {},
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
        pp = generate_response_linear(X, theta, theta_space, rng=rng)

    # Return information
    if return_pp and return_ll:
        return lpos, pp, ll_pointwise
    elif return_pp:
        return lpos, pp
    elif return_ll:
        return lpos, ll_pointwise
    else:
        return lpos


def log_posterior_logistic(
    theta_tr: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    theta_space: ThetaSpace,
    log_prior: PriorType = log_prior_linear_logistic,
    rng: Optional[RandomType] = None,
    return_pp: bool = False,
    return_ll: bool = False,
    prior_kwargs: Dict = {},
):
    if rng is None:
        rng = np.random.default_rng()

    # Compute log-prior
    lp = log_prior(theta_tr, X, y, theta_space, **prior_kwargs)

    if not np.isfinite(lp):
        if return_pp and return_ll:
            return (-np.inf, np.full_like(y, -1.0),
                    np.full_like(y, -1), np.full_like(y, -np.inf))
        elif return_pp:
            return -np.inf, np.full_like(y, -1.0), np.full_like(y, -1)
        elif return_ll:
            return -np.inf, np.full_like(y, -np.inf)
        else:
            return -np.inf

    # Compute log-likelihood (and possibly pointwise log-likelihood)
    if return_ll:
        ll, ll_pointwise = log_likelihood_logistic(
            theta_tr, X, y, theta_space, return_ll)
    else:
        ll = log_likelihood_logistic(theta_tr, X, y, theta_space, return_ll)

    # Compute log-posterior
    lpos = lp + ll

    # Compute posterior predictive samples
    if return_pp:
        theta = theta_space.backward(theta_tr)
        pp_p, pp_y = generate_response_logistic(
            X, theta, theta_space, return_prob=True, rng=rng)

    # Return information
    if return_pp and return_ll:
        return lpos, pp_p, pp_y, ll_pointwise
    elif return_pp:
        return lpos, pp_p, pp_y
    elif return_ll:
        return lpos, ll_pointwise
    else:
        return lpos


#
# Linear and logistic regression models for pymc
#

def make_model_linear_pymc(
    X,
    y,
    theta_space,
    *,
    b0,
    g,
    eta,
    prior_p=None,
):
    n, N = X.shape
    ts = theta_space
    p_max = theta_space.p_max

    with pm.Model() as model:
        X_pm = pm.MutableData('X_obs', X)

        if ts.include_p:
            p_cat = pm.Categorical(
                'p_cat',
                p=list(prior_p.values())
            )
            p = pm.Deterministic(ts.names[ts.p_idx], p_cat + 1)
        else:
            p = p_max

        alpha0_and_log_sigma = pm.Flat(
            ts.names[ts.alpha0_idx] + "_and_" + ts.names_ttr[ts.sigma2_idx],
            shape=2
        )

        alpha0 = pm.Deterministic(
            ts.names[ts.alpha0_idx], alpha0_and_log_sigma[0])

        log_sigma = alpha0_and_log_sigma[1]
        sigma = pm.math.exp(log_sigma)
        sigma2 = pm.Deterministic(ts.names[ts.sigma2_idx], sigma**2)

        tau_unordered = pm.Uniform(
            ts.names[ts.tau_idx_grouped] + "_unordered",
            0.0,
            1.0,
            shape=p_max,
        )

        tau = pm.Deterministic(
            ts.names[ts.tau_idx_grouped],
            at.sort(tau_unordered)
        )
        tau_red = tau[:p]

        idx = np.abs(ts.grid - tau_red[:, np.newaxis]).argmin(1)
        X_tau = X_pm[:, idx]

        G_tau = pm.math.matrix_dot(X_tau.T, X_tau)
        G_tau = (G_tau + G_tau.T)/2.  # Enforce symmetry
        G_tau_reg = G_tau + eta * \
            at.max(at.nlinalg.eigh(G_tau)[0])*at.eye(p)

        def beta_lprior(value, p, log_sigma, sigma2, G_tau_reg):
            b = (value - b0)[:p]
            G_log_det = pm.math.logdet(G_tau_reg)

            return (0.5*G_log_det
                    - p*log_sigma
                    - pm.math.matrix_dot(b.T, G_tau_reg, b)/(2.*g*sigma2))

        beta_unbounded = pm.DensityDist(
            ts.names[ts.beta_idx_grouped] + "_unbounded",
            p,
            log_sigma,
            sigma2,
            G_tau_reg,
            logp=beta_lprior,
            shape=3
        )

        # Restrict values of beta
        if ts.beta_range is not None:
            beta = pm.Deterministic(
                ts.names[ts.beta_idx_grouped],
                pm.math.clip(
                    beta_unbounded,
                    ts.beta_range[0],
                    ts.beta_range[1]
                )
            )

        else:
            beta = pm.Deterministic(
                ts.names[ts.beta_idx_grouped],
                beta_unbounded
            )

        beta_red = beta[:p]
        expected_obs = alpha0 + pm.math.matrix_dot(X_tau, beta_red)

        y_obs = pm.Normal('y_obs', mu=expected_obs, sigma=sigma, observed=y)

    return model


def make_model_logistic_pymc(
    X,
    y,
    theta_space,
    *,
    b0,
    g,
    eta,
    prior_p=None,
):
    n, N = X.shape
    ts = theta_space
    p_max = theta_space.p_max

    with pm.Model() as model:
        X_pm = pm.MutableData('X_obs', X)

        if ts.include_p:
            p_cat = pm.Categorical(
                'p_cat',
                p=list(prior_p.values())
            )
            p = pm.Deterministic(ts.names[ts.p_idx], p_cat + 1)
        else:
            p = p_max

        alpha0_and_log_sigma = pm.Flat(
            ts.names[ts.alpha0_idx] + "_and_" + ts.names_ttr[ts.sigma2_idx],
            shape=2
        )

        alpha0 = pm.Deterministic(
            ts.names[ts.alpha0_idx], alpha0_and_log_sigma[0])

        log_sigma = alpha0_and_log_sigma[1]
        sigma = pm.math.exp(log_sigma)
        sigma2 = pm.Deterministic(ts.names[ts.sigma2_idx], sigma**2)

        tau_unordered = pm.Uniform(
            ts.names[ts.tau_idx_grouped] + "_unordered",
            0.0,
            1.0,
            shape=p_max,
        )

        tau = pm.Deterministic(
            ts.names[ts.tau_idx_grouped],
            at.sort(tau_unordered)
        )
        tau_red = tau[:p]

        idx = np.abs(ts.grid - tau_red[:, np.newaxis]).argmin(1)
        X_tau = X_pm[:, idx]

        G_tau = pm.math.matrix_dot(X_tau.T, X_tau)
        G_tau = (G_tau + G_tau.T)/2.  # Enforce symmetry
        G_tau_reg = G_tau + eta * \
            at.max(at.nlinalg.eigh(G_tau)[0])*at.eye(p)

        def beta_lprior(value, p, log_sigma, sigma2, G_tau_reg):
            b = (value - b0)[:p]
            G_log_det = pm.math.logdet(G_tau_reg)

            return (0.5*G_log_det
                    - p*log_sigma
                    - pm.math.matrix_dot(b.T, G_tau_reg, b)/(2.*g*sigma2))

        beta_unbounded = pm.DensityDist(
            ts.names[ts.beta_idx_grouped] + "_unbounded",
            p,
            log_sigma,
            sigma2,
            G_tau_reg,
            logp=beta_lprior,
            shape=3
        )

        # Restrict values of beta
        if ts.beta_range is not None:
            beta = pm.Deterministic(
                ts.names[ts.beta_idx_grouped],
                pm.math.clip(
                    beta_unbounded,
                    ts.beta_range[0],
                    ts.beta_range[1]
                )
            )

        else:
            beta = pm.Deterministic(
                ts.names[ts.beta_idx_grouped],
                beta_unbounded
            )

        beta_red = beta[:p]

        px = pm.Deterministic(
            'p_star',
            pm.math.invlogit(alpha0 + pm.math.matrix_dot(X_tau, beta_red))
        )

        y_obs = pm.Bernoulli('y_obs', p=px, observed=y)

    return model
