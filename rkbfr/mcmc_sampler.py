# encoding: utf-8

"""
File with the source code of the MCMC samplers implemented, for functional
linear and logistic regression.

There are two sampler versions: one with `emcee` (the preferred one)
and one with `pymc`.
"""

from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from itertools import permutations
from multiprocessing import Pool
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import arviz as az
import emcee
import numpy as np
import pymc as pm
from pymc.step_methods.arraystep import BlockedStep
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.stats import beta as beta_dist
from scipy.stats import norm
from skfda.representation import FData
from skfda.representation.basis import FDataBasis
from skfda.representation.grid import FDataGrid
from sklearn.base import (BaseEstimator, ClassifierMixin, RegressorMixin,
                          TransformerMixin)
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.validation import check_is_fitted

from .autocorr import integrated_time
from .bayesian_model import (PriorType, RandomType, ThetaSpace, generate_pp,
                             log_posterior_linear, log_posterior_logistic,
                             log_prior_linear_logistic, make_model_linear_pymc,
                             make_model_logistic_pymc, point_estimate,
                             point_predict)
from .mle import compute_mle
from .utils import (HandleLogger, apply_threshold, check_random_state,
                    fdata_to_numpy, mode_fn, relabel_sample)

DataType = Union[
    FData,
    np.ndarray,
]

StepType = Callable[[Any], BlockedStep]

ModelType = Callable[
    # X, y, theta_space, **kwargs, rng
    [np.ndarray, np.ndarray, ThetaSpace, Any, Optional[RandomType]],
    pm.Model
]


class _BayesianRKHSFRegression(
    ABC,
    BaseEstimator,
    TransformerMixin
):
    """Bayesian functional regression based on a RKHS model.

    This estimator can be used both as an end-to-end regressor/classifier
    (via the 'predict' method) or as a variable selection
    procedure (via the 'transform' method).

    See https://github.com/antcc/tfm for more details on the model.
    """

    default_point_estimates = ['mean', 'median', 'mode']
    sd_beta_random: float = 10.0

    @abstractmethod
    def kind():
        ...

    @abstractmethod
    def fit(self, X, y):
        """Fit the estimator."""
        # Initialization tasks
        self.rng_ = check_random_state(self.random_state)
        self.n_jobs_ = self.n_jobs if self.n_jobs > 0 else os.cpu_count()
        self._mean_alpha0 = y.mean()
        self.mle_ = self.mle_precomputed
        if self.mle_ is not None:
            self._theta_space_fixed = self.theta_space.copy_p_fixed()
        else:
            self._theta_space_fixed = None
        if self.theta_space.include_p:
            self._n_components_default_pe = {}

    @abstractmethod
    def mean_acceptance(self):
        ...

    @abstractmethod
    def get_trace(self, burn, thin, flat):
        ...

    def predict(
        self,
        X: DataType,
        strategy='mode',
        bw='experimental'
    ) -> np.ndarray:
        """Predict using the posterior distribution.

        The parameter 'strategy' can be {'mean', 'median', 'mode',
        'posterior_mean'} or a callable representing a point estimate."""
        check_is_fitted(self)
        X = self._argcheck_X(X)
        ts = self.theta_space

        if callable(strategy) or strategy in self.default_point_estimates:
            y_pred, theta_hat = point_predict(
                X,
                self.idata_,
                ts,
                strategy,
                kind=self.kind,
                skipna=ts.include_p,
                th=self.threshold,
                bw=bw
            )

            if not callable(strategy) and ts.include_p:
                self._n_components_default_pe[strategy] = theta_hat[ts.p_idx]

        elif self.kind == 'linear' and strategy == 'posterior_mean':
            pp_test = generate_pp(
                self.idata_,
                X,
                ts,
                self.thin_pp,
                kind=self.kind,
                rng=self.rng_,
                verbose=self.verbose,
            )
            y_pred = pp_test.mean(axis=(0, 1))
        elif self.kind == 'logistic' and strategy in self.default_strategies:
            pp_test_p, pp_test_y = generate_pp(
                self.idata_,
                X,
                ts,
                self.thin_pp,
                kind=self.kind,
                rng=self.rng_,
                verbose=self.verbose,
            )
            if strategy == 'posterior_mean':
                y_pred = pp_test_p.mean(axis=(0, 1))
            elif strategy == 'posterior_vote':
                y_pred = pp_test_y.mean(axis=(0, 1))
            y_pred = apply_threshold(y_pred, self.threshold)
        else:
            raise ValueError(
                "Incorrect value for parameter 'strategy'.")

        return y_pred

    def transform(self, X: DataType, pe='mode', bw='experimental') -> DataType:
        """Use the estimator to transform the data set, performing
        variable selection according to the posterior of the time instants."""
        check_is_fitted(self)
        ts = self.theta_space

        if callable(pe) or pe in self.default_point_estimates:
            theta_hat = point_estimate(
                self.idata_,
                pe,
                ts.names,
                skipna=ts.include_p,
                bw=bw
            )

            n_comp = theta_hat[ts.p_idx] if ts.include_p else ts.p_max
            tau_hat = theta_hat[ts.tau_idx][:ts.round_p(n_comp)]

            if not callable(pe) and ts.include_p:
                self._n_components_default_pe[pe] = n_comp
        else:
            raise ValueError("Incorrect value for parameter 'pe'.")

        idx_hat = np.abs(ts.grid - tau_hat[:, np.newaxis]).argmin(1)
        idx_hat.sort()

        X_red = self._variable_selection(X, idx_hat)

        return X_red

    def score(self, X: DataType, y: np.ndarray, strategy='mode') -> float:
        """Score of the estimator."""
        X, y = self._argcheck_X_y(X, y)

        y_pred = self.predict(X, strategy)

        if self.kind == 'linear':
            score = r2_score(y, y_pred)
        else:
            score = accuracy_score(y, y_pred)

        return score

    def summary(self, stats='stats', bw='experimental', **kwargs):
        """Get a summary of the posterior distribution and the
        MCMC statistics."""
        check_is_fitted(self)

        skipna = self.theta_space.include_p
        additional_stats = {
            "min": np.nanmin if skipna else np.min,
            "max": np.nanmax if skipna else np.max,
            "median": np.nanmedian if skipna else np.median,
            "mode": lambda x: mode_fn(x, skipna=skipna, bw=bw),
        }

        return az.summary(
            self.idata_,
            extend=True,
            kind=stats,
            var_names=self.theta_space.names,
            skipna=skipna,
            labeller=self.theta_space.labeller,
            stat_funcs=additional_stats,
            **kwargs
        )

    def n_components(self, strategy, bw="experimental"):
        """Get how many components are actually used in each prediction
        strategy. Only relevant in the P_FREE case."""
        check_is_fitted(self)
        ts = self.theta_space

        if not ts.include_p:
            n_comp = ts.p_max
        elif 'posterior' in strategy:
            n_comp = np.max(
                self.idata_.posterior[ts.names[ts.p_idx]].to_numpy())
        elif (callable(strategy)
                or strategy not in self._n_components_default_pe.keys()):
            theta_hat = point_estimate(
                self.idata_,
                strategy,
                ts.names,
                skipna=ts.include_p,
                bw=bw
            )
            n_comp = theta_hat[ts.p_idx]
        else:
            n_comp = self._n_components_default_pe[strategy]

        return ts.round_p(n_comp)

    def get_idata(self):
        """Get the InferenceData object representing the posterior."""
        check_is_fitted(self)

        return self.idata_

    def total_samples(self):
        """Get the total number of samples."""
        check_is_fitted(self)

        n_samples = (self.idata_.posterior.dims["chain"]
                     * self.idata_.posterior.dims["draw"])

        return n_samples

    def autocorrelation_times(self, burn=None, thin=None):
        """Compute the autocorrelation times."""
        check_is_fitted(self)

        if burn is None:
            burn = self.burn_
        if thin is None:
            thin = self.thin

        x = self.get_trace(burn, thin)
        x = x.swapaxes(0, 1)  # (draw, chain, n_dim)

        with HandleLogger(self.verbose):
            if self.theta_space.include_p:
                n_dim = self.theta_space.n_dim
                autocorr = np.zeros(n_dim)

                for dim in range(n_dim):
                    x_dim = x[:, :, dim]
                    # Remove chains where all values are NaN
                    x_dim = x_dim[:, ~np.isnan(x_dim).all(axis=0)]
                    # Replace NaN with 0.0 for computing autocorrelation
                    x_dim = np.nan_to_num(x_dim, nan=0.0)
                    # Compute autocorrelation
                    autocorr[dim] = integrated_time(x_dim, quiet=True)
            else:
                autocorr = integrated_time(x, quiet=True)

        return thin*autocorr

    def _get_prior_kwargs(self, X, y):
        ts = self.theta_space

        if isinstance(self.b0, str):
            if self.b0 == 'mle':
                if self.mle_ is None:
                    self.mle_, self._theta_space_fixed = \
                        self._compute_mle(X, y)
                b0 = self.mle_[self._theta_space_fixed.beta_idx]
            elif self.b0 == 'random':
                b0 = self.rng_.normal(
                    0, self.sd_beta_random, ts.p_max
                )
            elif self.b0 == 'zero':
                b0 = np.zeros(ts.p_max)
            else:
                raise ValueError(
                    "%s is not a valid string for b0." %
                    self.b0
                )
        elif isinstance(self.b0, np.ndarray):
            b0 = self.b0
        else:
            raise ValueError("Invalid value for b0.")

        prior_kwargs = {
            "b0": b0,
            "g": self.g,
            "eta": self.eta,
        }

        if ts.include_p:
            if self.prior_p is None:
                raise ValueError(
                    "Expected a dictionary for the prior probabilities "
                    "of the parameter 'p'."
                )
            if (len(self.prior_p) != ts.p_max):
                raise ValueError(
                    "Must provide a prior probability for each "
                    "possible value of p."
                )
            if sum(self.prior_p.values()) != 1.0:
                raise ValueError(
                    "The prior probabilities of 'p' must add up to one."
                )
            prior_kwargs["prior_p"] = dict(sorted(self.prior_p.items()))

        if self.prior_tau is not None:
            if isinstance(self.prior_tau, np.ndarray):
                tau_params = self.prior_tau
            elif self.prior_tau == 'auto':
                tau_params = self._get_tau_params(X)
            else:
                raise ValueError("Invalid value for prior_tau.")

            prior_kwargs["tau_params"] = tau_params

        return prior_kwargs

    def _get_tau_params(self, X):
        ts = self.theta_space

        def f(x):
            """Variance of Beta(a, b)."""
            a, b = x
            return (a*b/((a+b)**2*(a+b+1)))

        tau_params = np.zeros((ts.p_max, 2))
        obj = X.var(axis=0)
        obj = obj/np.sum(obj)

        for i in range(ts.p_max):
            M = np.argmax(obj)/len(ts.grid)

            params = minimize(
                f,
                x0=[1.01, 1.01],
                constraints=[
                    {'type': 'eq',
                     'fun': lambda x: (1 - M)*x[0] - M*x[1] + 2*M - 1},
                    {'type': 'eq',
                     'fun': lambda x: (
                         beta_dist.ppf(0.75, x[0], x[1])
                         - beta_dist.ppf(0.25, x[0], x[1])
                         - 1./(ts.p_max**2))}],
                bounds=[(1.01, 50), (1.01, 50)]
            ).x

            tau_params[i] = params

            # Subtract normalized pdf to normalized var
            pdf = beta_dist.pdf(ts.grid, *params)
            pdf = pdf/np.sum(pdf)
            pdf[0] = pdf[-1] = 1.0
            obj -= pdf

        return tau_params

    def _discard_components(self, trace):
        # Set unused values to NaN on each sample
        _ = [[[self.theta_space.set_unused_nan(theta, inplace=True)
               for theta in draw] for draw in trace]]

    def _tidy_idata(self, idata):
        ts = self.theta_space

        if self.verbose > 0:
            print(f"[{self.__class__.__name__}] Discarding the "
                  f"first {self.burn_} samples...")

        # Burn-in and thinning
        idata_new = idata.sel(draw=slice(self.burn_, None, self.thin))

        # Set p values to integers
        if ts.include_p:
            idata_new.posterior[ts.names[ts.p_idx]] = \
                np.rint(idata_new.posterior[ts.names[ts.p_idx]]).astype(int)

        if self.kind == 'logistic' and 'posterior_predictive' in idata:
            idata_new.posterior_predictive['y_star'] = \
                np.rint(idata_new.posterior_predictive['y_star']).astype(int)

        return idata_new

    def _variable_selection(self, X: DataType, idx: np.ndarray) -> DataType:
        grid = self.theta_space.grid
        N = len(grid)

        if isinstance(X, np.ndarray):
            if X.shape[1] != N:
                raise ValueError(
                    "Data must be compatible with the specified "
                    "grid (i.e. 'self.theta_space.grid').")

            X_red = X[:, idx]
        elif isinstance(X, FDataBasis):
            X_data = X.to_grid(grid_points=grid).data_matrix
            X_red = FDataGrid(X_data[:, idx], grid[idx])
        elif isinstance(X, FDataGrid):
            X_data = X.data_matrix
            X_red = FDataGrid(X_data[:, idx], grid[idx])
        else:
            raise ValueError('Data type not supported for X.')

        return X_red

    def _compute_mle(self, X, y):
        if self.verbose > 1:
            print(f"[{self.__class__.__name__}] Computing MLE...")
        ts_fixed = self.theta_space.copy_p_fixed()

        mle, _ = compute_mle(
            X,
            y,
            ts_fixed,
            kind=self.kind,
            method=self.mle_method,
            strategy=self.mle_strategy,
            n_reps=self.n_reps_mle,
            n_jobs=self.n_jobs_,
            rng=self.rng_
        )

        return mle, ts_fixed

    def _argcheck_X(self, X: DataType) -> np.ndarray:
        grid = self.theta_space.grid
        return fdata_to_numpy(X, grid)

    def _argcheck_X_y(  # noqa: N802
        self,
        X: DataType,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if (len(y) != len(X)):
            raise ValueError(
                "The number of samples on independent and "
                "dependend variables should be the same."
            )

        X = self._argcheck_X(X)

        return X, y


class _BayesianRKHSFRegressionEmcee(_BayesianRKHSFRegression):
    """Bayesian functional regression based on a RKHS model.

    It uses 'emcee' [1] as the underlying MCMC algorithm for
    approximating the posterior distribution.

    Parameters
    ----------
    theta_space : ThetaSpace
        Object representing the finite-dimensional parameter space.
    n_walkers : int
        Number of walkers (chains).
    n_iter : int
        Number of iterations.
    log_prior : Optional[PriorType]
        A function representing the joint log-prior distribution.
    prior_kwargs : Dict
        Additional arguments for the log-prior distribution.
    b0 : Union[str, np.ndarray]
        Hyperparameter of the default prior distribution.
    g : float
        Hyperparameter of the default prior distribution.
    eta : float
        Hyperparameter of the default prior distribution.
    prior_p : Optional[Dict]
        Prior probabilities for each value of p (P_FREE).
    prior_tau : Optional[Union[str, np.ndarray]]
        If None, tau defaults to a uniform prior. If 'auto' or a np.ndarray,
        set a beta distribution on tau, with automatic or pre-specified
        shape parameters.
    n_iter_warmup : int
        Number of initial iterations (only as warm-up; they
        are discarded afterwards).
    initial_state : Union[str, np.ndarray]
        Initial state of the chains. Can be an approximation of the MLE ('mle'),
        random points ('random') or a given np.ndarray.
    frac_random : float
        Fraction of the chains that get a random initial point regardless
        of the value of 'initial_state'.
    relabel_strategy : str
        Strategy for relabeling the samples in order to avoid label
        switching. It can be one of {'auto', 'pivot', 'beta', 'tau'}.
    moves : Optional[Sequence[emcee.moves.Move]]
        Moves for the emcee sampler.
    compute_pp : bool
        Whether to compute the posterior predictive distribution "on-line".
    compute_ll : bool
        Whether to compute the point-wise log-likelihood of the generated
        samples.
    thin : int
        Number of consecutive samples to skip from the chains.
    thin_pp : int
        Number of consecutive samples to skip from the chains when computing
        the posterior predictive distribution.
    burn : int
        Number of samples to discard from the chains.
    burn_relative : int
        Number of samples to discard from the chains, relative to the
        autocorrelation times.
    mle_precomputed : Optional[np.ndarray]
        Whether the MLE is supplied as a precomputed value.
    mle_method : str
        Method to use when approximating the MLE ('local' or 'global').
    mle_strategy : str
        Local optimization method for MLE approximation.
    n_reps_mle : int
        Number of independent runs of the MLE approximation algorithm.
    threshold : float
        Threshold for generating predictions in logistic regression. Ignored
        in linear regression.
    n_jobs : int
        Number of cores ot use.
    verbose : int
        Verbosity level.
    progress_notebook : bool
        Whether to show a progress bar in notebook style.
    progress_kwargs : Optional[Dict]
        Additional arguments for the progress bar.
    random_state : Optional[RandomType]
        Random state for reproducibility.

    [1] https://emcee.readthedocs.io/
    """

    sd_beta: float = 1.0
    sd_tau: float = 0.2
    sd_alpha0: float = 1.0
    param_sigma2: float = 2.0  # Shape parameter in inv_gamma distribution
    sd_sigma2: float = 1.0

    def __init__(
        self,
        theta_space: ThetaSpace = ThetaSpace(),
        n_walkers: int = 100,
        n_iter: int = 1000,
        *,
        log_prior: Optional[PriorType] = None,
        prior_kwargs: Dict = {},
        b0: Union[str, np.ndarray] = 'mle',
        g: float = 5.0,
        eta: float = 1.0,
        prior_p: Optional[Dict] = None,
        prior_tau: Optional[Union[str, np.ndarray]] = None,
        n_iter_warmup: int = 100,
        initial_state: Union[str, np.ndarray] = 'mle',
        frac_random: float = 0.3,
        relabel_strategy: str = 'auto',
        moves: Optional[Sequence[emcee.moves.Move]] = None,
        compute_pp: bool = False,
        compute_ll: bool = False,
        thin: int = 1,
        thin_pp: int = 1,
        burn: int = 100,
        burn_relative: int = 3,
        mle_precomputed: Optional[np.ndarray] = None,
        mle_method: str = 'L-BFGS-B',
        mle_strategy: str = 'global',
        n_reps_mle: int = 4,
        threshold: float = 0.5,
        n_jobs: int = 1,
        verbose: int = 0,
        progress_notebook: bool = False,
        progress_kwargs: Optional[Dict] = None,
        random_state: Optional[RandomType] = None
    ) -> _BayesianRKHSFRegressionEmcee:
        self.theta_space = theta_space
        self.n_walkers = n_walkers
        self.n_iter = n_iter
        self.log_prior = log_prior
        self.prior_kwargs = prior_kwargs
        self.b0 = b0
        self.g = g
        self.eta = eta
        self.prior_p = prior_p
        self.prior_tau = prior_tau
        self.n_iter_warmup = n_iter_warmup
        self.initial_state = initial_state
        self.frac_random = frac_random
        self.relabel_strategy = relabel_strategy
        self.moves = moves
        self.compute_pp = compute_pp
        self.compute_ll = compute_ll
        self.thin = thin
        self.thin_pp = thin_pp
        self.burn = burn
        self.burn_relative = burn_relative
        self.mle_precomputed = mle_precomputed
        self.mle_method = mle_method
        self.mle_strategy = mle_strategy
        self.n_reps_mle = n_reps_mle
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.progress_notebook = progress_notebook
        self.progress_kwargs = progress_kwargs
        self.random_state = random_state

    def fit(
        self,
        X: DataType,
        y: np.ndarray
    ) -> _BayesianRKHSFRegressionEmcee:
        X, y = self._argcheck_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        super().fit(X, y)
        ts = self.theta_space

        # Prior

        if self.log_prior is None:
            log_prior = log_prior_linear_logistic
            prior_kwargs = self._get_prior_kwargs(X, y)
        elif callable(self.log_prior):
            log_prior = self.log_prior
            prior_kwargs = self.prior_kwargs
        else:
            raise ValueError("Invalid value for log_prior.")

        # Posterior

        if self.kind == 'linear':
            log_posterior = log_posterior_linear
        else:
            log_posterior = log_posterior_logistic

        posterior_args = (X, y)

        posterior_kwargs = {
            "theta_space": ts,
            "rng": self.rng_,
            "return_ll": self.compute_ll,
            "return_pp": self.compute_pp,
            "log_prior": log_prior,
            "prior_kwargs": prior_kwargs,
        }

        # Initial state of walkers

        if isinstance(self.initial_state, str):
            if self.initial_state == 'mle':
                if self.mle_ is None:
                    self.mle_, self._theta_space_fixed = \
                        self._compute_mle(X, y)
                n_random = int(self.frac_random*self.n_walkers)
                n_around = self.n_walkers - n_random
                initial_state = self._weighted_initial_guess_around_value(
                    self.mle_, n_random, n_around)
            elif self.initial_state == 'random':
                initial_state = self._initial_guess_random(self.n_walkers)
            else:
                raise ValueError(
                    "%s is not a valid string for initial_state." %
                    self.initial_state
                )
        elif isinstance(self.initial_state, np.ndarray):
            initial_state = self.initial_state
        else:
            raise ValueError("Invalid value for initial_state.")

        # Run sampler

        with Pool(self.n_jobs_) as pool:
            self._sampler = emcee.EnsembleSampler(
                self.n_walkers,
                ts.n_dim,
                log_posterior,
                pool=pool,
                args=posterior_args,
                kwargs=posterior_kwargs,
                moves=self.moves
            )

            # Set initial random state for the sampler
            random_seed = self.rng_.integers(2**32)
            self._sampler.random_state = \
                np.random.RandomState(random_seed).get_state()

            if self.verbose > 1:
                print(f"[{self.__class__.__name__}] MCMC warmup iterations...")

            # Warmup iterations
            if self.n_iter_warmup > 0:
                state_init_sampler = self._sampler.run_mcmc(
                    initial_state,
                    self.n_iter_warmup,
                    progress=False,
                    store=False)

                self._sampler.reset()
            else:
                state_init_sampler = initial_state

            if self.verbose > 0:
                progress = 'notebook' if self.progress_notebook else True
                if self.progress_kwargs is None:
                    self.progress_kwargs = \
                        {"desc": f"[{self.__class__.__name__}] MCMC"}
            else:
                progress = False

            # MCMC run
            self._sampler.run_mcmc(
                state_init_sampler,
                self.n_iter,
                progress=progress,
                progress_kwargs=self.progress_kwargs)

        # Transform back parameters and discard unused ones if applicable

        self._transform_trace()

        # Calculate burn-in

        self.burn_ = self._compute_burn()

        # Relabel elements to correct label switching

        self._relabel_trace(log_posterior, posterior_args, posterior_kwargs)

        # Get data (after burn-in and thinning)

        self.idata_ = self._emcee_to_idata()

        return self

    def mean_acceptance(self):
        check_is_fitted(self)

        return np.mean(self._sampler.acceptance_fraction)

    def get_trace(self, burn=None, thin=None, flat=False):
        """Get the trace of the sampler, with shape (n_iter, n_walkers, n_dim)."""
        check_is_fitted(self)

        if burn is None:
            burn = self.burn_
        if thin is None:
            thin = self.thin

        trace = np.copy(self._sampler.get_chain(discard=burn, thin=thin))
        trace = trace.swapaxes(0, 1)  # (chain, draw, n_dim)

        if flat:
            trace = trace.reshape(-1, trace.shape[-1])  # All chains combined

        return trace

    def _transform_trace(self):
        ts = self.theta_space
        trace = self._sampler.get_chain()

        # Transform back parameters
        trace[:, :, ts.tau_idx] = ts.tau_ttr.backward(
            trace[:, :, ts.tau_idx])
        trace[:, :, ts.sigma2_idx] = ts.sigma2_ttr.backward(
            trace[:, :, ts.sigma2_idx])

        if ts.include_p:
            self._discard_components(trace)

    def _relabel_trace(self, log_posterior=None, lp_args=None, lp_kwargs=None):
        chain = self._sampler.backend.chain[self.burn_:, :, :]
        chain_shape = chain.shape
        ts = self.theta_space

        if self.relabel_strategy == 'pivot':
            chain_flat = chain.reshape(-1, ts.n_dim)

            # Get reference theta (MAP)
            ref_idx = np.argmax(
                [log_posterior(x, *lp_args, **lp_kwargs) for x in chain_flat])
            ref_theta = chain_flat[ref_idx, :]

            # Get all permutations of β/τ
            all_perm = np.zeros((
                np.math.factorial(ts.p_max), *chain_flat.shape))
            for i, perm in enumerate(permutations(range(ts.p_max))):
                pp = np.array(perm)
                idx = np.concatenate((pp, pp + ts.p_max, [-2, -1]))
                all_perm[i] = chain_flat[:, idx]

            # For each permutation, get distance between each iteration
            # and the reference value
            all_perm_dist = np.linalg.norm(all_perm - ref_theta, axis=2)

            # For each iteration find the permutation that minimizes the distance
            idx_min = np.argmin(all_perm_dist, axis=0)

            # Select the corresponding permutations
            new_chain = np.take_along_axis(
                all_perm, idx_min[None, :, None], axis=0)

            # Overwrite trace with the original shape
            chain[:, :, :] = new_chain.reshape(*chain_shape)

        else:
            # Decide whether to order by β or τ
            if self.relabel_strategy == 'beta':
                order_by_beta = True
            elif self.relabel_strategy == 'tau':
                order_by_beta = False
            else:  # 'auto'
                _, beta, tau, _, _ = ts.slice_params(chain, clip=False)
                beta_flat = beta.reshape(-1, ts.p_max)
                tau_flat = tau.reshape(-1, ts.p_max)

                # Rescale parameters to common units
                beta_scale = np.nanmean(
                    norm.cdf(
                        beta_flat,
                        loc=np.nanmean(beta_flat),
                        scale=np.nanstd(beta_flat)),
                    axis=0)
                tau_scale = np.nanmean(
                    norm.cdf(
                        tau_flat,
                        loc=np.nanmean(tau_flat),
                        scale=np.nanstd(tau_flat)),
                    axis=0)

                # Look for the maximum pairwise distance
                if ts.p_max > 1:
                    pdist_beta_max = np.max(pdist(beta_scale.reshape(-1, 1)))
                    pdist_tau_max = np.max(pdist(tau_scale.reshape(-1, 1)))
                else:
                    pdist_beta_max = beta_scale[0]
                    pdist_tau_max = tau_scale[0]

                order_by_beta = True if pdist_beta_max > pdist_tau_max else False

            # Impose identifiability constraints
            relabel_sample(chain, self.theta_space, order_by_beta)

    def _compute_burn(self):
        if self.burn is None:
            autocorr = self.autocorrelation_times(burn=0, thin=1)
            max_autocorr = np.nanmax(autocorr)
            if np.isfinite(max_autocorr):
                burn = int(self.burn_relative*max_autocorr)
            else:
                burn = self.n_iter//10
        else:
            burn = self.burn

        return burn

    def _emcee_to_idata(self):
        ts = self.theta_space
        names = ts.names
        blob_names = []
        blob_groups = []
        dims = {f"{names[ts.beta_idx_grouped]}": [ts.dim_name],
                f"{names[ts.tau_idx_grouped]}": [ts.dim_name],
                "X_obs": ["observation", "value"],
                "y_obs": ["observation"]}

        if self.kind == 'linear':
            pp_names = ["y_star"] if self.compute_pp else []
        else:
            pp_names = ["p_star", "y_star"] if self.compute_pp else []
        n_pp = len(pp_names)

        if n_pp > 0:
            new_vars = {}
            for name in pp_names:
                new_vars[name] = ["prediction"]

            blob_names = pp_names
            blob_groups = n_pp*["posterior_predictive"]
            dims = {**dims, **new_vars}
        if self.compute_ll:
            blob_names += ["y_obs"]
            blob_groups += ["log_likelihood"]

        if len(blob_names) == 0:  # No blobs
            blob_names = None
            blob_groups = None

        slices = [
            ts.beta_idx,
            ts.tau_idx,
            ts.alpha0_idx,
            ts.sigma2_idx
        ]

        if ts.include_p:
            slices = [ts.p_idx] + slices

        idata = az.from_emcee(
            self._sampler,
            var_names=names,
            slices=slices,
            arg_names=["X_obs", "y_obs"],
            arg_groups=["constant_data", "observed_data"],
            blob_names=blob_names,
            blob_groups=blob_groups,
            dims=dims
        )

        idata = self._tidy_idata(idata)

        return idata

    def _initial_guess_random(self, n_samples):
        p = self.theta_space.p_max
        rng = self.rng_

        beta_init = self.sd_beta_random * \
            rng.standard_normal(size=(n_samples, p))
        tau_init = rng.uniform(size=(n_samples, p))
        alpha0_init = self._mean_alpha0 + self.sd_alpha0 * \
            rng.standard_normal(size=(n_samples, 1))
        sigma2_init = 1. / \
            rng.standard_gamma(self.param_sigma2, size=(n_samples, 1))

        init = np.hstack((
            beta_init,
            tau_init,
            alpha0_init,
            sigma2_init
        ))

        if self.theta_space.include_p:
            if self.prior_p is not None:
                prior_p = list(self.prior_p.values())
            else:
                prior_p = np.full(p, 1./p)

            p_init = rng.choice(
                np.arange(p) + 1,
                size=n_samples,
                p=prior_p
            )

            init = np.vstack((
                p_init,
                init.T
            )).T

        init_tr = self.theta_space.forward(init)

        return init_tr

    def _initial_guess_around_value(self, value, n_samples):
        """'value' is in the original space and does not include 'p'."""
        rng = self.rng_
        p = self.theta_space.p_max

        beta_jitter = self.sd_beta * \
            rng.standard_normal(size=(n_samples, p))
        tau_jitter = self.sd_tau*rng.standard_normal(size=(n_samples, p))
        alpha0_jitter = self.sd_alpha0 * \
            rng.standard_normal(size=(n_samples, 1))
        sigma2_jitter = self.sd_sigma2 * \
            rng.standard_normal(size=(n_samples, 1))

        jitter = np.hstack((
            beta_jitter,
            tau_jitter,
            alpha0_jitter,
            sigma2_jitter
        ))

        value_jitter = value[np.newaxis, :] + jitter

        if self.theta_space.include_p:
            p_init = rng.choice(
                np.arange(self.theta_space.p_max) + 1,
                size=n_samples,
                p=list(self.prior_p.values())
            )

            value_jitter = np.vstack((
                p_init,
                value_jitter.T
            )).T

        value_jitter = self.theta_space.clip_bounds(value_jitter)
        value_jitter_tr = self.theta_space.forward(value_jitter)

        return value_jitter_tr

    def _weighted_initial_guess_around_value(self, value, n_random, n_around):
        if n_random > 0:
            init_1 = self._initial_guess_random(n_random)
        else:
            init_1 = None

        if n_around > 0:
            init_2 = self._initial_guess_around_value(value, n_around)
        else:
            init_2 = None

        if init_1 is None:
            init = init_2
        elif init_2 is None:
            init = init_1
        else:
            init = np.vstack((init_1, init_2))

        self.rng_.shuffle(init)

        return init


class BFLinearEmcee(
    _BayesianRKHSFRegressionEmcee,
    RegressorMixin
):
    kind = 'linear'
    default_strategies = ['posterior_mean']


class BFLogisticEmcee(
    _BayesianRKHSFRegressionEmcee,
    ClassifierMixin
):
    kind = 'logistic'
    default_strategies = ['posterior_mean', 'posterior_vote']


class _BayesianRKHSFRegressionPymc(_BayesianRKHSFRegression):
    """Bayesian functional regression based on a RKHS model.

    It uses 'pymc' [1] as a library for approximating the
    posterior distribution.

    Parameters
    ----------
    theta_space : ThetaSpace
        Object representing the finite-dimensional parameter space.
    n_walkers : int
        Number of walkers (chains).
    n_iter : int
        Number of iterations.
    model_fn : Optional[ModelType]
        Function that generates a pymc model.
    model_kwargs : Dict
        Additional model arguments.
    b0 : Union[str, np.ndarray]
        Hyperparameter of the default prior distribution.
    g : float
        Hyperparameter of the default prior distribution.
    eta : float
        Hyperparameter of the default prior distribution.
    prior_p : Optional[Dict]
        Prior probabilities for each value of p (P_FREE).
    prior_tau : Optional[Union[str, np.ndarray]]
        [CURRENTLY NOT IMPLEMENTED] If None, tau defaults to a uniform
        prior. If 'auto' or a np.ndarray, set a beta distribution on tau,
        with automatic or pre-specified shape parameters.
    n_iter_warmup : int
        Number of initial iterations (only as warm-up; they
        are discarded afterwards).
    step_fn : Optional[StepType]
        MCMC sampler to use ("step" in pymc jargon)
    step_kwargs : Dict
        Additional sampler parameters.
    relabel_strategy : str
        Strategy for relabeling the samples in order to avoid label
        switching. It can be one of {'auto', 'pivot', 'beta', 'tau'}.
    thin : int
        Number of consecutive samples to skip from the chains.
    thin_pp : int
        Number of consecutive samples to skip from the chains when computing
        the posterior predictive distribution.
    burn : int
        Number of samples to discard from the chains.
    mle_precomputed : Optional[np.ndarray]
        Whether the MLE is supplied as a precomputed value.
    mle_method : str
        Method to use when approximating the MLE ('local' or 'global').
    mle_strategy : str
        Local optimization method for MLE approximation.
    n_reps_mle : int
        Number of independent runs of the MLE approximation algorithm.
    threshold : float
        Threshold for generating predictions in logistic regression.
    n_jobs : int
        Number of cores ot use.
    verbose : int
        Verbosity level.
    random_state : Optional[RandomType]
        Random state for reproducibility.

    [1] https://www.pymc.io/.
    """

    def __init__(
        self,
        theta_space: ThetaSpace = ThetaSpace(),
        n_walkers: int = 2,
        n_iter: int = 1000,
        *,
        model_fn: Optional[ModelType] = None,
        model_kwargs: Dict = {},
        b0: Union[str, np.ndarray] = 'mle',
        g: float = 5.0,
        eta: float = 1.0,
        prior_p: Optional[Dict] = None,
        prior_tau: Optional[Union[str, np.ndarray]] = None,
        n_iter_warmup: int = 100,
        step_fn: Optional[StepType] = None,
        step_kwargs: Dict = {},
        relabel_strategy: str = 'auto',
        thin: int = 1,
        thin_pp: int = 1,
        burn: Optional[int] = None,
        mle_precomputed: Optional[np.ndarray] = None,
        mle_method: str = 'L-BFGS-B',
        mle_strategy: str = 'global',
        n_reps_mle: int = 4,
        threshold: float = 0.5,
        n_jobs: int = 1,
        verbose: int = 0,
        random_state: Optional[RandomType] = None
    ) -> _BayesianRKHSFRegressionPymc:
        self.theta_space = theta_space
        self.n_walkers = n_walkers
        self.n_iter = n_iter
        self.model_fn = model_fn
        self.model_kwargs = model_kwargs
        self.b0 = b0
        self.g = g
        self.eta = eta
        self.prior_p = prior_p
        self.prior_tau = prior_tau
        self.step_fn = step_fn
        self.step_kwargs = step_kwargs
        self.n_iter_warmup = n_iter_warmup
        self.relabel_strategy = relabel_strategy
        self.thin = thin
        self.thin_pp = thin_pp
        self.burn = burn
        self.mle_precomputed = mle_precomputed
        self.mle_method = mle_method
        self.mle_strategy = mle_strategy
        self.n_reps_mle = n_reps_mle
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def fit(
        self,
        X: DataType,
        y: np.ndarray
    ) -> _BayesianRKHSFRegressionPymc:
        X, y = self._argcheck_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        super().fit(X, y)
        ts = self.theta_space

        # Model creation

        if self.model_fn is None:
            model_kwargs = self._get_prior_kwargs(X, y)
            if self.kind == 'linear':
                self.model_ = \
                    make_model_linear_pymc(X, y, ts, **model_kwargs)
            else:
                self.model_ = \
                    make_model_logistic_pymc(X, y, ts, **model_kwargs)
        elif callable(self.model_fn):
            self.model_ = self.model_fn(X, y, ts, **self.model_kwargs)
        else:
            raise ValueError("Invalid value for model_fn.")

        # Set dimension names

        dims = {f"{ts.names[ts.beta_idx_grouped]}": [ts.dim_name],
                f"{ts.names[ts.tau_idx_grouped]}": [ts.dim_name],
                "X_obs": ["observation", "value"],
                "y_obs": ["observation"]}

        # Select samplers

        if self.step_fn is None:
            step = None
        else:
            with self.model_:
                step = [self.step_fn(self.model_.cont_vars, **self.step_kwargs)]
                if ts.include_p:
                    step.append(
                        pm.CategoricalGibbsMetropolis(self.model_.disc_vars)
                    )

        # Select random seeds

        seed = [self.rng_.integers(2**32) for _ in range(self.n_walkers)]

        # Select initial points

        if ts.include_p:
            initvals = [{'p_cat': i % ts.p_max} for i in range(self.n_walkers)]
        else:
            initvals = None

        # Run MCMC procedure

        with self.model_:
            with HandleLogger(self.verbose):
                idata = pm.sample(
                    self.n_iter,
                    cores=self.n_jobs_,
                    init="jitter+adapt_diag_grad",
                    chains=self.n_walkers,
                    initvals=initvals,
                    tune=self.n_iter_warmup,
                    step=step,
                    random_seed=seed,
                    return_inferencedata=True,
                    progressbar=self.verbose > 0,
                    idata_kwargs={"dims": dims}
                )

            # Retain only relevant information
            idata.posterior = idata.posterior[ts.names]
            attrs = idata.posterior.attrs
            trace = idata.posterior.to_stacked_array(
                new_dim="stack",
                sample_dims=("chain", "draw")
            )
            if ts.include_p:
                self._discard_components(trace.values)
            idata.posterior = trace.to_unstacked_dataset(dim="stack")[ts.names]
            idata.posterior.attrs = attrs

            # Get data (after burn-in and thinning)
            self.burn_ = np.minimum(self.burn, self.n_iter//10)
            idata = self._tidy_idata(idata)

            # Save trace
            self.idata_ = idata
            self._trace = trace.values

        return self

    def mean_acceptance(self):
        """Get the mean acceptance of the sampler."""
        check_is_fitted(self)

        stats = self.idata_.sample_stats
        if "acceptance_rate" in stats:
            return stats.acceptance_rate.mean().values
        elif "accepted" in stats:
            return stats.accepted.mean().values
        else:
            warnings.warn("The sampler does not provide acceptance stats.")
            return np.nan

    def get_trace(self, burn=None, thin=None, flat=False):
        """Get the trace of the sampler."""
        check_is_fitted(self)
        if burn is None:
            burn = self.burn_
        if thin is None:
            thin = self.thin

        trace = np.copy(self._trace[:, burn::thin, :])

        if flat:
            trace = trace.reshape(-1, trace.shape[-1])

        return trace

    def to_graphviz(self):
        """Generate a graphical representation of the Bayesian model."""
        return pm.model_graph.model_to_graphviz(self.model_)


class BFLinearPymc(
    _BayesianRKHSFRegressionPymc,
    RegressorMixin
):
    kind = 'linear'
    default_strategies = ['posterior_mean']


class BFLogisticPymc(
    _BayesianRKHSFRegressionPymc,
    ClassifierMixin
):
    kind = 'logistic'
    default_strategies = ['posterior_mean', 'posterior_vote']
