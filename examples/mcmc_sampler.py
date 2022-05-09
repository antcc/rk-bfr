# encoding: utf-8

from __future__ import annotations

import os
from multiprocessing import Pool
from typing import Dict, Optional, Sequence, Tuple, Union

import arviz as az
import emcee
import numpy as np
import utils
from bayesian_model import (PriorType, ThetaSpace, generate_pp,
                            log_posterior_linear, log_prior_linear,
                            point_estimate, point_predict)
from mle import compute_mle
from skfda.representation import FData
from skfda.representation.basis import FDataBasis
from skfda.representation.grid import FDataGrid
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted

DataType = Union[
    FData,
    np.ndarray,
]

RandomType = Union[
    int,
    np.random.RandomState,
    np.random.Generator,
    None
]


class BayesianLinearRegressionEmcee(
    BaseEstimator,
    RegressorMixin,
    TransformerMixin
):
    """Bayesian functional linear regression.

    It uses 'emcee' as the underlying MCMC algorithm for
    approximating the posterior distribution.

    This estimator can be used both as an end-to-end regressor
    (via the 'predict' method) or as a variable selection
    procedure (via the 'transform' method).

    See [REFERENCE].
    """
    sd_beta: float = 1.0
    sd_beta_random: float = 10.0
    sd_tau: float = 0.2
    sd_alpha0: float = 1.0
    param_sigma2: float = 2.0  # Shape parameter in inv_gamma distribution
    sd_sigma2: float = 1.0

    default_point_estimates = ['mean', 'median', 'mode']

    def __init__(
        self,
        theta_space: ThetaSpace = ThetaSpace(),
        n_walkers: int = 100,
        *,
        log_prior: PriorType = None,
        prior_kwargs: Optional[Dict] = None,
        b0: Union[str, np.ndarray] = 'mle',
        g: float = 5.0,
        eta: float = 1.0,
        prior_p: Optional[Dict] = None,
        n_iter: int = 1000,
        n_iter_warmup: int = 100,
        initial_state: Union[str, np.ndarray] = 'mle',
        frac_random: float = 0.3,
        moves: Optional[Sequence[emcee.Move]] = None,
        compute_pp: bool = False,
        compute_ll: bool = False,
        thin: int = 1,
        thin_pp: int = 5,
        burn: int = 100,
        burn_relative: int = 3,
        mle_method: str = 'L-BFGS-B',
        mle_strategy: str = 'global',
        n_jobs: int = -1,
        verbose: int = 0,
        progress_notebook: bool = False,
        progress_kwargs: Optional[Dict] = None,
        random_state: RandomType = None
    ) -> None:
        self.theta_space = theta_space
        self.n_walkers = n_walkers
        self.log_prior = log_prior
        self.prior_kwargs = prior_kwargs
        self.b0 = b0
        self.g = g
        self.eta = eta
        self.prior_p = prior_p
        self.n_iter = n_iter
        self.n_iter_warmup = n_iter_warmup
        self.initial_state = initial_state
        self.frac_random = frac_random
        self.moves = moves
        self.compute_pp = compute_pp
        self.compute_ll = compute_ll
        self.thin = thin
        self.thin_pp = thin_pp
        self.burn = burn
        self.burn_relative = burn_relative
        self.mle_method = mle_method
        self.mle_strategy = mle_strategy
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.progress_notebook = progress_notebook
        self.progress_kwargs = progress_kwargs
        self.random_state = random_state

    def fit(
        self,
        X: DataType,
        y: np.ndarray
    ) -> BayesianLinearRegressionEmcee:

        X, y = self._argcheck_X_y(X, y)
        self.rng_ = utils.check_random_state(self.random_state)
        n_jobs = self.n_jobs if self.n_jobs > 0 else os.cpu_count()
        ts = self.theta_space
        self._mean_alpha0 = y.mean()
        self.mle_ = None
        self._theta_space_fixed = None

        if ts.include_p:
            self.n_components_default_pe = {}

        # Prior

        if self.log_prior is None:
            log_prior = log_prior_linear
        elif callable(self.log_prior):
            log_prior = self.log_prior
        else:
            raise ValueError("Invalid value for log_prior.")

        if self.prior_kwargs is None:
            if isinstance(self.b0, str):
                if self.b0 == 'mle':
                    if self.mle_ is None:
                        self.mle_, self._theta_space_fixed = \
                            self._compute_mle(X, y, n_jobs, self.rng_)
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
                if sum(self.prior_p.values()) != 1.0:
                    raise ValueError(
                        "The prior probabilities of 'p' must add up to one."
                    )
                prior_kwargs["prior_p"] = self.prior_p
        else:
            prior_kwargs = self.prior_kwargs

        # Posterior

        posterior_kwargs = {
            "theta_space": ts,
            "rng": self.rng_,
            "return_ll": self.compute_ll,
            "return_pp": self.compute_pp,
            "log_prior": log_prior,
            "prior_kwargs": prior_kwargs,
        }

        if isinstance(self.initial_state, str):
            if self.initial_state == 'mle':
                if self.mle_ is None:
                    self.mle_, self._theta_space_fixed = \
                        self._compute_mle(X, y, n_jobs, self.rng_)
                initial_state = self._weighted_initial_guess_around_mle(
                    self.rng_)
            elif self.initial_state == 'random':
                initial_state = self._initial_guess_random(
                    self.n_walkers, self.rng_)
            else:
                raise ValueError(
                    "%s is not a valid string for initial_state." %
                    self.initial_state
                )
        elif isinstance(self.initial_state, np.ndarray):
            initial_state = self.initial_state
        else:
            raise ValueError("Invalid value for initial_state.")

        if self.moves is None:
            moves = [
                (emcee.moves.StretchMove(), 0.7),
                (emcee.moves.WalkMove(), 0.3),
            ]
        else:
            moves = self.moves

        # Run sampler

        with Pool(n_jobs) as pool:
            self.sampler_ = emcee.EnsembleSampler(
                self.n_walkers,
                ts.n_dim,
                log_posterior_linear,
                pool=pool,
                args=(X, y,),
                kwargs=posterior_kwargs,
                moves=moves
            )

            # Set initial random state for the sampler
            random_seed = self.rng_.integers(2**32)
            self.sampler_.random_state = \
                np.random.RandomState(random_seed).get_state()

            if self.verbose > 1:
                print("[BFLinReg] MCMC warmup iterations...")

            # Warmup iterations
            if self.n_iter_warmup > 0:
                state = self.sampler_.run_mcmc(
                    initial_state, self.n_iter_warmup,
                    progress=False, store=False
                )
                self.sampler_.reset()
            else:
                state = initial_state

            if self.verbose > 0:
                progress = 'notebook' if self.progress_notebook else True
                if self.progress_kwargs is None:
                    self.progress_kwargs = {"desc": "[BFLinReg] MCMC"}
            else:
                progress = False

            # MCMC run
            self.sampler_.run_mcmc(
                state, self.n_iter,
                progress=progress,
                progress_kwargs=self.progress_kwargs
            )

        # Transform back parameters and discard unused ones if applicable
        self._transform_trace()

        # Calculate burn-in
        autocorr = self.autocorrelation_times(burn=0, thin=1)
        max_autocorr = np.nanmax(autocorr)

        if np.isfinite(max_autocorr):
            burn = int(self.burn_relative*max_autocorr)
        else:
            burn = self.burn

        self.burn_ = np.minimum(burn, self.n_iter//10)

        # Get data (after burn-in and thinning)
        self.idata_ = self._emcee_to_idata()

        return self

    def predict(
        self,
        X: DataType,
        strategy='mode',
        bw='experimental'
    ) -> np.ndarray:
        """The parameter 'strategy' can be {'mean', 'median', 'mode', 'posterior_mean'}
           or a callable representing a point estimate."""
        check_is_fitted(self)
        X = self._argcheck_X(X)
        ts = self.theta_space

        if callable(strategy) or strategy in self.default_point_estimates:
            y_pred, theta_hat = point_predict(
                X,
                self.idata_,
                ts,
                strategy,
                kind='linear',
                skipna=ts.include_p,
                bw=bw
            )

            if not callable(strategy) and ts.include_p:
                self.n_components_default_pe[strategy] = theta_hat[ts.p_idx]

        elif strategy == 'posterior_mean':
            if self.verbose > 0:
                progress = 'notebook' if self.progress_notebook else True
            else:
                progress = False

            pp_test = generate_pp(
                self.idata_,
                X,
                ts,
                self.thin_pp,
                kind='linear',
                rng=self.rng_,
                progress=progress,
            )
            y_pred = pp_test.mean(axis=(0, 1))

        else:
            raise ValueError("Incorrect value for parameter 'strategy'.")

        return y_pred

    def transform(self, X: DataType, pe='mode', bw='experimental') -> DataType:
        check_is_fitted(self)
        grid = self.theta_space.grid

        tau_hat = point_estimate(
            self.idata_,
            pe,
            self.theta_space.names,
            skipna=self.theta_space.include_p,
            bw=bw
        )[self.theta_space.tau_idx]
        idx_hat = np.abs(grid - tau_hat[:, np.newaxis]).argmin(1)
        idx_hat.sort()

        X_red = self._variable_selection(X, idx_hat)

        return X_red

    def score(self, X: DataType, y: np.ndarray, strategy='mode') -> float:
        X, y = self._argcheck_X_y(X, y)

        y_pred = self.predict(X, strategy)
        r2 = r2_score(y, y_pred)

        return r2

    def mean_acceptance(self):
        check_is_fitted(self)

        return np.mean(self.sampler_.acceptance_fraction)

    def autocorrelation_times(self, burn=None, thin=None):
        check_is_fitted(self)
        if burn is None:
            burn = self.burn_
        if thin is None:
            thin = self.thin

        """with utils.HandleLogger(self.verbose):
            autocorr = self.sampler_.get_autocorr_time(
                discard=burn, thin=thin, quiet=True)"""

        with utils.HandleLogger(self.verbose):
            if self.theta_space.include_p:
                x = self.sampler_.get_chain(discard=burn, thin=thin)
                n_dim = self.theta_space.n_dim
                autocorr = np.zeros(n_dim)

                for dim in range(n_dim):
                    x_dim = x[:, :, dim]
                    # Remove chains where all values are NaN
                    x_dim = x_dim[:, ~np.isnan(x_dim).all(axis=0)]
                    # Replace NaN with 0.0 for computing autocorrelation
                    x_dim = np.nan_to_num(x_dim)
                    # Compute autocorrelation
                    autocorr[dim] = thin*emcee.autocorr.integrated_time(
                        x_dim, quiet=True)
            else:
                autocorr = self.sampler_.get_autocorr_time(
                    discard=burn, thin=thin, quiet=True)

        return autocorr

    def total_samples(self, burn=None, thin=None):
        check_is_fitted(self)

        if burn is None:
            burn = self.burn_
        if thin is None:
            thin = self.thin

        return len(self.sampler_.get_chain(discard=burn, thin=thin, flat=True))

    def n_components(self, strategy, bw="experimental"):
        check_is_fitted(self)
        ts = self.theta_space

        if not ts.include_p:
            n_comp = ts.p_max
        elif strategy == "posterior_mean":
            n_comp = np.max(self.idata_.posterior[ts.names[ts.p_idx]].to_numpy())
        elif (callable(strategy)
                or strategy not in self.n_components_default_pe.keys()):
            theta_hat = point_estimate(
                self.idata_,
                strategy,
                ts.names,
                skipna=ts.include_p,
                bw=bw
            )
            n_comp = theta_hat[ts.p_idx]
        else:
            n_comp = self.n_components_default_pe[strategy]

        return ts.round_p(n_comp)

    def get_trace(self, burn=None, thin=None, flat=False):
        check_is_fitted(self)

        if burn is None:
            burn = self.burn_
        if thin is None:
            thin = self.thin

        trace = np.copy(self.sampler_.get_chain(discard=burn, thin=thin))

        if flat:
            trace = trace.reshape(-1, trace.shape[-1])  # All chains combined

        return trace

    def summary(self, bw='experimental', **kwargs):
        skipna = self.theta_space.include_p
        additional_stats = {
            "min": np.nanmin if skipna else np.min,
            "max": np.nanmax if skipna else np.max,
            "median": np.nanmedian if skipna else np.median,
            "mode": lambda x: utils.mode_fn(x, skipna=skipna, bw=bw),
        }

        return az.summary(
            self.idata_,
            extend=True,
            kind='stats',
            var_names=self.theta_space.names,
            skipna=skipna,
            labeller=self.theta_space.labeller,
            stat_funcs=additional_stats,
            **kwargs
        )

    def _transform_trace(self):
        ts = self.theta_space
        trace = self.sampler_.get_chain()

        # Transform back parameters
        trace[:, :, ts.tau_idx] = ts.tau_ttr.backward(
            trace[:, :, ts.tau_idx])
        trace[:, :, ts.sigma2_idx] = ts.sigma2_ttr.backward(
            trace[:, :, ts.sigma2_idx])

        # Set unused values to NaN on each sample
        if ts.include_p:
            trace = np.apply_along_axis(ts.set_unused_nan, -1, trace)

    def _compute_mle(self, X, y, n_jobs, rng):
        if self.verbose > 1:
            print("[BFLinReg] Computing MLE...")
        ts_fixed = self.theta_space.copy_p_fixed()

        mle, _ = compute_mle(
            X,
            y,
            ts_fixed,
            kind='linear',
            method=self.mle_method,
            strategy=self.mle_strategy,
            n_jobs=n_jobs,
            rng=rng
        )

        return mle, ts_fixed

    def _emcee_to_idata(self):
        ts = self.theta_space
        names = ts.names
        pp_names = ["y_star"] if self.compute_pp else []
        n_pp = len(pp_names)
        blob_names = []
        blob_groups = []
        dims = {f"{names[ts.beta_idx_grouped]}": [ts.coord_name],
                f"{names[ts.tau_idx_grouped]}": [ts.coord_name],
                "X_obs": ["observation", "value"],
                "y_obs": ["observation"]}

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
            self.sampler_,
            var_names=names,
            slices=slices,
            arg_names=["X_obs", "y_obs"],
            arg_groups=["constant_data", "observed_data"],
            blob_names=blob_names,
            blob_groups=blob_groups,
            dims=dims
        )

        if self.verbose > 0:
            print(f"[BFLinReg] Discarding the first {self.burn_} samples...")

        # Burn-in and thinning
        idata = idata.sel(draw=slice(self.burn_, None, self.thin))

        # Set p values to integers
        if ts.include_p:
            idata.posterior[names[ts.p_idx]] = \
                np.rint(idata.posterior[names[ts.p_idx]]).astype(int)

        return idata

    def _initial_guess_random(self, n_samples, rng):
        p = self.theta_space.p_max

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
                prior_p = sorted(self.prior_p.values())
            else:
                prior_p = np.full(p, 1./p)

            p_init = rng.choice(
                np.arange(self.theta_space.p_max) + 1,
                size=n_samples,
                p=prior_p
            )

            init = np.vstack((
                p_init,
                init.T
            )).T

        init_tr = self.theta_space.forward(init)

        return init_tr if n_samples > 1 else init_tr[0]

    def _initial_guess_around_value(self, theta_fixed, n_samples, rng):
        """'theta_fixed' is in the original space and does not include 'p'."""
        if rng is None:
            rng = np.random.default_rng()

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

        value_jitter = theta_fixed[np.newaxis, :] + jitter

        if self.theta_space.include_p:
            p_init = rng.choice(
                np.arange(self.theta_space.p_max) + 1,
                size=n_samples,
                p=sorted(self.prior_p.values())
            )

            value_jitter = np.vstack((
                p_init,
                value_jitter.T
            )).T

        value_jitter = self.theta_space.clip_bounds(value_jitter)
        value_jitter_tr = self.theta_space.forward(value_jitter)

        return value_jitter_tr if n_samples > 1 else value_jitter_tr[0]

    def _weighted_initial_guess_around_mle(self, rng):
        n_random = int(self.frac_random*self.n_walkers)
        n_around = self.n_walkers - n_random

        if n_random > 0:
            init_1 = self._initial_guess_random(n_random, rng)
        else:
            init_1 = None

        if n_around > 0:
            init_2 = self._initial_guess_around_value(self.mle_, n_around, rng)
        else:
            init_2 = None

        if init_1 is None:
            init = init_2
        elif init_2 is None:
            init = init_1
        else:
            init = np.vstack((init_1, init_2))

        rng.shuffle(init)

        return init

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

    def _argcheck_X(self, X: DataType) -> np.ndarray:
        grid = self.theta_space.grid
        N = len(grid)

        if isinstance(X, np.ndarray):
            if X.shape[1] != N:
                raise ValueError(
                    "Data must be compatible with the specified "
                    "grid (i.e. 'self.theta_space.grid').")
        elif isinstance(X, FDataBasis):
            X = X.to_grid(grid_points=grid).data_matrix.reshape(-1, N)
        elif isinstance(X, FDataGrid):
            X = X.data_matrix.reshape(-1, N)
        else:
            raise ValueError('Data type not supported for X.')

        return X

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
