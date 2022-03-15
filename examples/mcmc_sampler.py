# encoding: utf-8

from __future__ import annotations

from typing import Union, Optional, Dict, Sequence, Tuple
from multiprocessing import Pool
import os

import numpy as np
import emcee
import arviz as az

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import r2_score

from skfda.representation import FData
from skfda.representation.grid import FDataGrid
from skfda.representation.basis import FDataBasis

import utils
from mle_utils import compute_mle
from bayesian_model import (
    PriorType, log_posterior_linear,
    log_prior_linear, ThetaSpace
)

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
    sd_tau: float = 0.2
    sd_alpha0: float = 1.0
    param_sigma2: float = 2.0  # Shape parameter in inv_gamma distribution
    sd_sigma2: float = 1.0

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
        frac_random: float = 0.3,
        n_iter: int = 1000,
        n_iter_warmup: int = 100,
        initial_state: Union[str, np.ndarray] = 'mle',
        moves: Optional[Sequence[emcee.Move]] = None,
        compute_pp: bool = False,
        compute_ll: bool = False,
        thin: int = 1,
        burn: int = 100,
        frac_burn: int = 3,
        n_jobs: int = -1,
        verbose: int = 0,
        progress_notebook: bool = False,
        progress_kwargs: Optional[Dict] = None,
        random_state: RandomType = None
    ) -> None:
        """[TODO]"""
        self.theta_space = theta_space
        self.n_walkers = n_walkers
        self.log_prior = log_prior
        self.prior_kwargs = prior_kwargs
        self.b0 = b0
        self.g = g
        self.eta = eta
        self.frac_random = frac_random
        self.n_iter = n_iter
        self.n_iter_warmup = n_iter_warmup
        self.initial_state = initial_state
        self.moves = moves
        self.compute_pp = compute_pp
        self.compute_ll = compute_ll
        self.thin = thin
        self.burn = burn
        self.frac_burn = frac_burn
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
        """[TODO]"""

        X, y = self._argcheck_X_y(X, y)
        self._mean_alpha0 = y.mean()

        # MLE

        self._mle = None

        # Check parameters

        n_jobs = self.n_jobs if self.n_jobs > 0 else os.cpu_count()
        rng = utils.check_random_state(self.random_state)

        if isinstance(self.b0, str):
            if self.b0 == 'mle':
                if self._mle is None:
                    if self.verbose > 0:
                        print("[BFLR] Computing MLE...")
                    self._mle, _ = compute_mle(
                        self.theta_space,
                        X,
                        y,
                        False,
                        n_jobs=n_jobs,
                        rng=rng
                    )
                b0 = self._mle[self.theta_space.beta_idx]
            elif self.b0 == 'random':
                b0 = rng.normal(0, 10*self.sd_beta, self.theta_space.p)
            elif self.b0 == 'zero':
                b0 = np.zeros(self.theta_space.p)
            else:
                raise ValueError(
                    "%s is not a valid string for b0." %
                    self.b0
                )
        elif isinstance(self.b0, np.ndarray):
            b0 = self.b0
        else:
            raise ValueError("Invalid value for b0.")

        # Prior

        if self.log_prior is None:
            log_prior = log_prior_linear
        elif callable(self.log_prior):
            log_prior = self.log_prior
        else:
            raise ValueError("Invalid value for log_prior.")

        if self.prior_kwargs is None:
            prior_kwargs = {
                "b0": b0,
                "g": self.g,
                "eta": self.eta,
            }
        else:
            prior_kwargs = self.prior_kwargs

        # Posterior

        posterior_kwargs = {
            "theta_space": self.theta_space,
            "rng": rng,
            "return_ll": self.compute_ll,
            "return_pp": self.compute_pp,
            "log_prior": log_prior,
            "prior_kwargs": prior_kwargs,
        }

        if isinstance(self.initial_state, str):
            if self.initial_state == 'mle':
                if self._mle is None:
                    if self.verbose > 0:
                        print("[BFLR] Computing MLE...")
                    self._mle, _ = compute_mle(
                        self.theta_space,
                        X,
                        y,
                        False,
                        n_jobs=n_jobs,
                        rng=rng
                    )
                initial_state = utils.weighted_initial_guess_around_value(
                    self.theta_space, self._mle, self.sd_beta, self.sd_tau,
                    self._mean_alpha0, self.sd_alpha0, self.param_sigma2,
                    self.sd_sigma2, self.n_walkers, self.frac_random,
                    rng, 10*self.sd_beta
                )
            elif self.initial_state == 'random':
                initial_state = utils.initial_guess_random(
                    self.theta_space, 10*self.sd_beta, self._mean_alpha0,
                    self.sd_alpha0, self.param_sigma2, self.n_walkers,
                    rng
                )
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
            self.sampler = emcee.EnsembleSampler(
                self.n_walkers,
                self.theta_space.ndim,
                log_posterior_linear,
                pool=pool,
                args=(X, y,),
                kwargs=posterior_kwargs,
                moves=moves
            )

            # Initial state
            state = self.sampler.run_mcmc(
                initial_state, self.n_iter_warmup,
                progress=False, store=False
            )
            self.sampler.reset()

            if self.verbose > 0:
                progress = 'notebook' if self.progress_notebook else True
            else:
                progress = False

            # MCMC run
            self.sampler.run_mcmc(
                state, self.n_iter,
                progress=progress,
                progress_kwargs=self.progress_kwargs
            )

        # Get data

        with utils.HandleLogger(self.verbose):
            autocorr = self.sampler.get_autocorr_time(quiet=True)

        max_autocorr = np.max(autocorr)
        if np.isfinite(max_autocorr):
            burn = int(self.frac_burn*max_autocorr)
        else:
            burn = self.burn

        self.idata_ = self._emcee_to_idata(burn)

        return self

    def predict(self, X: DataType) -> np.ndarray:
        check_is_fitted(self)
        X = self._argcheck_X(X)

        return np.zeros(X.shape[0])

    def score(self, X: DataType, y: np.ndarray) -> float:
        X, y = self._argcheck_X_y(X, y)

        y_hat = self.predict(X)
        r2 = r2_score(y, y_hat)

        return r2

    def mle(self):
        check_is_fitted(self)

        if self._mle is not None:
            return self._mle
        else:
            raise AttributeError("The MLE was not computed during training.")

    def mean_acceptance(self):
        check_is_fitted(self)

        return np.mean(self.sampler.acceptance_fraction)

    def _emcee_to_idata(self, burn):
        names = self.theta_space.names
        names_ttr = self.theta_space.names_ttr
        p = self.theta_space.p
        pp_names = ["y_star"] if self.compute_pp else []
        n_pp = len(pp_names)
        blob_names = []
        blob_groups = []
        dims = {f"{names_ttr[0]}": ["vector"],
                f"{names_ttr[1]}": ["vector"],
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

        idata = az.from_emcee(
            self.sampler,
            var_names=names_ttr,
            slices=[slice(0, p), slice(p, 2*p), -2, -1],
            arg_names=["X_obs", "y_obs"],
            blob_names=blob_names,
            blob_groups=blob_groups,
            dims=dims
        )

        # Burn-in and thinning
        idata = idata.sel(draw=slice(burn, None, self.thin))

        idata.posterior[names[1]] = \
            self.theta_space.tau_ttr.backward(
                idata.posterior[names_ttr[1]])
        idata.posterior[names[-1]] = \
            self.theta_space.sigma2_ttr.backward(
                idata.posterior[names_ttr[-1]])

        return idata

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
