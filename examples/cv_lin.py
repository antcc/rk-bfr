# encoding: utf-8

import os
import logging
import sys
import time
from multiprocessing import Pool
from itertools import product

import numpy as np
import pandas as pd
import scipy

import emcee
import pymc3 as pm
import theano.tensor as tt

from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split, GridSearchCV

import utils
from _fpls import FPLS, APLS, FPLSBasis
# from _fpca_basis import FPCABasis

import skfda
from skfda.preprocessing.dim_reduction.variable_selection import (
    RecursiveMaximaHunting as RMH,
    # MinimumRedundancyMaximumRelevance as mRMR,
)
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from skfda.ml.regression import KNeighborsRegressor
from skfda.ml.regression import LinearRegression as FLinearRegression
from skfda.representation.basis import FDataBasis, Fourier, BSpline
from skfda.representation.grid import FDataGrid
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.preprocessing.smoothing.validation import (
    SmoothingParameterSearch,
    LinearSmootherGeneralizedCVScorer,
    akaike_information_criterion
)
from skfda.preprocessing.smoothing.kernel_smoothers import (
    NadarayaWatsonSmoother as NW
)

###################################################################
# CONFIGURATION
###################################################################

# Ignore warnings
os.environ["PYTHONWARNINGS"] = 'ignore'
np.seterr(over='ignore', divide='ignore')

# Floating point precision for display
np.set_printoptions(precision=3, suppress=True)
pd.set_option("display.precision", 3)
pd.set_option('display.max_columns', 80)

###################################################################
# GLOBAL OPTIONS
###################################################################

# MCMC algorithm
alg = sys.argv[1].lower()
if ":" in alg:
    MCMC_ALG = alg.split(":")[0]
    USE_NUTS = alg.split(":")[1] == "nuts"
else:
    MCMC_ALG = alg
    USE_NUTS = False

# Randomness and reproducibility
SEED = int(sys.argv[2])
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# Multiprocessing
N_CORES = os.cpu_count()

# Data
SYNTHETIC_DATA = True
MODEL_GEN = "RKHS"  # 'L2' or 'RKHS'
REAL_DATA = "Aemet"

kernel_fn = utils.fractional_brownian_kernel
beta_coef = utils.cholaquidis_scenario3

INITIAL_SMOOTHING = None  # None, 'NW' or 'Basis'
N_BASIS = 16
STANDARDIZE_PREDICTORS = False
STANDARDIZE_RESPONSE = False
TRANSFORM_TAU = False

basis = BSpline(n_basis=N_BASIS)
smoothing_params = np.logspace(-4, 4, 50)

folds = KFold(shuffle=True, random_state=SEED)

# Model
MIN_P = 2
MAX_P = 4

# Override options with command-line parameters
if len(sys.argv) > 3:
    if sys.argv[3][:2] == "s:":
        SYNTHETIC_DATA = True
        MODEL_GEN = sys.argv[3][2:]
    elif sys.argv[3][:2] == "r:":
        SYNTHETIC_DATA = False
        REAL_DATA = sys.argv[3][2:]
    if sys.argv[4] == "1":
        kernel_fn = utils.fractional_brownian_kernel
    elif sys.argv[4] == "2":
        kernel_fn = utils.ornstein_uhlenbeck_kernel
    else:
        kernel_fn = utils.squared_exponential_kernel
    if sys.argv[5] == "NW" or sys.argv[5] == "Basis":
        INITIAL_SMOOTHING = sys.argv[5]
    else:
        INITIAL_SMOOTHING = None

    if len(sys.argv) > 6:
        MIN_P = int(sys.argv[6])
        MAX_P = int(sys.argv[7])

# Results
FIT_REF_ALGS = False
REFIT_BEST_VAR_SEL = True
PRINT_RESULTS_ONLINE = False
PRINT_TO_FILE = False
SAVE_RESULTS = True
SAVE_TOP = 10

###################################################################
# HYPERPARAMETERS
###################################################################

# -- Cv and Model parameters

N_REPS = 5

mle_method = 'L-BFGS-B'
mle_strategy = 'global'

ps = np.arange(MIN_P, MAX_P + 1)
gs = [5]
etas = [0.01, 0.1, 1.0, 10.0]
num_ps = len(ps)
num_gs = len(gs)
num_etas = len(etas)

# -- MCMC sampler parameters

thin = 1
thin_pp = 5

# Emcee specific parameters
n_walkers = 64
n_iter_initial = 100
n_iter = 1000
return_pp = False
return_ll = False
frac_random = 0.3

sd_beta_init = 1.0
sd_tau_init = 0.2
sd_alpha0_init = 1.0
param_sigma2_init = 2.0  # shape parameter in inv_gamma distribution
sd_sigma2_init = 1.0

moves = [
    (emcee.moves.StretchMove(), 0.7),
    (emcee.moves.WalkMove(), 0.3),
]

# Pymc specific parameters

burn = 0

n_samples_nuts = 1000
tune_nuts = 1000
target_accept = 0.8
n_samples_metropolis = 6000
tune_metropolis = 4000


###################################################################
# AUXILIARY FUNCTIONS
###################################################################

# -- MLE computation

def neg_ll(theta_tr, X, Y, theta_space):
    """Transformed parameter vector 'theta_tr' is (β, logit τ, α0, log σ)."""

    n, N = X.shape
    grid = theta_space.grid

    assert len(theta_tr) == theta_space.ndim

    theta = theta_space.backward(theta_tr)
    beta, tau, alpha0, sigma2 = theta_space.get_params(theta)
    log_sigma = theta_space.get_sigma2(theta_tr)

    idx = np.abs(grid - tau[:, np.newaxis]).argmin(1)
    X_tau = X[:, idx]

    return -(-n*log_sigma
             - np.linalg.norm(Y - alpha0 - X_tau@beta)**2/(2*sigma2))


def optimizer_global(random_state, args):
    theta_init, X, Y, theta_space, method, bounds = args
    return scipy.optimize.basinhopping(
        neg_ll,
        x0=theta_init,
        seed=random_state,
        minimizer_kwargs={"args": (X, Y, theta_space),
                          "method": method,
                          "bounds": bounds}
    ).x


def compute_mle(
        theta_space, X, Y,
        method='Powell',
        strategy='global',
        rng=None):
    if rng is None:
        rng = np.random.default_rng()

    p = theta_space.p

    theta_init = theta_space.forward(
        np.array([0.0]*p + [0.5]*p + [Y.mean()] + [1.0]))

    if TRANSFORM_TAU:
        bounds = None
    else:
        bounds = ([(None, None)]*p
                  + [(theta_space.tau_lb, theta_space.tau_ub)]*p
                  + [(None, None)]
                  + [(None, None)])

    with utils.IgnoreWarnings():
        if strategy == 'local':
            mle_theta = scipy.optimize.minimize(
                neg_ll,
                x0=theta_init,
                bounds=bounds,
                method=method,
                args=(X, Y, theta_space)
            ).x
            bic = utils.compute_bic(theta_space, neg_ll, mle_theta, X, Y)
        elif strategy == 'global':
            mles = np.zeros((N_CORES, theta_space.ndim))

            with Pool(N_CORES) as pool:
                random_states = [rng.integers(2**32) for i in range(N_CORES)]
                args_optim = [theta_init, X, Y, theta_space, method, bounds]
                mles = pool.starmap(
                    optimizer_global, product(random_states, [args_optim]))
                bics = utils.bic = utils.compute_bic(
                    theta_space, neg_ll, mles, X, Y)
                mle_theta = mles[np.argmin(bics)]
                bic = bics[np.argmin(bics)]
        else:
            raise ValueError(
                "Parameter 'strategy' must be one of {'local', 'global'}.")

    return mle_theta, bic


# -- Log-posterior model for emcee

def log_prior(theta_tr):
    """Global parameters (for efficient parallelization):
        X, b0, g, eta, theta_space"""
    assert len(theta_tr) == theta_space.ndim

    n, N = X.shape
    p = theta_space.p
    grid = theta_space.grid

    theta = theta_space.backward(theta_tr)
    beta, tau, alpha0, sigma2 = theta_space.get_params(theta)
    log_sigma = theta_space.get_sigma2(theta_tr)

    if not TRANSFORM_TAU:
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


def log_likelihood(theta_tr, Y):
    """Global parameters (for efficient parallelization):
        X, theta_space, return_ll"""
    n, N = X.shape
    grid = theta_space.grid

    assert len(theta_tr) == theta_space.ndim

    theta = theta_space.backward(theta_tr)
    beta, tau, alpha0, sigma2 = theta_space.get_params(theta)
    log_sigma = theta_space.get_sigma2(theta_tr)

    idx = np.abs(grid - tau[:, np.newaxis]).argmin(1)
    X_tau = X[:, idx]

    ll = (-n*log_sigma
          - np.linalg.norm(Y - alpha0 - X_tau@beta)**2/(2*sigma2))

    if return_ll:
        # Add constant term so that it is the genuine log-probability
        ll_pointwise = (-log_sigma - 0.5*np.log(2*np.pi)
                        - (Y - alpha0 - X_tau@beta)**2/(2*sigma2))
        return ll, ll_pointwise
    else:
        return ll


def log_posterior(theta_tr, Y):
    """Global parameters (for efficient parallelization):
        X, rng, return_pp, return_ll, theta_space"""
    # Compute log-prior
    lp = log_prior(theta_tr)

    if not np.isfinite(lp):
        if return_pp and return_ll:
            return -np.inf, np.full_like(Y, -np.inf), np.full_like(Y, -np.inf)
        elif return_pp:
            return -np.inf, np.full_like(Y, -np.inf)
        elif return_ll:
            return -np.inf, np.full_like(Y, -np.inf)
        else:
            return -np.inf

    # Compute log-likelihood (and possibly pointwise log-likelihood)
    if return_ll:
        ll, ll_pointwise = log_likelihood(theta_tr, Y)
    else:
        ll = log_likelihood(theta_tr, Y)

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


# -- Log-posterior model for pymc

def make_model_pymc(
    theta_space, g, eta, X, Y,
    names, names_aux, mle_theta=None
):
    n, N = X.shape
    grid = theta_space.grid
    p = theta_space.p

    if mle_theta is not None:
        b0 = mle_theta[:p]
    else:
        b0 = g*rng.standard_normal(size=p)  # <-- Change if needed

    with pm.Model() as model:
        X_pm = pm.Data('X', X)

        alpha0_and_log_sigma = pm.DensityDist(
            names_aux[0], lambda x: 0, shape=(2,))

        alpha0 = pm.Deterministic(names[-2], alpha0_and_log_sigma[0])

        log_sigma = alpha0_and_log_sigma[1]
        sigma = pm.math.exp(log_sigma)
        sigma2 = pm.Deterministic(names[-1], sigma**2)

        tau = pm.Uniform(names[1], 0.0, 1.0, shape=(p,))

        idx = np.abs(grid - tau[:, np.newaxis]).argmin(1)
        X_tau = X_pm[:, idx]
        G_tau = pm.math.matrix_dot(X_tau.T, X_tau)
        G_tau = (G_tau + G_tau.T)/2.  # Enforce symmetry
        G_tau_reg = G_tau + eta * \
            tt.max(tt.nlinalg.eigh(G_tau)[0])*np.identity(p)

        def beta_lprior(x):
            b = x - b0

            return (0.5*pm.math.logdet(G_tau_reg)
                    - p*log_sigma
                    - pm.math.matrix_dot(b.T, G_tau_reg, b)/(2.*g*sigma2))

        beta = pm.DensityDist(names[0], beta_lprior, shape=(p,))

        expected_obs = alpha0 + pm.math.matrix_dot(X_tau, beta)

        y_obs = pm.Normal('y_obs', mu=expected_obs, sigma=sigma, observed=Y)

    return model


# -- Sklearn CV and transformers

def cv_sk(
    regressors, folds, X, Y, X_test, Y_test,
    columns_name, verbose=False
):

    df_metrics_sk = pd.DataFrame(columns=columns_name)

    for i, (name, pipe, params) in enumerate(regressors):
        if verbose:
            print(f"  Fitting {name}...")
        reg_cv = GridSearchCV(pipe, params, scoring="neg_mean_squared_error",
                              n_jobs=N_CORES, cv=folds)

        with utils.IgnoreWarnings():
            reg_cv.fit(X, Y)

        Y_hat_sk = reg_cv.predict(X_test)
        metrics_sk = utils.regression_metrics(Y_test, Y_hat_sk)

        if name == "sk_fknn":
            n_features = f"K={reg_cv.best_params_['reg__n_neighbors']}"
        elif "svm" in name:
            n_features = reg_cv.best_estimator_["reg"].n_features_in_
        elif "pls1" in name:
            n_features = reg_cv.best_estimator_["reg"].n_components
        else:
            if isinstance(reg_cv.best_estimator_["reg"].coef_[0], FDataBasis):
                coef = reg_cv.best_estimator_["reg"].coef_[0].coefficients[0]
            else:
                coef = reg_cv.best_estimator_["reg"].coef_

            n_features = sum(~np.isclose(coef, 0))

        df_metrics_sk.loc[i] = [
            name,
            n_features,
            metrics_sk["mse"],
            metrics_sk["rmse"],
            metrics_sk["r2"]]

    df_metrics_sk.sort_values(columns_name[-3], inplace=True)

    return df_metrics_sk, reg_cv


class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, p=1):
        self.p = p

    def fit(self, X, y=None):
        N = X.shape[1]
        self.idx_ = np.linspace(0, N - 1, self.p).astype(int)
        return self

    def transform(self, X, y=None):
        return X[:, self.idx_]


class DataMatrix(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.N = len(X.grid_points[0])
        return self

    def transform(self, X, y=None):
        return X.data_matrix.reshape(-1, self.N)


class Basis(BaseEstimator, TransformerMixin):

    def __init__(self, basis=Fourier()):
        self.basis = basis

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.to_basis(self.basis)


class VariableSelection(BaseEstimator, TransformerMixin):

    def __init__(self, grid=None, idx=None):
        self.grid = grid
        self.idx = idx
        self.idx.sort()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return FDataGrid(X.data_matrix[:, self.idx], self.grid[self.idx])


class PLSRegressionWrapper(PLSRegression):

    def transform(self, X, y=None):
        check_is_fitted(self)

        return super().transform(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def predict(self, X, copy=True):
        check_is_fitted(self)

        if self.coef_.shape[1] == 1:  # if n_targets == 1
            return super().predict(X, copy)[:, 0]
        else:
            return super().predict(X, copy)


# -- Bayesian variable selection

def bayesian_var_sel(idata, theta_space, names,
                     X, Y, X_test, Y_test, folds,
                     columns_name, prefix="emcee",
                     point_est='mode', verbose=False):
    grid = theta_space.grid
    p_hat = theta_space.p
    tau_hat = utils.point_estimate(idata, point_est, names)[p_hat:2*p_hat]
    idx_hat = np.abs(grid - tau_hat[:, np.newaxis]).argmin(1)

    regressors_var_sel = []
    alphas = np.logspace(-4, 4, 20)
    params_reg = {"reg__alpha": alphas}
    params_svm = {"reg__C": alphas,
                  "reg__gamma": ['auto', 'scale']}

    # MCMC+Lasso
    regressors_var_sel.append((f"{prefix}_{point_est}+sk_lasso",
                               Pipeline([
                                   ("var_sel", VariableSelection(grid, idx_hat)),
                                   ("data_matrix", DataMatrix()),
                                   ("reg", Lasso())]),
                               params_reg
                               ))

    # MCMC+Ridge
    regressors_var_sel.append((f"{prefix}_{point_est}+sk_ridge",
                               Pipeline([
                                   ("var_sel", VariableSelection(grid, idx_hat)),
                                   ("data_matrix", DataMatrix()),
                                   ("reg", Ridge())]),
                               params_reg
                               ))

    # MCMC+SVM RBF
    regressors_var_sel.append((f"{prefix}_{point_est}+sk_svm_rbf",
                               Pipeline([
                                   ("var_sel", VariableSelection(grid, idx_hat)),
                                   ("data_matrix", DataMatrix()),
                                   ("reg", SVR(kernel='rbf'))]),
                               params_svm
                               ))

    df_metrics_var_sel, _ = cv_sk(
        regressors_var_sel, folds, X, Y,
        X_test, Y_test, columns_name, verbose)

    return df_metrics_var_sel


# -- Utility functions

def iteration_count(total, current):
    """
    - total: list of total number of parameters
      (e.g. [num_ps, num_gs, num_etas] = [3, 3, 1]).
    - current: list of the current iteration for each parameter
      (e.g. [i, j, k] = [0, 1, 2]).
    """
    assert len(total) == len(current)

    num_params = len(total)
    aux = np.zeros(num_params)

    for i in range(num_params):
        aux[i] = np.prod(total[i + 1:])

    return np.sum(aux*current).astype(int) + 1


###################################################################
# GENERATE DATASET
###################################################################

print(f"Random seed: {SEED}")
print(f"Num. cores: {N_CORES}")

# -- Dataset generation

if SYNTHETIC_DATA:
    n_train, n_test = 100, 50
    N = 100
    grid = np.linspace(1./N, 1., N)

    beta_true = np.array([-5., 10.])
    tau_true = np.array([0.1, 0.8])
    alpha0_true = 5.
    sigma2_true = 0.5

    if MODEL_GEN == "L2":
        x, y = utils.generate_gp_l2_dataset(
            grid, kernel_fn,
            n_train + n_test, beta_coef, alpha0_true,
            sigma2_true, rng=rng
        )
    elif MODEL_GEN == "RKHS":
        x, y = utils.generate_gp_rkhs_dataset(
            grid, kernel_fn,
            n_train + n_test, beta_true, tau_true,
            alpha0_true, sigma2_true, rng=rng
        )
    else:
        raise ValueError("Invalid model generation strategy.")

    # Train/test split
    X, X_test, Y, Y_test = train_test_split(
        x, y, train_size=n_train, random_state=SEED)

    # Create FData object
    X_fd = skfda.FDataGrid(X, grid)
    X_test_fd = skfda.FDataGrid(X_test, grid)

else:
    if REAL_DATA == "Tecator":
        x, y = skfda.datasets.fetch_tecator(return_X_y=True)
        y = np.sqrt(y[:, 1])  # Sqrt-Fat
    elif REAL_DATA == "Aemet":
        data = skfda.datasets.fetch_aemet()['data']
        data_matrix = data.data_matrix
        temperature = data_matrix[:, :, 0]
        x = FDataGrid(temperature, data.grid_points)
        # Log-Sum of log-precipitation for each station
        y = np.log(np.exp(data_matrix[:, :, 1]).sum(axis=1))
    else:
        raise ValueError("REAL_DATA must be 'Tecator' or 'Aemet'.")

    X_fd, X_test_fd, Y, Y_test = train_test_split(
        x, y, train_size=0.8, random_state=SEED)

    N = len(X_fd.grid_points[0])
    grid = np.linspace(1./N, 1., N)  # TODO: use (normalized) real grid
    n_train, n_test = len(X_fd.data_matrix), len(X_test_fd.data_matrix)

if INITIAL_SMOOTHING is not None:
    print("\nSmoothing data...", end="")
    if INITIAL_SMOOTHING == "NW":
        smoother = NW()
    elif INITIAL_SMOOTHING == "Basis":
        smoother = BasisSmoother(basis)
    else:
        raise ValueError(
            f"Expected 'NW' or 'Basis' but got {INITIAL_SMOOTHING}.")

    best_smoother = SmoothingParameterSearch(
        smoother,
        smoothing_params,
        scoring=LinearSmootherGeneralizedCVScorer(
            akaike_information_criterion),
        n_jobs=-1,
    )

    with utils.IgnoreWarnings():
        best_smoother.fit(X_fd)

    X_fd = best_smoother.transform(X_fd)
    X_test_fd = best_smoother.transform(X_test_fd)
    print("done")

if STANDARDIZE_PREDICTORS:
    X_sd = np.sqrt(X_fd.var())
else:
    X_sd = 1.0

if STANDARDIZE_RESPONSE:
    Y_m = Y.mean()
    Y_sd = Y.std()
else:
    Y_m = 0.0
    Y_sd = 1.0

# Standardize data
X_m = X_fd.mean(axis=0)
X_fd = (X_fd - X_m)/X_sd
X = X_fd.data_matrix.reshape(-1, N)
X_test_fd = (X_test_fd - X_m)/X_sd
X_test = X_test_fd.data_matrix.reshape(-1, N)
Y = (Y - Y_m)/Y_sd
Y_test = (Y_test - Y_m)/Y_sd

# Mean of response variable
mean_alpha0_init = Y.mean()

# Names of parameters
theta_names = ["β", "τ", "α0", "σ2"]
if TRANSFORM_TAU:
    theta_names_ttr = ["β", "logit τ", "α0", "log σ"]
else:
    theta_names_ttr = ["β", "τ", "α0", "log σ"]
theta_names_aux = ["α0 and log σ"]

# Names of results
results_columns_mcmc = \
    ["Estimator", "Features", "g", "η", "Mean_accpt(%)", "MSE", "RMSE", "R2"]
results_columns_ref = ["Estimator", "Features", "MSE", "RMSE", "R2"]

# Transformations
if TRANSFORM_TAU:
    tau_ttr = utils.Logit()
else:
    tau_ttr = utils.Identity()

# -- Estimators

point_estimates = ["mode", "mean", "median"]
all_estimates = ["posterior_mean"] + point_estimates
num_estimates = len(all_estimates)

###################################################################
# RUN SAMPLER
###################################################################

mean_acceptance = np.zeros((N_REPS, num_ps, num_gs, num_etas))
mse = np.zeros((N_REPS, num_ps, num_gs, num_etas, num_estimates))
rmse = np.zeros((N_REPS, num_ps, num_gs, num_etas, num_estimates))
r2 = np.zeros((N_REPS, num_ps, num_gs, num_etas, num_estimates))
exec_times = np.zeros((N_REPS, num_ps, num_gs, num_etas))
bics = np.zeros((N_REPS, num_ps))
mles = dict.fromkeys(ps)
total_models = np.prod(exec_times.shape[1:])
df_metrics_mle = pd.DataFrame(columns=results_columns_ref)

logging.disable(sys.maxsize)  # Disable logger

try:
    for rep in range(N_REPS):
        print(f"\n(It. {rep + 1}/{N_REPS}) Running {total_models} models...")

        for i, p in enumerate(ps):
            # -- Parameter space

            theta_space = utils.ThetaSpace(
                p, grid, theta_names, theta_names_ttr, "", tau_ttr=tau_ttr)

            #  -- Compute MLE and b0

            if mles[p] is None:
                print(f"  [p={p}] Computing MLE...", end="")

                mle_theta, bic_theta = compute_mle(
                    theta_space, X, Y,
                    method=mle_method,
                    strategy=mle_strategy,
                    rng=rng)

                mles[p] = mle_theta
                bics[rep, i] = bic_theta

                # Simple model using only the MLE to predict

                mle_theta_back = theta_space.backward(mle_theta)
                Y_hat_mle = utils.generate_response(
                    X_test, mle_theta_back, noise=False)
                metrics_mle = utils.regression_metrics(Y_test, Y_hat_mle)
                df_metrics_mle.loc[i] = [
                    "mle",
                    p,
                    metrics_mle["mse"],
                    metrics_mle["rmse"],
                    metrics_mle["r2"]
                ]

                print("done")

            else:
                mle_theta = mles[p]

            b0 = mle_theta[theta_space.beta_idx]

            for j, g in enumerate(gs):
                for k, eta in enumerate(etas):

                    start = time.time()
                    it = iteration_count([num_ps, num_gs, num_etas], [i, j, k])

                    if not PRINT_RESULTS_ONLINE:
                        print(f"  * Launching model #{it}", end="\r")

                    # -- Run sampler

                    if MCMC_ALG == "emcee":

                        p0 = utils.weighted_initial_guess_around_value(
                            theta_space, mle_theta, sd_beta_init, sd_tau_init,
                            mean_alpha0_init, sd_alpha0_init, param_sigma2_init,
                            sd_sigma2_init, n_walkers=n_walkers, rng=rng,
                            frac_random=frac_random)

                        with Pool(N_CORES) as pool:
                            sampler = emcee.EnsembleSampler(
                                n_walkers, theta_space.ndim, log_posterior,
                                pool=pool, args=(Y,),
                                moves=moves
                            )
                            state = sampler.run_mcmc(
                                p0, n_iter_initial, progress=False, store=False
                            )
                            sampler.reset()
                            sampler.run_mcmc(
                                state, n_iter,
                                progress=PRINT_RESULTS_ONLINE,
                                progress_kwargs={
                                    "desc": f"({it}/{total_models})"}
                            )

                        # -- Get InferenceData object

                        autocorr = sampler.get_autocorr_time(quiet=True)
                        max_autocorr = np.max(autocorr)
                        if np.isfinite(max_autocorr):
                            burn = int(3*max_autocorr)
                        else:
                            burn = 500

                        idata = utils.emcee_to_idata(
                            sampler, theta_space, burn, thin,
                            ["y_star"] if return_pp else [], return_ll)

                        mean_acceptance[rep, i, j, k] = 100 * \
                            np.mean(sampler.acceptance_fraction)

                    elif MCMC_ALG == "pymc":

                        model = make_model_pymc(
                            theta_space, g, eta, X, Y, theta_names,
                            theta_names_aux[:1], mle_theta
                        )

                        with model:
                            if USE_NUTS:
                                idata_pymc = pm.sample(
                                    n_samples_nuts, cores=N_CORES,
                                    tune=tune_nuts,
                                    target_accept=target_accept,
                                    return_inferencedata=True,
                                    progressbar=PRINT_RESULTS_ONLINE)
                            else:
                                idata_pymc = pm.sample(
                                    n_samples_metropolis,
                                    cores=N_CORES,
                                    tune=tune_metropolis,
                                    step=pm.Metropolis(),
                                    return_inferencedata=True,
                                    progressbar=PRINT_RESULTS_ONLINE)

                            idata = idata_pymc.sel(
                                draw=slice(burn, None, thin))

                    else:
                        raise ValueError(
                            "Invalid MCMC algorithm. Must be 'emcee' or 'pymc'.")

                    exec_times[rep, i, j, k] = time.time() - start

                    # -- Save metrics

                    # Posterior mean estimate
                    if MCMC_ALG == "emcee":
                        pp_test = utils.generate_pp(
                            idata, X_test, theta_names,
                            thin_pp, rng=rng, progress=False)
                    elif MCMC_ALG == "pymc":
                        model_test = make_model_pymc(
                            theta_space, g, eta, X_test, Y_test,
                            theta_names, theta_names_aux[:1], mle_theta)
                        with model_test:
                            pp_test = utils.generate_pp(
                                idata, X_test,
                                theta_names,
                                rng=rng, progress=False)[:, ::thin_pp, :]

                    Y_hat_pp = pp_test.mean(axis=(0, 1))
                    metrics_pp = utils.regression_metrics(Y_test, Y_hat_pp)
                    mse[rep, i, j, k, 0] = metrics_pp["mse"]
                    rmse[rep, i, j, k, 0] = metrics_pp["rmse"]
                    r2[rep, i, j, k, 0] = metrics_pp["r2"]

                    # Point estimates
                    for m, pe in enumerate(point_estimates):
                        Y_hat_pe = utils.point_predict(
                            X_test, idata,
                            theta_names, pe)
                        metrics_pe = utils.regression_metrics(Y_test, Y_hat_pe)
                        mse[rep, i, j, k, m + 1] = metrics_pe["mse"]
                        rmse[rep, i, j, k, m + 1] = metrics_pe["rmse"]
                        r2[rep, i, j, k, m + 1] = metrics_pe["r2"]

                    if PRINT_RESULTS_ONLINE:
                        min_pe = np.argmin(mse[rep, i, j, k, :])
                        min_mse = mse[rep, i, j, k, min_pe]
                        print(
                            f"  [p={p}, g={g}, η={eta}]: Min MSE = {min_mse:.3f}"
                            f" with '{all_estimates[min_pe]}'")

        if not PRINT_RESULTS_ONLINE:
            print("")

except KeyboardInterrupt:
    print("\n[INFO] Process halted by user. Skipping...")
    rep = rep - 1


###################################################################
# COMPUTE AVERAGE RESULTS
###################################################################

df_metrics_mcmc = pd.DataFrame(columns=results_columns_mcmc)
min_mse = np.inf
min_mse_params = (-1, -1, -1)  # (i, j, k)

if rep + 1 > 0:
    for i, p in enumerate(ps):
        for j, g in enumerate(gs):
            for k, eta in enumerate(etas):
                for m, pe in enumerate(all_estimates):
                    it = iteration_count(
                        [num_ps, num_gs, num_etas, num_estimates], [i, j, k, m])

                    mean_mse = mse[:rep + 1, i, j, k, m].mean()
                    if mean_mse < min_mse:
                        min_mse = mean_mse
                        min_mse_params = (i, j, k)

                    df_metrics_mcmc.loc[it] = [
                        MCMC_ALG + "_" + pe,
                        p, g, eta,
                        (f"{mean_acceptance[:rep + 1, i, j, k].mean():.3f}"
                         f"±{mean_acceptance[:rep + 1, i, j, k].std():.3f}"
                            if MCMC_ALG == "emcee" else "-"),
                        f"{mse[:rep + 1, i, j, k, m].mean():.3f}"
                        f"±{mse[:rep + 1, i, j, k, m].std():.3f}",
                        f"{rmse[:rep + 1, i, j, k, m].mean():.3f}"
                        f"±{rmse[:rep + 1, i, j, k, m].std():.3f}",
                        f"{r2[:rep + 1, i, j, k, m].mean():.3f}"
                        f"±{r2[:rep + 1, i, j, k, m].std():.3f}"
                    ]

df_metrics_mle.sort_values(results_columns_ref[-3], inplace=True)
df_metrics_mcmc.sort_values(results_columns_mcmc[-3], inplace=True)

###################################################################
# BAYESIAN VARIABLE SELECTION ON BEST MODEL
###################################################################

df_metrics_var_sel = pd.DataFrame(columns=results_columns_ref)

if REFIT_BEST_VAR_SEL and rep + 1 > 0:
    i, j, k = min_mse_params
    p, g, eta = ps[i], gs[j], etas[k]
    mle_theta = mles[p]
    theta_space = utils.ThetaSpace(
        p, grid, theta_names, theta_names_ttr, "", tau_ttr=tau_ttr)
    b0 = mle_theta[theta_space.beta_idx]

    # -- Run sampler

    print(f"\nRefitting best model (p={p}, g={g}, η={eta})...")

    if MCMC_ALG == "emcee":

        p0 = utils.weighted_initial_guess_around_value(
            theta_space, mle_theta, sd_beta_init, sd_tau_init,
            mean_alpha0_init, sd_alpha0_init, param_sigma2_init,
            sd_sigma2_init, n_walkers=n_walkers, rng=rng,
            frac_random=frac_random)

        with Pool(N_CORES) as pool:
            sampler = emcee.EnsembleSampler(
                n_walkers, theta_space.ndim, log_posterior,
                pool=pool, args=(Y,),
                moves=moves)
            state = sampler.run_mcmc(
                p0, n_iter_initial, progress=False, store=False)
            sampler.reset()
            sampler.run_mcmc(state, n_iter, progress=False)

        # -- Get InferenceData object

        autocorr = sampler.get_autocorr_time(quiet=True)
        max_autocorr = np.max(autocorr)
        if np.isfinite(max_autocorr):
            burn = int(3*max_autocorr)
        else:
            burn = 500

        logging.disable(logging.NOTSET)  # Re-enable logger

        idata = utils.emcee_to_idata(
            sampler, theta_space, burn, thin,
            ["y_star"] if return_pp else [], return_ll)

    elif MCMC_ALG == "pymc":

        model = make_model_pymc(
            theta_space, g, eta, X, Y, theta_names,
            theta_names_aux[:1], mle_theta
        )

        with model:
            if USE_NUTS:
                idata_pymc = pm.sample(
                    n_samples_nuts, cores=N_CORES,
                    tune=tune_nuts,
                    target_accept=target_accept,
                    return_inferencedata=True,
                    progressbar=False)
            else:
                step = pm.Metropolis()
                idata_pymc = pm.sample(
                    n_samples_metropolis,
                    cores=N_CORES,
                    tune=tune_metropolis, step=step,
                    return_inferencedata=True,
                    progressbar=False)

            idata = idata_pymc.sel(
                draw=slice(burn, None, thin))

    else:
        raise ValueError("Invalid MCMC algorithm. Must be 'emcee' or 'pymc'.")

    # -- Bayesian variable selection

    print("Fitting sklearn models with Bayesian variable selection...")

    for pe in point_estimates:
        df_metrics_var_sel = df_metrics_var_sel.append(
            bayesian_var_sel(
                idata, theta_space, theta_names, X_fd,
                Y, X_test_fd, Y_test, folds, results_columns_ref,
                prefix=MCMC_ALG, point_est=pe))

    df_metrics_var_sel.sort_values(results_columns_ref[-3], inplace=True)


###################################################################
# FIT SKLEARN MODELS
###################################################################

if FIT_REF_ALGS:

    # -- Select family of regressors

    regressors = []

    alphas = np.logspace(-4, 4, 20)
    n_selected = [5, 10, 15, 20, 25, X.shape[1]]
    n_components = [2, 3, 4, 5, 10]
    n_basis_bsplines = [8, 10, 12, 14, 16]
    n_basis_fourier = [3, 5, 7, 9, 11]
    n_neighbors = [3, 5, 7]

    basis_bspline = [BSpline(n_basis=p) for p in n_basis_bsplines]
    basis_fourier = [Fourier(n_basis=p) for p in n_basis_fourier]

    basis_fpls = []
    for p in n_components:
        try:
            basis_fpls.append(FPLSBasis(X_fd, Y, n_basis=p))
        except ValueError:
            print(f"Can't create FPLSBasis with n_basis={p}")
            continue

    params_reg = {"reg__alpha": alphas}
    params_svm = {"reg__C": alphas,
                  "reg__gamma": ['auto', 'scale']}
    params_select = {"selector__p": n_selected}
    params_pls = {"reg__n_components": n_components}
    params_dim_red = {"dim_red__n_components": n_components}
    params_basis = {"basis__basis": basis_bspline + basis_fourier}
    params_basis_fpca = {"basis__n_basis": n_components}
    params_basis_fpls = {"basis__basis": basis_fpls}
    params_knn = {"reg__n_neighbors": n_neighbors,
                  "reg__weights": ['uniform', 'distance']}
    params_mrmr = {"var_sel__method": ["MID", "MIQ"],
                   "var_sel__n_features_to_select": n_components}

    """
    MULTIVARIATE MODELS
    """

    # Lasso
    regressors.append(("sk_lasso",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("reg", Lasso())]),
                       params_reg
                       ))

    # PLS1 regression
    regressors.append(("sk_pls1",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("reg", PLSRegressionWrapper())]),
                       params_pls
                       ))

    """
    VARIABLE SELECTION + MULTIVARIATE MODELS
    """

    # Manual+Ridge
    regressors.append(("manual_sel+sk_ridge",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("selector", FeatureSelector()),
                           ("reg", Ridge())]),
                       {**params_reg, **params_select}
                       ))

    # FPCA+Ridge
    regressors.append(("fpca+sk_ridge",
                       Pipeline([
                           ("dim_red", FPCA()),  # Retains scores only
                           ("reg", Ridge())]),
                       {**params_dim_red, **params_reg}
                       ))

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # FPLS (fixed basis)+Ridge
    regressors.append(("fpls_basis+sk_ridge",
                       Pipeline([
                           ("basis", Basis()),
                           ("dim_red", FPLS()),
                           ("reg", Ridge())]),
                       {**params_basis, **params_dim_red, **params_reg}
                       ))

    """

    # PCA+Ridge
    regressors.append(("pca+sk_ridge",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PCA(random_state=SEED)),
                           ("reg", Ridge())]),
                       {**params_dim_red, **params_reg}
                       ))

    # PLS+Ridge
    regressors.append(("pls+sk_ridge",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PLSRegressionWrapper()),
                           ("reg", Ridge())]),
                       {**params_dim_red, **params_reg}
                       ))

    # RMH+Ridge
    regressors.append(("rmh+sk_ridge",
                       Pipeline([
                           ("var_sel", RMH()),
                           ("reg", Ridge())]),
                       params_reg
                       ))

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # mRMR+Ridge
    regressors.append(("mRMR+sk_ridge",
                       Pipeline([
                           ("var_sel", mRMR()),
                           ("reg", Ridge())]),
                       {**params_mrmr, **params_reg}
                       ))
    """

    # Manual+SVM RBF
    regressors.append(("manual_sel+sk_svm_rbf",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("selector", FeatureSelector()),
                           ("reg", SVR(kernel='rbf'))]),
                       {**params_select, **params_svm}
                       ))

    # FPCA+SVM RBF
    regressors.append(("fpca+sk_svm_rbf",
                       Pipeline([
                           ("dim_red", FPCA()),  # Retains scores only
                           ("reg", SVR(kernel='rbf'))]),
                       {**params_dim_red, **params_svm}
                       ))

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # FPLS (fixed basis)+SMV RBF
    regressors.append(("fpls_basis+sk_svm_rbf",
                       Pipeline([
                           ("basis", Basis()),
                           ("dim_red", FPLS()),
                           ("reg", SVR(kernel='rbf'))]),
                       {**params_basis, **params_dim_red, **params_svm}
                       ))
    """

    # PCA+SVM RBF
    regressors.append(("pca+sk_svm_rbf",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PCA(random_state=SEED)),
                           ("reg", SVR(kernel='rbf'))]),
                       {**params_dim_red, **params_svm}
                       ))

    # PLS+SMV RBF
    regressors.append(("pls+sk_svm_rbf",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PLSRegressionWrapper()),
                           ("reg", SVR(kernel='rbf'))]),
                       {**params_dim_red, **params_svm}
                       ))

    # RMH+SVM RBF
    regressors.append(("rmh+sk_svm_rbf",
                       Pipeline([
                           ("var_sel", RMH()),
                           ("reg", SVR(kernel='rbf'))]),
                       params_svm
                       ))

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # mRMR+SVM RBF
    regressors.append(("mRMR+sk_svm_rbf",
                       Pipeline([
                           ("var_sel", mRMR()),
                           ("reg", SVR(kernel='rbf'))]),
                       {**params_mrmr, **params_svm}
                       ))
    """

    """
    FUNCTIONAL MODELS
    """

    regressors.append(("sk_apls",
                       Pipeline([
                           ("reg", APLS())]),
                       params_pls
                       ))

    # NOTE: while not strictly necessary, the test data undergoes the
    # same basis expansion process as the training data. This is more
    # computationally efficient and seems to improve the performance.

    # Fixed basis + Functional Linear Regression
    regressors.append(("sk_flin_basis",
                       Pipeline([
                           ("basis", Basis()),
                           ("reg", FLinearRegression())]),
                       params_basis
                       ))

    """
    TARDA BASTANTE (cálculo de Gram matrix costoso en la base)

    # FPCA basis + Functional Linear Regression
    regressors.append(("sk_flin_khl",
                       Pipeline([
                           ("basis", FPCABasis()),
                           ("reg", FLinearRegression())]),
                       params_basis_fpca
                       ))
    """

    """
    TARDA BASTANTE (cálculo de Gram matrix costoso en la base)

    # FPLS basis + Functional Linear Regression
    regressors.append(("sk_flin_fpls",
                       Pipeline([
                           ("basis", Basis()),
                           ("reg", FLinearRegression())]),
                       params_basis_fpls
                       ))
    """

    # Fixed basis + FPLS1 regression
    regressors.append(("sk_fpls1_basis",
                       Pipeline([
                           ("basis", Basis()),
                           ("reg", FPLS())]),
                       {**params_basis, **params_pls}
                       ))

    # KNeighbors Functional Regression
    regressors.append(("sk_fknn",
                       Pipeline([
                           ("reg", KNeighborsRegressor())]),
                       params_knn
                       ))

    # -- Fit model and save metrics

    print("Fitting reference sklearn models...")
    df_metrics_sk, reg_cv = cv_sk(regressors, folds, X_fd, Y, X_test_fd,
                                  Y_test, results_columns_ref, verbose=True)

###################################################################
# PRINT RESULTS
###################################################################

if SYNTHETIC_DATA:
    data_name = kernel_fn.__name__ + "_" + MODEL_GEN
    if MODEL_GEN == "L2":
        data_name += ("_" + beta_coef.__name__)
else:
    data_name = REAL_DATA

if INITIAL_SMOOTHING == "NW":
    smoothing_str = best_smoother.best_estimator_.__class__.__name__
elif INITIAL_SMOOTHING == "Basis":
    smoothing_str = basis.__class__.__name__
else:
    smoothing_str = "none"

filename = ("reg_" + MCMC_ALG + "_" + data_name + "_smoothing_" + smoothing_str
            + "_frac_random_" + str(int(100*frac_random)) + "_walkers_"
            + str(n_walkers) + "_iters_" + str(n_iter) + "_reps_"
            + str(rep + 1) + "_seed_" + str(SEED))

if PRINT_TO_FILE:
    print(f"\nSaving results to file '{filename}'")
    f = open(filename + ".results", 'w')
    sys.stdout = f  # Change the standard output to the file we created
else:
    print("\nShowing results...\n")

print(" --> Bayesian Functional Linear Regression <--\n")
print("-- MODEL GENERATION --")
print(f"Train/test size: {n_train}/{n_test}")
if STANDARDIZE_PREDICTORS:
    print("Standardized predictors")
if STANDARDIZE_RESPONSE:
    print("Standardized response")
if INITIAL_SMOOTHING == "NW":
    print(
        f"Smoothing: {best_smoother.best_estimator_.__class__.__name__}"
        f"(λ={best_smoother.best_params_['smoothing_parameter']:.3f})")
elif INITIAL_SMOOTHING == "Basis":
    print(f"Smoothing: {basis.__class__.__name__}(n={N_BASIS})")
else:
    print("Smoothing: none")
print(f"Transform tau: {'true' if TRANSFORM_TAU else 'false'}")

if SYNTHETIC_DATA:
    print(f"Model type: {MODEL_GEN}")
    print(f"X ~ GP(0, {kernel_fn.__name__})")
    print("True parameters:")
    if MODEL_GEN == "RKHS":
        print(f"  β: {beta_true}\n  τ: {tau_true}")
    else:
        print(f"  β(t): {beta_coef.__name__}")
    print(f"  α0: {alpha0_true}\n  σ2: {sigma2_true}")
else:
    print(f"Data name: {REAL_DATA}")

if rep + 1 > 0:
    print("\n-- MLE PERFORMANCE --")
    bics_mean = bics[:rep + 1, ...].mean(axis=0)
    for i, p in enumerate(ps):
        print(f"BIC [p={p}]: {bics_mean[i]:.3f}")
    print("")
    print(df_metrics_mle.to_string(index=False, col_space=6))

    if MCMC_ALG == "emcee":
        print("\n-- EMCEE SAMPLER --")
        print(f"N_walkers: {n_walkers}")
        print(f"N_iters: {n_iter}")
        print(f"MLE: {mle_method} + {mle_strategy}")
        print(f"Frac_random: {frac_random}")
        print(f"Burn-in: {n_iter_initial} + 3*max_autocorr")
        print(f"Thinning chain/pp: {thin}/{thin_pp}")
        print("Moves:")
        for move, prob in moves:
            print(f"  {move.__class__.__name__}, {prob}")
    elif MCMC_ALG == "pymc":
        print("\n-- PYMC SAMPLER --")
        print(f"N_walkers: {N_CORES}")
        print("N_iters: "
              + (f"{n_samples_nuts + tune_nuts}" if USE_NUTS else
                 f"{n_samples_metropolis + tune_metropolis}"))
        print(f"MLE: {mle_method} + {mle_strategy}")
        print("Burn-in: "
              + (f"{tune_nuts + burn}" if USE_NUTS else
                 f"{tune_metropolis + burn}"))
        print(f"Thinning chain/pp: {thin}/{thin_pp}")
        print("Underlying algorithm: " + ("NUTS" if USE_NUTS else "Metropolis"))
        if USE_NUTS:
            print(f"  Target accept: {target_accept}")
    else:
        raise ValueError("Invalid MCMC algorithm. Must be 'emcee' or 'pymc'.")

    print(f"\n-- RESULTS {MCMC_ALG.upper()} --")
    print(f"Random iterations: {rep + 1}")
    print(
        f"Mean execution time: {exec_times[:rep + 1, ...].mean():.3f}"
        f"±{exec_times[:rep + 1, ...].std():.3f} s")
    print(f"Total execution time: {exec_times.sum()/60.:.3f} min\n")
    print(df_metrics_mcmc.to_string(index=False, col_space=6))

    if REFIT_BEST_VAR_SEL:
        print("\n-- RESULTS BAYESIAN VARIABLE SELECTION --")
        i, j, k = min_mse_params
        p, g, eta = ps[i], gs[j], etas[k]
        print(f"Base model: p={p}, g={g}, η={eta}\n")
        print(df_metrics_var_sel.to_string(index=False, col_space=6))

if FIT_REF_ALGS:
    print("\n-- RESULTS SKLEARN --\n")
    print(df_metrics_sk.to_string(index=False, col_space=6))

###################################################################
# SAVE RESULTS
###################################################################

try:
    if SAVE_RESULTS and rep + 1 > 0:
        # Save all the results dataframe in one CSV file
        df_all = [df_metrics_mcmc]
        if REFIT_BEST_VAR_SEL:
            df_all += [df_metrics_var_sel]
        if FIT_REF_ALGS:
            df_all += [df_metrics_sk]

        df = pd.concat(
            df_all,
            axis=0,
            ignore_index=True)
        df.to_csv("out/" + filename + ".csv", index=False)

        # Save the top MSE values to arrays
        mcmc_best = df_metrics_mcmc["MSE"][:SAVE_TOP].apply(
            lambda s: s.split("±")[0]).to_numpy(dtype=float)
        if REFIT_BEST_VAR_SEL:
            var_sel_best = df_metrics_var_sel["MSE"][:SAVE_TOP]
        else:
            var_sel_best = np.zeros(1)
        if FIT_REF_ALGS:
            sk_best = df_metrics_sk["MSE"][:SAVE_TOP]
        else:
            sk_best = np.zeros(1)

        np.savez(
            "out/" + filename + ".npz",
            mcmc_best=mcmc_best,
            var_sel_best=var_sel_best,
            sk_best=sk_best,
        )
except Exception as ex:
    print(ex)
