# encoding: utf-8

import numpy as np
import pandas as pd
import warnings
import sys
import os
import logging
from itertools import product
import time
import scipy
from multiprocessing import Pool

import skfda
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from skfda.ml.regression import LinearRegression as FLinearRegression
from skfda.ml.regression import KNeighborsRegressor
from skfda.representation.basis import FDataBasis, Fourier
from skfda.representation.grid import FDataGrid

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

import utils

import emcee

# Ignore warnings
os.environ["PYTHONWARNINGS"] = 'ignore::UserWarning'
np.seterr(over='ignore', divide='ignore')

# Floating point precision for display
np.set_printoptions(precision=3, suppress=True)
pd.set_option("display.precision", 3)
pd.set_option('display.max_columns', 80)

###################################################################
# GLOBAL OPTIONS
###################################################################

# Randomness and reproducibility
SEED = int(sys.argv[1])
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# Multiprocessing
N_CORES = os.cpu_count()

# Data
SYNTHETIC_DATA = True
MODEL_GEN = "RKHS"  # 'L2' or 'RKHS
REAL_DATA = "Tecator"
STANDARDIZE_PREDICTORS = False
STANDARDIZE_RESPONSE = False
BASIS_REPRESENTATION = True
N_BASIS = 11
TRANSFORM_TAU = False

kernel_fn = utils.fractional_brownian_kernel
beta_coef = utils.cholaquidis_scenario3
basis = Fourier(n_basis=N_BASIS)
folds = KFold(shuffle=True, random_state=SEED)

# Results
FIT_REF_ALGS = True
REFIT_BEST_VAR_SEL = True
PRINT_RESULTS_ONLINE = False
PRINT_TO_FILE = False
SAVE_RESULTS = False

###################################################################
# HYPERPARAMETERS
###################################################################

# -- Cv and Model parameters

N_REPS = 5

mle_method = 'L-BFGS-B'
mle_strategy = 'global'

ps = [2, 3, 4]
gs = [5]
etas = [0.01, 0.1, 1.0, 10.0]
num_ps = len(ps)
num_gs = len(gs)
num_etas = len(etas)

# -- Emcee sampler parameters

n_walkers = 64
n_iter_initial = 100
n_iter = 1000
return_pp = False
return_ll = False
thin = 1
thin_pp = 5
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


def optimizer_global(rng, args):
    theta_init, X, Y, theta_space, method, bounds = args
    return scipy.optimize.basinhopping(
        neg_ll,
        x0=theta_init,
        seed=rng,
        minimizer_kwargs={"args": (X, Y, theta_space),
                          "method": method,
                          "bounds": bounds}
    ).x


def compute_mle(
        theta_space, X, Y,
        method='Powell',
        strategy='global',
        rng=None):
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

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
                rngs = [np.random.default_rng(SEED + i)
                        for i in range(N_CORES)]
                args_optim = [theta_init, X, Y, theta_space, method, bounds]
                mles = pool.starmap(
                    optimizer_global, product(rngs, [args_optim]))
                bics = utils.bic = utils.compute_bic(
                    theta_space, neg_ll, mles, X, Y)
                mle_theta = mles[np.argmin(bics)]
                bic = bics[np.argmin(bics)]
        else:
            raise ValueError(
                "Parameter 'strategy' must be one of {'local', 'global'}.")

    return mle_theta, bic


# -- Log-posterior model

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


# -- Sklearn CV and transformers

def cv_sk(regressors, folds, X, Y, X_test, Y_test, columns_name, verbose=False):

    df_metrics_sk = pd.DataFrame(columns=columns_name)

    for i, (name, pipe, params) in enumerate(regressors):
        if verbose:
            print(f"  Fitting {name}...")
        reg_cv = GridSearchCV(pipe, params, scoring="neg_mean_squared_error",
                              n_jobs=N_CORES, cv=folds)
        reg_cv.fit(X, Y)
        Y_hat_sk = reg_cv.predict(X_test)
        metrics_sk = utils.regression_metrics(Y_test, Y_hat_sk)

        if name == "sk_fknn":
            n_features = f"K={reg_cv.best_params_['reg__n_neighbors']}"
        elif "svm" in name:
            n_features = reg_cv.best_estimator_["reg"].n_features_in_
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

    df_metrics_sk.sort_values(columns_name[-2], inplace=True)

    return df_metrics_sk


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

    # Emcee+Lasso
    regressors_var_sel.append((f"{prefix}_{point_est}+sk_lasso",
                               Pipeline([
                                   ("var_sel", VariableSelection(grid, idx_hat)),
                                   ("data_matrix", DataMatrix()),
                                   ("reg", Lasso())]),
                               params_reg
                               ))

    # Emcee+Ridge
    regressors_var_sel.append((f"{prefix}_{point_est}+sk_ridge",
                               Pipeline([
                                   ("var_sel", VariableSelection(grid, idx_hat)),
                                   ("data_matrix", DataMatrix()),
                                   ("reg", Ridge())]),
                               params_reg
                               ))

    # Emcee+SVM RBF
    regressors_var_sel.append((f"{prefix}_{point_est}+sk_svm_rbf",
                               Pipeline([
                                   ("var_sel", VariableSelection(grid, idx_hat)),
                                   ("data_matrix", DataMatrix()),
                                   ("reg", SVR(kernel='rbf'))]),
                               params_svm
                               ))

    df_metrics_var_sel = cv_sk(regressors_var_sel, folds, X, Y,
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
        raise ValueError(
            f"Model generation must be 'L2' or 'RKHS', but got {MODEL_GEN}")

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

if BASIS_REPRESENTATION:
    X_fd = X_fd.to_basis(basis).to_grid(X_fd.grid_points[0])
    X_test_fd = X_test_fd.to_basis(basis).to_grid(X_fd.grid_points[0])

if STANDARDIZE_PREDICTORS:
    X_sd = X_fd.data_matrix.std(axis=0)
else:
    X_sd = np.ones(X_fd.data_matrix.shape[1:])

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
results_columns_emcee = \
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

print(f"Random seed: {SEED}")
print(f"Num. cores: {N_CORES}")

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
                        sampler.run_mcmc(state, n_iter, progress=PRINT_RESULTS_ONLINE,
                                         progress_kwargs={"desc": f"({it}/{total_models})"})

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

                    exec_times[rep, i, j, k] = time.time() - start

                    # -- Save metrics

                    mean_acceptance[rep, i, j, k] = 100 * \
                        np.mean(sampler.acceptance_fraction)

                    # Posterior mean estimate
                    pp_test = utils.generate_pp(
                        idata, X_test, theta_names, thin_pp, rng=rng, progress=False)
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
                            f"  [p={p}, g={g}, η={eta}]: Min MSE = {min_mse:.3f} "
                            f"with '{all_estimates[min_pe]}'")

        if not PRINT_RESULTS_ONLINE:
            print("")

except KeyboardInterrupt:
    print("\n[INFO] Process halted by user. Skipping...")
    rep = rep - 1

finally:
    rep = N_REPS - 1

###################################################################
# COMPUTE AVERAGE RESULTS
###################################################################

df_metrics_emcee = pd.DataFrame(columns=results_columns_emcee)
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

                    df_metrics_emcee.loc[it] = [
                        "emcee_" + pe,
                        p, g, eta,
                        f"{mean_acceptance[:rep + 1, i, j, k].mean():.3f}"
                        f"±{mean_acceptance[:rep + 1, i, j, k].std():.3f}",
                        f"{mse[:rep + 1, i, j, k, m].mean():.3f}"
                        f"±{mse[:rep + 1, i, j, k, m].std():.3f}",
                        f"{rmse[:rep + 1, i, j, k, m].mean():.3f}"
                        f"±{rmse[:rep + 1, i, j, k, m].std():.3f}",
                        f"{r2[:rep + 1, i, j, k, m].mean():.3f}"
                        f"±{r2[:rep + 1, i, j, k, m].std():.3f}"
                    ]

df_metrics_mle.sort_values(results_columns_ref[-2], inplace=True)
df_metrics_emcee.sort_values(results_columns_emcee[-2], inplace=True)

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

    # -- Bayesian variable selection

    print("Fitting sklearn models with Bayesian variable selection...")

    for pe in point_estimates:
        df_metrics_var_sel = df_metrics_var_sel.append(
            bayesian_var_sel(
                idata, theta_space, theta_names, X_fd,
                Y, X_test_fd, Y_test, folds, results_columns_ref,
                prefix="emcee", point_est=pe))

    df_metrics_var_sel.sort_values(results_columns_ref[-2], inplace=True)

###################################################################
# FIT SKLEARN MODELS
###################################################################

if FIT_REF_ALGS:

    # -- Select family of regressors

    regressors = []

    alphas = np.logspace(-4, 4, 20)
    n_selected = [5, 10, 15, 20, 25, X.shape[1]]
    n_components = [2, 3, 4, 5, 6, 10]
    n_basis = [3, 5, 7, 9, 11]
    basis_fourier = [Fourier(n_basis=p) for p in n_basis]
    n_neighbors = [3, 5, 7]

    params_reg = {"reg__alpha": alphas}
    params_svm = {"reg__C": alphas,
                  "reg__gamma": ['auto', 'scale']}
    params_select = {"selector__p": n_selected}
    params_fpca = {"dim_red__n_components": n_components}
    params_basis = {"basis__basis": basis_fourier}
    params_knn = {"reg__n_neighbors": n_neighbors,
                  "reg__weights": ['uniform', 'distance']}

    # Lasso
    regressors.append(("sk_lasso",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("reg", Lasso())]),
                       params_reg
                       ))

    # Manual+Ridge
    regressors.append(("manual_sel+sk_ridge",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("selector", FeatureSelector()),
                           ("reg", Ridge())]),
                       {**params_reg, **params_select}
                       ))

    # FPCA+Lin
    regressors.append(("fpca+sk_lin",
                       Pipeline([
                           ("dim_red", FPCA()),  # Retains scores only
                           ("reg", LinearRegression())]),
                       params_fpca
                       ))

    # FPCA+Ridge
    regressors.append(("fpca+sk_ridge",
                       Pipeline([
                           ("dim_red", FPCA()),  # Retains scores only
                           ("reg", Ridge())]),
                       {**params_fpca, **params_reg}
                       ))

    # FPCA+SVM RBF
    regressors.append(("fpca+sk_svm_rbf",
                       Pipeline([
                           ("dim_red", FPCA()),  # Retains scores only
                           ("reg", SVR(kernel='rbf'))]),
                       {**params_fpca, **params_svm}
                       ))

    # Manual+SVM RBF
    regressors.append(("manual_sel+sk_svm_rbf",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("selector", FeatureSelector()),
                           ("reg", SVR(kernel='rbf'))]),
                       {**params_select, **params_svm}
                       ))

    # Functional Linear Regression
    regressors.append(("sk_flin",
                       Pipeline([
                           ("basis", Basis()),
                           ("reg", FLinearRegression())]),
                       params_basis
                       ))

    # KNeighbors Functional Regression
    regressors.append(("sk_fknn",
                       Pipeline([
                           ("reg", KNeighborsRegressor())]),
                       params_knn
                       ))

    # -- Fit model and save metrics

    print("Fitting reference sklearn models...")
    df_metrics_sk = cv_sk(regressors, folds, X_fd, Y, X_test_fd,
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

filename = ("emcee_" + data_name + "_basis_"
            + (basis.__class__.__name__ if BASIS_REPRESENTATION else "None")
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
if BASIS_REPRESENTATION:
    print(f"Basis: {basis.__class__.__name__}(n={N_BASIS})")
else:
    print("Basis: None")
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

    print("\n-- RESULTS EMCEE --")
    print(f"Random iterations: {rep + 1}")
    print(
        f"Mean execution time: {exec_times[:rep + 1, ...].mean():.3f}"
        f"±{exec_times[:rep + 1, ...].std():.3f} s")
    print(f"Total execution time: {exec_times.sum()/60.:.3f} min\n")
    print(df_metrics_emcee.to_string(index=False, col_space=6))

    if REFIT_BEST_VAR_SEL:
        print("\n-- RESULTS BAYESIAN VARIABLE SELECTION --\n")
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

if SAVE_RESULTS and rep + 1 > 0:
    np.savez(
        filename + ".npz",
        mean_acceptance=mean_acceptance[:rep + 1, ...],
        mse=mse[:rep + 1, ...],
        rmse=rmse[:rep + 1, ...],
        r2=r2[:rep + 1, ...]
    )
