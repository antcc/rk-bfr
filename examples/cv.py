# encoding: utf-8

import numpy as np
import pandas as pd
import sys
import os
import logging
import time
import scipy
from multiprocessing import Pool

import skfda
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from skfda.ml.regression import LinearRegression as FLinearRegression
from skfda.ml.regression import KNeighborsRegressor
from skfda.representation.basis import FDataBasis, Fourier

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

# Floating point precision for display
np.set_printoptions(precision=3, suppress=True)
pd.set_option("display.precision", 3)
pd.set_option('display.max_columns', 80)


# -- MLE computation

def neg_ll(theta_tr, X, Y, theta_space):
    """Transformed parameter vector 'theta_tr' is (β, logit τ, α0, log σ)."""

    n, N = X.shape
    grid = np.linspace(1./N, 1., N)

    assert len(theta_tr) == theta_space.ndim

    theta = theta_space.backward(theta_tr)
    beta, tau, alpha0, sigma2 = theta_space.get_params(theta)
    log_sigma = theta_space.get_sigma2(theta_tr)

    idx = np.abs(grid - tau[:, np.newaxis]).argmin(1)
    X_tau = X[:, idx]

    return -(-n*log_sigma
             - np.linalg.norm(Y - alpha0 - X_tau@beta)**2/(2*sigma2))


def compute_mle(
        theta_space, X, Y,
        method='Powell',
        strategy='global',
        rng=None):
    p = theta_space.p

    theta_init = theta_space.forward(
        np.array([0.0]*p + [0.5]*p + [mean_alpha0_init] + [1.0]))

    if TRANSFORM_TAU:
        bounds = None
    else:
        bounds = ([(None, None)]*p
                  + [(theta_space.tau_lb, theta_space.tau_ub)]*p
                  + [(None, None)]
                  + [(None, None)])

    if strategy == 'local':
        mle = scipy.optimize.minimize(
            neg_ll,
            x0=theta_init,
            bounds=bounds,
            method=method,
            args=(X, Y, theta_space)
        )
    elif strategy == 'global':
        mle = scipy.optimize.basinhopping(
            neg_ll,
            x0=theta_init,
            seed=rng,
            minimizer_kwargs={"args": (X, Y, theta_space),
                              "method": method,
                              "bounds": bounds}
        )
    else:
        raise ValueError("Parameter 'strategy' must be one of {'local', 'global'}.")

    return mle.x

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
                 - (p + 2)*log_sigma
                 - b.T@G_tau_reg@b/(2*g*sigma2))

    return log_prior


def log_likelihood(theta_tr, Y):
    """Global parameters (for efficient parallelization): X"""
    return -neg_ll(theta_tr, X, Y, theta_space)


def log_posterior(theta_tr, Y):
    """Global parameters (for efficient parallelization): X, rng, return_pps"""
    lp = log_prior(theta_tr)

    if not np.isfinite(lp):
        if return_pps:
            return -np.inf, np.full_like(Y, -np.inf)
        else:
            return -np.inf

    ll = log_likelihood(theta_tr, Y)
    lpos = lp + ll

    if return_pps:
        theta = theta_space.backward(theta_tr)
        pps = utils.generate_response(X, theta, rng=rng)
        return lpos, pps
    else:
        return lpos

# -- Sklearn transformers


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
# GLOBAL OPTIONS
###################################################################

# Randomness and reproducibility
SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# Multiprocessing
N_CORES = 4

# Data
SYNTHETIC_DATA = False
MODEL_GEN = "L2"
kernel_fn = utils.fractional_brownian_kernel
beta_coef = utils.grollemund_smooth
TRANSFORM_TAU = False

# Results
FIT_SKLEARN = False
PRINT_RESULTS_ONLINE = False
PRINT_TO_FILE = True
SAVE_RESULTS = False

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
        X, Y = utils.generate_gp_l2_dataset(
            grid, kernel_fn,
            n_train, beta_coef, alpha0_true,
            sigma2_true, rng=rng
        )

        X_test, Y_test = utils.generate_gp_l2_dataset(
            grid, kernel_fn,
            n_test, beta_coef, alpha0_true,
            sigma2_true, rng=rng
        )

    elif MODEL_GEN == "RKHS":
        X, Y = utils.generate_gp_rkhs_dataset(
            grid, kernel_fn,
            n_train, beta_true, tau_true,
            alpha0_true, sigma2_true, rng=rng
        )

        X_test, Y_test = utils.generate_gp_rkhs_dataset(
            grid, kernel_fn,
            n_test, beta_true, tau_true,
            alpha0_true, sigma2_true, rng=rng
        )

    else:
        raise ValueError(
            f"Model generation must be 'L2' or 'RKHS', but got {MODEL_GEN}")

    # Create FData object
    X_fd = skfda.FDataGrid(X, grid)
    X_test_fd = skfda.FDataGrid(X_test, grid)

else:
    real_data_name = "Tecator"
    X_tecator, Y_tecator = skfda.datasets.fetch_tecator(return_X_y=True)
    Y_tecator = Y_tecator[:, 1]  # Fat percentage

    X_fd, X_test_fd, Y, Y_test = train_test_split(
        X_tecator, Y_tecator, train_size=0.8, random_state=SEED)

    N = len(X_fd.grid_points[0])
    grid = np.linspace(1./N, 1., N)  # TODO: use (normalized) real grid
    n_train, n_test = len(X_fd.data_matrix), len(X_test_fd.data_matrix)

# Standardize data
X_m = X_fd.mean(axis=0)
X_fd = X_fd - X_m
X = X_fd.data_matrix.reshape(-1, N)
X_test_fd = X_test_fd - X_m
X_test = X_test_fd.data_matrix.reshape(-1, N)

# Names of parameters
theta_names = ["β", "τ", "α0", "σ2"]
if TRANSFORM_TAU:
    theta_names_ttr = ["β", "logit τ", "α0", "log σ"]
else:
    theta_names_ttr = ["β", "τ", "α0", "log σ"]
theta_names_aux = ["α0 and log σ"]

# Names of results
results_columns_emcee = \
    ["Estimator", "Features", "g", "η", "Mean_accpt (%)", "MSE", "RMSE", "R2"]
results_columns_sk = ["Estimator", "Features", "MSE", "RMSE", "R2"]

# Transformations
if TRANSFORM_TAU:
    tau_ttr = utils.Logit()
else:
    tau_ttr = utils.Identity()

###################################################################
# HYPERPARAMETERS
###################################################################

# -- Cv and Model parameters

N_REPS = 10

mle_method = 'L-BFGS-B'
mle_strategy = 'global'

ps = [6, 7, 8]
gs = [5, 20]
etas = [0.001, 0.01, 0.1, 1.0, 10.0]
num_ps = len(ps)
num_gs = len(gs)
num_etas = len(etas)

# -- Emcee sampler parameters

n_walkers = 100
n_iter_initial = 100
n_iter = 1000
return_pps = False
thin = 1
frac_random = 0.2

sd_beta_init = 10.0
sd_tau_init = 0.2
mean_alpha0_init = Y.mean()
sd_alpha0_init = 1.0
param_sigma2_init = 2.0  # shape parameter in inv_gamma distribution
sd_sigma2_init = 1.0

# -- Metrics parameters

point_estimates = ["mode", "mean", "median"]
num_estimates = len(point_estimates)

###################################################################
# RUN SAMPLER
###################################################################

print(f"Random seed: {SEED}")
print(f"Num. cores: {N_CORES}")

mean_acceptance = np.zeros((N_REPS, num_ps, num_gs, num_etas))
mse = np.zeros((N_REPS, num_ps, num_gs, num_etas, num_estimates))
r2 = np.zeros((N_REPS, num_ps, num_gs, num_etas, num_estimates))
exec_times = np.zeros((N_REPS, num_ps, num_gs, num_etas))
bic = np.zeros((N_REPS, num_ps))
total_models = np.prod(exec_times.shape[1:])

logging.disable(sys.maxsize)  # Disable logger

try:
    for rep in range(N_REPS):
        print(f"\n(It. {rep + 1}/{N_REPS}) Running {total_models} models...")
        for i, p in enumerate(ps):
            # -- Parameter space

            theta_space = utils.ThetaSpace(
                p, grid, theta_names, theta_names_ttr, "", tau_ttr=tau_ttr)

            #  -- Compute MLE and initial points

            print(f"[p={p}] Computing MLE")

            mle_theta = compute_mle(
                theta_space, X, Y,
                method=mle_method,
                strategy=mle_strategy,
                rng=rng)

            b0 = mle_theta[theta_space.beta_idx]

            bic[rep, i] = (theta_space.ndim*np.log(n_train)
                           + 2*neg_ll(mle_theta, X, Y, theta_space))

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
                            n_walkers, 2*p + 2, log_posterior, pool=pool, args=(Y,))
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
                        sampler, theta_space, burn, thin, return_pps)

                    exec_times[rep, i, j, k] = time.time() - start

                    # -- Save metrics

                    mean_acceptance[rep, i, j, k] = 100 * \
                        np.mean(sampler.acceptance_fraction)

                    for m, pe in enumerate(point_estimates):
                        Y_hat = utils.point_predict(
                            X_test, idata,
                            theta_names, pe)
                        metrics = utils.regression_metrics(Y_test, Y_hat)
                        mse[rep, i, j, k, m] = metrics["mse"]
                        r2[rep, i, j, k, m] = metrics["r2"]

                    if PRINT_RESULTS_ONLINE:
                        print(f"  p={p}, g={g}, η={eta}")
                        print(
                            f"  --> Smallest MSE: {np.min(mse[rep, i, j, k, :]):.3f}")

except KeyboardInterrupt:
    print("\n[INFO] Process halted by user. Skipping...")
    rep = rep - 1

logging.disable(logging.NOTSET)  # Re-enable logger

###################################################################
# COMPUTE AVERAGE RESULTS
###################################################################

df_metrics_emcee = pd.DataFrame(columns=results_columns_emcee)

for i, p in enumerate(ps):
    for j, g in enumerate(gs):
        for k, eta in enumerate(etas):
            for m, pe in enumerate(point_estimates):
                it = iteration_count(
                    [num_ps, num_gs, num_etas, num_estimates], [i, j, k, m])

                df_metrics_emcee.loc[it] = [
                    "emcee_" + pe,
                    p, g, eta,
                    f"{mean_acceptance[:, i, j, k].mean():.3f}±{mean_acceptance[:, i, j, k].std():.3f}",
                    f"{mse[:, i, j, k, m].mean():.3f}±{mse[:, i, j, k, m].std():.3f}",
                    f"{np.sqrt(mse[:, i, j, k, m]).mean():.3f}±{np.sqrt(mse[:, i, j, k, m]).std():.3f}",
                    f"{r2[:, i, j, k, m].mean():.3f}±{r2[:, i, j, k, m].std():.3f}"
                ]

df_metrics_emcee.sort_values(results_columns_emcee[-2], inplace=True)

###################################################################
# FIT SKLEARN MODELS
###################################################################

if FIT_SKLEARN:

    # -- Select family of regressors

    regressors = []
    folds = KFold(shuffle=True, random_state=SEED)

    alphas = np.logspace(-4, 4, 20)
    n_selected = [5, 10, 15, 20, 25]
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

    # Ridge
    regressors.append(("sk_ridge",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("reg", Ridge())]),
                       params_reg
                       ))

    # Ridge+manual
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

    # SVM RBF+manual
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

    df_metrics_sk = pd.DataFrame(columns=results_columns_sk)

    print("\nFitting sklearn models...")

    for i, (name, pipe, params) in enumerate(regressors):
        print(f"  {name}")
        reg_cv = GridSearchCV(pipe, params, scoring="neg_mean_squared_error",
                              n_jobs=N_CORES, cv=folds)
        reg_cv.fit(X_fd, Y)
        Y_hat = reg_cv.predict(X_test_fd)
        metrics = utils.regression_metrics(Y_test, Y_hat)

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
            metrics["mse"],
            np.sqrt(metrics["mse"]),
            metrics["r2"]]

    df_metrics_sk.sort_values(results_columns_sk[-2], inplace=True)

###################################################################
# PRINT RESULTS
###################################################################

if SYNTHETIC_DATA:
    data_name = kernel_fn.__name__ + "_" + MODEL_GEN + "_"
    if MODEL_GEN == "L2":
        data_name += beta_coef.__name__
else:
    data_name = real_data_name

filename = ("emcee_" + data_name + "_transform_tau_" + str(TRANSFORM_TAU)
            + "_mle_" + mle_strategy + "_" + mle_method + "_frac_random_"
            + str(int(100*frac_random)) + "_walkers_" + str(n_walkers)
            + "_iters_" + str(n_iter) + "_thin_" + str(thin) + "_reps_"
            + str(N_REPS) + "_seed_" + str(SEED))

if PRINT_TO_FILE:
    print(f"\nSaving results to file '{filename}'")
    f = open(filename + ".results", 'w')
    sys.stdout = f  # Change the standard output to the file we created
else:
    print("\nShowing results...\n")

print("-- MODEL GENERATION --")
print(f"Train,test size: {n_train},{n_test}")
if SYNTHETIC_DATA:
    print(f"GP kernel: {kernel_fn.__name__}")
    print(f"Model type: {MODEL_GEN}")
    print("True parameters:")
    if MODEL_GEN == "RKHS":
        print(f"  β: {beta_true}\n  τ: {tau_true}")
    else:
        print(f"  β(t): {beta_coef.__name__}")
    print(f"  α0: {alpha0_true}\n  σ2: {sigma2_true}")
else:
    print(f"Data name: {real_data_name}")
print(f"Transform tau: {'true' if TRANSFORM_TAU else 'false'}")

print("\n-- BIC --")
bic_mean = bic.mean(axis=0)
for i, p in enumerate(ps):
    print(f"BIC [p={p}]: {bic_mean[i]:.3f}")

print("\n-- EMCEE SAMPLER --")
print(f"N_walkers: {n_walkers}")
print(f"N_iters: {n_iter}")
print(f"MLE: {mle_method} + {mle_strategy}")
print(f"Frac_random: {frac_random}")
print(f"Burn-in: {n_iter_initial}")
print(f"Thinning: {thin}")

print("\n-- RESULTS EMCEE --")
print(f"Random iterations: {rep + 1}")
print(
    f"Mean execution time: {exec_times.mean():.3f}±{exec_times.std():.3f} s")
print(f"Total execution time: {exec_times.sum()/60.:.3f} min\n")
print(df_metrics_emcee.to_string(index=False, col_space=6))

if FIT_SKLEARN:
    print("\n-- RESULTS SKLEARN --\n")
    print(df_metrics_sk.to_string(index=False, col_space=6))


###################################################################
# SAVE RESULTS
###################################################################

if SAVE_RESULTS:
    np.savez(
        filename + ".npz",
        exec_times=exec_times,
        mean_acceptance=mean_acceptance,
        mse=mse,
        r2=r2
    )
