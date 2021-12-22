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
from skfda.preprocessing.dim_reduction.variable_selection import (
    RKHSVariableSelection as RKVS,
    RecursiveMaximaHunting as RMH
)
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from skfda.representation.basis import FDataBasis, Fourier
from skfda.representation.grid import FDataGrid
from skfda.ml.classification import (
    MaximumDepthClassifier, KNeighborsClassifier, NearestCentroid
)

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
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
MODEL_GEN = "RKHS"  # 'L2', 'RKHS' or 'MIXTURE'
REAL_DATA = "Medflies"
STANDARDIZE_PREDICTORS = False
BASIS_REPRESENTATION = True
N_BASIS = 9
NOISE = 0.05
TRANSFORM_TAU = False

kernel_fn = utils.fractional_brownian_kernel
kernel_fn2 = utils.squared_exponential_kernel
beta_coef = utils.cholaquidis_scenario3
basis = Fourier(n_basis=N_BASIS)
folds = StratifiedKFold(shuffle=True, random_state=SEED)

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
mean_alpha0_init = 0.0
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
    """Transformed parameter vector 'theta_tr' is (β, τ, α0, log σ)."""
    n, N = X.shape
    grid = theta_space.grid

    assert len(theta_tr) == theta_space.ndim

    beta, tau, alpha0, _ = theta_space.get_params(theta_tr)

    idx = np.abs(grid - tau[:, np.newaxis]).argmin(1)
    X_tau = X[:, idx]
    lin_comp = alpha0 + X_tau@beta

    return -np.sum(lin_comp*Y - np.logaddexp(0, lin_comp))


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
        np.array([0.0]*p + [0.5]*p + [0.0] + [1.0]))

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

    beta, tau, alpha0, _ = theta_space.get_params(theta_tr)

    idx = np.abs(grid - tau[:, np.newaxis]).argmin(1)
    X_tau = X[:, idx]
    lin_comp = alpha0 + X_tau@beta
    ll_pointwise = lin_comp*Y - np.logaddexp(0, lin_comp)
    ll = np.sum(ll_pointwise)

    if return_ll:
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
            return (-np.inf, np.full_like(Y, -1.0),
                    np.full_like(Y, -1), np.full_like(Y, -np.inf))
        elif return_pp:
            return -np.inf, np.full_like(Y, -1.0), np.full_like(Y, -1)
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
        pp_y, pp_p = utils.generate_response_logistic(
            X, theta, return_p=True, rng=rng)

    # Return information
    if return_pp and return_ll:
        return lpos, pp_p, pp_y, ll_pointwise
    elif return_pp:
        return lpos, pp_p, pp_y
    elif return_ll:
        return lpos, ll_pointwise
    else:
        return lpos


# -- Sklearn CV and transformers

def cv_sk(classifiers, folds, X, Y, X_test, Y_test, columns_name, verbose=False):
    df_metrics_sk = pd.DataFrame(columns=columns_name)

    for i, (name, pipe, params) in enumerate(classifiers):
        if verbose:
            print(f"  Fitting {name}...")
        clf_cv = GridSearchCV(pipe, params, scoring="accuracy",
                              n_jobs=N_CORES, cv=folds)
        clf_cv.fit(X, Y)
        Y_hat_sk = clf_cv.predict(X_test)
        metrics_sk = utils.classification_metrics(Y_test, Y_hat_sk)

        if name == "sk_fknn":
            n_features = f"K={clf_cv.best_params_['clf__n_neighbors']}"
        elif name == "sk_mdc" or name == "sk_fnc":
            n_features = X.data_matrix.shape[1]
        elif "svm" in name:
            n_features = clf_cv.best_estimator_["clf"].n_features_in_
        else:
            if isinstance(clf_cv.best_estimator_["clf"].coef_[0], FDataBasis):
                coef = clf_cv.best_estimator_["clf"].coef_[0].coefficients[0]
            else:
                coef = clf_cv.best_estimator_["clf"].coef_[0]

            n_features = sum(~np.isclose(coef, 0))

        df_metrics_sk.loc[i] = [
            name,
            n_features,
            metrics_sk["acc"]]

        df_metrics_sk.sort_values(
            columns_name[-1], inplace=True, ascending=False)

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
    tau_hat = utils.point_estimate(
        idata, point_est, names)[p_hat:2*p_hat]
    idx_hat = np.abs(grid - tau_hat[:, np.newaxis]).argmin(1)

    classifiers_var_sel = []
    Cs = np.logspace(-4, 4, 20)
    params_clf = {"clf__C": Cs}
    params_svm = {"clf__gamma": ['auto', 'scale']}

    # Emcee+LR
    classifiers_var_sel.append((f"{prefix}_{point_est}+sk_logistic",
                                Pipeline([
                                    ("var_sel", VariableSelection(grid, idx_hat)),
                                    ("data_matrix", DataMatrix()),
                                    ("clf", LogisticRegression(random_state=SEED))]),
                                params_clf
                                ))

    # Emcee+SVM Linear
    classifiers_var_sel.append((f"{prefix}_{point_est}+sk_svm_lin",
                                Pipeline([
                                    ("var_sel", VariableSelection(grid, idx_hat)),
                                    ("data_matrix", DataMatrix()),
                                    ("clf", LinearSVC(random_state=SEED))]),
                                params_clf
                                ))

    # Emcee+SVM RBF
    classifiers_var_sel.append((f"{prefix}_{point_est}+sk_svm_rbf",
                                Pipeline([
                                    ("var_sel", VariableSelection(grid, idx_hat)),
                                    ("data_matrix", DataMatrix()),
                                    ("clf", SVC(kernel='rbf'))]),
                                {**params_svm, **params_clf}
                                ))

    df_metrics_var_sel = cv_sk(classifiers_var_sel, folds,
                               X_fd, Y, X_test_fd, Y_test, columns_name,
                               verbose)

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

    mean_vector = None
    mean_vector2 = np.ones(N)

    beta_true = np.array([-5., 10.])
    tau_true = np.array([0.1, 0.8])
    alpha0_true = -0.5

    if MODEL_GEN == "MIXTURE":
        x, y = utils.generate_classification_dataset(
            grid, kernel_fn, kernel_fn2,
            n_train + n_test, rng,
            mean_vector, mean_vector2)
    else:
        if MODEL_GEN == "L2":
            x, y_lin = utils.generate_gp_l2_dataset(
                grid, kernel_fn,
                n_train + n_test, beta_coef,
                alpha0_true, 0.0, rng=rng
            )
        elif MODEL_GEN == "RKHS":
            x, y_lin = utils.generate_gp_rkhs_dataset(
                grid, kernel_fn,
                n_train + n_test, beta_true, tau_true,
                alpha0_true, 0.0, rng=rng
            )
        else:
            raise ValueError("Invalid model generation strategy.")

        # Transform linear response for logistic model
        y = utils.transform_linear_response(y_lin, noise=NOISE, rng=rng)

    # Train/test split
    X, X_test, Y, Y_test = train_test_split(
        x, y, train_size=n_train, stratify=y,
        random_state=SEED)

    # Create FData object
    X_fd = skfda.FDataGrid(X, grid)
    X_test_fd = skfda.FDataGrid(X_test, grid)

else:
    if REAL_DATA == "Medflies":
        x, y = skfda.datasets.fetch_medflies(return_X_y=True)
    elif REAL_DATA == "Growth":
        x, y = skfda.datasets.fetch_growth(return_X_y=True)
    else:
        raise ValueError("REAL_DATA must be 'Medflies' or 'Growth'.")

    X_fd, X_test_fd, Y, Y_test = train_test_split(
        x, y, train_size=0.8, stratify=y, random_state=SEED)

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

# Standardize data
X_m = X_fd.mean(axis=0)
X_fd = (X_fd - X_m)/X_sd
X = X_fd.data_matrix.reshape(-1, N)
X_test_fd = (X_test_fd - X_m)/X_sd
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
    ["Estimator", "Features", "g", "η", "Mean_accpt(%)", "Accuracy"]
results_columns_ref = ["Estimator", "Features", "Accuracy"]

# Transformations
if TRANSFORM_TAU:
    tau_ttr = utils.Logit()
else:
    tau_ttr = utils.Identity()

# -- Estimators

point_estimates = ["mode", "mean", "median"]
all_estimates = ["posterior_mean", "posterior_vote"] + point_estimates
num_estimates = len(all_estimates)

###################################################################
# RUN SAMPLER
###################################################################

print(f"Random seed: {SEED}")
print(f"Num. cores: {N_CORES}")

mean_acceptance = np.zeros((N_REPS, num_ps, num_gs, num_etas))
acc = np.zeros((N_REPS, num_ps, num_gs, num_etas, num_estimates))
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
                Y_hat_mle = utils.generate_response_logistic(
                    X_test, mle_theta_back)
                metrics_mle = utils.classification_metrics(Y_test, Y_hat_mle)
                df_metrics_mle.loc[i] = [
                    "mle",
                    p,
                    metrics_mle["acc"]
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
                        ["p_star", "y_star"] if return_pp else [],
                        return_ll)

                    exec_times[rep, i, j, k] = time.time() - start

                    # -- Save metrics

                    mean_acceptance[rep, i, j, k] = 100 * \
                        np.mean(sampler.acceptance_fraction)

                    # Posterior mean estimates
                    pp_test_p, pp_test_y = utils.generate_pp(
                        idata, X_test, theta_names,
                        thin_pp, rng=rng,
                        kind='classification',
                        progress=False)

                    Y_hat_pp_mean = [utils.threshold(y)
                                     for y in pp_test_p.mean(axis=(0, 1))]
                    Y_hat_pp_vote = [utils.threshold(y)
                                     for y in pp_test_y.mean(axis=(0, 1))]
                    metrics_pp_mean = utils.classification_metrics(
                        Y_test, Y_hat_pp_mean)
                    metrics_pp_vote = utils.classification_metrics(
                        Y_test, Y_hat_pp_vote)

                    acc[rep, i, j, k, 0] = metrics_pp_mean["acc"]
                    acc[rep, i, j, k, 1] = metrics_pp_vote["acc"]

                    # Point estimates
                    for m, pe in enumerate(point_estimates):
                        Y_hat_pe = utils.point_predict(
                            X_test, idata,
                            theta_names, pe,
                            kind='classification')
                        metrics_pe = utils.classification_metrics(
                            Y_test, Y_hat_pe)
                        acc[rep, i, j, k, m + 2] = metrics_pe["acc"]

                    if PRINT_RESULTS_ONLINE:
                        max_acc = np.argmax(acc[rep, i, j, k, :])
                        max_acc = acc[rep, i, j, k, max_acc]
                        print(
                            f"  [p={p}, g={g}, η={eta}]: Max Acc = {max_acc:.3f} "
                            f"with '{all_estimates[max_acc]}'")

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
max_acc = -np.inf
max_acc_params = (-1, -1, -1)  # (i, j, k)

if rep + 1 > 0:
    for i, p in enumerate(ps):
        for j, g in enumerate(gs):
            for k, eta in enumerate(etas):
                for m, pe in enumerate(all_estimates):
                    it = iteration_count(
                        [num_ps, num_gs, num_etas, num_estimates], [i, j, k, m])

                    mean_acc = acc[:rep + 1, i, j, k, m].mean()
                    if mean_acc > max_acc:
                        max_acc = mean_acc
                        max_acc_params = (i, j, k)

                    df_metrics_emcee.loc[it] = [
                        "emcee_" + pe,
                        p, g, eta,
                        f"{mean_acceptance[:rep + 1, i, j, k].mean():.3f}"
                        f"±{mean_acceptance[:rep + 1, i, j, k].std():.3f}",
                        f"{acc[:rep + 1, i, j, k, m].mean():.3f}"
                        f"±{acc[:rep + 1, i, j, k, m].std():.3f}"
                    ]

df_metrics_mle.sort_values(
    results_columns_ref[-1], inplace=True, ascending=False)
df_metrics_emcee.sort_values(
    results_columns_emcee[-1], inplace=True, ascending=False)

###################################################################
# BAYESIAN VARIABLE SELECTION ON BEST MODEL
###################################################################

df_metrics_var_sel = pd.DataFrame(columns=results_columns_ref)

if REFIT_BEST_VAR_SEL and rep + 1 > 0:
    i, j, k = max_acc_params
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
        ["p_star", "y_star"] if return_pp else [],
        return_ll)

    # -- Bayesian variable selection

    print("Fitting sklearn models with Bayesian variable selection...")

    for pe in point_estimates:
        df_metrics_var_sel = df_metrics_var_sel.append(
            bayesian_var_sel(
                idata, theta_space, theta_names, X_fd,
                Y, X_test_fd, Y_test, folds, results_columns_ref,
                prefix="emcee", point_est=pe))

    df_metrics_var_sel.sort_values(
        results_columns_ref[-1], inplace=True, ascending=False)

###################################################################
# FIT SKLEARN MODELS
###################################################################

if FIT_REF_ALGS:

    # -- Select family of regressors

    classifiers = []

    Cs = np.logspace(-4, 4, 20)
    n_selected = [5, 10, 15, 20, 25, X.shape[1]]
    n_components = [2, 3, 4, 5, 6, 10]
    n_neighbors = [3, 5, 7]

    params_clf = {"clf__C": Cs}
    params_svm = {"clf__gamma": ['auto', 'scale']}
    params_select = {"selector__p": n_selected}
    params_fpca = {"dim_red__n_components": n_components}
    params_var_sel = {"var_sel__n_features_to_select": n_components}
    params_knn = {"clf__n_neighbors": n_neighbors,
                  "clf__weights": ['uniform', 'distance']}
    params_depth = {"clf__depth_method": [skfda.exploratory.depth.ModifiedBandDepth(),
                                          skfda.exploratory.depth.IntegratedDepth()]}

    # Manual+LR
    classifiers.append(("manual_sel+sk_logistic",
                        Pipeline([
                            ("data_matrix", DataMatrix()),
                            ("selector", FeatureSelector()),
                            ("clf", LogisticRegression(random_state=SEED))]),
                        {**params_clf, **params_select}
                        ))

    # FPCA+LR
    classifiers.append(("fpca+sk_logistic",
                        Pipeline([
                            ("dim_red", FPCA()),  # Retains scores only
                            ("clf", LogisticRegression(random_state=SEED))]),
                        {**params_fpca, **params_clf}
                        ))

    # RKVS+LR
    classifiers.append(("rkvs+sk_logistic",
                        Pipeline([
                            ("var_sel", RKVS()),
                            ("clf", LogisticRegression(random_state=SEED))]),
                        params_var_sel
                        ))

    # RMH+LR
    classifiers.append(("rmh+sk_logistic",
                        Pipeline([
                            ("var_sel", RMH()),
                            ("clf", LogisticRegression(random_state=SEED))]),
                        {}
                        ))

    # Manual+SVM Linear
    classifiers.append(("manual_sel+sk_svm_lin",
                        Pipeline([
                            ("data_matrix", DataMatrix()),
                            ("selector", FeatureSelector()),
                            ("clf", LinearSVC(random_state=SEED))]),
                        {**params_select, **params_clf}
                        ))

    # FPCA+SVM Linear
    classifiers.append(("fpca+sk_svm_lin",
                        Pipeline([
                            ("dim_red", FPCA()),  # Retains scores only
                            ("clf", LinearSVC(random_state=SEED))]),
                        {**params_fpca, **params_clf}
                        ))

    # RKVS+SVM Linear
    classifiers.append(("rkvs+sk_svm_lin",
                        Pipeline([
                            ("var_sel", RKVS()),
                            ("clf", LinearSVC(random_state=SEED))]),
                        {**params_var_sel, **params_clf}
                        ))

    # RMH+SVM Linear
    classifiers.append(("rmh+sk_svm_lin",
                        Pipeline([
                            ("var_sel", RMH()),
                            ("clf", LinearSVC(random_state=SEED))]),
                        params_clf
                        ))

    # Manual+SVM RBF
    classifiers.append(("manual_sel+sk_svm_rbf",
                        Pipeline([
                            ("data_matrix", DataMatrix()),
                            ("selector", FeatureSelector()),
                            ("clf", SVC(kernel='rbf'))]),
                        {**params_select, **params_clf, **params_svm}
                        ))

    # FPCA+SVM RBF
    classifiers.append(("fpca+sk_svm_rbf",
                        Pipeline([
                            ("dim_red", FPCA()),  # Retains scores only
                            ("clf", SVC(kernel='rbf'))]),
                        {**params_fpca, **params_clf, **params_svm}
                        ))

    # RKVS+SVM RBF
    classifiers.append(("rkvs+sk_svm_rbf",
                        Pipeline([
                            ("var_sel", RKVS()),
                            ("clf", SVC(kernel='rbf'))]),
                        {**params_var_sel, **params_clf, **params_svm}
                        ))

    # RMH+SVM RBF
    classifiers.append(("rmh+sk_svm_rbf",
                        Pipeline([
                            ("var_sel", RMH()),
                            ("clf", SVC(kernel='rbf'))]),
                        {**params_clf, **params_svm}
                        ))

    # Maximum Depth Classifier
    classifiers.append(("sk_mdc",
                        Pipeline([
                            ("clf", MaximumDepthClassifier())]),
                        params_depth
                        ))

    # KNeighbors Functional Classification
    classifiers.append(("sk_fknn",
                        Pipeline([
                            ("clf", KNeighborsClassifier())]),
                        params_knn
                        ))

    # Nearest Centroid Functional Classification
    classifiers.append(("sk_fnc",
                        Pipeline([
                            ("clf", NearestCentroid())]),
                        {}
                        ))

    # -- Fit model and save metrics

    print("Fitting reference sklearn models...")
    df_metrics_sk = cv_sk(classifiers, folds, X_fd, Y, X_test_fd,
                          Y_test, results_columns_ref, verbose=True)

###################################################################
# PRINT RESULTS
###################################################################

if SYNTHETIC_DATA:
    data_name = kernel_fn.__name__ + "_" + MODEL_GEN
    if MODEL_GEN == "L2":
        data_name += ("_" + beta_coef.__name__)
    elif MODEL_GEN == "MIXTURE":
        data_name += ("_" + kernel_fn2.__name__)
else:
    data_name = REAL_DATA

filename = ("emcee_" + data_name + "_noise_" + str(int(100*NOISE)) + "_basis_"
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

print(" --> Bayesian Functional Logistic Regression <--\n")
print("-- MODEL GENERATION --")
print(f"Train/test size: {n_train}/{n_test}")
if STANDARDIZE_PREDICTORS:
    print("Standardized predictors")
if BASIS_REPRESENTATION:
    print(f"Basis: {basis.__class__.__name__}(n={N_BASIS})")
else:
    print("Basis: None")
print(f"Noise: {NOISE}")
print(f"Transform tau: {'true' if TRANSFORM_TAU else 'false'}")

if SYNTHETIC_DATA:
    print(f"Model type: {MODEL_GEN}")
    if MODEL_GEN == "MIXTURE":
        print(f"X|Y=0 ~ GP(0, {kernel_fn.__name__})")
        print(f"X|Y=1 ~ GP(0, {kernel_fn2.__name__})")
    else:
        print(f"X ~ GP(0, {kernel_fn.__name__})")
        print("True parameters:")
        if MODEL_GEN == "RKHS":
            print(f"  β: {beta_true}\n  τ: {tau_true}")
        elif MODEL_GEN == "L2":
            print(f"  β(t): {beta_coef.__name__}")
        print(f"  α0: {alpha0_true}")
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
        i, j, k = max_acc_params
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
        acc=acc[:rep + 1, ...]
    )
