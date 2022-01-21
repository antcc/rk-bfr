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

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold
)
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LogisticRegression, Ridge, Lasso

import utils
from _lda import LDA
from _fpls import FPLS, APLS, FPLSBasis
# from _fpca_basis import FPCABasis

import skfda
from skfda.preprocessing.dim_reduction.variable_selection import (
    RKHSVariableSelection as RKVS,
    RecursiveMaximaHunting as RMH,
    # MinimumRedundancyMaximumRelevance as mRMR,
)
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
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
from skfda.ml.classification import (
    MaximumDepthClassifier, KNeighborsClassifier,
    NearestCentroid,
)
from skfda.exploratory.depth import IntegratedDepth, ModifiedBandDepth

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
NOISE = 0.10

kernel_fn = utils.fractional_brownian_kernel
kernel_fn2 = utils.squared_exponential_kernel
beta_coef = utils.cholaquidis_scenario3

INITIAL_SMOOTHING = "NW"  # None, 'NW' or 'Basis'
N_BASIS = 16
STANDARDIZE_PREDICTORS = False
TRANSFORM_TAU = False

basis = BSpline(n_basis=N_BASIS)
smoothing_params = np.logspace(-4, 4, 50)

folds = StratifiedKFold(shuffle=True, random_state=SEED)

# Override options with command-line parameters
if len(sys.argv) > 2:
    if SYNTHETIC_DATA:
        MODEL_GEN = sys.argv[2]
    else:
        REAL_DATA = sys.argv[2]
    if sys.argv[3] == "1":
        kernel_fn = utils.fractional_brownian_kernel
    elif sys.argv[3] == "2":
        kernel_fn = utils.ornstein_uhlenbeck_kernel
    else:
        kernel_fn = utils.squared_exponential_kernel
    if sys.argv[4] == "NW" or sys.argv[4] == "Basis":
        INITIAL_SMOOTHING = sys.argv[4]
    else:
        INITIAL_SMOOTHING = None

# Results
FIT_REF_ALGS = True
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
        np.array([0.0]*p + [0.5]*p + [0.0] + [1.0]))

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

        with utils.IgnoreWarnings():
            clf_cv.fit(X, Y)

        Y_hat_sk = clf_cv.predict(X_test)
        metrics_sk = utils.classification_metrics(Y_test, Y_hat_sk)

        if name == "sk_fknn":
            n_features = f"K={clf_cv.best_params_['clf__n_neighbors']}"
        elif name == "sk_mdc" or name == "sk_fnc":
            n_features = X.data_matrix.shape[1]
        elif name == "sk_flr":
            n_features = clf_cv.best_estimator_["clf"].p
        elif "pls1" in name:
            n_features = \
                clf_cv.best_estimator_["clf"].base_regressor.n_components
        elif "svm" in name:
            n_features = clf_cv.best_estimator_["clf"].n_features_in_
        else:
            if isinstance(clf_cv.best_estimator_["clf"].coef_[0], FDataBasis):
                coef = clf_cv.best_estimator_["clf"].coef_[0].coefficients[0]
            elif "sk_logistic" in name:
                coef = clf_cv.best_estimator_["clf"].coef_[0]
            else:
                coef = clf_cv.best_estimator_["clf"].coef_

            n_features = sum(~np.isclose(coef, 0))

        df_metrics_sk.loc[i] = [
            name,
            n_features,
            metrics_sk["acc"]]

        df_metrics_sk.sort_values(
            columns_name[-1], inplace=True, ascending=False)

    return df_metrics_sk, clf_cv


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

    df_metrics_var_sel, clf_cv_var_sel = cv_sk(
        classifiers_var_sel, folds,
        X_fd, Y, X_test_fd, Y_test,
        columns_name, verbose)

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
                    X_test, mle_theta_back, prob=False)
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
                        sampler.run_mcmc(
                            state, n_iter,
                            progress=PRINT_RESULTS_ONLINE,
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
                            f"  [p={p}, g={g}, η={eta}]: Max Acc = {max_acc:.3f}"
                            f" with '{all_estimates[max_acc]}'")

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

df_metrics_sk = pd.DataFrame(columns=results_columns_ref)

if FIT_REF_ALGS:

    # -- Select family of classifiers

    classifiers = []
    Cs = np.logspace(-4, 4, 20)
    n_selected = [5, 10, 15, 20, 25, X.shape[1]]
    n_components = [2, 3, 4, 5, 10]
    n_basis_bsplines = [8, 10, 12, 14, 16]
    n_basis_fourier = [3, 5, 7, 9, 11]
    basis_bspline = [BSpline(n_basis=p) for p in n_basis_bsplines]
    basis_fourier = [Fourier(n_basis=p) for p in n_basis_fourier]
    basis_fpls = [FPLSBasis(X_fd, Y, n_basis=p) for p in n_components]

    ridge_regressors = [Ridge(alpha=C) for C in Cs]
    lasso_regressors = [Lasso(alpha=C) for C in Cs]
    pls_regressors = [PLSRegressionWrapper(
        n_components=p) for p in n_components]
    fpls_regressors = [FPLS(n_components=p) for p in n_components]
    apls_regressors = [APLS(n_components=p) for p in n_components]
    n_neighbors = [3, 5, 7]

    params_clf = {"clf__C": Cs}
    params_svm = {"clf__gamma": ['auto', 'scale']}
    params_select = {"selector__p": n_selected}
    params_dim_red = {"dim_red__n_components": n_components}
    params_basis = {"basis__basis": basis_bspline + basis_fourier}
    params_basis_fpca = {"basis__n_basis": n_components}
    params_basis_fpls = {"basis__basis": basis_fpls}
    params_var_sel = {"var_sel__n_features_to_select": n_components}
    params_knn = {"clf__n_neighbors": n_neighbors,
                  "clf__weights": ['uniform', 'distance']}
    params_depth = {"clf__depth_method": [
        ModifiedBandDepth(), IntegratedDepth()]}
    params_mrmr = {"var_sel__method": ["MID", "MIQ"]}
    params_base_regressors_ridge = {"clf__base_regressor": ridge_regressors}
    params_base_regressors_lasso = {"clf__base_regressor": lasso_regressors}
    params_base_regressors_pls = {"clf__base_regressor": pls_regressors}
    params_base_regressors_fpls = {"clf__base_regressor": fpls_regressors}
    params_base_regressors_apls = {"clf__base_regressor": apls_regressors}

    """
    MULTIVARIATE MODELS
    """

    # LDA (based on FPCA+Ridge regression)
    classifiers.append(("sk_lda_fpca+ridge",
                       Pipeline([
                           ("dim_red", FPCA()),
                           ("clf", LDA())]),
                       {**params_dim_red, **params_base_regressors_ridge}
                        ))

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # LDA (based on FPLS (fixed basis)+Ridge regression)
    classifiers.append(("sk_lda_fpls_basis+ridge",
                       Pipeline([
                           ("basis", Basis()),
                           ("dim_red", FPLS()),
                           ("clf", LDA())]),
                       {**params_basis,
                        **params_dim_red,
                        **params_base_regressors_ridge}
                        ))
    """

    # LDA (based on Lasso regression)
    classifiers.append(("sk_lda_lasso",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("clf", LDA())]),
                       params_base_regressors_lasso
                        ))

    # LDA (based on PLS1 regression)
    classifiers.append(("sk_lda_pls1",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("clf", LDA())]),
                       params_base_regressors_pls
                        ))

    # LDA (based on Manual+Ridge regression)
    classifiers.append(("sk_lda_manual+ridge",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("selector", FeatureSelector()),
                           ("clf", LDA())]),
                       {**params_select, **params_base_regressors_ridge}
                        ))

    # LDA (based on PCA+Ridge regression)
    classifiers.append(("sk_lda_pca+ridge",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PCA(random_state=SEED)),
                           ("clf", LDA())]),
                       {**params_dim_red, **params_base_regressors_ridge}
                        ))

    # LDA (based on PLS+Ridge regression)
    classifiers.append(("sk_lda_pls+ridge",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PLSRegressionWrapper()),
                           ("clf", LDA())]),
                       {**params_dim_red, **params_base_regressors_ridge}
                        ))

    # LDA (based on RMH+Ridge regression)
    classifiers.append(("sk_lda_rmh+ridge",
                       Pipeline([
                           ("var_sel", RMH()),
                           ("clf", LDA())]),
                       params_base_regressors_ridge
                        ))

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # LDA (based on mRMR+Ridge regression)
    classifiers.append(("sk_lda_mRMR+ridge",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("var_sel", mRMR()),
                           ("clf", LDA())]),
                       {**params_var_sel,
                        **params_mrmr,
                        **params_base_regressors_ridge}
                       ))
    """

    """
    VARIABLE SELECTION + MULTIVARIATE MODELS
    """

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
                       {**params_dim_red, **params_clf}
                        ))

    # PCA+LR
    classifiers.append(("pca+sk_logistic",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PCA(random_state=SEED)),
                           ("clf", LogisticRegression(random_state=SEED))]),
                       {**params_dim_red, **params_clf}
                        ))

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # FPLS (fixed basis)+LR
    classifiers.append(("fpls_basis+sk_logistic",
                       Pipeline([
                           ("basis", Basis()),
                           ("dim_red", FPLS()),
                           ("clf", LogisticRegression(random_state=SEED))]),
                       {**params_basis, **params_dim_red, **params_clf}
                        ))
    """

    # PLS+LR
    classifiers.append(("pls+sk_logistic",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PLSRegressionWrapper()),
                           ("clf", LogisticRegression(random_state=SEED))]),
                       {**params_dim_red, **params_clf}
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

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # mRMR+LR
    classifiers.append(("mRMR+sk_logistic",
                       Pipeline([
                           ("var_sel", mRMR()),
                           ("clf", LogisticRegression(random_state=SEED))]),
                       {**params_var_sel, **params_mrmr}
                        ))
    """

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
                       {**params_dim_red, **params_clf}
                        ))

    # PCA+SVM Linear
    classifiers.append(("pca+sk_svm_lin",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PCA(random_state=SEED)),
                           ("clf", LinearSVC(random_state=SEED))]),
                       {**params_dim_red, **params_clf}
                        ))
    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # FPLS (fixed basis)+SVM Linear
    classifiers.append(("fpls_basis+sk_svm_lin",
                       Pipeline([
                           ("basis", Basis()),
                           ("dim_red", FPLS()),
                           ("clf", LinearSVC(random_state=SEED))]),
                       {**params_basis, **params_dim_red, **params_clf}
                        ))
    """

    # PLS+SVM Linear
    classifiers.append(("pls+sk_svm_lin",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PLSRegressionWrapper()),
                           ("clf", LinearSVC(random_state=SEED))]),
                       {**params_dim_red, **params_clf}
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

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # mRMR+SVM Linear
    classifiers.append(("mRMR+sk_svm_lin",
                       Pipeline([
                           ("var_sel", mRMR()),
                           ("clf", LinearSVC(random_state=SEED))]),
                       {**params_var_sel, **params_mrmr, **params_clf}
                        ))
    """

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
                       {**params_dim_red, **params_clf, **params_svm}
                        ))

    # PCA+SVM RBF
    classifiers.append(("pca+sk_svm_rbf",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PCA(random_state=SEED)),
                           ("clf", SVC(kernel='rbf'))]),
                       {**params_dim_red, **params_clf, **params_svm}
                        ))

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # FPLS (fixed basis)+SVM RBF
    classifiers.append(("fpls_basis+sk_svm_rbf",
                       Pipeline([
                           ("basis", Basis()),
                           ("dim_red", FPLS()),
                           ("clf", SVC(kernel='rbf'))]),
                       {**params_basis,
                        **params_dim_red,
                        **params_clf,
                        **params_svm}
                        ))
    """

    # PLS+SVM RBF
    classifiers.append(("pls+sk_svm_rbf",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PLSRegressionWrapper()),
                           ("clf", SVC(kernel='rbf'))]),
                       {**params_dim_red, **params_clf, **params_svm}
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

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # mRMR+SVM RBF
    classifiers.append(("mRMR+sk_svm_rbf",
                       Pipeline([
                           ("var_sel", mRMR()),
                           ("clf", SVC(kernel='rbf'))]),
                       {**params_var_sel,
                        **params_mrmr,
                        **params_clf,
                        **params_svm}
                        ))
    """

    """
    FUNCTIONAL MODELS
    """

    """
    TARDA BASTANTE

    # Functional Logistic Regression Model (testing)
    from _logistic_regression_TEMP import LogisticRegression as FLogisticRegression
    params_flr = {"clf__p": n_components}

    classifiers.append(("sk_flr",
                       Pipeline([
                           ("clf", FLogisticRegression())]),
                       params_flr
                        ))
    """

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

    # NOTE: while not strictly necessary, the test data undergoes the
    # same basis expansion process as the training data. This is more
    # computationally efficient and seems to improve the performance.

    # Functional LDA (based on L^2-regression with fixed basis)
    classifiers.append(("sk_flda_l2_basis",
                       Pipeline([
                           ("basis", Basis()),
                           ("clf", LDA())]),
                       params_basis
                        ))

    """
    TARDA BASTANTE (cálculo de Gram matrix costoso en la base)

    # Functional LDA (based on L^2-regression with FPCA basis)
    classifiers.append(("sk_flda_l2_khl",
                       Pipeline([
                           ("basis", FPCABasis()),
                           ("clf", LDA())]),
                       params_basis_fpca
                        ))
    """

    """
    TARDA BASTANTE (cálculo de Gram matrix costoso en la base)

    # Functional LDA (based on L^2-regression with FPLS basis)
    classifiers.append(("sk_flda_l2_fpls",
                       Pipeline([
                           ("basis", Basis()),
                           ("clf", LDA())]),
                       params_basis_fpls
                        ))
    """

    # Functional LDA (based on FPLS1 regression with fixed basis)
    classifiers.append(("sk_flda_fpls1_basis",
                       Pipeline([
                           ("basis", Basis()),
                           ("clf", LDA())]),
                       {**params_basis, **params_base_regressors_fpls}
                        ))

    # Functional LDA (based on APLS regression)
    classifiers.append(("sk_flda_apls",
                       Pipeline([
                           ("clf", LDA())]),
                       params_base_regressors_apls
                        ))

    # -- Fit model and save metrics

    print("Fitting reference sklearn models...")
    df_metrics_sk, clf_cv = cv_sk(classifiers, folds, X_fd, Y, X_test_fd,
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

if INITIAL_SMOOTHING == "NW":
    smoothing_str = best_smoother.best_estimator_.__class__.__name__
elif INITIAL_SMOOTHING == "Basis":
    smoothing_str = basis.__class__.__name__
else:
    smoothing_str = "none"

filename = ("clf_" + "emcee_" + data_name + "_noise_" + str(int(100*NOISE))
            + "_smoothing_" + smoothing_str + "_frac_random_"
            + str(int(100*frac_random)) + "_walkers_"
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
if INITIAL_SMOOTHING == "NW":
    print(
        f"Smoothing: {best_smoother.best_estimator_.__class__.__name__}"
        f"(λ={best_smoother.best_params_['smoothing_parameter']:.3f})")
elif INITIAL_SMOOTHING == "Basis":
    print(f"Smoothing: {basis.__class__.__name__}(n=N_BASIS)")
else:
    print("Smoothing: none")
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
        print("\n-- RESULTS BAYESIAN VARIABLE SELECTION --")
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
    # Save all the results dataframe in one CSV file
    df = pd.concat(
        [df_metrics_emcee, df_metrics_var_sel, df_metrics_sk],
        axis=0,
        ignore_index=True)
    df.to_csv(filename + ".csv", index=False)

    # Save the top Accuracy values to arrays
    emcee_best = df_metrics_emcee["Accuracy"][:SAVE_TOP].apply(
        lambda s: s.split("±")[0]).to_numpy(dtype=float)
    var_sel_best = df_metrics_var_sel["Accuracy"][:SAVE_TOP].to_numpy(
        dtype=float)
    sk_best = df_metrics_sk["Accuracy"][:SAVE_TOP].to_numpy(dtype=float)

    np.savez(
        filename + ".npz",
        emcee_best=emcee_best,
        var_sel_best=var_sel_best,
        sk_best=sk_best,
    )
