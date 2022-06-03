# encoding: utf-8

import argparse
import os
import sys
import time
from collections import defaultdict
from itertools import product

import emcee
import numpy as np
import pandas as pd
import preprocessing
import pymc3 as pm
import simulation
from _fpls import APLS, FPLS, FPLSBasis
from bayesian_model import ThetaSpace, probability_to_label
from mcmc_sampler import (BFLinearEmcee, BFLinearPymc, BFLogisticEmcee,
                          BFLogisticPymc)
from mle import compute_mle
from skfda.datasets import (fetch_cran, fetch_growth, fetch_medflies,
                            fetch_tecator)
from skfda.exploratory.depth import IntegratedDepth, ModifiedBandDepth
from skfda.preprocessing.smoothing.kernel_smoothers import \
    NadarayaWatsonSmoother as NW
from skfda.representation.basis import BSpline, Fourier
from skfda.representation.grid import FDataGrid
from sklearn.linear_model import Lasso, LogisticRegressionCV, Ridge, RidgeCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn_utils import DataMatrix, PLSRegressionWrapper
from utils import (bayesian_variable_selection_predict, cv_sk,
                   linear_regression_comparison_suite,
                   logistic_regression_comparison_suite)

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

# Script behavior
RUN_REF_ALGS = True
VERBOSE = True
PRECOMPUTE_MLE = True
PRINT_TO_FILE = False
SAVE_RESULTS = False
PRINT_PATH = "../results/"  # /home/antcc/bayesian-functional-regression/results/
SAVE_PATH = PRINT_PATH + "out/"


###################################################################
# UTILITY FUNCTIONS
###################################################################

def enumerated_product(*args):
    """Get cartesian product of lists along with their individual indices."""
    yield from zip(product(*(range(len(x)) for x in args)), product(*args))


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
# PARSING FUNCTIONS
###################################################################

def get_arg_parser():
    parser = argparse.ArgumentParser("Bayesian-RKHS Functional Regression")
    data_group = parser.add_mutually_exclusive_group(required=True)
    p_group = parser.add_mutually_exclusive_group(required=True)

    # Optional misc. arguments
    parser.add_argument("-s", "--seed", type=int, help="random seed")
    parser.add_argument(
        "-v", "--verbose", type=int, choices=[0, 1, 2], const=1,
        default=0, nargs='?', help="increase verbosity")
    parser.add_argument(
        "-c", "--n-cores", type=int, default=1,
        help="number of cores for paralellization (-1 to use all)")
    parser.add_argument(
        "-r", "--n-reps", type=int, default=1,
        help="number of random train/test splits for robustness")
    parser.add_argument(
        "-k", "--n-folds", type=int, default=2,
        help="number of folds for CV in hyperparameter tuning"
    )

    # Optional dataset arguments
    parser.add_argument(
        "-n", "--n-samples", type=int, default=150,
        help="number of functional samples"
    )
    parser.add_argument(
        "-N", "--n-grid", type=int, default=100,
        help="number of grid points for functional regressors"
    )
    parser.add_argument(
        "--smoothing", default="nw", choices=["none", "nw"],
        help="smooth functional data as part of preprocessing"
    )
    parser.add_argument(
        "--train-size", type=float, default=0.7,
        help="fraction of data used for training"
    )
    parser.add_argument(
        "--noise", type=float, default=0.1,
        help="fraction of noise for logistic synthetic data"
    )

    # Optional MCMC arguments
    parser.add_argument(
        "--n-walkers", type=int, default=4,
        help="number of independent chains in MCMC algorithm"
    )
    parser.add_argument(
        "--n-iters", type=int, default=100,
        help="number of iterations in MCMC algorithm"
    )
    parser.add_argument(
        "--n-tune", type=int, default=100,
        help="number of tune/warmup iterations in MCMC algorithm"
    )
    parser.add_argument(
        "--n-reps-mle", type=int, default=8,
        help="number of random repetitions of MLE computation"
    )
    parser.add_argument(
        "--eta-range", type=int, metavar=("ETA_MIN", "ETA_MAX"),
        nargs=2, default=[-1, 1],
        help="range of the parameter η (in logarithmic space)"
    )
    parser.add_argument(
        "--g", type=float, default=5.0,
        help="value of the parameter g"
    )

    # Optional arguments for emcee sampler
    parser.add_argument(
        "--frac-random", type=float, default=0.3,
        help="fraction of initial points randomly generated in emcee sampler"
    )
    parser.add_argument(
        "--moves", choices=["sw", "de", "s"], default="sw",
        help="types of moves in emcee sampler"
    )

    # Optional arguments for pymc sampler
    parser.add_argument(
        "--step", choices=["metropolis", "nuts"],
        default="metropolis", help="sampling step in pymc sampler"
    )
    parser.add_argument(
        "--target-accept", type=float, default=0.8,
        help="target_accept for NUTS step in pymc sampler"
    )

    # Mandatory arguments
    parser.add_argument(
        "kind",
        help="type of problem to solve",
        choices=["linear", "logistic"]
    )
    parser.add_argument(
        "method",
        help="MCMC method to approximate the posterior",
        choices=["emcee", "pymc"]
    )
    parser.add_argument(
        "data",
        help="type of data to use",
        choices=["rkhs", "l2", "real"]
    )
    data_group.add_argument(
        "--kernel",
        help="name of kernel to use in simulations",
        choices=["ou", "sqexp", "fbm"]
    )
    data_group.add_argument(
        "--data-name",
        help="name of data set to use as real data",
        choices=["tecator", "moisture", "medflies", "growth"]
    )
    p_group.add_argument(
        "--p-range",
        type=int,
        nargs=2,
        help="range for the parameter p (number of components)",
        metavar=("P_MIN", "P_MAX")
    )
    p_group.add_argument(
        "--p-prior",
        type=float,
        nargs='+',
        help="prior distribution for the parameter p (number of components), "
             "from 1 to P_MAX.",
    )

    return parser


###################################################################
# DATA FUNCTIONS
###################################################################

def get_data_linear(
    is_simulated_data,
    model_type,
    n_samples=150,
    n_grid=100,
    kernel_fn=None,
    beta_coef=None,
    initial_smoothing=False,
    tau_range=(0, 1),
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    if is_simulated_data:
        if kernel_fn is None:
            raise ValueError("Must provide a kernel function.")

        grid = np.linspace(tau_range[0] + 1./n_grid, tau_range[1], n_grid)
        alpha0_true = 5.
        sigma2_true = 0.5

        if model_type == "l2":
            if beta_coef is None:
                raise ValueError("Must provide a coefficient function.")

            X, y = simulation.generate_gp_l2_dataset(
                grid,
                kernel_fn,
                n_samples,
                beta_coef,
                alpha0_true,
                sigma2_true,
                rng=rng
            )
        elif model_type == "rkhs":
            beta_true = [-5., 1., 10.]
            tau_true = [0.1, 0.4, 0.8]
            X, y = simulation.generate_gp_rkhs_dataset(
                grid,
                kernel_fn,
                n_samples,
                beta_true,
                tau_true,
                alpha0_true,
                sigma2_true,
                rng=rng
            )
        else:
            raise ValueError("Invalid model generation strategy.")

        # Create FData object
        X_fd = FDataGrid(X, grid)

    else:  # Real data
        if model_type == "tecator":
            X_fd, y = fetch_tecator(return_X_y=True)
            y = y[:, 1]  # Fat level
        elif model_type == "moisture":
            data = fetch_cran("Moisturespectrum", "fds")["Moisturespectrum"]
            y = fetch_cran("Moisturevalues", "fds")["Moisturevalues"]
            X_fd = FDataGrid(data["y"].T[:, ::5], data["x"][::5])
        else:
            raise ValueError("Real data set must be 'tecator' or 'aemet'.")

        grid = preprocessing.normalize_grid(
            X_fd.grid_points[0], tau_range[0], tau_range[1])

    # Smooth data
    if initial_smoothing:
        smoother = NW()
        smoothing_params = np.logspace(-4, 4, 50)

        X_fd, _ = preprocessing.smooth_data(
            X_fd,
            smoother,
            smoothing_params
        )

    return X_fd, y, grid


def get_data_logistic(
    is_simulated_data,
    model_type,
    n_samples=150,
    n_grid=100,
    kernel_fn=None,
    beta_coef=None,
    noise=0.1,
    initial_smoothing=False,
    tau_range=(0, 1),
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    if is_simulated_data:
        if kernel_fn is None:
            raise ValueError("Must provide a kernel function.")

        grid = np.linspace(tau_range[0] + 1./n_grid, tau_range[1], n_grid)
        alpha0_true = -0.5

        if model_type == "l2":
            if beta_coef is None:
                raise ValueError("Must provide a coefficient function.")

            X, y_lin = simulation.generate_gp_l2_dataset(
                grid,
                kernel_fn,
                n_samples,
                beta_coef,
                alpha0_true,
                0.0,
                rng=rng
            )
        elif model_type == "rkhs":
            beta_true = [-5., 1., 10.]
            tau_true = [0.1, 0.4, 0.8]
            X, y_lin = simulation.generate_gp_rkhs_dataset(
                grid,
                kernel_fn,
                n_samples,
                beta_true,
                tau_true,
                alpha0_true,
                0.0,
                rng=rng
            )
        else:
            raise ValueError("Invalid model generation strategy.")

        # Transform linear response for logistic model
        y = probability_to_label(
            y_lin, random_noise=noise, rng=rng)

        # Create FData object
        X_fd = FDataGrid(X, grid)

    else:  # Real data
        if model_type == "medflies":
            X_fd, y = fetch_medflies(return_X_y=True)
        elif model_type == "growth":
            X_fd, y = fetch_growth(return_X_y=True)
        else:
            raise ValueError("Real data set must be 'medflies' or 'growth'.")

        grid = preprocessing.normalize_grid(
            X_fd.grid_points[0], tau_range[0], tau_range[1])

    # Smooth data
    if initial_smoothing:
        smoother = NW()
        smoothing_params = np.logspace(-4, 4, 50)

        X_fd, _ = preprocessing.smooth_data(
            X_fd,
            smoother,
            smoothing_params
        )

    return X_fd, y, grid


###################################################################
# MODEL FUNCTIONS
###################################################################

def get_reference_models_linear(X, y, seed):
    alphas = np.logspace(-4, 4, 20)
    n_selected = [5, 10, 15, 20, 25, len(X.grid_points[0])]
    n_components = [2, 3, 4, 5, 10]
    n_basis_bsplines = [8, 10, 12, 14, 16]
    n_basis_fourier = [3, 5, 7, 9, 11]
    n_neighbors = [3, 5, 7]

    basis_bspline = [BSpline(n_basis=p) for p in n_basis_bsplines]
    basis_fourier = [Fourier(n_basis=p) for p in n_basis_fourier]

    basis_fpls = []
    for p in n_components:
        try:
            basis_fpls.append(FPLSBasis(X, y, n_basis=p))
        except ValueError:
            # print(f"Can't create FPLSBasis with n_basis={p}")
            continue

    params_regularizer = {"reg__alpha": alphas}
    params_svm = {"reg__C": alphas,
                  "reg__gamma": ['auto', 'scale']}
    params_select = {"selector__p": n_selected}
    params_pls = {"reg__n_components": n_components}
    params_dim_red = {"dim_red__n_components": n_components}
    params_basis = {"basis__basis": basis_bspline + basis_fourier}
    # params_basis_fpca = {"basis__n_basis": n_components}
    # params_basis_fpls = {"basis__basis": basis_fpls}
    params_knn = {"reg__n_neighbors": n_neighbors,
                  "reg__weights": ['uniform', 'distance']}
    # params_mrmr = {"var_sel__method": ["MID", "MIQ"],
    #                "var_sel__n_features_to_select": n_components}

    regressors = linear_regression_comparison_suite(
        params_regularizer,
        params_select,
        params_dim_red,
        params_svm,
        params_basis,
        params_pls,
        params_knn,
        random_state=seed
    )

    return regressors


def get_reference_models_logistic(X, y, seed):
    Cs = np.logspace(-4, 4, 20)
    n_selected = [5, 10, 15, 20, 25, len(X.grid_points[0])]
    n_components = [2, 3, 4, 5, 10]
    n_basis_bsplines = [8, 10, 12, 14, 16]
    n_basis_fourier = [3, 5, 7, 9, 11]
    n_neighbors = [3, 5, 7]

    basis_bspline = [BSpline(n_basis=p) for p in n_basis_bsplines]
    basis_fourier = [Fourier(n_basis=p) for p in n_basis_fourier]

    basis_fpls = []
    for p in n_components:
        try:
            basis_fpls.append(FPLSBasis(X, y, n_basis=p))
        except ValueError:
            print(f"Can't create FPLSBasis with n_basis={p}")
            continue

    ridge_regressors = [Ridge(alpha=C) for C in Cs]
    lasso_regressors = [Lasso(alpha=C) for C in Cs]
    pls_regressors = [PLSRegressionWrapper(
        n_components=p) for p in n_components]
    fpls_regressors = [FPLS(n_components=p) for p in n_components]
    apls_regressors = [APLS(n_components=p) for p in n_components]

    params_clf = {"clf__C": Cs}
    params_svm = {"clf__gamma": ['auto', 'scale']}
    params_select = {"selector__p": n_selected}
    params_dim_red = {"dim_red__n_components": n_components}
    params_basis = {"basis__basis": basis_bspline + basis_fourier}
    # params_basis_fpca = {"basis__n_basis": n_components}
    # params_basis_fpls = {"basis__basis": basis_fpls}
    params_var_sel = {"var_sel__n_features_to_select": n_components}
    params_knn = {"clf__n_neighbors": n_neighbors,
                  "clf__weights": ['uniform', 'distance']}
    params_depth = {"clf__depth_method": [
        ModifiedBandDepth(), IntegratedDepth()]}
    # params_mrmr = {"var_sel__method": ["MID", "MIQ"]}
    params_base_regressors_ridge = {"clf__base_regressor": ridge_regressors}
    params_base_regressors_lasso = {"clf__base_regressor": lasso_regressors}
    params_base_regressors_pls = {"clf__base_regressor": pls_regressors}
    params_base_regressors_fpls = {"clf__base_regressor": fpls_regressors}
    params_base_regressors_apls = {"clf__base_regressor": apls_regressors}

    classifiers = logistic_regression_comparison_suite(
        params_clf,
        params_base_regressors_ridge,
        params_base_regressors_lasso,
        params_base_regressors_pls,
        params_base_regressors_fpls,
        params_base_regressors_apls,
        params_select,
        params_dim_red,
        params_var_sel,
        params_svm,
        params_depth,
        params_knn,
        params_basis,
        random_state=seed,
    )

    return classifiers


def get_theta_space_wrapper(grid, include_p, theta_names, tau_range):
    return lambda p: ThetaSpace(
        p,
        grid,
        include_p=include_p,
        names=theta_names,
        tau_range=tau_range,
    )


def get_mle_wrapper(method, strategy, kind, n_reps, n_jobs, rng):
    return lambda X, y, theta_space: compute_mle(
        X,
        y,
        theta_space,
        kind=kind,
        method=method,
        strategy=strategy,
        n_reps=n_reps,
        n_jobs=n_jobs,
        rng=rng
    )[0]


def get_bayesian_model_wrapper(
    args,
    prior_p,
    rng,
    moves=None,
    step_fn=None,
    step_kwargs=None
):
    kwargs_mcmc = {
        "b0": 'mle',
        "g": args.g,
        "prior_p": prior_p,
        "n_iter_warmup": args.n_tune,
        "n_reps_mle": args.n_reps_mle,
        "n_jobs": args.n_cores,
        "verbose": args.verbose,
        "random_state": rng
    }

    if args.method == "emcee":
        kwargs_emcee = {
            "frac_random": args.frac_random,
            "moves": moves
        }

        if args.kind == "linear":
            return lambda theta_space, kwargs: BFLinearEmcee(
                theta_space,
                args.n_walkers,
                args.n_iters,
                **kwargs_mcmc,
                **kwargs_emcee,
                **kwargs,
            )
        else:  # logistic
            return lambda theta_space, kwargs: BFLogisticEmcee(
                theta_space,
                args.n_walkers,
                args.n_iters,
                **kwargs_mcmc,
                **kwargs_emcee,
                **kwargs,
            )
    else:  # pymc
        kwargs_pymc = {
            "step_fn": step_fn,
            "step_kwargs": step_kwargs
        }

        if args.kind == "linear":
            return lambda theta_space, kwargs: BFLinearPymc(
                theta_space,
                args.n_walkers,
                args.n_iters,
                **kwargs_mcmc,
                **kwargs_pymc,
                **kwargs
            )
        else:  # logistic
            return lambda theta_space, kwargs: BFLogisticPymc(
                theta_space,
                args.n_walkers,
                args.n_iters,
                **kwargs_mcmc,
                **kwargs_pymc,
                **kwargs
            )


###################################################################
# CV FUNCTIONS
###################################################################

def bayesian_cv(
    X,
    y,
    cv_folds,
    params_cv,
    params_cv_names,
    params_cv_shape,
    theta_space_wrapper,
    mle_wrapper,
    bayesian_model_wrapper,
    include_p,
    p_max,
    all_estimates,
    point_estimates,
    kind,
    est_multiple,
    precompute_mle=False,
    verbose=False
):
    # Record score for all [fold, p, eta]
    score_bayesian_cv = defaultdict(
        lambda: np.zeros((cv_folds.n_splits, *params_cv_shape)))
    score_var_sel_cv = defaultdict(
        lambda: np.zeros((cv_folds.n_splits, *params_cv_shape)))

    # For each combination of [p, eta], save the corresponding estimator
    est_cv = np.empty(params_cv_shape, dtype=object)

    if include_p:
        theta_space = theta_space_wrapper(p_max)
        if precompute_mle:
            ts_fixed = theta_space.copy_p_fixed()
            mle_theta = mle_wrapper(X, y, ts_fixed)

    # Perform K-fold cross-validation for the parameters 'p' and 'η'
    for i, (train_cv, test_cv) in enumerate(cv_folds.split(X, y)):
        X_train_cv, X_test_cv = X[train_cv], X[test_cv]
        y_train_cv, y_test_cv = y[train_cv], y[test_cv]

        # Iterate over all possible pairs of hyperparameters
        for idx, param in enumerated_product(*params_cv):
            if include_p:
                param_without_p = param
                param_names_without_p = params_cv_names
            else:
                param_without_p = param[1:]
                param_names_without_p = params_cv_names[1:]
                theta_space = theta_space_wrapper(param[0])
                if precompute_mle:
                    mle_theta = mle_wrapper(X, y, theta_space)

            named_params = {
                k: v
                for k, v in zip(param_names_without_p, param_without_p)
            }

            if precompute_mle:
                named_params = {**named_params, "mle_precomputed": mle_theta}

            if verbose:
                it = iteration_count(
                    [cv_folds.n_splits, *params_cv_shape], [i, *idx])
                print(f"  * Launching model #{it}", end="\r")

            # Get models
            est_bayesian = bayesian_model_wrapper(theta_space, named_params)

            # Save estimator for eventual refitting
            est_cv[idx] = est_bayesian

            # Fit models
            est_bayesian.fit(X_train_cv, y_train_cv)

            # Bayesian models: compute score on test_cv
            for strategy in all_estimates:
                y_pred_cv = est_bayesian.predict(
                    X_test_cv, strategy=strategy)
                if kind == "linear":
                    score = -mean_squared_error(y_test_cv, y_pred_cv)
                else:
                    score = accuracy_score(y_test_cv, y_pred_cv)
                score_bayesian_cv[strategy][(i, *idx)] = score

            # Variable selection: compute score on test_cv
            for pe in point_estimates:
                y_pred_cv = bayesian_variable_selection_predict(
                    X_train_cv, y_train_cv, X_test_cv,
                    pe, est_bayesian, est_multiple)
                if kind == "linear":
                    score = -mean_squared_error(y_test_cv, y_pred_cv)
                else:
                    score = accuracy_score(y_test_cv, y_pred_cv)
                score_var_sel_cv[pe][(i, *idx)] = score

    return score_bayesian_cv, score_var_sel_cv, est_cv


def find_best_estimator_cv(mean_score_cv, est_cv):
    best_score = -np.inf
    best_estimator = None
    best_strategy = None
    for k, v in mean_score_cv.items():
        max_score_idx = np.unravel_index(v.argmax(), v.shape)
        if v[max_score_idx] > best_score:
            best_score = v[max_score_idx]
            best_strategy = k
            best_estimator = est_cv[max_score_idx]

    return best_estimator, best_strategy


###################################################################
# MAIN FUNCTION
###################################################################

def main():
    """Bayesian-RKHS Functional Regression experiments."""

    ##
    # SET PARAMETERS VALUES
    ##

    # Parse command-line arguments
    parser = get_arg_parser()
    args = parser.parse_args()

    # Randomness and reproducibility
    seed = args.seed
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Dataset generation parameters
    tau_range = (0, 1)
    beta_coef = simulation.cholaquidis_scenario3
    if args.kind == "linear":
        cv_folds = KFold(args.n_folds, shuffle=True, random_state=seed)
    else:
        cv_folds = StratifiedKFold(
            args.n_folds, shuffle=True, random_state=seed)

    # Decide if p is an hyperparameter or part of the model
    include_p = args.p_prior is not None

    # Main hyperparameters
    mle_method = 'L-BFGS-B'
    mle_strategy = 'global'
    etas = [10**i for i in range(args.eta_range[0], args.eta_range[1] + 1)]
    params_cv = [etas]
    params_cv_names = ["eta"]

    if include_p:
        prior_p = dict(enumerate(args.p_prior, start=1))
        p_max = len(prior_p)
    else:
        ps = [p for p in range(args.p_range[0], args.p_range[1] + 1)]
        prior_p = None
        p_max = None
        params_cv = [ps] + params_cv
        params_cv_names = ["p"] + params_cv_names

    # MCMC parameters
    moves = None
    step_fn = None
    step_kwargs = None
    if args.method == "emcee":
        if args.moves == "sw":
            moves = [
                (emcee.moves.StretchMove(), 0.7),
                (emcee.moves.WalkMove(), 0.3)]
        elif args.moves == "de":
            moves = [
                (emcee.moves.DEMove(), 0.8),
                (emcee.moves.DESnookerMove(), 0.2)]
        else:
            moves = [(emcee.moves.StretchMove(), 1.0)]
    else:
        if args.step == "nuts":
            step_fn = pm.NUTS
            step_kwargs = {"target_accept": args.target_accept}
        else:
            step_fn = pm.Metropolis
            step_kwargs = {}

    # Misc. parameters
    theta_names = ["β", "τ", "α0", "σ2"]
    if include_p:
        theta_names = ["p"] + theta_names
    point_estimates = ["mean", "median", "mode"]
    if args.kind == "linear":
        score_column = "MSE"
        all_estimates = ["posterior_mean"] + point_estimates
    else:
        score_column = "Acc"
        all_estimates = ["posterior_mean", "posterior_vote"] + point_estimates

    ##
    # GET DATASET
    ##

    if VERBOSE:
        print("Getting data...\n")

    # Get dataset parameters
    is_simulated_data = not args.data == "real"
    if is_simulated_data:
        model_type = args.data
    else:
        model_type = args.data_name

    if args.kernel == "ou":
        kernel_fn = simulation.ornstein_uhlenbeck_kernel
    elif args.kernel == "sqexp":
        kernel_fn = simulation.squared_exponential_kernel
    else:
        kernel_fn = simulation.fractional_brownian_kernel

    # Retrieve data
    if args.kind == "linear":
        X_fd, y, grid = get_data_linear(
            is_simulated_data,
            model_type,
            args.n_samples,
            args.n_grid,
            kernel_fn=kernel_fn,
            beta_coef=beta_coef,
            initial_smoothing=args.smoothing == "nw",
            tau_range=tau_range,
            rng=rng
        )
    else:
        X_fd, y, grid = get_data_logistic(
            is_simulated_data,
            model_type,
            args.n_samples,
            args.n_grid,
            kernel_fn=kernel_fn,
            beta_coef=beta_coef,
            noise=args.noise,
            initial_smoothing=args.smoothing == "nw",
            tau_range=tau_range,
            rng=rng
        )

    ##
    # RANDOM SPLITS LOOP
    ##

    score_ref_best = defaultdict(lambda: [])
    score_bayesian_best = np.zeros(args.n_reps)
    score_var_sel_best = np.zeros(args.n_reps)
    score_bayesian_all = []
    score_var_sel_all = []
    bayesian_strategy_count = defaultdict(lambda: 0)
    var_sel_strategy_count = defaultdict(lambda: 0)
    exec_times = np.zeros((args.n_reps, 2))  # (splits, (ref, bayesian))

    # Get wrappers for parameter space and bayesian regressor
    theta_space_wrapper = get_theta_space_wrapper(
        grid, include_p, theta_names, tau_range)
    if PRECOMPUTE_MLE:
        mle_wrapper = get_mle_wrapper(
            mle_method, mle_strategy, args.kind,
            args.n_reps_mle, args.n_cores, rng)
    else:
        mle_wrapper = None
    bayesian_model_wrapper = get_bayesian_model_wrapper(
        args, prior_p, rng, moves, step_fn, step_kwargs)

    # Multiple-regression estimator for variable selection algorithm
    if args.kind == "linear":
        cv_est_multiple = KFold(5, shuffle=True, random_state=seed)
        est_multiple = Pipeline([
            ("data", DataMatrix()),
            ("reg", RidgeCV(
                alphas=np.logspace(-3, 3, 10),
                scoring="neg_mean_squared_error",
                cv=cv_est_multiple))]
        )
    else:
        cv_est_multiple = StratifiedKFold(5, shuffle=True, random_state=seed)
        est_multiple = Pipeline([
            ("data", DataMatrix()),
            ("clf", LogisticRegressionCV(
                Cs=np.logspace(-3, 3, 10),
                scoring="accuracy",
                cv=cv_est_multiple))]
        )

    try:
        for rep in range(args.n_reps):
            # Train/test split
            if args.kind == "linear":
                X_train, X_test, y_train, y_test = train_test_split(
                    X_fd, y, train_size=args.train_size,
                    random_state=seed + rep)
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_fd, y, train_size=args.train_size,
                    stratify=y, random_state=seed + rep)

            # Standardize data
            X_train, X_test = preprocessing.standardize_predictors(
                X_train, X_test)

            ##
            # RUN REFERENCE ALGORITHMS
            ##

            if RUN_REF_ALGS:
                start = time.time()

                # Get reference models
                if args.kind == "linear":
                    est_ref = get_reference_models_linear(
                        X_train, y_train, seed + rep)
                else:
                    est_ref = get_reference_models_logistic(
                        X_train, y_train, seed + rep)

                if VERBOSE:
                    print(f"(It. {rep + 1}/{args.n_reps}) "
                          f"Running {len(est_ref)} reference models...")

                # Fit models (through CV+refitting) and predict on test set
                df_ref_split, _ = cv_sk(
                    est_ref,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    cv_folds,
                    kind=args.kind,
                    n_jobs=args.n_cores,
                    sort_by=0,
                    verbose=False
                )

                # Save score of best models
                ref_models_score = df_ref_split[["Estimator", score_column]]
                for name, score in ref_models_score.values:
                    score_ref_best[name].append(score)

                exec_times[rep, 0] = time.time() - start

            ##
            # RUN BAYESIAN ALGORITHM
            ##

            start = time.time()

            # Compute number of models
            params_cv_shape = tuple([len(param) for param in params_cv])
            total_models = np.prod([cv_folds.n_splits, *params_cv_shape])

            if VERBOSE:
                print(f"(It. {rep + 1}/{args.n_reps}) Running {total_models} "
                      "bayesian RKHS models...")

            score_bayesian_cv, score_var_sel_cv, est_cv = bayesian_cv(
                X_train,
                y_train,
                cv_folds,
                params_cv,
                params_cv_names,
                params_cv_shape,
                theta_space_wrapper,
                mle_wrapper,
                bayesian_model_wrapper,
                include_p,
                p_max,
                all_estimates,
                point_estimates,
                args.kind,
                est_multiple,
                precompute_mle=PRECOMPUTE_MLE,
                verbose=VERBOSE
            )

            # Compute mean mse across folds
            mean_score_bayesian_cv = {
                k: v.mean(axis=0) for k, v in score_bayesian_cv.items()}
            mean_score_var_sel_cv = {
                k: v.mean(axis=0) for k, v in score_var_sel_cv.items()}

            # Save CV results
            score_bayesian_all.append(mean_score_bayesian_cv)
            score_var_sel_all.append(mean_score_var_sel_cv)

            # Get best bayesian model (i.e. the best strategy)
            best_estimator_bayesian, best_strategy_bayesian = \
                find_best_estimator_cv(mean_score_bayesian_cv, est_cv)

            # Get best variable selection model (i.e. the best point estimate)
            best_estimator_var_sel, best_pe_var_sel = \
                find_best_estimator_cv(mean_score_var_sel_cv, est_cv)

            # Keep track of how often each strategy was the best
            bayesian_strategy_count[best_strategy_bayesian] += 1
            var_sel_strategy_count[best_pe_var_sel] += 1

            # Refit best models on the whole training set
            if VERBOSE:
                print("  * Refitting best models")

            best_estimator_bayesian.fit(X_train, y_train)
            best_estimator_var_sel.fit(X_train, y_train)

            # Compute predictions on the hold-out set
            y_pred_bayesian = best_estimator_bayesian.predict(
                X_test, strategy=best_strategy_bayesian)
            y_pred_var_sel = bayesian_variable_selection_predict(
                X_train, y_train, X_test,
                best_pe_var_sel, best_estimator_var_sel,
                est_multiple)

            # Save mse of best models
            if args.kind == "linear":
                score_bayesian = mean_squared_error(y_test, y_pred_bayesian)
                score_var_sel = mean_squared_error(y_test, y_pred_var_sel)
            else:
                score_bayesian = accuracy_score(y_test, y_pred_bayesian)
                score_var_sel = accuracy_score(y_test, y_pred_var_sel)

            score_bayesian_best[rep] = score_bayesian
            score_var_sel_best[rep] = score_var_sel

            exec_times[rep, 1] = time.time() - start

    except KeyboardInterrupt:
        print("\n[INFO] Process halted by user. Skipping...")
        rep = rep - 1

    ##
    # AVERAGE RESULTS ACROSS SPLITS
    ##

    mean_score_ref = [(k, np.mean(v), np.std(v))
                      for k, v in score_ref_best.items()]

    mean_score_bayesian_var_sel = [
        (
            args.method,
            np.mean(score_bayesian_best[:rep + 1]),
            np.std(score_bayesian_best[:rep + 1]),
            max(bayesian_strategy_count, key=bayesian_strategy_count.get)
        ),
        (
            args.method + "+sk_ridge",
            np.mean(score_var_sel_best[:rep + 1]),
            np.std(score_var_sel_best[:rep + 1]),
            max(var_sel_strategy_count, key=var_sel_strategy_count.get)
        )]

    df_metrics_ref = pd.DataFrame(
        mean_score_ref,
        columns=["Estimator", "Mean " + score_column, "SD"]
    ).sort_values("Mean " + score_column, ascending=args.kind == "linear")

    df_metrics_bayesian_var_sel = pd.DataFrame(
        mean_score_bayesian_var_sel,
        columns=["Estimator", "Mean " + score_column, "SD", "Main strategy"]
    ).sort_values("Mean " + score_column, ascending=args.kind == "linear")

    ##
    # PRINT RESULTS
    ##

    # Get filename
    if is_simulated_data:
        data_name = args.data + "_" + kernel_fn.__name__
    else:
        data_name = args.data_name
    smoothing = "_smoothing" if args.smoothing == "nw" else ""

    if args.kind == "linear":
        prefix_kind = "reg"
    else:
        prefix_kind = "clf"

    filename = (prefix_kind + "_" + args.method + "_"
                + (args.moves if args.method == "emcee" else args.step)
                + "_" + data_name + "_" + str(len(X_fd)) + smoothing
                + ("_p_free" if include_p else "")
                + "_nw_" + str(args.n_walkers) + "_ni_" + str(args.n_iters)
                + "_seed_" + str(seed))

    if PRINT_TO_FILE:
        print(f"\nSaving results to file '{filename}.results'")
        f = open(PRINT_PATH + filename + ".results", 'w')
        sys.stdout = f  # Change the standard output to the file we created

    print(f"\n*** Bayesian-RKHS Functional {args.kind.capitalize()} "
          "Regression ***\n")

    # Print dataset information

    print("-- GENERAL INFORMATION --")
    print(f"Random seed: {seed}")
    print(f"N_cores: {args.n_cores}")
    print(f"Random train/test splits: {rep + 1}")
    print(f"CV folds: {args.n_folds}")
    print("N_reps MLE:", args.n_reps_mle)

    print("\n-- MODEL GENERATION --")
    print(f"Total samples: {len(X_fd)}")
    print(f"Grid size: {len(X_fd.grid_points[0])}")
    print(f"Train size: {len(X_train)}")
    if args.smoothing == "nw":
        print("Smoothing: Nadaraya-Watson")
    else:
        print("Smoothing: None")

    if is_simulated_data:
        print(f"Model type: {args.data.upper()}")
        print(f"X ~ GP(0, {kernel_fn.__name__})")
    else:
        print(f"Data name: {args.data_name}")

    if args.kind == "logistic":
        print(f"Noise: {int(100*args.noise)}%")

    print("\n-- BAYESIAN RKHS MODEL --")
    print("Number of components (p):", (prior_p if include_p else ps))
    print("Values of η:", etas)
    print(f"g = {args.g}")

    if rep + 1 > 0:
        # Print MCMC method information
        if args.method == "emcee":
            print("\n-- EMCEE SAMPLER --")
            print(f"N_walkers: {args.n_walkers}")
            print(f"N_iters: {args.n_iters} + {args.n_tune}")
            print(f"Frac_random: {args.frac_random}")
            print("Moves:")
            for move, prob in moves:
                print(f"  {move.__class__.__name__}, {prob}")
        else:
            print("\n-- PYMC SAMPLER --")
            print(f"N_walkers: {args.n_walkers}")
            print(f"N_iters: {args.n_iters} + {args.n_tune}")
            print("Step method: "
                  + ("NUTS" if args.step == "nuts" else "Metropolis"))
            if args.step == "nuts":
                print(f"  Target accept: {args.target_accept}")

        # Print results

        if RUN_REF_ALGS:
            print("\n-- RESULTS REFERENCE ALGORITHMS --")
            print(
                "Mean split execution time: "
                f"{exec_times[:rep + 1, 0].mean():.3f}"
                f"±{exec_times[:rep + 1, 0].std():.3f} s")
            print("Total splits execution time: "
                  f"{exec_times[:rep + 1, 0].sum()/60.:.3f} min\n")
            print(df_metrics_ref.to_string(index=False, col_space=7))

        print(f"\n-- RESULTS {args.method.upper()} --")
        print(
            "Mean split execution time: "
            f"{exec_times[:rep + 1, 1].mean():.3f}"
            f"±{exec_times[:rep + 1, 1].std():.3f} s")
        print("Total splits execution time: "
              f"{exec_times[:rep + 1, 1].sum()/60.:.3f} min\n")
        print(df_metrics_bayesian_var_sel.to_string(index=False, col_space=7))

    ##
    # SAVE RESULTS
    ##

    try:
        if SAVE_RESULTS and rep + 1 > 0:
            # Save all the results dataframe in one CSV file
            df_all = [df_metrics_bayesian_var_sel]
            if RUN_REF_ALGS:
                df_all += [df_metrics_ref]

            df = pd.concat(
                df_all,
                axis=0,
                ignore_index=True
            )
            df.to_csv(SAVE_PATH + filename + ".csv", index=False)

            # Save the mean CV results and strategy statistics to disk
            np.savez(
                SAVE_PATH + filename + ".npz",
                mse_bayesian_all=score_bayesian_all,
                mse_var_sel_all=score_var_sel_all,
                bayesian_strategy_count=dict(bayesian_strategy_count),
                var_sel_strategy_count=dict(var_sel_strategy_count)
            )
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()
