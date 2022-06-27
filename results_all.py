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
import pymc as pm
from rkbfr import preprocessing, simulation
from rkbfr.bayesian_model import ThetaSpace, probability_to_label
from rkbfr.mcmc_sampler import (BFLinearEmcee, BFLinearPymc, BFLogisticEmcee,
                                BFLogisticPymc)
from rkbfr.mle import compute_mle
from run_utils import bayesian_variable_selection_predict
from skfda.datasets import (fetch_cran, fetch_growth, fetch_medflies,
                            fetch_phoneme, fetch_tecator)
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.preprocessing.smoothing.kernel_smoothers import \
    NadarayaWatsonSmoother as NW
from skfda.representation.basis import BSpline
from skfda.representation.grid import FDataGrid
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn_utils import DataMatrix

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
VERBOSE = True
PRECOMPUTE_MLE = True
PRINT_TO_FILE = False
PRINT_PATH = "results/"


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
        "--smoothing", default="nw", choices=["none", "nw", "basis"],
        help="smooth functional data as part of preprocessing"
    )
    parser.add_argument(
        "--train-size", type=float, default=0.5,
        help="fraction of data used for training"
    )
    parser.add_argument(
        "--noise", type=float, default=0.1,
        help="fraction of noise for logistic synthetic data"
    )
    parser.add_argument(
        "--standardize", action="store_true",
        help="whether to consider predictors with unit variance."
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
        "--n-burn", type=int, default=100,
        help="number of initial samples to discard in MCMC algorithm"
    )
    parser.add_argument(
        "--n-reps-mle", type=int, default=4,
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
        "--frac-random", type=float, default=0.1,
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
        choices=["rkhs", "l2", "real", "mixture"]
    )
    data_group.add_argument(
        "--kernel",
        help="name of kernel to use in simulations",
        choices=["ou", "sqexp", "fbm", "bm",
                 "gbm", "homoscedastic", "heteroscedastic"]
    )
    data_group.add_argument(
        "--data-name",
        help="name of data set to use as real data",
        choices=["tecator", "moisture", "sugar",
                 "medflies", "growth", "phoneme"]
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
    X_type,
    model_type,
    n_samples=150,
    n_grid=100,
    kernel_fn=None,
    beta_coef=None,
    initial_smoothing="none",
    tau_range=(0, 1),
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    if is_simulated_data:
        grid = np.linspace(tau_range[0] + 1./n_grid, tau_range[1], n_grid)
        mean_vector = None
        alpha0_true = 5.
        sigma2_true = 0.5

        # Generate X
        if X_type == "gbm":
            X = np.exp(
                simulation.gp(
                    grid,
                    mean_vector,
                    simulation.brownian_kernel,
                    n_samples,
                    rng
                ))
        else:  # GP
            X = simulation.gp(
                grid,
                mean_vector,
                kernel_fn,
                n_samples,
                rng
            )

        # Generate Y
        if model_type == "l2":
            if beta_coef is None:
                raise ValueError("Must provide a coefficient function.")

            y = simulation.generate_l2_dataset(
                X,
                grid,
                beta_coef,
                alpha0_true,
                sigma2_true,
                rng=rng
            )
        elif model_type == "rkhs":
            beta_true = [-5., 1., 10.]
            tau_true = [0.1, 0.4, 0.8]

            y = simulation.generate_rkhs_dataset(
                X,
                grid,
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
            data = X_fd.data_matrix[..., 0]
            u, idx = np.unique(data, axis=0, return_index=True)  # Find repeated
            X_fd = FDataGrid(data[idx], X_fd.grid_points[0]).derivative(order=2)
            y = y[idx, 1]  # Fat level
        elif model_type == "moisture":
            data = fetch_cran("Moisturespectrum", "fds")["Moisturespectrum"]
            y = fetch_cran("Moisturevalues", "fds")["Moisturevalues"]
            X_fd = FDataGrid(data["y"].T[:, ::7], data["x"][::7])
        elif model_type == "sugar":
            data = np.load('data/sugar.npz')
            X_fd = FDataGrid(data['x'][:, ::5])
            y = data['y']
        else:
            raise ValueError("Real data set must be 'tecator', "
                             "'moisture' or 'sugar'.")

        grid = preprocessing.normalize_grid(
            X_fd.grid_points[0], tau_range[0], tau_range[1])

        X_fd = FDataGrid(X_fd.data_matrix, grid)

    # Smooth data
    if initial_smoothing != "none":
        if initial_smoothing == "nw":
            smoother = NW()
        else:
            smoother = BasisSmoother(BSpline(n_basis=16))

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
    kernel_fn2=None,
    mean_vector2=None,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    if is_simulated_data:
        grid = np.linspace(tau_range[0] + 1./n_grid, tau_range[1], n_grid)
        mean_vector = None
        alpha0_true = -0.5

        if model_type == "mixture":
            X, y = simulation.generate_mixture_dataset(
                grid, mean_vector, mean_vector2,
                kernel_fn, kernel_fn2,
                n_samples, rng,
            )

        else:  # Logistic model (RKHS or L2)
            # Generate X
            X = simulation.gp(
                grid,
                mean_vector,
                kernel_fn,
                n_samples,
                rng
            )

            # Generate y
            if model_type == "l2":
                if beta_coef is None:
                    raise ValueError("Must provide a coefficient function.")

                y_lin = simulation.generate_l2_dataset(
                    X,
                    grid,
                    beta_coef,
                    alpha0_true,
                    sigma2=0.0,
                    rng=rng
                )
            elif model_type == "rkhs":
                beta_true = [-5., 1., 10.]
                tau_true = [0.1, 0.4, 0.8]
                y_lin = simulation.generate_rkhs_dataset(
                    X,
                    grid,
                    beta_true,
                    tau_true,
                    alpha0_true,
                    sigma2=0.0,
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
        elif model_type == "phoneme":
            X_fd, y = fetch_phoneme(return_X_y=True)
            y_idx = np.where(y < 2)[0]  # Only 2 classes
            rand_idx = rng.choice(y_idx, size=200)  # Choose 200 random curves
            X_fd = FDataGrid(
                X_fd.data_matrix[rand_idx, ::2, 0],  # Half the grid resolution
                X_fd.grid_points[0][::2]
            )
            y = y[rand_idx]
        else:
            raise ValueError("Real data set must be 'medflies', "
                             "'growth' or 'phoneme'.")

        grid = preprocessing.normalize_grid(
            X_fd.grid_points[0], tau_range[0], tau_range[1])

        X_fd = FDataGrid(X_fd.data_matrix, grid)

    # Smooth data
    if initial_smoothing != "none":
        if initial_smoothing == "nw":
            smoother = NW()
        else:
            smoother = BasisSmoother(BSpline(n_basis=16))

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

def get_theta_space_wrapper(
    grid,
    include_p,
    theta_names,
    tau_range,
    beta_range,
    sigma2_ub
):
    return lambda p: ThetaSpace(
        p,
        grid,
        include_p=include_p,
        names=theta_names,
        tau_range=tau_range,
        beta_range=beta_range,
        sigma2_ub=sigma2_ub
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
        "burn": args.n_burn,
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

    # Decide if p is an hyperparameter or part of the model
    include_p = args.p_prior is not None

    # Main hyperparameters
    mle_method = 'L-BFGS-B'
    mle_strategy = 'global'
    etas = [10**i for i in range(args.eta_range[0], args.eta_range[1] + 1)]
    params = [etas]
    params_names = ["eta"]
    params_symbols = ["η"]

    if include_p:
        prior_p = dict(enumerate(args.p_prior, start=1))
        p_max = len(prior_p)
    else:
        ps = [p for p in range(args.p_range[0], args.p_range[1] + 1)]
        prior_p = None
        p_max = None
        params = [ps] + params
        params_names = ["p"] + params_names
        params_symbols = ["p"] + params_symbols

    # MCMC parameters
    beta_range = (-1000, 1000) if include_p else None
    sigma2_ub = np.inf
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

    # Names
    theta_names = ["β", "τ", "α0", "σ2"]
    if include_p:
        theta_names = ["p"] + theta_names
    point_estimates = ["mean", "median", "mode"]
    if args.kind == "linear":
        score_column = "MSE"
        all_estimates = ["posterior_mean"] + point_estimates
        columns_name = [
            "Estimator",
            *params_symbols,
            "Mean MSE", "SD MSE",
            "Mean rMSE", "SD rMSE"
        ]
    else:
        score_column = "Acc"
        all_estimates = ["posterior_mean", "posterior_vote"] + point_estimates
        columns_name = ["Estimator", *params_symbols, "Mean Acc", "SD Acc"]

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
    elif args.kernel == "bm":
        kernel_fn = simulation.brownian_kernel
    elif args.kernel == "fbm":
        kernel_fn = simulation.fractional_brownian_kernel
    else:  # gbm or mixture or real data
        kernel_fn = None

    # Retrieve data
    if args.kind == "linear":
        X_type = "gbm" if args.kernel == "gbm" else "gp"
        X_fd, y, grid = get_data_linear(
            is_simulated_data,
            X_type,
            model_type,
            args.n_samples,
            args.n_grid,
            kernel_fn=kernel_fn,
            beta_coef=beta_coef,
            initial_smoothing=args.smoothing,
            tau_range=tau_range,
            rng=rng
        )
    else:  # logistic
        if args.data == "mixture":
            kernel_fn = simulation.brownian_kernel
            if args.kernel == "homoscedastic":
                kernel_fn2 = kernel_fn
                half_n_grid = args.n_grid//2
                mean_vector2 = np.concatenate((  # 0 until 0.5, then 0.75t
                    np.full(half_n_grid, 0),
                    0.75*np.linspace(
                        tau_range[0], tau_range[1], args.n_grid - half_n_grid)
                ))
            else:  # heteroscedastic
                mean_vector2 = None

                def kernel_fn2(s, t):
                    return simulation.brownian_kernel(s, t, 1.5)
        else:
            mean_vector2 = None
            kernel_fn2 = None

        X_fd, y, grid = get_data_logistic(
            is_simulated_data,
            model_type,
            args.n_samples,
            args.n_grid,
            kernel_fn=kernel_fn,
            beta_coef=beta_coef,
            noise=args.noise,
            initial_smoothing=args.smoothing,
            tau_range=tau_range,
            kernel_fn2=kernel_fn2,
            mean_vector2=mean_vector2,
            rng=rng
        )

    ##
    # RANDOM SPLITS LOOP
    ##

    score_bayesian_all = defaultdict(list)
    score_var_sel_all = defaultdict(list)

    if args.kind == "linear":
        rmse_bayesian_all = defaultdict(list)
        rmse_var_sel_all = defaultdict(list)

    exec_times = np.zeros(args.n_reps)

    # Get wrappers for parameter space and bayesian regressor
    theta_space_wrapper = get_theta_space_wrapper(
        grid, include_p, theta_names, tau_range, beta_range, sigma2_ub)
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
                alphas=np.logspace(-4, 4, 10),
                scoring="neg_mean_squared_error",
                cv=cv_est_multiple))]
        )
    else:
        cv_est_multiple = StratifiedKFold(5, shuffle=True, random_state=seed)
        est_multiple = Pipeline([
            ("data", DataMatrix()),
            ("clf", LogisticRegressionCV(
                Cs=np.logspace(-4, 4, 10),
                scoring="accuracy",
                cv=cv_est_multiple))]
        )

    # Train/test splits
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
                X_train, X_test, scale=args.standardize)

            ##
            # RUN BAYESIAN ALGORITHM
            ##

            start = time.time()

            # Compute number of models
            params_shape = tuple([len(param) for param in params])
            total_models = np.prod([*params_shape])

            if VERBOSE:
                print(f"(It. {rep + 1}/{args.n_reps}) Running {total_models} "
                      f"bayesian RKHS {args.kind} models...")

            if include_p:
                theta_space = theta_space_wrapper(p_max)
                if PRECOMPUTE_MLE:
                    ts_fixed = theta_space.copy_p_fixed()

            # Perform K-fold cross-validation for the parameters 'p' and 'η'
            if PRECOMPUTE_MLE:
                # Save precomputed mles
                mle_dict = {}
                if include_p:
                    mle_theta = mle_wrapper(X_train, y_train, ts_fixed)

            # Iterate over all possible pairs of hyperparameters
            for idx, param in enumerated_product(*params):
                if include_p:
                    param_without_p = param
                    param_names_without_p = params_names
                    param_values = [p_max, *param]
                else:
                    param_values = param
                    p = param[0]
                    param_without_p = param[1:]
                    param_names_without_p = params_names[1:]
                    theta_space = theta_space_wrapper(p)
                    if PRECOMPUTE_MLE:
                        if p not in mle_dict:
                            mle_dict[p] = mle_wrapper(
                                X_train, y_train, theta_space)
                        mle_theta = mle_dict[p]

                estimator_kwargs = {
                    k: v
                    for k, v in zip(param_names_without_p, param_without_p)
                }

                if PRECOMPUTE_MLE:
                    estimator_kwargs = {
                        **estimator_kwargs, "mle_precomputed": mle_theta}

                if VERBOSE:
                    it = iteration_count([*params_shape], [*idx])
                    print(f"  * Launching model #{it}", end="\r")

                # Get models
                estimator = bayesian_model_wrapper(
                    theta_space, estimator_kwargs)

                # Fit models
                estimator.fit(X_train, y_train)

                # Bayesian models: compute score on test set
                for strategy in all_estimates:
                    y_pred = estimator.predict(
                        X_test, strategy=strategy)
                    if args.kind == "linear":
                        score = mean_squared_error(y_test, y_pred)
                        rmse_bayesian_all[(strategy, *param_values)].append(
                            score/np.var(y_test))
                    else:
                        score = accuracy_score(y_test, y_pred)
                    score_bayesian_all[(strategy, *param_values)].append(score)

                # Variable selection: compute score on test set
                for pe in point_estimates:
                    y_pred = bayesian_variable_selection_predict(
                        X_train, y_train, X_test,
                        pe, estimator, est_multiple)
                    if args.kind == "linear":
                        score = mean_squared_error(y_test, y_pred)
                        rmse_var_sel_all[(pe, *param_values)].append(
                            score/np.var(y_test))
                    else:
                        score = accuracy_score(y_test, y_pred)
                    score_var_sel_all[(pe, *param_values)].append(score)

            exec_times[rep] = time.time() - start

    except KeyboardInterrupt:
        print("\n[INFO] Process halted by user. Skipping...")
        rep = rep - 1

    ##
    # AVERAGE RESULTS ACROSS SPLITS
    ##

    mean_scores = []

    if args.kind == "linear":
        dict_results = [
            (args.method + "_", "", score_bayesian_all, rmse_bayesian_all),
            (args.method + "_", "+ridge", score_var_sel_all, rmse_var_sel_all)
        ]

        for prefix, suffix, d1, d2 in dict_results:
            # Average MSE and relative MSE
            mean_scores.append([
                (
                    prefix + k + suffix,
                    *params,
                    np.mean(d1[(k, *params)]), np.std(d1[(k, *params)]),
                    np.mean(d2[(k, *params)]), np.std(d2[(k, *params)])
                )
                for (k, *params) in d1.keys()])

    else:  # logistic
        # Average accuracy
        mean_scores.append([
            (
                args.method + "_" + k,
                *params,
                np.mean(v), np.std(v)
            )
            for (k, *params), v in score_bayesian_all.items()])
        mean_scores.append([
            (
                args.method + "_" + k + "+logistic",
                *params,
                np.mean(v), np.std(v)
            )
            for (k, *params), v in score_var_sel_all.items()])

    df_metrics_bayesian = pd.DataFrame(
        mean_scores[0],
        columns=columns_name
    ).sort_values("Mean " + score_column, ascending=args.kind == "linear")

    df_metrics_var_sel = pd.DataFrame(
        mean_scores[1],
        columns=columns_name
    ).sort_values("Mean " + score_column, ascending=args.kind == "linear")

    ##
    # PRINT RESULTS
    ##

    # Get filename
    if is_simulated_data:
        if args.data == "mixture":
            data_name = "mixture_" + args.kernel
        elif args.kernel == "gbm":
            data_name = "gbm_" + args.data
        else:
            data_name = "gp_" + kernel_fn.__name__ + "_" + args.data
    else:
        data_name = args.data_name

    if args.smoothing == "nw":
        smoothing = "_smoothing_nw"
    elif args.smoothing == "basis":
        smoothing = "_smoothing_basis"
    else:
        smoothing = ""

    if args.kind == "linear":
        prefix_kind = "reg"
    else:
        prefix_kind = "clf"

    filename = (prefix_kind + "_all_" + args.method + "_"
                + (args.moves if args.method == "emcee" else args.step)
                + "_" + data_name + "_" + str(len(X_fd)) + smoothing
                + ("_p_free" if include_p else "")
                + "_nw_" + str(args.n_walkers) + "_ni_" + str(args.n_iters)
                + "_seed_" + str(seed))

    if PRINT_TO_FILE:
        print(f"\nSaving results to file '{filename}.results'")
        f = open(PRINT_PATH + filename + ".results", 'w')
        sys.stdout = f  # Change the standard output to the file we created

    print(f"\n\n*** Bayesian-RKHS Functional {args.kind.capitalize()} "
          "Regression ***\n")

    # Print dataset information

    print("-- GENERAL INFORMATION --")
    print(f"Random seed: {seed}")
    print(f"N_cores: {args.n_cores}")
    print(f"Random train/test splits: {rep + 1}")
    print("N_reps MLE:", args.n_reps_mle)

    print("\n-- MODEL GENERATION --")
    print(f"Total samples: {len(X_fd)}")
    print(f"Grid size: {len(X_fd.grid_points[0])}")
    print(f"Train size: {len(X_train)}")
    if args.smoothing == "nw":
        print("Smoothing: Nadaraya-Watson")
    elif args.smoothing == "basis":
        print("Smoothing: BSpline(16)")
    else:
        print("Smoothing: None")

    if is_simulated_data:
        if args.data == "mixture":
            if args.kernel == "homoscedastic":
                print("Model type: BM(0, 1) + BM(m(t), 1)")
            else:
                print("Model type: BM(0, 1) + BM(0, 1.5)")
        else:
            if args.kernel == "gbm":
                print("X ~ GBM(0, 1)")
            else:
                print(f"X ~ GP(0, {kernel_fn.__name__})")
            print(f"Model type: {args.data.upper()}")

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

        print(f"\n-- RESULTS {args.method.upper()} --")
        print(
            "Mean split execution time: "
            f"{exec_times[:rep + 1].mean():.3f}"
            f"±{exec_times[:rep + 1].std():.3f} s")
        print("Total splits execution time: "
              f"{exec_times[:rep + 1].sum()/60.:.3f} min\n")

        print("Functional methods:\n")
        print(df_metrics_bayesian.to_string(index=False, col_space=4))
        print("\nVariable selection methods:\n")
        print(df_metrics_var_sel.to_string(index=False, col_space=4))


if __name__ == "__main__":
    main()
