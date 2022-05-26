# encoding: utf-8

import logging
import numbers
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from _fpls import APLS, FPLS
from arviz import concat, convert_to_inference_data, kde, make_ufunc
from IPython.display import display
from scipy.stats import mode
from skfda.ml.regression import KNeighborsRegressor
from skfda.ml.regression import LinearRegression as FLinearRegression
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from skfda.preprocessing.dim_reduction.variable_selection import \
    RecursiveMaximaHunting as RMH
from skfda.representation.basis import FDataBasis
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn_utils import (Basis, DataMatrix, FeatureSelector,
                           PLSRegressionWrapper)


# Custom context managers for handling warnings

class IgnoreWarnings():
    key = "PYTHONWARNINGS"

    def __enter__(self):
        if self.key in os.environ:
            self.state = os.environ["PYTHONWARNINGS"]
        else:
            self.state = "default"
        os.environ["PYTHONWARNINGS"] = "ignore"
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        os.environ["PYTHONWARNINGS"] = self.state


class HandleLogger():
    def __init__(self, verbose: int = 0):
        self.verbose = verbose

    def __enter__(self):
        if self.verbose < 2:
            logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


def check_random_state(seed):
    """Turn seed into a np.random.Generator instance.

    For compatibility with sklearn, the case in which the
    seed is a np.random.RandomState is also considered.

    Parameters
    ----------
    seed : None, int, np.random.RandomState or Generator.
        If seed is None, return a Generator with default initialization.
        If seed is an int, return a new Generator instance seeded with seed.
        If seed is an instance of RandomState, convert it to Generator.
        If seed is already a Generator instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, numbers.Integral):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.RandomState):
        return np.random.default_rng(seed.get_state()[1])
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.Generator instance" % seed
    )


def pp_to_idata(pps, idata, var_names, y_obs=None, merge=False):
    """All the pp arrays must have the same shape (the shape of y_obs)."""
    dim_name = "prediction"
    coords = idata.posterior[["chain", "draw"]].coords
    coords.update({dim_name: np.arange(0, pps[0].shape[-1])})
    data_vars = {}

    for pp, var_name in zip(pps, var_names):
        data_vars[var_name] = (("chain", "draw", dim_name), pp)

    idata_pp = convert_to_inference_data(
        xr.Dataset(data_vars=data_vars, coords=coords),
        group="posterior_predictive",
    )

    if merge:
        idata.extend(idata_pp)
    else:
        if y_obs is None:
            idata_aux = convert_to_inference_data(
                idata.observed_data, group="observed_data")
        else:
            idata_aux = convert_to_inference_data(
                xr.Dataset(data_vars={"y_obs": ("observation", y_obs)},
                           coords=coords),
                group="observed_data")

        concat(idata_pp, idata_aux, inplace=True)

        return idata_pp


def mode_fn(values, skipna=False, bw='experimental'):
    """Note that NaN values are always ignored."""
    if not skipna and np.isnan(values).any():
        warnings.warn("Your data appears to have NaN values.")

    if values.dtype.kind == "f":
        x, density = kde(values, bw=bw)
        return x[np.argmax(density)]
    else:
        return mode(values)[0][0]


def compute_mode_xarray(
    data,
    dim=("chain", "draw"),
    skipna=False,
    bw='experimental'
):
    def mode_fn_args(x):
        return mode_fn(x, skipna=skipna, bw=bw)

    return xr.apply_ufunc(
        make_ufunc(mode_fn_args), data,
        input_core_dims=(dim,)
    )


def linear_regression_metrics(
    y_true,
    y_pred,
    n_features,
    predictor_name,
    df=None,
    sort_by=-2,
):
    if df is None:
        results_columns = ["Estimator", "Features", "MSE", r"$R^2$"]
        df = pd.DataFrame(columns=results_columns)

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    df.loc[len(df)] = [
        predictor_name,
        n_features,
        mse,
        r2
    ]

    df.sort_values(df.columns[sort_by], inplace=True)

    return df


def bayesian_variable_selection_predict(
    X_train,
    y_train,
    X_test,
    pe,
    reg_bayesian,
    reg_linear
):
    X_train_red = reg_bayesian.transform(X_train, pe=pe)
    X_test_red = reg_bayesian.transform(X_test, pe=pe)
    reg_linear.fit(X_train_red, y_train)
    y_pred = reg_linear.predict(X_test_red)

    return y_pred


def cv_sk(
    regressors,
    X,
    y,
    X_test,
    y_test,
    folds,
    n_jobs=1,
    df=None,
    sort_by=-2,
    verbose=False,
):
    for name, pipe, params in regressors:
        if verbose:
            print(f"  Fitting {name}...")

        reg_cv = GridSearchCV(
            pipe,
            params,
            scoring="neg_mean_squared_error",
            n_jobs=n_jobs,
            cv=folds
        )

        with IgnoreWarnings():
            reg_cv.fit(X, y)

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

        y_pred = reg_cv.predict(X_test)
        df = linear_regression_metrics(
            y_test,
            y_pred,
            n_features,
            name,
            df,
            sort_by
        )

    return df, reg_cv


def multiple_linear_regression_cv(
    X,
    y,
    X_test,
    y_test,
    folds,
    regressors=[],
    n_jobs=1,
    prefix="emcee",
    pe='mode',
    df=None,
    sort_by=-2,
    verbose=False,
    random_state=None,
):
    if len(regressors) == 0:
        reg_lst = []
        alphas = np.logspace(-4, 4, 20)
        params_regularizer = {"reg__alpha": alphas}
        params_svm = {"reg__C": alphas,
                      "reg__gamma": ['auto', 'scale']}

        # MCMC+Lasso
        reg_lst.append((
            f"{prefix}_{pe}+sk_lasso",
            Pipeline([("reg", Lasso())]),
            params_regularizer
        ))

        # MCMC+Ridge
        reg_lst.append((
            f"{prefix}_{pe}+sk_ridge",
            Pipeline([("reg", Ridge(random_state=random_state))]),
            params_regularizer
        ))

        # MCMC+SVM RBF
        reg_lst.append((
            f"{prefix}_{pe}+sk_svm_rbf",
            Pipeline([("reg", SVR(kernel='rbf'))]),
            params_svm
        ))

    else:
        reg_lst = regressors

    df_metrics, _ = cv_sk(
        reg_lst,
        X,
        y,
        X_test,
        y_test,
        folds,
        n_jobs,
        df,
        sort_by,
        verbose
    )

    return df_metrics


def run_bayesian_model(
    estimator,
    X,
    y,
    X_test,
    y_test,
    folds,
    n_jobs=1,
    sort_by=-2,
    prefix="emcee",
    compute_metrics=True,
    verbose=False,
    notebook=False,
    random_state=None,
):
    estimator.fit(X, y)

    if verbose:
        fit_summary = estimator.summary()
        print(f"Mean acceptance: {100*estimator.mean_acceptance():.3f}%")
        if notebook:
            display(fit_summary)
        else:
            print(fit_summary)

        print("\nComputing metrics...\n")

    if compute_metrics:

        # -- Compute metrics using several point estimates

        y_pred_pp = estimator.predict(X_test, strategy='posterior_mean')
        df_metrics = linear_regression_metrics(
            y_test,
            y_pred_pp,
            estimator.n_components("posterior_mean"),
            prefix + "_posterior_mean",
            sort_by=sort_by
        )

        for pe in estimator.default_point_estimates:
            y_pred_pe = estimator.predict(X_test, strategy=pe)
            df_metrics = linear_regression_metrics(
                y_test,
                y_pred_pe,
                estimator.n_components(pe),
                prefix + "_" + pe,
                df=df_metrics,
                sort_by=sort_by
            )

        # -- Bayesian variable selection

        for pe in estimator.default_point_estimates:
            X_red = estimator.transform(X, pe=pe)
            X_test_red = estimator.transform(X_test, pe=pe)

            df_metrics = multiple_linear_regression_cv(
                X_red,
                y,
                X_test_red,
                y_test,
                folds,
                n_jobs=n_jobs,
                prefix=prefix,
                pe=pe,
                df=df_metrics,
                sort_by=sort_by,
                verbose=False,
                random_state=random_state
            )

    else:
        df_metrics = pd.DataFrame()

    return df_metrics


def linear_regression_comparison_suite(
    params_regularizer,
    params_select,
    params_dim_red,
    params_svm,
    params_basis,
    params_pls,
    params_knn,
    random_state=None
):
    regressors = []

    """
    MULTIVARIATE MODELS
    """

    # Lasso
    regressors.append(("sk_lasso",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("reg", Lasso())]),
                       params_regularizer
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
                       {**params_regularizer, **params_select}
                       ))

    # FPCA+Ridge
    regressors.append(("fpca+sk_ridge",
                       Pipeline([
                           ("dim_red", FPCA()),  # Retains scores only
                           ("reg", Ridge())]),
                       {**params_dim_red, **params_regularizer}
                       ))

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # FPLS (fixed basis)+Ridge
    regressors.append(("fpls_basis+sk_ridge",
                       Pipeline([
                           ("basis", Basis()),
                           ("dim_red", FPLS()),
                           ("reg", Ridge())]),
                       {**params_basis, **params_dim_red, **params_regularizer}
                       ))

    """

    # PCA+Ridge
    regressors.append(("pca+sk_ridge",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PCA(random_state=random_state)),
                           ("reg", Ridge())]),
                       {**params_dim_red, **params_regularizer}
                       ))

    # PLS+Ridge
    regressors.append(("pls+sk_ridge",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PLSRegressionWrapper()),
                           ("reg", Ridge())]),
                       {**params_dim_red, **params_regularizer}
                       ))

    # RMH+Ridge
    regressors.append(("rmh+sk_ridge",
                       Pipeline([
                           ("var_sel", RMH()),
                           ("reg", Ridge())]),
                       params_regularizer
                       ))

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # mRMR+Ridge
    regressors.append(("mRMR+sk_ridge",
                       Pipeline([
                           ("var_sel", mRMR()),
                           ("reg", Ridge())]),
                       {**params_mrmr, **params_regularizer}
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
                           ("dim_red", PCA(random_state=random_state)),
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

    """NOTE: while not strictly necessary, the test data undergoes the
             same basis expansion process as the training data. This is more
             computationally efficient and seems to improve the performance."""

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

    return regressors
