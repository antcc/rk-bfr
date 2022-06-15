# encoding: utf-8

import logging
import numbers
import os
import warnings

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from ._fpls import APLS, FPLS
from ._lda import LDA

try:
    from IPython.display import display
except ImportError:
    pass
from scipy.stats import mode
from skfda.ml.classification import (KNeighborsClassifier,
                                     MaximumDepthClassifier, NearestCentroid)
from skfda.ml.regression import KNeighborsRegressor
from skfda.ml.regression import LinearRegression as FLinearRegression
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from skfda.preprocessing.dim_reduction.variable_selection import \
    RecursiveMaximaHunting as RMH
from skfda.preprocessing.dim_reduction.variable_selection import \
    RKHSVariableSelection as RKVS
from skfda.representation.basis import FDataBasis
from skfda.representation.grid import FDataGrid
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .sklearn_utils import (Basis, DataMatrix, FeatureSelector,
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


def fdata_to_numpy(X, grid):
    N = len(grid)

    if isinstance(X, np.ndarray):
        if X.shape[1] != N:
            raise ValueError(
                "Data must be compatible with the specified grid")
    elif isinstance(X, FDataBasis):
        X = X.to_grid(grid_points=grid).data_matrix.reshape(-1, N)
    elif isinstance(X, FDataGrid):
        X = X.data_matrix.reshape(-1, N)
    else:
        raise ValueError('Data type not supported for X.')

    return X


def apply_threshold(y, th=0.5):
    y_th = np.copy(y).astype(int)
    y_th[..., y >= th] = 1
    y_th[..., y < th] = 0

    return y_th


def pp_to_idata(pps, idata, var_names, y_obs=None, merge=False):
    """All the pp arrays must have the same shape (the shape of y_obs)."""
    dim_name = "prediction"
    coords = idata.posterior[["chain", "draw"]].coords
    coords.update({dim_name: np.arange(0, pps[0].shape[-1])})
    data_vars = {}

    for pp, var_name in zip(pps, var_names):
        data_vars[var_name] = (("chain", "draw", dim_name), pp)

    idata_pp = az.convert_to_inference_data(
        xr.Dataset(data_vars=data_vars, coords=coords),
        group="posterior_predictive",
    )

    if merge:
        idata.extend(idata_pp)
    else:
        if y_obs is None:
            idata_aux = az.convert_to_inference_data(
                idata.observed_data, group="observed_data")
        else:
            idata_aux = az.convert_to_inference_data(
                xr.Dataset(data_vars={"y_obs": ("observation", y_obs)},
                           coords=coords),
                group="observed_data")

        az.concat(idata_pp, idata_aux, inplace=True)

        return idata_pp


def mode_fn(values, skipna=False, bw='experimental'):
    """Note that NaN values are always ignored."""
    if not skipna and np.isnan(values).any():
        warnings.warn("Your data appears to have NaN values.")

    if values.dtype.kind == "f":
        x, density = az.kde(values, bw=bw)
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
        az.make_ufunc(mode_fn_args), data,
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


def logistic_regression_metrics(
    y_true,
    y_pred,
    n_features,
    predictor_name,
    df=None,
    sort_by=-1,
):
    if df is None:
        results_columns = ["Estimator", "Features", "Acc"]
        df = pd.DataFrame(columns=results_columns)

    acc = accuracy_score(y_true, y_pred)
    df.loc[len(df)] = [
        predictor_name,
        n_features,
        acc
    ]

    df.sort_values(df.columns[sort_by], inplace=True, ascending=False)

    return df


def bayesian_variable_selection_predict(
    X_train,
    y_train,
    X_test,
    pe,
    est_bayesian,
    est_multiple
):
    X_train_red = est_bayesian.transform(X_train, pe=pe)
    X_test_red = est_bayesian.transform(X_test, pe=pe)
    est_multiple.fit(X_train_red, y_train)
    y_pred = est_multiple.predict(X_test_red)

    return y_pred


def cv_sk(
    estimators,
    X,
    y,
    X_test,
    y_test,
    folds,
    kind='linear',
    n_jobs=1,
    df=None,
    sort_by=-2,
    verbose=False,
):
    if kind == 'linear':
        scoring = "neg_mean_squared_error"
        est_name = "reg"
    else:
        scoring = "accuracy"
        est_name = "clf"

    for name, pipe, params in estimators:
        if verbose:
            print(f"  Fitting {name}...")

        est_cv = GridSearchCV(
            pipe,
            params,
            scoring=scoring,
            n_jobs=n_jobs,
            cv=folds
        )

        with IgnoreWarnings():
            est_cv.fit(X, y)

        if name == "fknn":
            K = est_cv.best_params_[f"{est_name}__n_neighbors"]
            n_features = f"K={K}"
        elif name == "mdc" or name == "fnc" or name == "optimal_bayes":
            n_features = X.data_matrix.shape[1]
        elif name == "flr":
            n_features = est_cv.best_estimator_[est_name].p
        elif "svm" in name:
            n_features = est_cv.best_estimator_[est_name].n_features_in_
        elif "pls1" in name:
            if kind == 'linear':
                n_features = est_cv.best_estimator_[est_name].n_components
            else:
                n_features = \
                    est_cv.best_estimator_[est_name].base_regressor.n_components
        else:
            if isinstance(est_cv.best_estimator_[est_name].coef_[0], FDataBasis):
                coef = \
                    est_cv.best_estimator_[est_name].coef_[0].coefficients[0]
            elif "logistic" in name:
                coef = est_cv.best_estimator_[est_name].coef_[0]
            else:
                coef = est_cv.best_estimator_[est_name].coef_

            n_features = sum(~np.isclose(coef, 0))

        y_pred = est_cv.predict(X_test)

        if kind == 'linear':
            df = linear_regression_metrics(
                y_test,
                y_pred,
                n_features,
                name,
                df,
                sort_by
            )
        else:
            df = logistic_regression_metrics(
                y_test,
                y_pred,
                n_features,
                name,
                df,
                sort_by
            )

    return df, est_cv


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
        """
        params_svm = {"reg__C": alphas,
                      "reg__gamma": ['auto', 'scale']}
        """

        """
        # MCMC+Lasso
        reg_lst.append((
            f"{prefix}_{pe}+lasso",
            Pipeline([("reg", Lasso())]),
            params_regularizer
        ))
        """

        # MCMC+Ridge
        reg_lst.append((
            f"{prefix}_{pe}+ridge",
            Pipeline([("reg", Ridge(random_state=random_state))]),
            params_regularizer
        ))

        """
        # MCMC+SVM RBF
        reg_lst.append((
            f"{prefix}_{pe}+svm_rbf",
            Pipeline([("reg", SVR(kernel='rbf'))]),
            params_svm
        ))
        """

    else:
        reg_lst = regressors

    df_metrics, _ = cv_sk(
        reg_lst,
        X,
        y,
        X_test,
        y_test,
        folds,
        kind='linear',
        n_jobs=n_jobs,
        df=df,
        sort_by=sort_by,
        verbose=verbose
    )

    return df_metrics


def multiple_logistic_regression_cv(
    X,
    y,
    X_test,
    y_test,
    folds,
    classifiers=[],
    n_jobs=1,
    prefix="emcee",
    pe='mode',
    df=None,
    sort_by=-1,
    verbose=False,
    random_state=None,
):
    if len(classifiers) == 0:
        clf_lst = []
        Cs = np.logspace(-4, 4, 20)
        params_clf = {"clf__C": Cs}
        # params_svm = {"clf__gamma": ['auto', 'scale']}

        # Emcee+LR
        clf_lst.append((
            f"{prefix}_{pe}+logistic",
            Pipeline([
                ("clf", LogisticRegression(random_state=random_state))]),
            params_clf
        ))

        """
        # Emcee+SVM Linear
        clf_lst.append((
            f"{prefix}_{pe}+svm_lin",
            Pipeline([
                ("clf", LinearSVC(random_state=random_state))]),
            params_clf
        ))
        """

        """
        # Emcee+SVM RBF
        clf_lst.append((
            f"{prefix}_{pe}+svm_rbf",
            Pipeline([
                ("clf", SVC(kernel='rbf'))]),
            {**params_svm, **params_clf}
        ))
        """

    else:
        clf_lst = classifiers

    df_metrics, _ = cv_sk(
        clf_lst,
        X,
        y,
        X_test,
        y_test,
        folds,
        kind='logistic',
        n_jobs=n_jobs,
        df=df,
        sort_by=sort_by,
        verbose=verbose
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
    kind='linear',
    prefix='emcee',
    compute_metrics=True,
    verbose=False,
    notebook=False,
    random_state=None,
):
    if kind == 'linear':
        metrics_fn = linear_regression_metrics
        multiple_regression_cv_fn = multiple_linear_regression_cv
    else:
        metrics_fn = logistic_regression_metrics
        multiple_regression_cv_fn = multiple_logistic_regression_cv

    estimator.fit(X, y)

    if verbose:
        fit_summary = estimator.summary()
        print(f"Mean acceptance: {100*estimator.mean_acceptance():.3f}%")
        if notebook:
            display(fit_summary)
        else:
            print(fit_summary)

    if compute_metrics:

        if verbose:
            print("\nComputing metrics...\n")

        # -- Compute metrics using several point estimates

        df_metrics = None

        for strategy in estimator.default_strategies:
            y_pred_pp = estimator.predict(X_test, strategy=strategy)
            df_metrics = metrics_fn(
                y_test,
                y_pred_pp,
                estimator.n_components(strategy),
                prefix + "_" + strategy,
                df=df_metrics,
                sort_by=sort_by
            )

        for pe in estimator.default_point_estimates:
            y_pred_pe = estimator.predict(X_test, strategy=pe)
            df_metrics = metrics_fn(
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

            df_metrics = multiple_regression_cv_fn(
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
    regressors.append(("lasso",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("reg", Lasso())]),
                       params_regularizer
                       ))

    # PLS1 regression
    regressors.append(("pls1",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("reg", PLSRegressionWrapper())]),
                       params_pls
                       ))

    """
    VARIABLE SELECTION + MULTIVARIATE MODELS
    """

    # Manual+Ridge
    regressors.append(("manual_sel+ridge",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("selector", FeatureSelector()),
                           ("reg", Ridge())]),
                       {**params_regularizer, **params_select}
                       ))

    # FPCA+Ridge
    regressors.append(("fpca+ridge",
                       Pipeline([
                           ("dim_red", FPCA()),  # Retains scores only
                           ("reg", Ridge())]),
                       {**params_dim_red, **params_regularizer}
                       ))

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # FPLS (fixed basis)+Ridge
    regressors.append(("fpls_basis+ridge",
                       Pipeline([
                           ("basis", Basis()),
                           ("dim_red", FPLS()),
                           ("reg", Ridge())]),
                       {**params_basis, **params_dim_red, **params_regularizer}
                       ))

    """

    # PCA+Ridge
    regressors.append(("pca+ridge",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PCA(random_state=random_state)),
                           ("reg", Ridge())]),
                       {**params_dim_red, **params_regularizer}
                       ))

    # PLS+Ridge
    regressors.append(("pls+ridge",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PLSRegressionWrapper()),
                           ("reg", Ridge())]),
                       {**params_dim_red, **params_regularizer}
                       ))

    # RMH+Ridge
    regressors.append(("rmh+ridge",
                       Pipeline([
                           ("var_sel", RMH()),
                           ("reg", Ridge())]),
                       params_regularizer
                       ))

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # mRMR+Ridge
    regressors.append(("mRMR+ridge",
                       Pipeline([
                           ("var_sel", mRMR()),
                           ("reg", Ridge())]),
                       {**params_mrmr, **params_regularizer}
                       ))
    """

    """
    # Manual+SVM RBF
    regressors.append(("manual_sel+svm_rbf",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("selector", FeatureSelector()),
                           ("reg", SVR(kernel='rbf'))]),
                       {**params_select, **params_svm}
                       ))

    # FPCA+SVM RBF
    regressors.append(("fpca+svm_rbf",
                       Pipeline([
                           ("dim_red", FPCA()),  # Retains scores only
                           ("reg", SVR(kernel='rbf'))]),
                       {**params_dim_red, **params_svm}
                       ))
    """

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # FPLS (fixed basis)+SMV RBF
    regressors.append(("fpls_basis+svm_rbf",
                       Pipeline([
                           ("basis", Basis()),
                           ("dim_red", FPLS()),
                           ("reg", SVR(kernel='rbf'))]),
                       {**params_basis, **params_dim_red, **params_svm}
                       ))
    """

    """
    # PCA+SVM RBF
    regressors.append(("pca+svm_rbf",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PCA(random_state=random_state)),
                           ("reg", SVR(kernel='rbf'))]),
                       {**params_dim_red, **params_svm}
                       ))

    # PLS+SMV RBF
    regressors.append(("pls+svm_rbf",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PLSRegressionWrapper()),
                           ("reg", SVR(kernel='rbf'))]),
                       {**params_dim_red, **params_svm}
                       ))

    # RMH+SVM RBF
    regressors.append(("rmh+svm_rbf",
                       Pipeline([
                           ("var_sel", RMH()),
                           ("reg", SVR(kernel='rbf'))]),
                       params_svm
                       ))

    """
    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # mRMR+SVM RBF
    regressors.append(("mRMR+svm_rbf",
                       Pipeline([
                           ("var_sel", mRMR()),
                           ("reg", SVR(kernel='rbf'))]),
                       {**params_mrmr, **params_svm}
                       ))
    """

    """
    FUNCTIONAL MODELS
    """

    regressors.append(("apls",
                       Pipeline([
                           ("reg", APLS())]),
                       params_pls
                       ))

    """NOTE: while not strictly necessary, the test data undergoes the
             same basis expansion process as the training data. This is more
             computationally efficient and seems to improve the performance."""

    # Fixed basis + Functional Linear Regression
    regressors.append(("flin_basis",
                       Pipeline([
                           ("basis", Basis()),
                           ("reg", FLinearRegression())]),
                       params_basis
                       ))

    """
    TARDA BASTANTE (cálculo de Gram matrix costoso en la base)

    # FPCA basis + Functional Linear Regression
    regressors.append(("flin_khl",
                       Pipeline([
                           ("basis", FPCABasis()),
                           ("reg", FLinearRegression())]),
                       params_basis_fpca
                       ))
    """

    """
    TARDA BASTANTE (cálculo de Gram matrix costoso en la base)

    # FPLS basis + Functional Linear Regression
    regressors.append(("flin_fpls",
                       Pipeline([
                           ("basis", Basis()),
                           ("reg", FLinearRegression())]),
                       params_basis_fpls
                       ))
    """

    # Fixed basis + FPLS1 regression
    regressors.append(("fpls1_basis",
                       Pipeline([
                           ("basis", Basis()),
                           ("reg", FPLS())]),
                       {**params_basis, **params_pls}
                       ))

    # KNeighbors Functional Regression
    regressors.append(("fknn",
                       Pipeline([
                           ("reg", KNeighborsRegressor())]),
                       params_knn
                       ))

    return regressors


def logistic_regression_comparison_suite(
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
    random_state=None
):
    classifiers = []

    """
    MULTIVARIATE MODELS
    """

    """
    DATA SHOULD NOT BE CENTERED
    # Optimal bayes rule in the discrete case (QDA)
    classifiers.append(("optimal_bayes",
                       Pipeline([
                           ("clf", NaiveGPClassifier(np.mean(Y == 1)))]),
                       {}
                        ))
    """

    # LDA (based on FPCA+Ridge regression)
    classifiers.append(("lda_fpca+ridge",
                       Pipeline([
                           ("dim_red", FPCA()),
                           ("clf", LDA())]),
                       {**params_dim_red, **params_base_regressors_ridge}
                        ))

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # LDA (based on FPLS (fixed basis)+Ridge regression)
    classifiers.append(("lda_fpls_basis+ridge",
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
    classifiers.append(("lda_lasso",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("clf", LDA())]),
                       params_base_regressors_lasso
                        ))

    # LDA (based on PLS1 regression)
    classifiers.append(("lda_pls1",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("clf", LDA())]),
                       params_base_regressors_pls
                        ))

    # LDA (based on Manual+Ridge regression)
    classifiers.append(("lda_manual+ridge",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("selector", FeatureSelector()),
                           ("clf", LDA())]),
                       {**params_select, **params_base_regressors_ridge}
                        ))

    # LDA (based on PCA+Ridge regression)
    classifiers.append(("lda_pca+ridge",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PCA(random_state=random_state)),
                           ("clf", LDA())]),
                       {**params_dim_red, **params_base_regressors_ridge}
                        ))

    # LDA (based on PLS+Ridge regression)
    classifiers.append(("lda_pls+ridge",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PLSRegressionWrapper()),
                           ("clf", LDA())]),
                       {**params_dim_red, **params_base_regressors_ridge}
                        ))

    # LDA (based on RMH+Ridge regression)
    classifiers.append(("lda_rmh+ridge",
                       Pipeline([
                           ("var_sel", RMH()),
                           ("clf", LDA())]),
                       params_base_regressors_ridge
                        ))

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # LDA (based on mRMR+Ridge regression)
    classifiers.append(("lda_mRMR+ridge",
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
    classifiers.append(("manual_sel+logistic",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("selector", FeatureSelector()),
                           ("clf", LogisticRegression(
                               random_state=random_state))]),
                       {**params_clf, **params_select}
                        ))

    # FPCA+LR
    classifiers.append(("fpca+logistic",
                       Pipeline([
                           ("dim_red", FPCA()),  # Retains scores only
                           ("clf", LogisticRegression(
                               random_state=random_state))]),
                       {**params_dim_red, **params_clf}
                        ))

    # PCA+LR
    classifiers.append(("pca+logistic",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PCA(random_state=random_state)),
                           ("clf", LogisticRegression(
                               random_state=random_state))]),
                       {**params_dim_red, **params_clf}
                        ))

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # FPLS (fixed basis)+LR
    classifiers.append(("fpls_basis+logistic",
                       Pipeline([
                           ("basis", Basis()),
                           ("dim_red", FPLS()),
                           ("clf", LogisticRegression(random_state=SEED))]),
                       {**params_basis, **params_dim_red, **params_clf}
                        ))
    """

    # PLS+LR
    classifiers.append(("pls+logistic",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PLSRegressionWrapper()),
                           ("clf", LogisticRegression(
                               random_state=random_state))]),
                       {**params_dim_red, **params_clf}
                        ))

    # RKVS+LR
    classifiers.append(("rkvs+logistic",
                       Pipeline([
                           ("var_sel", RKVS()),
                           ("clf", LogisticRegression(
                               random_state=random_state))]),
                       params_var_sel
                        ))

    # RMH+LR
    classifiers.append(("rmh+logistic",
                       Pipeline([
                           ("var_sel", RMH()),
                           ("clf", LogisticRegression(
                               random_state=random_state))]),
                       {}
                        ))

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # mRMR+LR
    classifiers.append(("mRMR+logistic",
                       Pipeline([
                           ("var_sel", mRMR()),
                           ("clf", LogisticRegression(random_state=SEED))]),
                       {**params_var_sel, **params_mrmr}
                        ))
    """

    """
    # Manual+SVM Linear
    classifiers.append(("manual_sel+svm_lin",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("selector", FeatureSelector()),
                           ("clf", LinearSVC(random_state=random_state))]),
                       {**params_select, **params_clf}
                        ))

    # FPCA+SVM Linear
    classifiers.append(("fpca+svm_lin",
                       Pipeline([
                           ("dim_red", FPCA()),  # Retains scores only
                           ("clf", LinearSVC(random_state=random_state))]),
                       {**params_dim_red, **params_clf}
                        ))

    # PCA+SVM Linear
    classifiers.append(("pca+svm_lin",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PCA(random_state=random_state)),
                           ("clf", LinearSVC(random_state=random_state))]),
                       {**params_dim_red, **params_clf}
                        ))
    """

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # FPLS (fixed basis)+SVM Linear
    classifiers.append(("fpls_basis+svm_lin",
                       Pipeline([
                           ("basis", Basis()),
                           ("dim_red", FPLS()),
                           ("clf", LinearSVC(random_state=SEED))]),
                       {**params_basis, **params_dim_red, **params_clf}
                        ))
    """

    """
    # PLS+SVM Linear
    classifiers.append(("pls+svm_lin",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PLSRegressionWrapper()),
                           ("clf", LinearSVC(random_state=random_state))]),
                       {**params_dim_red, **params_clf}
                        ))

    # RKVS+SVM Linear
    classifiers.append(("rkvs+svm_lin",
                       Pipeline([
                           ("var_sel", RKVS()),
                           ("clf", LinearSVC(random_state=random_state))]),
                       {**params_var_sel, **params_clf}
                        ))

    # RMH+SVM Linear
    classifiers.append(("rmh+svm_lin",
                       Pipeline([
                           ("var_sel", RMH()),
                           ("clf", LinearSVC(random_state=random_state))]),
                       params_clf
                        ))
    """

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # mRMR+SVM Linear
    classifiers.append(("mRMR+svm_lin",
                       Pipeline([
                           ("var_sel", mRMR()),
                           ("clf", LinearSVC(random_state=SEED))]),
                       {**params_var_sel, **params_mrmr, **params_clf}
                        ))
    """

    """
    # Manual+SVM RBF
    classifiers.append(("manual_sel+svm_rbf",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("selector", FeatureSelector()),
                           ("clf", SVC(kernel='rbf'))]),
                       {**params_select, **params_clf, **params_svm}
                        ))

    # FPCA+SVM RBF
    classifiers.append(("fpca+svm_rbf",
                       Pipeline([
                           ("dim_red", FPCA()),  # Retains scores only
                           ("clf", SVC(kernel='rbf'))]),
                       {**params_dim_red, **params_clf, **params_svm}
                        ))

    # PCA+SVM RBF
    classifiers.append(("pca+svm_rbf",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PCA(random_state=random_state)),
                           ("clf", SVC(kernel='rbf'))]),
                       {**params_dim_red, **params_clf, **params_svm}
                        ))
    """

    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # FPLS (fixed basis)+SVM RBF
    classifiers.append(("fpls_basis+svm_rbf",
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

    """
    # PLS+SVM RBF
    classifiers.append(("pls+svm_rbf",
                       Pipeline([
                           ("data_matrix", DataMatrix()),
                           ("dim_red", PLSRegressionWrapper()),
                           ("clf", SVC(kernel='rbf'))]),
                       {**params_dim_red, **params_clf, **params_svm}
                        ))

    # RKVS+SVM RBF
    classifiers.append(("rkvs+svm_rbf",
                       Pipeline([
                           ("var_sel", RKVS()),
                           ("clf", SVC(kernel='rbf'))]),
                       {**params_var_sel, **params_clf, **params_svm}
                        ))

    # RMH+SVM RBF
    classifiers.append(("rmh+svm_rbf",
                       Pipeline([
                           ("var_sel", RMH()),
                           ("clf", SVC(kernel='rbf'))]),
                       {**params_clf, **params_svm}
                        ))

    """
    """
    TARDA DEMASIADO (búsqueda en CV demasiado grande?)

    # mRMR+SVM RBF
    classifiers.append(("mRMR+svm_rbf",
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
    from _logistic_regression_TEMP import LogisticRegression \
        as FLogisticRegression
    params_flr = {"clf__p": n_components}

    classifiers.append(("flr",
                       Pipeline([
                           ("clf", FLogisticRegression())]),
                       params_flr
                        ))
    """

    # Maximum Depth Classifier
    classifiers.append(("mdc",
                       Pipeline([
                           ("clf", MaximumDepthClassifier())]),
                       params_depth
                        ))

    # KNeighbors Functional Classification
    classifiers.append(("fknn",
                       Pipeline([
                           ("clf", KNeighborsClassifier())]),
                       params_knn
                        ))

    # Nearest Centroid Functional Classification
    classifiers.append(("fnc",
                       Pipeline([
                           ("clf", NearestCentroid())]),
                       {}
                        ))

    # NOTE: while not strictly necessary, the test data undergoes the
    # same basis expansion process as the training data. This is more
    # computationally efficient and seems to improve the performance.

    # Functional LDA (based on L^2-regression with fixed basis)
    classifiers.append(("flda_l2_basis",
                       Pipeline([
                           ("basis", Basis()),
                           ("clf", LDA())]),
                       params_basis
                        ))

    """
    TARDA BASTANTE (cálculo de Gram matrix costoso en la base)

    # Functional LDA (based on L^2-regression with FPCA basis)
    classifiers.append(("flda_l2_khl",
                       Pipeline([
                           ("basis", FPCABasis()),
                           ("clf", LDA())]),
                       params_basis_fpca
                        ))
    """

    """
    TARDA BASTANTE (cálculo de Gram matrix costoso en la base)

    # Functional LDA (based on L^2-regression with FPLS basis)
    classifiers.append(("flda_l2_fpls",
                       Pipeline([
                           ("basis", Basis()),
                           ("clf", LDA())]),
                       params_basis_fpls
                        ))
    """

    # Functional LDA (based on FPLS1 regression with fixed basis)
    classifiers.append(("flda_fpls1_basis",
                       Pipeline([
                           ("basis", Basis()),
                           ("clf", LDA())]),
                       {**params_basis, **params_base_regressors_fpls}
                        ))

    # Functional LDA (based on APLS regression)
    classifiers.append(("flda_apls",
                       Pipeline([
                           ("clf", LDA())]),
                       params_base_regressors_apls
                        ))

    return classifiers
