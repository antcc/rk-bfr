# encoding: utf-8

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.integrate import simpson
from scipy.linalg import sqrtm
from skfda.representation import FData
from skfda.representation.basis import Basis, FDataBasis
from skfda.representation.grid import FDataGrid
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted


##
# HELPER FUNCTIONS
##

def _cov_operator(f, K, grid):
    return simpson(K*f, x=grid)


def _scalar_product(f, g, K, grid):
    inner_integral = _cov_operator(f, K, grid)
    return simpson(g*inner_integral, x=grid)


def _modified_gram_schmidt(fs, K, grid):
    basis = []

    for j, f in enumerate(fs):
        g = f
        for i in range(j):
            g_i = basis[i]
            g = g - _scalar_product(g, g_i, K, grid)*g_i

        norm_sq = _scalar_product(g, g, K, grid)
        if norm_sq > 0:
            basis.append(g/np.sqrt(norm_sq))
        else:
            raise ValueError("Invalid input matrix")

    return np.array(basis)


##
# FPLS BASIS
##

class FPLSBasis(Basis):
    r"""Defines a basis via the FPLS expansion of a process X(t)."""

    def __init__(
        self,
        X: FData,
        y: np.ndarray,
        *,
        n_basis: int = 2,
    ) -> None:
        """
        Construct a FPLSBasis object.

        Args:
            X: Functional data from which to construct the basis.
            y: Scalar response for each functional predictor.
            n_basis: Number of basis functions.
        """
        self.basis = None
        self.coefficients = None
        self.X = X

        super().__init__(domain_range=X.domain_range, n_basis=n_basis)

        # Compute basis
        fpls = APLS(n_components=self.n_basis)
        fpls.fit(self.X, y)
        self.basis = fpls.basis_

    def _evaluate(self, eval_points: np.ndarray) -> np.ndarray:
        eval_points = eval_points[..., 0]  # Input is scalar

        if isinstance(self.X, FDataGrid):
            grid = self.X.grid_points[0]
        else:
            grid = np.linspace(1./100, 1, 100)

        eval_points_idx = np.abs(
            grid - eval_points[:, np.newaxis]).argmin(1)

        return self.basis[:, eval_points_idx]


##
# FPLS ESTIMATORS
##


class APLS(
    BaseEstimator,     # type: ignore
    RegressorMixin,    # type: ignore
    TransformerMixin   # type: ignore
):
    """Implements the APLS algorithm proposed by Delaigle and Hall [1].

    [1] Delaigle, A., & Hall, P. (2012). Methodology and theory for partial
    least squares applied to functional data. The Annals of Statistics,
    40(1), 322-352."""

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components

    def fit(
        self,
        X: FData,
        y: np.ndarray,
        eval_points: Optional[np.ndarray] = None
    ) -> APLS:
        X = self._argcheck_X(X, eval_points)

        # Get mean information of data
        self.X_mean_ = X.mean(axis=0)
        self.y_mean_ = y.mean()

        # Get centered data
        data, self.grid_ = self._extract_centered_data(X)
        y = y - self.y_mean_

        # Estimate K(s, t)
        K = X.cov().data_matrix[0, ..., 0]

        # Estimate K^i(b) for i=1,...,p
        kb = self._fpls_span(data, y)
        Kb_iterations = [kb]
        for i in range(1, self.n_components):
            Kb_next = _cov_operator(Kb_iterations[i - 1], K, self.grid_)
            Kb_iterations.append(Kb_next)

        # Orthonormalize {K^i(b)} to obtain an FPLS basis
        self.basis_ = _modified_gram_schmidt(Kb_iterations, K, self.grid_)

        # Compute the coordinates of each sample with respect to the FPLS basis
        coord_matrix = self._compute_coord_matrix(data, self.grid_)

        # Obtain coefficients by solving a least squares problem
        self.coef_ = np.linalg.lstsq(coord_matrix, y, rcond=None)[0]

        return self

    def transform(
        self,
        X: FData,
        eval_points: Optional[np.ndarray] = None
    ) -> np.ndarray:
        check_is_fitted(self)
        X = self._argcheck_X(X, eval_points)

        data, grid = self._extract_centered_data(X)
        coord_matrix = self._compute_coord_matrix(data, grid)

        return coord_matrix

    def predict(
        self,
        X: FData,
        eval_points: Optional[np.ndarray] = None
    ) -> np.ndarray:
        check_is_fitted(self)
        X = self._argcheck_X(X, eval_points)

        data, grid = self._extract_centered_data(X)
        coord_matrix = self._compute_coord_matrix(data, grid)

        return self.y_mean_ + coord_matrix@self.coef_

    def _extract_centered_data(self, X, eval_points=None):
        # Center X and evaluate on a grid if needed
        X = X - self.X_mean_

        # Retain only the data matrix
        data = X.data_matrix[..., 0]
        grid = X.grid_points[0]

        return data, grid

    def _compute_coord_matrix(self, data: np.ndarray, grid: np.ndarray):
        return simpson(data[:, None]*self.basis_[None, :], x=grid)

    def _fpls_span(self, X, y):
        # X is assumed to be centered
        return X.T@(y - y.mean())/(X.shape[0] - 1)

    def _argcheck_X(self, X, eval_points):
        if isinstance(X, FDataBasis):
            X = X.to_grid(eval_points)

        if (hasattr(self, "grid_")
                and not np.array_equal(X.grid_points[0], self.grid_)):
            raise ValueError("Grid must coincide for training and test data.")

        return X


class FPLS(
    BaseEstimator,     # type: ignore
    TransformerMixin,  # type: ignore
    RegressorMixin,    # type: ignore
):
    """Functional PLS regressor and transformer.

    The algorithm has been implemented following the reference [1].
    The training data needs to be expressed as a finite basis expansion.

    [1] Aguilera, A. M., Escabias, M., Preda, C., & Saporta, G. (2010).
    Using basis expansions for estimating functional PLS regression:
    applications with chemometric data. Chemometrics and Intelligent
    Laboratory Systems, 104(2), 289-305.
    """

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components

    def _transform_data(self, X: FDataBasis) -> np.ndarray:
        A = X.coefficients
        Phi = sqrtm(X.basis.gram_matrix())

        return A@Phi

    def fit(self, X: FDataBasis, y: np.ndarray) -> FPLS:
        K = X.coefficients.shape[1]
        if self.n_components > K:
            raise AttributeError(
                f"n_components cannot be higher that the number of "
                f"basis elements ({self.n_components} > {K}).")

        self.X_ = X
        self.n_targets = 1 if len(y.shape) == 1 else y.shape[1]
        self.transformed_matrix_ = self._transform_data(X)

        self.base_regressor = PLSRegression(n_components=self.n_components)
        self.base_regressor.fit(self.transformed_matrix_, y)
        self.coef_ = self.base_regressor.coef_

        return self

    def transform(self, X: FDataBasis) -> np.ndarray:
        check_is_fitted(self)

        if X.equals(self.X_):
            transformed_matrix = self.transformed_matrix_
        else:
            transformed_matrix = self._transform_data(X)

        return self.base_regressor.transform(transformed_matrix)

    def predict(self, X: FDataBasis) -> np.ndarray:
        check_is_fitted(self)

        if X.equals(self.X_):
            transformed_matrix = self.transformed_matrix_
        else:
            transformed_matrix = self._transform_data(X)

        preds = self.base_regressor.predict(transformed_matrix)

        if self.n_targets == 1:
            return preds[:, 0]
        else:
            return preds
