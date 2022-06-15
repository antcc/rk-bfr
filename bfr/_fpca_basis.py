# encoding: utf-8

from __future__ import annotations

import numpy as np
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from skfda.representation import FData
from skfda.representation.basis import Basis, FDataBasis
from skfda.representation.grid import FDataGrid
from sklearn.base import BaseEstimator, TransformerMixin


class KarhunenLoeve(Basis):
    r"""Defines a basis via the Karhunen-Loève expansion of a process X(t)."""

    def __init__(
        self,
        X: FData,
        *,
        n_basis: int = 2,
    ) -> None:
        """
        Construct a KarhunenLoeve object.

        Args:
            X: Functional data from which to construct the basis.
            n_basis: Number of basis functions.
        """
        self.basis = None
        self.coefficients = None
        self.X = X

        super().__init__(domain_range=X.domain_range, n_basis=n_basis)

        # Compute FPCs
        self._compute_fpcs()

    def _compute_fpcs(self):
        """See [1] for details on the implementation of FPCA.

        [1] Ramsay, J., Silverman, B. W. (2005). Discretizing the functions.
        In Functional Data Analysis (p. 161). Springer.
        """
        fpca = FPCA(n_components=self.n_basis)
        fpca.fit(self.X)
        fpca_components = fpca.components_

        if isinstance(self.X, FDataGrid):
            fpca_components = fpca.components_.data_matrix
            fpca_components = fpca_components.reshape(
                fpca_components.shape[:-1])

        self.basis = fpca_components
        self.coefficients = fpca.transform(self.X)

    def _evaluate(self, eval_points: np.ndarray) -> np.ndarray:
        eval_points = eval_points[..., 0]  # Input is scalar

        if isinstance(self.X, FDataGrid):
            eval_points_idx = np.abs(
                self.X.grid_points[0] - eval_points[:, np.newaxis]).argmin(1)

            return self.basis[:, eval_points_idx]
        else:
            return self.basis.evaluate(eval_points)


class FPCABasis(
    BaseEstimator,  # type: ignore
    TransformerMixin,  # type: ignore
):
    def __init__(self, n_basis: int = 2) -> None:
        self.n_basis = n_basis

    def fit(self, X: FData, y: None = None) -> FPCABasis:
        self.X = X
        self.expansion = KarhunenLoeve(X, n_basis=self.n_basis)

        return self

    def transform(self, X: FData, y: None = None) -> FDataBasis:
        if X.equals(self.X):
            return FDataBasis(
                self.expansion,
                self.expansion.coefficients)

        return X.to_basis(self.expansion)
