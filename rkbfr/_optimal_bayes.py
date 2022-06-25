# encoding: utf-8

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from skfda._utils._utils import _classifier_get_classes
from skfda.representation import FData
from skfda.representation.basis import FDataBasis
from skfda.representation.grid import FDataGrid
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

AcceptedDataType = Union[
    FData,
    np.ndarray,
]


class NaiveGPClassifier(BaseEstimator, ClassifierMixin):
    """
    The parameter 'p' controls the a priori probability
    of the positive class.
    """

    def __init__(
        self,
        p: int = 0.5,
    ) -> None:
        self.p = p

    def fit(
        self,
        X: AcceptedDataType,
        y: np.ndarray
    ) -> NaiveGPClassifier:
        """
        In the case of finite monitorization times, the class-
        conditional distributions are multivariate normals,
        and the optimal rule is the quadratic discriminant
        (Hastie et al., 2009).

        The data is NOT supposed to be centered.
        """
        self.grid_ = None
        X, classes, y_ind = self._argcheck_X_y(X, y)
        self.classes_ = classes

        X0 = X[y == classes[0]]
        X1 = X[y == classes[1]]

        # Compute sample means
        m0 = X0.mean(axis=0)
        m = X1.mean(axis=0) - m0
        X = X - m0

        X0 = X[y == classes[0]]
        X1 = X[y == classes[1]]

        # Save sample statistics
        self.m0_ = m0.data_matrix[..., 0].ravel()
        self.m_ = m.data_matrix[..., 0].ravel()
        self.K0_ = X0.cov().data_matrix[0, ..., 0]
        self.K1_ = X1.cov().data_matrix[0, ..., 0]

        # Compute logdets using SVD decomposition
        self.logdet0_ = np.linalg.slogdet(self.K0_)[1]
        self.logdet1_ = np.linalg.slogdet(self.K1_)[1]

        # Compute the inverse covariance matrices
        N = len(self.grid_)
        self.K0_inv_ = np.linalg.solve(self.K0_, np.identity(N))
        self.K1_inv_ = np.linalg.solve(self.K1_, np.identity(N))

        return self

    def predict(self, X: AcceptedDataType) -> np.ndarray:
        scores = self.decision_function(X)
        preds = [self.classes_[1] if s > 0 else self.classes_[0]
                 for s in scores]

        return preds

    def decision_function(self, X: AcceptedDataType) -> np.ndarray:
        """A positive score means self.classes_[1] would be predicted"""
        check_is_fitted(self)
        X = self._argcheck_X(X)
        X = X.data_matrix[..., 0] - self.m0_

        return (-0.5*(self.logdet1_ - self.logdet0_
                      + (X@(self.K1_inv_ - self.K0_inv_)*X).sum(axis=1)
                      + self.m_.T@self.K1_inv_@self.m_)
                + X@self.K1_inv_@self.m_
                - np.log((1 - self.p)/self.p)
                )

    def _argcheck_X(self, X: AcceptedDataType) -> FDataGrid:
        # Convert X to FDataGrid
        if isinstance(X, np.ndarray):
            try:
                X = FDataGrid(X, grid_points=self.grid_)
            except ValueError:
                print("Data must be compatible with the grid used "
                      "for training (i.e. 'self.grid_').")
                raise
        elif isinstance(X, FDataBasis):
            X = X.to_grid(grid_points=self.grid_)
        elif isinstance(X, FDataGrid):
            if (self.grid_ is not None
                    and not np.array_equal(X.grid_points[0], self.grid_)):
                raise ValueError(
                    "Grid must be the same one used for training "
                    "(i.e. 'self.grid_').")
        else:
            raise ValueError('Data type not supported for X.')

        # Save grid
        if self.grid_ is None:
            self.grid_ = X.grid_points[0]

        return X

    def _argcheck_X_y(  # noqa: N802
        self,
        X: AcceptedDataType,
        y: np.ndarray,
    ) -> Tuple[FDataGrid, np.ndarray, np.ndarray]:
        classes, y_ind = _classifier_get_classes(y)

        if classes.size > 2:
            raise ValueError(
                f"The number of classes must be two "
                f"but got {classes.size} classes instead."
            )

        if (len(y) != len(X)):
            raise ValueError(
                "The number of samples on independent variables "
                "and classes should be the same."
            )

        X = self._argcheck_X(X)

        return X, classes, y_ind
