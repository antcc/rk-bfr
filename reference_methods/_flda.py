# encoding: utf-8

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from skfda._utils._utils import _classifier_get_classes
from skfda.ml.regression import LinearRegression as LinearRegressionL2
from skfda.representation import FData
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

AcceptedDataType = Union[
    FData,
    np.ndarray,
]


class FLDA(
    BaseEstimator,  # type: ignore
    ClassifierMixin,  # type: ignore
):
    r"""Linear Discriminant Analysis classifier for functional data.

    This class implements the linear discriminant analysis method
    for functional classification proposed in Preda et al. (2007).
    The predictions are obtained after a linear regression of the
    target values on the functional covariates, possibly after a
    convenient encoding of the target values.

    .. warning::
        For now, only binary classification for functional
        data is supported.

    Args:
        base_regressor:
            functional linear regressor to perform regression of $Y$ on $X_t$.
            Defaults to the standard functional $L^2$ regression model.
        encode_y:
            whether to encode the target values. Defaults to True.
        threshold:
            if it is None, use 0.0 as the score threshold for class membership.
            If it is the string 'tailored', use the mid point of the two class
            labels as threshold. Otherwise use the supplied number as the
            threshold. Defaults to None.

    Attributes:
        classes\_: A list containing the labels of the classes.
        threshold\_: Selected threshold for distinguishing the two classes.
        coef\_: A container for the coefficients of the linear regression.
        intercept\_: Independent term.

    @see Preda, C., Saporta, G., & Lévéder, C. (2007). PLS classification of
    functional data. Computational Statistics, 22(2), 223-235.
    """

    def __init__(
        self,
        base_regressor: RegressorMixin = LinearRegressionL2(),
        *,
        encode_y: bool = True,
        threshold: Optional[int] = None,
    ) -> None:
        self.base_regressor = base_regressor
        self.encode_y = encode_y
        self.threshold = threshold

    def fit(  # noqa: D102
        self,
        X: AcceptedDataType,
        y: np.ndarray,
        **kwargs,
    ) -> FLDA:
        # Check parameters
        X, classes, y_ind = self._argcheck_X_y(X, y)
        p0 = np.mean(y_ind == 0)
        p1 = 1 - p0

        # Encode target values if needed
        if self.encode_y:
            y_new = np.array(
                [-np.sqrt(p1/p0) if yy == 0 else np.sqrt(p0/p1)
                 for yy in y_ind])
        else:
            y_new = y_ind

        # Fit base regressor
        self.base_regressor.fit(X, y_new, **kwargs)

        # Save attributes
        self.coef_ = self.base_regressor.coef_
        if hasattr(self.base_regressor, "intercept_"):
            self.intercept_ = self.base_regressor.intercept_
        else:
            self.intercept_ = 0.0
        self.classes_ = classes

        # Set threshold
        if self.threshold is None:
            y_unique = np.unique(y_new)
            self.threshold_ = p0*y_unique[0] + p1*y_unique[1]
        elif isinstance(self.threshold, (int, float)):
            self.threshold_ = self.threshold
        else:
            raise ValueError(
                "Expected None or a numeric value for "
                "parameter 'threshold'.")

        return self

    def predict(  # noqa: D102
        self,
        X: AcceptedDataType,
    ) -> np.ndarray:
        check_is_fitted(self)

        scores_centered = self.decision_function(X)
        y_hat = np.array([self.classes_[0] if yy < 0.0 else self.classes_[1]
                          for yy in scores_centered])

        return y_hat

    def decision_function(  # noqa: D102
        self,
        X: AcceptedDataType,
    ) -> np.ndarray:
        r"""Predict confidence scores for samples.

         In the binary case, confidence score for `self.classes_[1]`,
         where >0 means this class would be predicted.
        """
        check_is_fitted(self)

        scores = self.base_regressor.predict(X)
        scores_centered = scores - self.threshold_

        return np.asarray(scores_centered)

    def _argcheck_X_y(  # noqa: N802
        self,
        X: AcceptedDataType,
        y: np.ndarray,
    ) -> Tuple[AcceptedDataType, np.ndarray, np.ndarray]:
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

        return X, classes, y_ind
