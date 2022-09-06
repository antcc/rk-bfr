# encoding: utf-8

"""
Utility functions.
"""

import logging
import numbers
import os
import warnings

import arviz as az
import numpy as np
import xarray as xr
from scipy.stats import mode
from skfda.representation.basis import FDataBasis
from skfda.representation.grid import FDataGrid


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
    """Convert FData object to numpy array."""
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
    """Convert probabilities to class labels."""
    y_th = np.copy(y).astype(int)
    y_th[..., y >= th] = 1
    y_th[..., y < th] = 0

    return y_th


def relabel_sample(sample, theta_space, order_by_beta):
    """Relabel sample by imposing an ordering constraint."""
    _, beta, tau, _, _ = theta_space.slice_params(sample, clip=False)
    arr = beta if order_by_beta else tau

    sorted_idx = np.argsort(arr, axis=-1)
    sample[..., theta_space.beta_idx] = np.take_along_axis(
        beta, sorted_idx, axis=-1)
    sample[..., theta_space.tau_idx] = np.take_along_axis(
        tau, sorted_idx, axis=-1)


def pp_to_idata(pps, idata, var_names, y_obs=None, merge=False):
    """Convert posterior predictive arrays to InferenceData.

    All the pp arrays must have the same shape (the shape of y_obs).
    """
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
    """Compute the mode of the traces.

    Note that NaN values are always ignored.
    """
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
    """Compute the mode of a xarray object."""
    def mode_fn_args(x):
        return mode_fn(x, skipna=skipna, bw=bw)

    return xr.apply_ufunc(
        az.make_ufunc(mode_fn_args), data,
        input_core_dims=(dim,)
    )
