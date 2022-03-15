# encoding: utf-8

import numpy as np
import os
from matplotlib import pyplot as plt
import arviz as az
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm
import numbers
from sklearn.metrics import r2_score
from arviz.stats.stats_utils import make_ufunc
from scipy.special import expit
from arviz.plots.plot_utils import calculate_point_estimate
import xarray as xr
from scipy.integrate import trapz
import logging


class HandleLogger():
    def __init__(self, verbose: int = 0):
        self.verbose = verbose

    def __enter__(self):
        if self.verbose < 2:
            logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


# Custom transformations

class Identity():

    def forward(self, x):
        return x

    def backward(self, y):
        return y


class Logit():
    eps = 1e-6

    def forward(self, x):
        return np.log(x/(1 - x + self.eps) + self.eps)

    def backward(self, y):
        return expit(y)   # 1./(1 + np.exp(-y))


class LogSq():
    eps = 1e-6

    def forward(self, x):
        return np.log(np.sqrt(x) + self.eps)

    def backward(self, y):
        return np.exp(y)**2


def check_random_state(seed):
    """Turn seed into a np.random.Generator instance.

    For compatibility with sklearn, the case in which the
    seed is a np.random.RandomState is also considered.

    Parameters
    ----------
    seed : None, int, instance of np.random.RandomState or instance of Generator.
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


# Custom context manager for handling warnings

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


def brownian_kernel(s, t, sigma=1.0):
    return sigma*np.minimum(s, t)


def fractional_brownian_kernel(s, t, H=0.8):
    return 0.5*(s**(2*H) + t**(2*H) - np.abs(s - t)**(2*H))


def ornstein_uhlenbeck_kernel(s, t, v=1.0):
    return np.exp(-np.abs(s - t)/v)


def squared_exponential_kernel(s, t, v=0.2):
    return np.exp(-np.abs(s - t)**2/(2*v**2))


def grollemund_smooth(t):
    return (5*np.exp(-20*(t-0.25)**2)
            - 2*np.exp(-20*(t-0.5)**2)
            + 2*np.exp(-20*(t-0.75)**2))


def cholaquidis_scenario3(t):
    return np.log(1 + 4*t)


def cov_matrix(kernel_fn, s, t):
    ss, tt = np.meshgrid(s, t, indexing='ij')

    # Evaluate the kernel over meshgrid (vectorized operation)
    K = kernel_fn(ss, tt)

    return K


def generate_response(X, theta, noise=True, rng=None):
    """Generate a response when the parameter vector 'theta' is (β, τ, α0, σ2)."""
    assert len(theta) % 2 == 0

    n, N = X.shape
    grid = np.linspace(1./N, 1., N)
    p = (len(theta) - 2)//2

    beta = theta[:p]
    tau = theta[p:2*p]
    alpha0 = theta[-2]
    sigma = np.sqrt(theta[-1])

    idx = np.abs(grid - tau[:, np.newaxis]).argmin(1)
    X_tau = X[:, idx]
    Y = alpha0 + X_tau@beta

    if noise:
        if rng is None:
            rng = np.random.default_rng()
        Y += sigma*rng.standard_normal(size=n)

    return Y


def threshold_lin(x):
    """sigmoid(x) >= 0.5 iff x >= 0"""
    return 1 if x >= 0 else 0


def threshold(x):
    """y_hat == 1 iff sigmoid(x) >= 0.5"""
    return 1 if x >= 0.5 else 0


def generate_response_logistic(X, theta, prob=True, return_p=False, rng=None):
    """Generate a response when the parameter vector 'theta' is (β, τ, α0, σ2).
       Return the response vector and (possibly) the probabilities associated."""
    assert len(theta) % 2 == 0

    Y_lin = generate_response(X, theta, noise=False)

    if prob:
        Y = transform_linear_response(Y_lin, rng=rng)
    else:
        Y = [threshold_lin(y) for y in Y_lin]

    if return_p:
        return Y, expit(Y_lin)
    else:
        return Y


def gp(grid, mean_vector, kernel_fn, n_samples, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    kernel_matrix = cov_matrix(kernel_fn, grid, grid)

    X = rng.multivariate_normal(mean_vector, kernel_matrix, size=n_samples)
    X = X - X.mean(axis=0)  # Center data
    return X


def generate_gp_l2_dataset(
        grid, kernel_fn, n_samples,
        beta_coef, alpha0, sigma2, rng=None,
        mean_vector=None
):
    """Generate dataset based on GP with a given kernel function."""
    if rng is None:
        rng = np.random.default_rng()

    if mean_vector is None:
        mean_vector = np.zeros(len(grid))

    beta = beta_coef(grid)

    X = gp(grid, mean_vector, kernel_fn, n_samples, rng)
    Y = alpha0 + trapz(y=X*beta, x=grid)

    if sigma2 > 0.0:
        Y += np.sqrt(sigma2)*rng.standard_normal(size=n_samples)

    return X, Y


def generate_gp_rkhs_dataset(
        grid, kernel_fn, n_samples,
        beta, tau, alpha0, sigma2, rng=None,
        mean_vector=None
):
    """Generate dataset based on GP with a given kernel function."""
    if mean_vector is None:
        mean_vector = np.zeros(len(grid))

    X = gp(grid, mean_vector, kernel_fn, n_samples, rng)

    theta_true = np.concatenate((
        beta, tau,
        [alpha0], [sigma2]
    ))

    add_noise = sigma2 > 0.0
    Y = generate_response(X, theta_true, noise=add_noise, rng=rng)

    return X, Y


def generate_classification_dataset(
        grid, kernel_fn, kernel_fn2,
        n_samples, rng=None, mean_vector=None,
        mean_vector2=None
):
    """Generate dataset based on a known distribution on X|Y."""
    if mean_vector is None:
        mean_vector = np.zeros(len(grid))
    if mean_vector2 is None:
        mean_vector2 = np.zeros(len(grid))

    X1 = gp(grid, mean_vector, kernel_fn, n_samples//2, rng)
    X2 = gp(grid, mean_vector2, kernel_fn2, n_samples - n_samples//2, rng)

    X = np.vstack((X1, X2))
    Y = (n_samples//2 <= np.arange(n_samples)).astype(int)

    return X, Y


def transform_linear_response(Y_lin, noise=None, rng=None):
    """Convert probabilities into class labels with a possible random
       noise."""
    if rng is None:
        rng = np.random.default_rng()

    Y = rng.binomial(1, expit(Y_lin))

    if noise is not None:
        n_permute = int(len(Y)*noise)

        idx_0 = rng.choice(np.where(Y == 0)[0], size=n_permute)
        idx_1 = rng.choice(np.where(Y == 1)[0], size=n_permute)

        Y[idx_0] = 1
        Y[idx_1] = 0

    return Y


def plot_dataset(X, Y, plot_means=True, n_samples=None, figsize=(9, 4)):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    n, N = X.shape
    grid = np.linspace(1./N, 1., N)

    if n_samples is None:
        n_samples = n

    axs[0].set_title(r"Functional regressors $X_i(t)$")
    axs[0].set_xlabel(r"$t$")
    axs[0].plot(grid, X.T[:, :n_samples], alpha=0.8)

    axs[1].set_yticks([])
    axs[1].set_title(r"Scalar values $Y_i$")
    az.plot_dist(Y, ax=axs[1])
    axs[1].plot(Y, np.zeros_like(Y), '|', color='k', alpha=0.5)

    if plot_means:
        axs[0].plot(
            grid, np.mean(X, axis=0),
            linewidth=3, color='k',
            label="Sample mean")
        axs[0].legend()

        axs[1].axvline(Y.mean(), ls="--", lw=2, color="r", label="Sample mean")
        axs[1].legend()


def plot_dataset_classification(
    X,
    Y,
    plot_means=True,
    n_samples=None,
    figsize=(5, 4),
    ax=None
):
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    n, N = X.shape
    grid = np.linspace(1./N, 1., N)
    n_samples_0 = (Y == 0).sum()
    n_samples_1 = (Y == 1).sum()

    if n_samples is None:
        plot_n0 = n_samples_0
        plot_n1 = n_samples_1
    else:
        plot_n0 = int(n_samples_0*n_samples/len(Y))
        plot_n1 = int(n_samples_1*n_samples/len(Y))

    axs[0].set_title(r"Labeled functional regressors $X_i(t)$")
    axs[0].set_xlabel(r"$t$")
    if plot_n0 > 0:
        axs[0].plot(grid, X.T[:, Y == 0][:, :plot_n0], alpha=0.5, color="blue",
                    label=["Class 0"] + [""]*(plot_n0 - 1))
    if plot_n1 > 0:
        axs[0].plot(grid, X.T[:, Y == 1][:, :plot_n1], alpha=0.5, color="red",
                    label=["Class 1"] + [""]*(plot_n1 - 1))

    if plot_means:
        axs[0].plot(
            grid, np.mean(X, axis=0),
            linewidth=3, color='k',
            label="Sample mean")

    axs[0].legend(fontsize=8)

    axs[1].set_title("Class distribution")
    axs[1].set_xlabel("Class")
    axs[1].set_xticks([0, 1])
    counts = [n_samples_0, n_samples_1]
    freq = counts/np.sum(counts)
    if counts[0] > 0:
        axs[1].bar(0, freq[0], color="blue",
                   label="Class 0", width=.3)
    if counts[1] > 0:
        axs[1].bar(1, freq[1], color="red",
                   label="Class 1", width=.3)
    axs[1].legend()


def initial_guess_random(
        theta_space, sd_beta, mean_alpha0, sd_alpha0,
        param_sigma2, n_walkers=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    p = theta_space.p

    beta_init = sd_beta*rng.standard_normal(size=(n_walkers, p))
    tau_init = rng.uniform(size=(n_walkers, p))
    alpha0_init = mean_alpha0 + sd_alpha0 * \
        rng.standard_normal(size=(n_walkers, 1))
    sigma2_init = 1./rng.standard_gamma(param_sigma2, size=(n_walkers, 1))

    init = np.hstack((
        beta_init,
        tau_init,
        alpha0_init,
        sigma2_init
    ))

    init_tr = theta_space.forward(init)

    return init_tr if n_walkers > 1 else init_tr[0]


def initial_guess_around_value(
        theta_space, value_tr, sd_beta, sd_alpha0, sd_sigma2,
        sd_tau, n_walkers=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    p = theta_space.p
    assert len(value_tr) == theta_space.ndim

    beta_jitter = sd_beta*rng.standard_normal(size=(n_walkers, p))
    tau_jitter = sd_tau*rng.standard_normal(size=(n_walkers, p))
    alpha0_jitter = sd_alpha0*rng.standard_normal(size=(n_walkers, 1))
    sigma2_jitter = sd_sigma2*rng.standard_normal(size=(n_walkers, 1))

    jitter = np.hstack((
        beta_jitter,
        tau_jitter,
        alpha0_jitter,
        sigma2_jitter
    ))

    value = theta_space.backward(value_tr)
    value_jitter = theta_space.clip_bounds(value[np.newaxis, :] + jitter)
    value_jitter_tr = theta_space.forward(value_jitter)

    return value_jitter_tr if n_walkers > 1 else value_jitter_tr[0]


def weighted_initial_guess_around_value(
        theta_space, value_tr, sd_beta, sd_tau,
        mean_alpha0, sd_alpha0, param_sigma2, sd_sigma2,
        n_walkers=1, frac_random=0.5, rng=None, sd_beta_random=None):

    if sd_beta_random is None:
        sd_beta_random = sd_beta

    n_random = int(frac_random*n_walkers)
    n_around = n_walkers - n_random

    if n_random > 0:
        init_1 = initial_guess_random(
            theta_space, sd_beta_random, mean_alpha0,
            sd_alpha0, param_sigma2,
            n_walkers=n_random, rng=rng)
    else:
        init_1 = None

    if n_around > 0:
        init_2 = initial_guess_around_value(
            theta_space, value_tr,
            sd_beta, sd_alpha0,
            sd_sigma2, sd_tau,
            n_walkers=n_around,
            rng=rng
        )
    else:
        init_2 = None

    if init_1 is None:
        init = init_2
    elif init_2 is None:
        init = init_1
    else:
        init = np.vstack((init_1, init_2))

    rng.shuffle(init)

    return init


def logdet(M):
    s = np.linalg.svd(M, compute_uv=False)
    return np.sum(np.log(np.abs(s)))


def get_trace_emcee(sampler, theta_space, burn, thin, flat=False):
    trace = np.copy(sampler.get_chain(discard=burn, thin=thin))
    trace[:, :, theta_space.tau_idx] = theta_space.tau_ttr.backward(
        trace[:, :, theta_space.tau_idx])
    trace[:, :, theta_space.sigma2_idx] = theta_space.sigma2_ttr.backward(
        trace[:, :, theta_space.sigma2_idx])

    if flat:
        trace = trace.reshape(-1, trace.shape[-1])  # All chains combined

    return trace


def plot_histogram(samples, nrows, ncols, labels, figsize=(10, 10)):
    """Plot histogram of 'samples', which is an ndarray of (nchains, n_dim)."""
    n_dim = samples.shape[-1]
    fig = plt.figure(figsize=figsize)

    plt.suptitle(
        r"Histogram of parameters using samples from all chains", y=1.05,
        fontsize=15)

    for i in range(n_dim):
        plt.subplot(nrows, ncols, i + 1)
        nn, bins, _ = plt.hist(
            samples[:, i],
            bins=100,
            density=True)
        max_bin = nn.argmax()
        mode = bins[max_bin:max_bin + 2].mean()
        plt.xlabel(labels[i])
        plt.axvline(mode, label=f"Mode: {mode:.3f}",
                    color="red", lw=2, alpha=0.8)  # mode
        plt.legend()
        plt.ylabel(r"$\pi$(" + labels[i] + ")")
        plt.yticks([])


def plot_evolution(trace, labels):
    n_dim = trace.shape[-1]
    fig, axes = plt.subplots(n_dim, figsize=(5, 12), sharex=True)

    for i in range(n_dim):
        ax = axes[i]
        ax.plot(trace[:, :, i], "k", alpha=0.3)
        ax.set_ylabel(labels[i])

    axes[0].set_title("Evolution of parameters for all chains")
    axes[-1].set_xticks([])
    axes[-1].set_xlabel("step")


def emcee_to_idata(
    sampler,
    theta_space,
    burn,
    thin,
    pp_names=[],
    is_blob_ll=False
):
    names = theta_space.names
    names_ttr = theta_space.names_ttr
    p = theta_space.p
    n_pp = len(pp_names)
    blob_names = []
    blob_groups = []
    dims = {f"{names_ttr[0]}": ["vector"],
            f"{names_ttr[1]}": ["vector"],
            "y_obs": ["observation"]}

    if n_pp > 0:
        new_vars = {}
        for name in pp_names:
            new_vars[name] = ["prediction"]

        blob_names = pp_names
        blob_groups = n_pp*["posterior_predictive"]
        dims = {**dims, **new_vars}
    if is_blob_ll:
        blob_names += ["y_obs"]
        blob_groups += ["log_likelihood"]

    if len(blob_names) == 0:  # No blobs
        blob_names = None
        blob_groups = None

    idata = az.from_emcee(
        sampler,
        var_names=names_ttr,
        slices=[slice(0, p), slice(p, 2*p), -2, -1],
        arg_names=["y_obs"],
        blob_names=blob_names,
        blob_groups=blob_groups,
        dims=dims
    )

    # Burn-in and thinning
    idata = idata.sel(draw=slice(burn, None, thin))

    idata.posterior[names[1]] = \
        theta_space.tau_ttr.backward(idata.posterior[names_ttr[1]])
    idata.posterior[names[-1]] = \
        theta_space.sigma2_ttr.backward(idata.posterior[names_ttr[-1]])

    return idata


# TODO: improve inner for loop and change generate_response so that it can handle
# multiple thetas and return a matrix of (n_thetas, n) (or the transpose)
def generate_pp(
        idata, X, var_names, thin=1, rng=None,
        kind='regression', progress='notebook'):
    if rng is None:
        rng = np.random.default_rng()

    n = X.shape[0]
    posterior_trace = idata.posterior
    n_chain = len(posterior_trace["chain"])
    n_draw = len(posterior_trace["draw"])
    range_draws = range(0, n_draw, thin)
    pp_y = np.zeros((n_chain, len(range_draws), n))

    if kind == 'classification':
        pp_p = np.zeros((n_chain, len(range_draws), n))

    if progress is True:
        chain_range = tqdm(range(n_chain), "Posterior predictive samples")
    elif progress == 'notebook':
        chain_range = tqdm_notebook(
            range(n_chain), "Posterior predictive samples")
    else:
        chain_range = range(n_chain)

    for i in chain_range:
        for j, jj in enumerate(range_draws):
            theta_ds = posterior_trace[var_names].isel(
                chain=i, draw=jj).data_vars.values()
            theta = np.concatenate([param.values.ravel()
                                    for param in theta_ds])

            if kind == 'classification':
                Y_star, p_star = generate_response_logistic(
                    X, theta, return_p=True, rng=rng)
                pp_p[i, j, :] = p_star
            else:
                Y_star = generate_response(X, theta, rng=rng)

            pp_y[i, j, :] = Y_star

    if kind == 'classification':
        return pp_p, pp_y.astype(int)
    else:
        return pp_y


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


def plot_ppc(idata, n_samples=None, ax=None, legend=False, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    az.plot_ppc(idata, ax=ax, num_pp_samples=n_samples, **kwargs)

    if legend and "observed_data" in idata:
        ax.axvline(idata.observed_data["y_obs"].mean(), ls="--",
                   color="r", lw=2, label="Observed mean")
        ax.legend()


def bpv(pp, Y, t_stat):
    """Compute bayesian p-values for a given statistic.
       - t_stat is a vectorized function that accepts an 'axis' parameter.
       - pp is an ndarray of shape (..., len(Y)) representing the
         posterior samples."""
    pp_flat = pp.reshape(-1, len(Y))
    t_stat_pp = t_stat(pp_flat, axis=-1)
    t_stat_observed = t_stat(Y)

    return np.mean(t_stat_pp <= t_stat_observed)


def get_mode_func():
    def mode(x):
        return calculate_point_estimate('mode', x)

    return mode


def compute_mode(data):
    mode_func = get_mode_func()
    return xr.apply_ufunc(
        make_ufunc(mode_func), data,
        input_core_dims=(("chain", "draw"),))


def summary(data, **kwargs):
    additional_stats = {
        "mode": get_mode_func(),
        "median": np.median
    }

    return az.summary(data, extend=True,
                      stat_funcs=additional_stats,
                      **kwargs)


def point_estimate(idata, pe, names):
    posterior_trace = idata.posterior
    if pe == 'mean':
        theta_ds = posterior_trace[names].mean(
            dim=("chain", "draw")).data_vars.values()
    elif pe == 'mode':
        theta_ds = compute_mode(posterior_trace[names]).data_vars.values()
    elif pe == 'median':
        theta_ds = posterior_trace[names].median(
            dim=("chain", "draw")).data_vars.values()
    else:
        raise ValueError(
            "'pe' must be one of {mean, median, mode}.")

    theta = np.concatenate([param.values.ravel() for param in theta_ds])

    return theta


def point_predict(X, idata, names, pe='mean', kind='regression'):
    theta = point_estimate(idata, pe, names)
    if kind == 'regression':
        Y_hat = generate_response(X, theta, noise=False)
    else:
        Y_hat = generate_response_logistic(X, theta, prob=False)

    return Y_hat


def regression_metrics(Y, Y_hat):
    """Quantify the goodness-of-fit of a regression model."""
    mse = np.mean((Y - Y_hat)**2)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y, Y_hat)

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }

    return metrics


def classification_metrics(Y, Y_hat):
    """Quantify the goodness-of-fit of a classification model."""
    acc = np.mean(Y == Y_hat)

    metrics = {
        "acc": acc
    }

    return metrics
