# encoding: utf-8

import numpy as np
from matplotlib import pyplot as plt
import arviz as az
from tqdm.notebook import tqdm
from sklearn.metrics import r2_score
from arviz.stats.stats_utils import make_ufunc
from arviz.plots.plot_utils import calculate_point_estimate
import xarray as xr


def cov_matrix(kernel_fn, s, t):
    ss, tt = np.meshgrid(s, t, indexing='ij')

    # Evaluate the kernel over meshgrid (vectorized operation)
    K = kernel_fn(ss, tt)

    return K


def generate_response(X, theta, noise=True, rng=None):
    assert len(theta) % 2 == 0

    n, N = X.shape
    grid = np.linspace(0., 1., N)
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


def generate_gp_dataset(grid, kernel_fn, n_samples, theta, rng=None):
    """Generate dataset based on GP with a given kernel function."""
    if rng is None:
        rng = np.random.default_rng()

    N = len(grid)

    mean_vector = np.zeros(N)
    kernel_matrix = cov_matrix(kernel_fn, grid, grid)

    X = rng.multivariate_normal(mean_vector, kernel_matrix, size=n_samples)
    Y = generate_response(X, theta, rng=rng)

    return X, Y


def plot_dataset(X, Y, plot_means=True):
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    n, N = X.shape
    grid = np.linspace(0., 1., N)

    axs[0].set_title(r"(Some) functional regressors $X_i(t)$")
    axs[0].set_xlabel(r"$t$")
    axs[0].plot(grid, X.T[:, :n//2], alpha=0.8)

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


def initial_guess_random(p, sd_beta, sd_alpha0, sd_log_sigma, n_walkers=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    beta_init = sd_beta*rng.standard_normal(size=(n_walkers, p))
    tau_init = rng.uniform(size=(n_walkers, p))
    alpha0_init = sd_alpha0*rng.standard_normal(size=(n_walkers, 1))
    log_sigma_init = sd_log_sigma*rng.standard_normal(size=(n_walkers, 1))

    init = np.hstack((
        beta_init,
        tau_init,
        alpha0_init,
        log_sigma_init
    ))

    return init if n_walkers > 1 else init[0]


def intial_guess_around_value(
        value, sd_beta=1, sd_tau=0.1, sd_alpha0=1,
        sd_log_sigma=0.5, n_walkers=1, rng=None):
    assert len(value) % 2 == 0

    if rng is None:
        rng = np.random.default_rng()

    p = (len(value) - 2)//2

    beta_jitter = sd_beta*rng.standard_normal(size=(n_walkers, p))
    tau_jitter = sd_tau*rng.standard_normal(size=(n_walkers, p))
    alpha0_jitter = sd_alpha0*rng.standard_normal(size=(n_walkers, 1))
    log_sigma_jitter = sd_log_sigma*rng.standard_normal(size=(n_walkers, 1))

    jitter = np.hstack((
        beta_jitter,
        tau_jitter,
        alpha0_jitter,
        log_sigma_jitter
    ))

    init_jitter = value[np.newaxis, :] + jitter

    # Restrict tau to [0, 1]
    init_jitter[:, p:2*p] = np.clip(init_jitter[:, p:2*p], 0.0, 1.0)

    return init_jitter if n_walkers > 1 else init_jitter[0]


def logdet(M):
    s = np.linalg.svd(M, compute_uv=False)
    return np.sum(np.log(np.abs(s)))


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




def emcee_to_idata(sampler, p, names, names_aux, burn, thin, ppc=False):
    var_names = names[:-1] + names_aux

    if ppc:
        idata = az.from_emcee(
            sampler,
            var_names=var_names,
            slices=[slice(0, p), slice(p, 2*p), -2, -1],
            arg_names=["y_obs"],
            blob_names=["y_rec"],
            blob_groups=["posterior_predictive"],
            dims={f"{names[0]}": ["projection"],
                  f"{names[1]}": ["projection"],
                  "y_obs": ["observed"],
                  "y_rec": ["recovered"]}
        )
    else:
        idata = az.from_emcee(
            sampler,
            var_names=var_names,
            slices=[slice(0, p), slice(p, 2*p), -2, -1],
            arg_names=["y_obs"],
            dims={f"{names[0]}": ["projection"],
                  f"{names[1]}": ["projection"],
                  "y_obs": ["observed"]},
        )

    # Burn-in and thinning
    idata = idata.sel(draw=slice(burn, None, thin))

    # Recover sigma^2
    idata.posterior[names[-1]] = np.exp(idata.posterior[names_aux[0]])**2

    return idata


# TODO: improve inner for loop and change generate_response so that it can handle
# multiple thetas and return a matrix of (n_thetas, n) (or the transpose)
def generate_ppc(idata, X, var_names, thin=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    n = X.shape[0]
    posterior_trace = idata.posterior
    n_chain = len(posterior_trace["chain"])
    n_draw = len(posterior_trace["draw"])
    ppc = np.zeros((n_chain, n_draw, n))

    for i in tqdm(range(n_chain), "Posterior predictive samples"):
        for j in range(0, n_draw, thin):
            theta_ds = posterior_trace[var_names].isel(
                chain=i, draw=j).data_vars.values()
            theta = np.concatenate([param.values.ravel()
                                    for param in theta_ds])
            ppc[i, j, :] = generate_response(X, theta, rng=rng)

    return ppc


def ppc_to_idata(ppc, idata, y_name, y_obs=None):
    n = ppc.shape[-1]
    dim_name = "recovered" if y_name == "y_rec" else "predicted"
    coords = idata.posterior[["chain", "draw"]].coords
    coords.update({dim_name: np.arange(0, n)})

    idata_ppc = az.convert_to_inference_data(
        xr.Dataset(data_vars={y_name: (("chain", "draw", dim_name), ppc)},
                   coords=coords),
        group="posterior_predictive",
    )

    if y_obs is None:
        idata_obs = az.convert_to_inference_data(
            idata.observed_data, group="observed_data")
    else:
        idata_obs = az.convert_to_inference_data(
            xr.Dataset(data_vars={"y_obs": ("observed", y_obs)},
                       coords=coords),
            group="observed_data")

    az.concat(idata_ppc, idata_obs, inplace=True)

    return idata_ppc


def plot_ppc(idata, n_samples=None, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    az.plot_ppc(idata, ax=ax, num_pp_samples=n_samples, **kwargs)

    if "observed_data" in idata:
        ax.axvline(idata.observed_data["y_obs"].mean(), ls="--",
                   color="r", lw=2, label="Observed mean")
        ax.legend()


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
    return az.summary(data, extend=True,
                      stat_funcs={"mode": get_mode_func()}, **kwargs)


def point_predict(X, idata, names, point_estimate='mean', rng=None):
    if rng is None:
        rng = np.random.default_rng()

    posterior_trace = idata.posterior
    if point_estimate == 'mean':
        theta_ds = posterior_trace[names].mean(
            dim=("chain", "draw")).data_vars.values()
    elif point_estimate == 'mode':
        theta_ds = compute_mode(posterior_trace[names]).data_vars.values()
    elif point_estimate == 'median':
        theta_ds = posterior_trace[names].median(
            dim=("chain", "draw")).data_vars.values()
    else:
        raise ValueError(
            "'point_estimate' must be one of {mean, median, mode}.")

    theta = np.concatenate([param.values.ravel() for param in theta_ds])

    Y_hat = generate_response(X, theta, noise=False, rng=rng)

    return Y_hat


def regression_metrics(y, y_hat):
    """Quantify the goodness-of-fit of a regression model."""
    mse = np.mean((y - y_hat)**2)
    r2 = r2_score(y, y_hat)

    metrics = {
        "mse": mse,
        "r2": r2
    }
    return metrics
