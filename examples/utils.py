# encoding: utf-8

import numpy as np
from matplotlib import pyplot as plt
import arviz as az


def cov_matrix(kernel_fn, s, t):
    ss, tt = np.meshgrid(s, t, indexing='ij')

    # Evaluate the kernel over meshgrid (vectorized operation)
    K = kernel_fn(ss, tt)

    return K


def generate_gp_dataset(rng, grid, kernel_fn, n_samples, beta, tau, mu, var_error):
    """Generate dataset based on GP with a given kernel function."""
    N = len(grid)
    mean_vector = np.zeros(N)
    kernel_matrix = cov_matrix(kernel_fn, grid, grid)
    idx = np.abs(np.subtract.outer(grid, tau)).argmin(0)

    X = rng.multivariate_normal(mean_vector, kernel_matrix, size=n_samples)
    X_tau = X[:, idx]
    error = np.sqrt(var_error)*rng.standard_normal(size=n_samples)
    Y = mu*np.ones(n_samples) + X_tau@beta + error

    return X, Y


def plot_dataset(X, Y, grid, plot_means=True):
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    n = len(Y)

    axs[0].set_title(r"(Some) functional regressors $X_i(t)$")
    axs[0].set_xlabel(r"$t$")
    axs[0].plot(grid, X.T[:, :n//4], alpha=0.8)

    axs[1].set_yticks([])
    axs[1].set_title(r"Scalar values $Y_i$")
    az.plot_kde(Y, ax=axs[1])
    axs[1].plot(Y, np.zeros_like(Y), '|', color='k', alpha=0.5)

    if plot_means:
        axs[0].plot(
            grid, np.mean(X, axis=0),
            linewidth=3, color='k',
            label="Sample mean")
        axs[0].legend()

        axs[1].axvline(Y.mean(), ls="--", lw=2, color="r", label="Sample mean")
        axs[1].legend()


def plot_histogram(samples, nrows, ncols, labels, figsize=(10, 10)):
    """'samples' is an ndrray of (nchains, n_dim)."""

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


def plot_ppc(idata, n_samples, var_names, y_obs=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    az.plot_ppc(idata, ax=ax, num_pp_samples=n_samples, var_names=var_names)

    if y_obs is not None:
        ax.axvline(y_obs.mean(), ls="--", color="r", lw=2, label="Observed mean")
        ax.legend(loc="upper left")


def summary(data, names, **kwargs):
    return az.summary(data, var_names=names, extend=True, stat_funcs={
        "mode": lambda x: az.plots.plot_utils.calculate_point_estimate('mode', x)}, **kwargs)


def logdet(M):
    s = np.linalg.svd(M, compute_uv=False)
    return np.sum(np.log(np.abs(s)))
