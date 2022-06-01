# encoding: utf-8

import arviz as az
import matplotlib.pyplot as plt
import numpy as np


def plot_dataset_regression(
    X,
    y,
    plot_means=True,
    n_samples=None,
    figsize=(9, 4)
):
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
    az.plot_dist(y, ax=axs[1])
    axs[1].plot(y, np.zeros_like(y), '|', color='k', alpha=0.5)

    if plot_means:
        axs[0].plot(
            grid, np.mean(X, axis=0),
            linewidth=3, color='k',
            label="Sample mean")
        axs[0].legend()

        axs[1].axvline(y.mean(), ls="--", lw=2, color="r", label="Sample mean")
        axs[1].legend()


def plot_dataset_classification(
    X,
    y,
    plot_means=True,
    n_samples=None,
    figsize=(5, 4),
    ax=None
):
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    n, N = X.shape
    grid = np.linspace(1./N, 1., N)
    n_samples_0 = (y == 0).sum()
    n_samples_1 = (y == 1).sum()

    if n_samples is None:
        plot_n0 = n_samples_0
        plot_n1 = n_samples_1
    else:
        plot_n0 = int(n_samples_0*n_samples/len(y))
        plot_n1 = int(n_samples_1*n_samples/len(y))

    axs[0].set_title(r"Labeled functional regressors $X_i(t)$")
    axs[0].set_xlabel(r"$t$")
    if plot_n0 > 0:
        axs[0].plot(grid, X.T[:, y == 0][:, :plot_n0], alpha=0.5, color="blue",
                    label=["Class 0"] + [""]*(plot_n0 - 1))
    if plot_n1 > 0:
        axs[0].plot(grid, X.T[:, y == 1][:, :plot_n1], alpha=0.5, color="red",
                    label=["Class 1"] + [""]*(plot_n1 - 1))

    if plot_means:
        axs[0].plot(
            grid, np.mean(X, axis=0),
            linewidth=3, color='k',
            label="Sample mean")

    axs[0].legend(fontsize=10)

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


def plot_autocorr(idata, theta_space, gridsize):
    az.plot_autocorr(
        idata.posterior.dropna(theta_space.dim_name, how="all").fillna(0.0),
        combined=True,
        var_names=theta_space.names,
        grid=gridsize,
        labeller=theta_space.labeller
    )


def plot_posterior(idata, theta_space, gridsize, textsize, pe='mode'):
    az.plot_posterior(
        idata.posterior.dropna(theta_space.dim_name, how="all"),
        labeller=theta_space.labeller,
        point_estimate=pe,
        grid=gridsize,
        textsize=textsize,
        skipna=theta_space.include_p,
        var_names=theta_space.names
    )


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


def plot_evolution(trace, names):
    n_dim = trace.shape[-1]
    fig, axes = plt.subplots(n_dim, figsize=(5, 12), sharex=True)

    for i in range(n_dim):
        ax = axes[i]
        ax.plot(trace[:, :, i], "k", alpha=0.3)
        ax.set_ylabel(names[i])

    axes[0].set_title("Evolution of parameters for all chains")
    axes[-1].set_xticks([])
    axes[-1].set_xlabel("step")


def plot_ppc(idata, n_samples=None, ax=None, legend=False, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    az.plot_ppc(idata, ax=ax, num_pp_samples=n_samples, **kwargs)

    if legend and "observed_data" in idata:
        ax.axvline(idata.observed_data["y_obs"].mean(), ls="--",
                   color="r", lw=2, label="Observed mean")
        ax.legend()
