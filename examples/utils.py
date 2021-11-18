# encoding: utf-8

import numpy as np
from matplotlib import pyplot as plt
import arviz as az
from tqdm.notebook import tqdm
from sklearn.metrics import r2_score
from arviz.stats.stats_utils import make_ufunc
from arviz.plots.plot_utils import calculate_point_estimate
import xarray as xr
from scipy.integrate import trapz
from scipy.special import expit


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


class ThetaSpace():
    eps = 1e-6

    def __init__(
            self, p, grid, names, names_ttr, labels,
            tau_ttr=Logit(), sigma2_ttr=LogSq()):
        self.p = p
        self.grid = grid
        self.ndim = 2*p + 2
        self.names = names
        self.names_ttr = names_ttr
        self.labels = labels

        self.tau_ttr = tau_ttr
        self.sigma2_ttr = sigma2_ttr

        self.tau_lb = 0.0 + self.eps
        self.tau_ub = 1.0 - self.eps

        self.beta_idx = np.arange(0, self.p)
        self.tau_idx = np.arange(self.p, 2*self.p)
        self.alpha0_idx = -2
        self.sigma2_idx = -1

    def get_beta(self, theta):
        return theta[self.beta_idx]

    def get_tau(self, theta):
        return theta[self.tau_idx]

    def get_alpha0(self, theta):
        return theta[self.alpha0_idx]

    def get_sigma2(self, theta):
        return theta[self.sigma2_idx]

    def get_params(self, theta):
        beta = theta[self.beta_idx]
        tau = theta[self.tau_idx]
        alpha0 = theta[self.alpha0_idx]
        sigma2 = theta[self.sigma2_idx]

        return beta, tau, alpha0, sigma2

    def _perform_ttr(self, theta, ttrs):
        ndim = len(theta.shape)

        if ndim < 2:
            theta = theta[np.newaxis, :]

        tau_ttr = ttrs[0]
        sigma2_ttr = ttrs[1]

        theta_tr = np.hstack((
            theta[:, :self.p],
            tau_ttr(theta[:, self.p:2*self.p]),
            np.atleast_2d(theta[:, -2]).T,
            np.atleast_2d(sigma2_ttr(theta[:, -1])).T
        ))

        return theta_tr[0] if ndim == 1 else theta_tr

    def forward(self, theta):
        """Parameter is (β, τ, α0, σ2)."""
        theta_tr = self._perform_ttr(
            theta,
            [self.tau_ttr.forward, self.sigma2_ttr.forward])

        return theta_tr

    def backward(self, theta_tr):
        """Parameter 'theta_tr' is (β, logit τ, α0, log σ)."""
        theta = self._perform_ttr(
            theta_tr,
            [self.tau_ttr.backward, self.sigma2_ttr.backward])

        return theta

    def clip_bounds(self, theta):
        """Clip variables to their bounds."""

        theta_clp = np.copy(theta)

        # Restrict τ to [0, 1]
        theta_clp[:, self.tau_idx] = \
            np.clip(theta[:, self.tau_idx], self.tau_lb, self.tau_ub)

        # Restrict σ2 to the positive reals
        theta_clp[:, self.sigma2_idx] = \
            np.clip(theta_clp[:, self.sigma2_idx], 0.0, None)

        return theta_clp


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


def gp(grid, kernel_fn, n_samples, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    N = len(grid)
    mean_vector = np.zeros(N)
    kernel_matrix = cov_matrix(kernel_fn, grid, grid)

    X = rng.multivariate_normal(mean_vector, kernel_matrix, size=n_samples)
    X = X - X.mean(axis=0)
    return X


def generate_gp_l2_dataset(
        grid, kernel_fn, n_samples,
        beta_coef, alpha0, sigma2, rng=None
):
    """Generate dataset based on GP with a given kernel function."""
    if rng is None:
        rng = np.random.default_rng()

    beta = beta_coef(grid)

    X = gp(grid, kernel_fn, n_samples, rng)
    Y = (alpha0 + trapz(y=X*beta, x=grid)
         + np.sqrt(sigma2)*rng.standard_normal(size=n_samples))

    return X, Y


def generate_gp_rkhs_dataset(
        grid, kernel_fn, n_samples,
        beta, tau, alpha0, sigma2, rng=None
):
    """Generate dataset based on GP with a given kernel function."""
    X = gp(grid, kernel_fn, n_samples, rng)

    theta_true = np.concatenate((
        beta, tau,
        [alpha0], [sigma2]
    ))

    Y = generate_response(X, theta_true, rng=rng)

    return X, Y


def plot_dataset(X, Y, plot_means=True):
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    n, N = X.shape
    grid = np.linspace(1./N, 1., N)

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
        n_walkers=1, frac_random=0.5, rng=None):

    n_random = int(frac_random*n_walkers)
    n_around = n_walkers - n_random

    if n_random > 0:
        init_1 = initial_guess_random(
            theta_space, sd_beta, mean_alpha0,
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


def emcee_to_idata(sampler, theta_space, burn, thin, ppc=False):
    names = theta_space.names
    names_ttr = theta_space.names_ttr
    p = theta_space.p

    if ppc:
        idata = az.from_emcee(
            sampler,
            var_names=names_ttr,
            slices=[slice(0, p), slice(p, 2*p), -2, -1],
            arg_names=["y_obs"],
            blob_names=["y_rec"],
            blob_groups=["posterior_predictive"],
            dims={f"{names_ttr[0]}": ["projection"],
                  f"{names_ttr[1]}": ["projection"],
                  "y_obs": ["observed"],
                  "y_rec": ["recovered"]}
        )
    else:
        idata = az.from_emcee(
            sampler,
            var_names=names_ttr,
            slices=[slice(0, p), slice(p, 2*p), -2, -1],
            arg_names=["y_obs"],
            dims={f"{names_ttr[0]}": ["projection"],
                  f"{names_ttr[1]}": ["projection"],
                  "y_obs": ["observed"]},
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
    additional_stats = {
        "mode": get_mode_func(),
        "median": np.median
    }

    return az.summary(data, extend=True,
                      stat_funcs=additional_stats,
                      **kwargs)


def point_predict(X, idata, names, point_estimate='mean'):
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

    Y_hat = generate_response(X, theta, noise=False)

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
