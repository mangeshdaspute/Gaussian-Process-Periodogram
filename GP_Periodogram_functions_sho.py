"""
GP_Periodogram_functions_sho.py
================================
Single-SHO kernel variants of the GP Periodogram pipeline functions.

Exposes the same public interface as GP_Periodogram_functions.py so that
GP_Periodogram_main.py can import from either module transparently:

    gridsearch_initial_params
    run_gp_periodogram
    build_results_dataframe
    gp_predict_best
    plot_gp_periodogram
    plot_lifetimes_transparent

All other functions (null model, bootstrap FAP, GLS plots, timeseries plot,
GP prediction plot) are unchanged and imported directly from
GP_Periodogram_functions.

Reference
---------
Daspute et al. (2026) — Interpretable Gaussian Process Periodogram.
"""

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import celerite
from celerite import terms
from scipy.optimize import minimize
from joblib import Parallel, delayed
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Private helper — negative log-likelihood (identical to the double-SHO file)
# ---------------------------------------------------------------------------

def _neg_log_like(params: np.ndarray, y: np.ndarray, gp) -> float:
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)


# ---------------------------------------------------------------------------
# Grid-search for GP initial parameters — single SHO
# ---------------------------------------------------------------------------

def _compute_for_w0_gridsearch(w0: float,
                                t: np.ndarray,
                                y: np.ndarray,
                                yerr: np.ndarray,
                                weighted_mean: float,
                                null_log_likelihood: float,
                                log_S0_grid: np.ndarray,
                                log_Q_grid: np.ndarray,
                                log_sigma_grid: np.ndarray) -> pd.DataFrame:
    """Evaluate the single-SHO GP log-likelihood on a parameter grid for
    a fixed angular frequency *w0*.

    Returns a single-row DataFrame with the parameter combination that
    maximises Δ lnL.
    """
    term1 = terms.SHOTerm(log_S0=0, log_Q=0, log_omega0=np.log(w0))
    term1.freeze_parameter("log_omega0")
    jitter_term = terms.JitterTerm(log_sigma=-2)
    kernel = term1 + jitter_term

    gp = celerite.GP(kernel, mean=weighted_mean)
    gp.compute(t, yerr)

    records = []
    for log_S0_1 in log_S0_grid:
        for log_Q1 in log_Q_grid:
            for log_sigma in log_sigma_grid:
                gp.set_parameter_vector([log_S0_1, log_Q1, log_sigma])
                log_like = gp.log_likelihood(y)
                delta_ll = log_like - null_log_likelihood
                records.append({
                    "w0": w0,
                    "log_S0_1": log_S0_1,
                    "log_Q1": log_Q1,
                    "log_sigma": log_sigma,
                    "delta_ll": delta_ll,
                })

    df_w0 = pd.DataFrame(records)
    best_row = df_w0.loc[[df_w0["delta_ll"].idxmax()]].reset_index(drop=True)
    return best_row


def gridsearch_initial_params(t: np.ndarray,
                               y: np.ndarray,
                               yerr: np.ndarray,
                               w0_list: np.ndarray,
                               weighted_mean: float,
                               null_log_likelihood: float,
                               name: str,
                               rms_scatter: float,
                               n_points: int = 7) -> pd.DataFrame:
    """Grid-search the initial single-SHO kernel parameters over a frequency
    grid.

    For each angular frequency in *w0_list*, searches over a coarse grid of
    (log_S0, log_Q, log_sigma) and retains the combination that gives the
    highest Δ lnL.  Results are saved to CSV and a diagnostic Q-vs-frequency
    plot is produced.

    Parameters
    ----------
    t, y, yerr : array-like
    w0_list : array-like
        Angular frequency grid [rad/day].
    weighted_mean : float
    null_log_likelihood : float
    name : str
        Label used in filenames.
    rms_scatter : float
        Unused; kept to match the double-SHO signature.
    n_points : int
        Number of grid points per parameter axis.

    Returns
    -------
    optimal_df : pd.DataFrame
        One row per entry in *w0_list* with the best-fit initial parameters.
    """
    n_points=n_points*2
    log_S0_grid = np.linspace(-15, 15, n_points)
    log_Q_grid = np.linspace(np.log(0.502), 15, n_points)
    log_sigma_grid = np.linspace(-15, 15, n_points)

    all_results = Parallel(n_jobs=-2)(
        delayed(_compute_for_w0_gridsearch)(
            w0, t, y, yerr, weighted_mean, null_log_likelihood,
            log_S0_grid, log_Q_grid, log_sigma_grid,
        )
        for w0 in tqdm(w0_list, desc="Grid-search initial parameters (single SHO)")
    )

    optimal_df = pd.concat(all_results, ignore_index=True)
    optimal_df.to_csv(f"optimal_parameters {name}.csv", index=False)

    # Diagnostic: Q1 vs frequency
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(
        optimal_df["w0"] / (2 * np.pi),
        np.exp(optimal_df["log_Q1"]),
        "o", label="Q1",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [1/d]")
    ax.set_ylabel("Q")
    ax.legend()
    ax.set_title("Optimal Q1 vs frequency")
    fig.savefig(f"Q_vs_frequency {name}.png", dpi=200)
    plt.close(fig)

    return optimal_df


# ---------------------------------------------------------------------------
# GP periodogram — per-frequency optimisation — single SHO
# ---------------------------------------------------------------------------

def _process_single_w0(w0: float,
                        y: np.ndarray,
                        t: np.ndarray,
                        yerr: np.ndarray,
                        weighted_mean: float,
                        null_log_likelihood: float,
                        df_initial: pd.DataFrame) -> list:
    """Optimise the single-SHO GP kernel at a fixed angular frequency *w0*.

    Uses the pre-computed grid-search result (``df_initial``) to warm-start
    the L-BFGS-B optimisation.

    Returns
    -------
    list
        [S0_1, Q_1, delta_log_like, w0, jitter,]
    """
    log_Q_lowest = np.log(0.501)
    bounds_sho = {
        "log_S0": (-15, 15),
        "log_Q": (log_Q_lowest, 15),
        "log_omega0": (-15, 15),
    }

    # -- Retrieve warm-start parameters from the grid-search CSV ------------
    closest_idx = (df_initial["w0"] - w0).abs().idxmin()
    log_S0_1 = df_initial.loc[closest_idx, "log_S0_1"]
    log_Q1 = df_initial.loc[closest_idx, "log_Q1"]
    log_sigma = df_initial.loc[closest_idx, "log_sigma"]

    # -- Build single-SHO kernel at the current frequency ------------------
    term1 = terms.SHOTerm(
        log_S0=log_S0_1, log_Q=log_Q1, log_omega0=np.log(w0),
        bounds=bounds_sho,
    )
    term1.freeze_parameter("log_omega0")
    jitter_term = terms.JitterTerm(
        log_sigma=log_sigma,
        bounds={"log_sigma": (-15, 15)},
    )
    kernel = term1 + jitter_term

    gp = celerite.GP(kernel, mean=weighted_mean)
    gp.compute(t, yerr)

    # -- Optimise ----------------------------------------------------------
    r = minimize(
        _neg_log_like,
        gp.get_parameter_vector(),
        method="L-BFGS-B",
        bounds=gp.get_parameter_bounds(),
        args=(y, gp),
    )
    gp.set_parameter_vector(r.x)
    params = gp.get_parameter_dict()

    S0_1 = np.exp(params["kernel:terms[0]:log_S0"])
    Q_1 = np.exp(params["kernel:terms[0]:log_Q"])
    jitter = np.exp(params["kernel:terms[1]:log_sigma"])

    delta_log_like = gp.log_likelihood(y) - null_log_likelihood

    

    return [
        S0_1, Q_1, delta_log_like, w0, jitter,
    ]


def run_gp_periodogram(t: np.ndarray,
                       y: np.ndarray,
                       yerr: np.ndarray,
                       w0_list: np.ndarray,
                       weighted_mean: float,
                       null_log_likelihood: float,
                       df_initial: pd.DataFrame) -> list:
    """Run the single-SHO GP periodogram over the full frequency grid in
    parallel.

    Parameters
    ----------
    t, y, yerr : array-like
    w0_list : array-like
        Angular frequency grid [rad/day].
    weighted_mean : float
    null_log_likelihood : float
    df_initial : pd.DataFrame
        Output of :func:`gridsearch_initial_params`.

    Returns
    -------
    result : list of lists
        One list per frequency (see :func:`_process_single_w0`).
    """
    result = Parallel(n_jobs=-2)(
        delayed(_process_single_w0)(
            w0, y, t, yerr, weighted_mean, null_log_likelihood, df_initial
        )
        for w0 in tqdm(w0_list, desc="GP periodogram (single SHO)")
    )
    return result


# ---------------------------------------------------------------------------
# Derived quantities from the raw periodogram result — single SHO
# ---------------------------------------------------------------------------

def build_results_dataframe(result: list, rms_scatter: float) -> pd.DataFrame:
    """Convert the raw list returned by :func:`run_gp_periodogram` into a
    tidy DataFrame with all derived quantities (lifetime, amplitude, etc.).

    Parameters
    ----------
    result : list of lists
    rms_scatter : float
        RMS of the input RV time series [m/s].

    Returns
    -------
    df_results : pd.DataFrame
    """
    cols = [
        "S0_1", "Q_1", "delta_log_like", "w0", "jitter",
    ]
    df = pd.DataFrame(result, columns=cols)

    df["Frequency_1"] = df["w0"] / (2.0 * np.pi)
    df["Period_1"] = 2.0 * np.pi / df["w0"]
    df["lifetime_1"] = 2.0 * df["Q_1"] / df["w0"]
    df["Amplitude_1[m/s]"] = np.sqrt(2.0 * df["Q_1"] * df["w0"] * df["S0_1"])

    norm_factor = math.sqrt(2.0) * rms_scatter
    df["fraction_RMS1[m/s]"] = df["Amplitude_1[m/s]"] / norm_factor

    return df


# ---------------------------------------------------------------------------
# GP prediction at the best-fit frequency — single SHO
# ---------------------------------------------------------------------------

def gp_predict_best(t: np.ndarray,
                    y: np.ndarray,
                    yerr: np.ndarray,
                    df_results: pd.DataFrame,
                    weighted_mean: float) -> tuple:
    """Reconstruct the single-SHO GP at the highest-Δ lnL frequency and
    return predictions on a dense grid.

    Returns
    -------
    t_pred : np.ndarray
    mu : np.ndarray
    std : np.ndarray
    residuals : np.ndarray  (evaluated at the original time stamps)
    """
    best = df_results.loc[df_results["delta_log_like"].idxmax()]

    S0_1 = float(best["S0_1"])
    Q_1 = float(best["Q_1"])
    w0 = float(best["w0"])
    jitter = float(best["jitter"])

    log_Q_lowest = np.log(0.501)
    bounds = {
        "log_S0": (-15, 15),
        "log_Q": (log_Q_lowest, 15),
        "log_omega0": (-15, 15),
    }

    term1 = terms.SHOTerm(
        log_S0=np.log(S0_1), log_Q=np.log(Q_1), log_omega0=np.log(w0),
        bounds=bounds,
    )
    term1.freeze_parameter("log_omega0")
    jitter_term = terms.JitterTerm(
        log_sigma=np.log(jitter), bounds={"log_sigma": (-15, 15)}
    )
    kernel = term1 + jitter_term

    gp = celerite.GP(kernel, mean=weighted_mean)
    gp.compute(t, yerr)

    t_pred = np.linspace(np.min(t), np.max(t), len(t) * 10)
    mu, var = gp.predict(y, t_pred, return_var=True)
    std = np.sqrt(var)

    mu_at_t, _ = gp.predict(y, t, return_var=True)
    residuals = y - mu_at_t

    return t_pred, mu, std, residuals


# ---------------------------------------------------------------------------
# Plotting — single SHO
# ---------------------------------------------------------------------------

def plot_gp_periodogram(Frequency_val: np.ndarray,
                        delta_log_like: np.ndarray,
                        df_results: pd.DataFrame,
                        name: str,
                        xlim_full: tuple = (0.0, 1.0),
                        xlim_zoom: tuple = (0.0, 0.1),
                        fap_1pct: float | None = None,
                        fap_10pct: float | None = None) -> None:
    """Plot the single-SHO GP periodogram coloured by the RMS fraction of the
    oscillator, at full range and zoomed in.

    Parameters
    ----------
    fap_1pct, fap_10pct : float or None
        If provided, draw horizontal FAP threshold lines on the plot.
    """
    for xlim, suffix in [(xlim_full, ""), (xlim_zoom, "_zoomed")]:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        ax.plot(Frequency_val, delta_log_like, color="gray",
                linewidth=0.5, alpha=0.5, zorder=1)
        sc = ax.scatter(
            Frequency_val, delta_log_like,
            c=df_results["fraction_RMS1[m/s]"],
            cmap="rainbow", s=10, edgecolor="none", zorder=2,
        )
        cbar = fig.colorbar(sc, cax=plt.gca().inset_axes([0.30, 0.96, 0.4, 0.035]),
                    orientation="horizontal")#[left, bottom, width, height]
        cbar.set_label("RMS fraction of oscillator", fontsize=10)
        cbar.ax.tick_params(labelsize=10)
        if fap_1pct is not None:
            ax.axhline(fap_1pct, color="black", linestyle="--",
                       label=f"1% FAP ({fap_1pct:.2f})", zorder=3)
        if fap_10pct is not None:
            ax.axhline(fap_10pct, color="black", linestyle="-.",
                       label=f"10% FAP ({fap_10pct:.2f})", zorder=3)
        if fap_1pct is not None or fap_10pct is not None:
            ax.legend(loc="upper right")
        ax.set_xlabel("Frequency of oscillator [1/d]")
        ax.set_ylabel("Δ lnL")
        ax.set_xlim(xlim)
        ax.grid(which="major", linestyle="-")
        ax.minorticks_on()
        ax.grid(which="minor", color="#DDDDDD", linestyle="--", alpha=0.6)
        fig.tight_layout()
        fig.savefig(f"GP_periodogram_{name}{suffix}.png")
        plt.close(fig)


def plot_lifetimes_transparent(Frequency_val: np.ndarray,
                                lifetime0_values: np.ndarray,
                                df_results: pd.DataFrame,
                                timespan_obs: float,
                                name: str,
                                xlim_full: tuple = (0.0, 1.0),
                                xlim_zoom: tuple = (0.0, 0.1)) -> None:
    """Plot single-SHO kernel lifetime using RMS fraction as marker
    transparency, at full range and zoomed in."""
    for xlim, suffix in [(xlim_full, ""), (xlim_zoom, "_zoomed")]:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

        ax.plot(Frequency_val, lifetime0_values,
                color="red", linewidth=0.5, alpha=0.5, zorder=2)
        alpha1 = np.clip(df_results["fraction_RMS1[m/s]"].values, None, 1.0)
        rgba1 = np.zeros((len(alpha1), 4))
        rgba1[:, 0] = 1.0   # red channel
        rgba1[:, 3] = alpha1
        ax.scatter(Frequency_val, lifetime0_values,
                   facecolors=rgba1, edgecolors="none",
                   marker="o", s=15, zorder=4, label="Oscillator")

        ax.axhline(timespan_obs, color="black", linestyle="--",
                   label="Timespan_obs")
        ax.set_xlabel("Frequency of oscillator [1/d]")
        ax.set_ylabel("Lifetime [d]")
        ax.set_xlim(xlim)
        ax.set_yscale("log")
        ax.grid(which="major", linestyle="-")
        ax.minorticks_on()
        ax.grid(which="minor", color="#DDDDDD", linestyle="--", alpha=0.6)
        leg = ax.legend()
        for handle in leg.legend_handles:
            handle.set_alpha(1.0)
        fig.tight_layout()
        fig.savefig(f"lifetimes_{name}_transparent{suffix}.png")
        plt.close(fig)

def compute_initial_params_guess(w0_list: np.ndarray,
                                  rms_scatter: float) -> pd.DataFrame:
    """Build a DataFrame of analytical initial parameters for the single-SHO
    GP, one row per frequency, without running a grid-search.

    Initial values
    --------------
    jitter  = rms_scatter / 2
    Q       = 10
    S       = (2π / w0) * rms_scatter² * π  = 2π² * rms_scatter² / w0
    """
    log_Q = np.log(1.0)
    log_sigma = np.log(rms_scatter / 2.0)

    records = []
    for w0 in w0_list:
        S = 2.0 * np.pi ** 2 * rms_scatter ** 2 / w0
        log_S0 = np.log(S)
        records.append({
            "w0":        w0,
            "log_S0_1":  log_S0,
            "log_Q1":    log_Q,
            "log_sigma": log_sigma,
        })
    return pd.DataFrame(records)
