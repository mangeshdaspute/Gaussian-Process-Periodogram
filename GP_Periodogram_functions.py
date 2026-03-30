"""
gp_periodogram.py
=================
All computation and plotting functions for the GP Periodogram pipeline.

Reference
---------
Daspute et al. (2026) — Interpretable Gaussian Process Periodogram for
identifying and differentiating periodic and quasi-periodic signals in
time series.
"""

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec

import celerite
from celerite import terms
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm
from joblib import Parallel, delayed
from tqdm import tqdm

import mlp  # local GLS implementation — not refactored, required as-is
from astropy.timeseries import LombScargle


# ---------------------------------------------------------------------------
# Null-model helpers
# ---------------------------------------------------------------------------

def _neg_null_log_likelihood_with_jitter(ln_sigma_j_sq: float,
                                         y: np.ndarray,
                                         yerr: np.ndarray) -> float:
    """Negative log-likelihood of the constant null model with a free jitter.

    Returns the *negative* value so that scipy minimisers can find the
    maximum.

    Parameters
    ----------
    ln_sigma_j_sq : float
        Natural log of the jitter variance.
    y : array-like
        Observed RV values.
    yerr : array-like
        Per-observation measurement uncertainties.
    """
    sigma_j_sq = np.exp(ln_sigma_j_sq)
    total_var = yerr ** 2 + sigma_j_sq
    weights = 1.0 / total_var
    weighted_mean = np.average(y, weights=weights)
    log_like = -0.5 * np.sum(
        (y - weighted_mean) ** 2 / total_var + np.log(2.0 * np.pi * total_var)
    )
    return -log_like


def compute_null_model(y: np.ndarray,
                       yerr: np.ndarray) -> tuple:
    """Maximise the log-likelihood of the constant (null) model.

    Optimises over the jitter term analytically using bounded scalar
    minimisation.

    Parameters
    ----------
    y, yerr : array-like

    Returns
    -------
    null_log_likelihood : float
    weighted_mean : float
    optimal_jitter : float  (standard deviation, not variance)
    """
    min_ln = np.log(1e-10)
    max_ln = np.log(10.0 * np.max(yerr ** 2))

    result = minimize_scalar(
        _neg_null_log_likelihood_with_jitter,
        args=(y, yerr),
        bounds=(min_ln, max_ln),
        method="bounded",
    )
    null_log_likelihood = -result.fun
    optimal_jitter_sq = np.exp(result.x)
    optimal_jitter = np.sqrt(optimal_jitter_sq)

    optimal_weights = 1.0 / (yerr ** 2 + optimal_jitter_sq)
    weighted_mean = np.average(y, weights=optimal_weights)

    return null_log_likelihood, weighted_mean, optimal_jitter


# ---------------------------------------------------------------------------
# Bootstrap FAP for GLS
# ---------------------------------------------------------------------------

def bootstrap_fap(t: np.ndarray,
                  y: np.ndarray,
                  yerr: np.ndarray,
                  freq: np.ndarray,
                  jitter: float,
                  n_boot: int = 100) -> tuple:
    """Estimate GLS false-alarm-probability thresholds via bootstrap.

    Parameters
    ----------
    t, y, yerr : array-like
    freq : array-like
        Frequency grid used for the GLS computation.
    jitter : float
        Jitter standard deviation added to the noise model.
    n_boot : int
        Number of bootstrap realisations.

    Returns
    -------
    threshold_1pct : float
        Δ lnL threshold corresponding to FAP = 0.01.
    threshold_10pct : float
        Δ lnL threshold corresponding to FAP = 0.10.
    """

    def _single_bootstrap():
        noise = np.random.normal(
            loc=0.0,
            scale=np.sqrt(yerr ** 2 + jitter ** 2),
            size=t.shape,
        )
        gls_sim = mlp.Gls(
            [(t, noise, yerr)],
            fbeg=freq[0],
            fend=freq[-1],
            freq=freq,
            norm="dlnL",
        )
        return gls_sim.power.max()

    max_dlnL = Parallel(n_jobs=-2)(
        delayed(_single_bootstrap)()
        for _ in tqdm(range(n_boot), desc="GLS Bootstrap FAP")
    )
    thresholds = np.percentile(max_dlnL, [99.0, 90.0])
    return float(thresholds[0]), float(thresholds[1])



def _neg_log_like(params: np.ndarray, y: np.ndarray, gp) -> float:
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)



# ---------------------------------------------------------------------------
# Grid-search for GP initial parameters
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
    """Evaluate the double-SHO GP log-likelihood on a parameter grid for
    a fixed angular frequency *w0*.

    Returns a single-row DataFrame with the parameter combination that
    maximises Δ lnL.
    """
    log_Q_lowest = np.log(0.502)

    term1 = terms.SHOTerm(log_S0=0, log_Q=0, log_omega0=np.log(w0))
    term1.freeze_parameter("log_omega0")
    term2 = terms.SHOTerm(log_S0=0, log_Q=0, log_omega0=np.log(2.0 * w0))
    term2.freeze_parameter("log_omega0")
    jitter_term = terms.JitterTerm(log_sigma=-2)
    kernel = term1 + term2 + jitter_term

    gp = celerite.GP(kernel, mean=weighted_mean)
    gp.compute(t, yerr)

    records = []
    for log_S0_1 in log_S0_grid:
        for log_Q1 in log_Q_grid:
            for log_S0_2 in log_S0_grid:
                for log_Q2 in log_Q_grid:
                    for log_sigma in log_sigma_grid:
                        gp.set_parameter_vector(
                            [log_S0_1, log_Q1, log_S0_2, log_Q2, log_sigma]
                        )
                        log_like = gp.log_likelihood(y)
                        delta_ll = log_like - null_log_likelihood
                        records.append({
                            "w0": w0,
                            "log_S0_1": log_S0_1,
                            "log_Q1": log_Q1,
                            "log_S0_2": log_S0_2,
                            "log_Q2": log_Q2,
                            "log_sigma": log_sigma,
                            "delta_ll": delta_ll,
                        })

    df_w0 = pd.DataFrame(records)
    # Keep only the row with the highest Δ lnL
    best_row = df_w0.loc[[df_w0["delta_ll"].idxmax()]].reset_index(drop=True)
    return best_row


def gridsearch_initial_params(t: np.ndarray,
                               y: np.ndarray,
                               yerr: np.ndarray,
                               w0_list: np.ndarray,
                               weighted_mean: float,
                               null_log_likelihood: float,
                               name: str,
                               rms_scatter,
                               n_points: int = 7,
                               ) -> pd.DataFrame:
    """Grid-search the initial GP kernel parameters over a frequency grid.

    For each angular frequency in *w0_list*, the function searches over a
    coarse grid of (log_S0, log_Q, log_sigma) and retains the parameter
    combination that gives the highest Δ lnL.  Results are saved to CSV
    and a diagnostic plot (Q vs frequency) is produced.

    Parameters
    ----------
    t, y, yerr : array-like
    w0_list : array-like
        Angular frequency grid [rad/day].
    weighted_mean : float
    null_log_likelihood : float
    name : str
        Label used in filenames.
    n_points : int
        Number of grid points per parameter axis.

    Returns
    -------
    optimal_df : pd.DataFrame
        One row per entry in *w0_list* with the best-fit initial parameters.
    """
    log_S0_grid = np.linspace(-15, 15, n_points)
    log_Q_grid = np.linspace(np.log(0.502), 15, n_points)
    log_sigma_grid = np.linspace(-15, 15, n_points)

    all_results = Parallel(n_jobs=-2)(
        delayed(_compute_for_w0_gridsearch)(
            w0, t, y, yerr, weighted_mean, null_log_likelihood,
            log_S0_grid, log_Q_grid, log_sigma_grid,
        )
        for w0 in tqdm(w0_list, desc="Grid-search initial parameters")
    )

    optimal_df = pd.concat(all_results, ignore_index=True)
    optimal_df.to_csv(f"optimal_parameters {name}.csv", index=False)

    # Diagnostic: Q values vs frequency
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(
        optimal_df["w0"] / (2 * np.pi),
        np.exp(optimal_df["log_Q1"]),
        "o", label="Q1",
    )
    ax.plot(
        optimal_df["w0"] / (2 * np.pi),
        np.exp(optimal_df["log_Q2"]),
        "o", label="Q2",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [1/d]")
    ax.set_ylabel("Q")
    ax.legend()
    ax.set_title("Optimal Q1 and Q2 vs frequency")
    fig.savefig(f"Q_vs_frequency {name}.png", dpi=200)
    plt.close(fig)

    return optimal_df


# ---------------------------------------------------------------------------
# GP periodogram — per-frequency optimisation
# ---------------------------------------------------------------------------

def _process_single_w0(w0: float,
                        y: np.ndarray,
                        t: np.ndarray,
                        yerr: np.ndarray,
                        weighted_mean: float,
                        null_log_likelihood: float,
                        df_initial: pd.DataFrame) -> list:
    """Optimise the double-SHO GP kernel at a fixed angular frequency *w0*.

    Uses the pre-computed grid-search result (``df_initial``) to warm-start
    the L-BFGS-B optimisation.

    Returns
    -------
    list
        [S0_1, S0_2, Q_1, Q_2, delta_log_like, w0, w_2, jitter,
         null_S0_1, null_S0_2, null_Q_1, null_Q_2, null_jitter,
         null_log_like_val]
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
    log_S0_2 = df_initial.loc[closest_idx, "log_S0_2"]
    log_Q2 = df_initial.loc[closest_idx, "log_Q2"]
    log_sigma = df_initial.loc[closest_idx, "log_sigma"]

    # -- Build double-SHO kernel at the current frequency -------------------
    term1 = terms.SHOTerm(
        log_S0=log_S0_1, log_Q=log_Q1, log_omega0=np.log(w0),
        bounds=bounds_sho,
    )
    term2 = terms.SHOTerm(
        log_S0=log_S0_2, log_Q=log_Q2, log_omega0=np.log(2.0 * w0),
        bounds=bounds_sho,
    )
    term1.freeze_parameter("log_omega0")
    term2.freeze_parameter("log_omega0")
    jitter_term = terms.JitterTerm(
        log_sigma=log_sigma,
        bounds={"log_sigma": (-15, 15)},
    )
    kernel = term1 + term2 + jitter_term

    gp = celerite.GP(kernel, mean=weighted_mean)
    gp.compute(t, yerr)

    # -- Optimise -----------------------------------------------------------
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
    S0_2 = np.exp(params["kernel:terms[1]:log_S0"])
    Q_2 = np.exp(params["kernel:terms[1]:log_Q"])
    w_2 = 2.0 * w0
    jitter = np.exp(params["kernel:terms[2]:log_sigma"])

    delta_log_like = gp.log_likelihood(y) - null_log_likelihood


    return [
        S0_1, S0_2, Q_1, Q_2, delta_log_like, w0, w_2, jitter,
    ]


def run_gp_periodogram(t: np.ndarray,
                       y: np.ndarray,
                       yerr: np.ndarray,
                       w0_list: np.ndarray,
                       weighted_mean: float,
                       null_log_likelihood: float,
                       df_initial: pd.DataFrame) -> list:
    """Run the double-SHO GP periodogram over the full frequency grid in
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
        One list per frequency with 14 elements (see
        :func:`_process_single_w0`).
    """
    result = Parallel(n_jobs=-2)(
        delayed(_process_single_w0)(
            w0, y, t, yerr, weighted_mean, null_log_likelihood, df_initial
        )
        for w0 in tqdm(w0_list, desc="GP periodogram")
    )
    return result


# ---------------------------------------------------------------------------
# Derived quantities from the raw periodogram result
# ---------------------------------------------------------------------------

def build_results_dataframe(result: list, rms_scatter: float) -> pd.DataFrame:
    """Convert the raw list returned by :func:`run_gp_periodogram` into a
    tidy DataFrame with all derived quantities (lifetimes, amplitudes, etc.).

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
        "S0_1", "S0_2", "Q_1", "Q_2", "delta_log_like",
        "w0", "w_2", "jitter",
    ]
    df = pd.DataFrame(result, columns=cols)

    df["Frequency_1"] = df["w0"] / (2.0 * np.pi)
    df["Frequency_2"] = df["w_2"] / (2.0 * np.pi)
    df["Period_1"] = 2.0 * np.pi / df["w0"]
    df["Period_2"] = 2.0 * np.pi / df["w_2"]

    df["lifetime_1"] = 2.0 * df["Q_1"] / df["w0"]
    df["lifetime_2"] = 2.0 * df["Q_2"] / df["w_2"]

    df["Amplitude_1[m/s]"] = np.sqrt(2.0 * df["Q_1"] * df["w0"] * df["S0_1"])
    df["Amplitude_2[m/s]"] = np.sqrt(2.0 * df["Q_2"] * df["w_2"] * df["S0_2"])

    norm_factor = math.sqrt(2.0) * rms_scatter
    df["fraction_RMS1[m/s]"] = df["Amplitude_1[m/s]"] / norm_factor
    df["fraction_RMS2[m/s]"] = df["Amplitude_2[m/s]"] / norm_factor

    return df


# ---------------------------------------------------------------------------
# GP prediction at the best-fit frequency
# ---------------------------------------------------------------------------

def gp_predict_best(t: np.ndarray,
                    y: np.ndarray,
                    yerr: np.ndarray,
                    df_results: pd.DataFrame,
                    weighted_mean: float) -> tuple:
    """Reconstruct the GP at the highest-Δ lnL frequency and return
    predictions on a dense grid.

    Returns
    -------
    t_pred : np.ndarray
    mu : np.ndarray
    std : np.ndarray
    residuals : np.ndarray (evaluated at the original time stamps)
    """
    best = df_results.loc[df_results["delta_log_like"].idxmax()]

    S0_1 = float(best["S0_1"])
    Q_1 = float(best["Q_1"])
    S0_2 = float(best["S0_2"])
    Q_2 = float(best["Q_2"])
    w0 = float(best["w0"])
    w_2 = float(best["w_2"])
    jitter = float(best["jitter"])

    log_Q_lowest = np.log(0.501)
    bounds = {"log_S0": (-15, 15), "log_Q": (log_Q_lowest, 15), "log_omega0": (-15, 15)}

    term1 = terms.SHOTerm(
        log_S0=np.log(S0_1), log_Q=np.log(Q_1), log_omega0=np.log(w0),
        bounds=bounds,
    )
    term2 = terms.SHOTerm(
        log_S0=np.log(S0_2), log_Q=np.log(Q_2), log_omega0=np.log(w_2),
        bounds=bounds,
    )
    term1.freeze_parameter("log_omega0")
    term2.freeze_parameter("log_omega0")
    jitter_term = terms.JitterTerm(
        log_sigma=np.log(jitter), bounds={"log_sigma": (-15, 15)}
    )
    kernel = term1 + term2 + jitter_term

    gp = celerite.GP(kernel, mean=weighted_mean)
    gp.compute(t, yerr)

    t_pred = np.linspace(np.min(t), np.max(t), len(t) * 10)
    mu, var = gp.predict(y, t_pred, return_var=True)
    std = np.sqrt(var)

    mu_at_t, _ = gp.predict(y, t, return_var=True)
    residuals = y - mu_at_t

    return t_pred, mu, std, residuals


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_dt(dt: pd.Series, name: str) -> None:
    """Plot time sampling intervals vs observation index."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(dt, "r.")
    ax.set_yscale("log")
    ax.set_ylabel("dt [d]")
    ax.set_xlabel("Index of observation")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{name} dt.png")
    plt.close(fig)


def plot_uncertainty(yerr: np.ndarray, name: str) -> None:
    """Plot per-observation RV uncertainty vs observation index."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(yerr, "r.")
    ax.set_xlabel("Index of observation")
    ax.set_ylabel("RV uncertainty [m/s]")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{name} uncertainty.png")
    plt.close(fig)


def plot_timeseries(t: np.ndarray,
                    y: np.ndarray,
                    yerr: np.ndarray,
                    rms_scatter: float,
                    name: str) -> None:
    """Plot the RV time series with error bars."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.errorbar(
        t, y, yerr=yerr, fmt=".k", capsize=0,
        label=f"RMS scatter = {rms_scatter:.3f} m/s",
    )
    ax.set_xlabel("Time [d]")
    ax.set_ylabel("RV [m/s]")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{name} timeseries.png")
    plt.close(fig)


def plot_gls_astropy(frequency: np.ndarray,
                     power: np.ndarray,
                     power_fap_1pct: float,
                     power_fap_10pct: float,
                     name: str) -> None:
    """Plot the Astropy GLS periodogram with FAP levels."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(frequency, power, color="black", lw=0.5)
    ax.axhline(
        power_fap_1pct, color="blue", linestyle="-",
        label=f"1% FAP ({power_fap_1pct:.2f})",
    )
    ax.axhline(
        power_fap_10pct, color="green", linestyle="--",
        label=f"10% FAP ({power_fap_10pct:.2f})",
    )
    ax.set_xlabel("Frequency [1/days]")
    ax.set_ylabel("Power")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"GLS_astropy_{name}.png")
    plt.close(fig)


def plot_gls_dlnl(gls_freq: np.ndarray,
                  gls_power: np.ndarray,
                  name: str,
                  xlim_full: tuple = (0.0, 1.0),
                  xlim_zoom: tuple = (0.0, 0.1),
                  fap_1pct: float | None = None,
                  fap_10pct: float | None = None) -> None:
    """Plot the mlp GLS Δ lnL periodogram at full range and zoomed in.

    Parameters
    ----------
    fap_1pct, fap_10pct : float or None
        If provided, draw horizontal FAP threshold lines on the plot.
    """
    for xlim, suffix in [(xlim_full, ""), (xlim_zoom, "_zoomed")]:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        ax.plot(gls_freq, gls_power, "k-")
        if fap_1pct is not None:
            ax.axhline(fap_1pct, color="black", linestyle="--",
                       label=f"1% FAP ({fap_1pct:.2f})")
        if fap_10pct is not None:
            ax.axhline(fap_10pct, color="black", linestyle="-.",
                       label=f"10% FAP ({fap_10pct:.2f})")
        if fap_1pct is not None or fap_10pct is not None:
            ax.legend()
        ax.set_xlabel("Frequency [1/d]")
        ax.set_ylabel("Δ lnL")
        ax.set_xlim(xlim)
        fig.tight_layout()
        fig.savefig(f"GLS_{name}_dlnL{suffix}.png")
        plt.close(fig)


def plot_gls_diagnostics(gls, name: str) -> None:
    """Plot mlp GLS offset and jitter vs frequency and save results to CSV."""
    offsets = gls._off[:, 0]
    jitters = np.array([p[-gls.Nj:] for p in gls.par])[:, 0]
    amplitudes = np.sqrt(gls._a ** 2 + gls._b ** 2)

    for values, ylabel, suffix in [
        (offsets, "Offset [m/s]", "offsets"),
        (jitters, "Jitter [m/s]", "jitters"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.plot(gls.freq, values)
        ax.set_xlabel("Frequency [1/d]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Frequency vs {ylabel.split(' ')[0]} — {name}")
        fig.tight_layout()
        fig.savefig(f"GLS_{name}_{suffix}.png")
        plt.close(fig)

    pd.DataFrame({
        "frequency": gls.freq,
        "jitter": jitters,
        "delta_log_likelihood": gls.power,
        "offset": offsets,
        "amplitudes": amplitudes,
    }).to_csv(f"gls_mlp_results_{name}.csv", index=False)


def plot_gp_periodogram(Frequency_val: np.ndarray,
                        delta_log_like: np.ndarray,
                        df_results: pd.DataFrame,
                        name: str,
                        xlim_full: tuple = (0.0, 1.0),
                        xlim_zoom: tuple = (0.0, 0.1),
                        fap_1pct: float | None = None,
                        fap_10pct: float | None = None) -> None:
    """Plot the GP periodogram coloured by the RMS fraction of the first
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
                    orientation="horizontal") #[left, bottom, width, height]
        cbar.set_label("RMS fraction of first oscillator", fontsize=10)
        cbar.ax.tick_params(labelsize=10)
        if fap_1pct is not None:
            ax.axhline(fap_1pct, color="black", linestyle="--",
                       label=f"1% FAP ({fap_1pct:.2f})", zorder=3)
        if fap_10pct is not None:
            ax.axhline(fap_10pct, color="black", linestyle="-.",
                       label=f"10% FAP ({fap_10pct:.2f})", zorder=3)
        if fap_1pct is not None or fap_10pct is not None:
            ax.legend(loc="upper right")
        ax.set_xlabel("Frequency of first oscillator [1/d]")
        ax.set_ylabel("Δ lnL")
        ax.set_xlim(xlim)
        ax.grid(which="major", linestyle="-")
        ax.minorticks_on()
        ax.grid(which="minor", color="#DDDDDD", linestyle="--", alpha=0.6)
        fig.tight_layout()
        fig.savefig(f"GP_periodogram_{name}{suffix}.png")
        plt.close(fig)


def plot_lifetimes_transparent(Frequency_val: np.ndarray,
                                Frequency_val2: np.ndarray,
                                lifetime0_values: np.ndarray,
                                lifetime2_values: np.ndarray,
                                df_results: pd.DataFrame,
                                timespan_obs: float,
                                name: str,
                                xlim_full: tuple = (0.0, 1.0),
                                xlim_zoom: tuple = (0.0, 0.1)) -> None:
    """Plot GP kernel lifetimes using RMS fraction as marker transparency,
    at full range and zoomed in."""
    for xlim, suffix in [(xlim_full, ""), (xlim_zoom, "_zoomed")]:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

        # First oscillator — red markers
        ax.plot(Frequency_val, lifetime0_values,
                color="red", linewidth=0.5, alpha=0.5, zorder=2)
        alpha1 = np.clip(df_results["fraction_RMS1[m/s]"].values, None, 1.0)
        rgba1 = np.zeros((len(alpha1), 4))
        rgba1[:, 0] = 1.0   # red channel
        rgba1[:, 3] = alpha1
        ax.scatter(Frequency_val, lifetime0_values,
                   facecolors=rgba1, edgecolors="none",
                   marker="o", s=15, zorder=4, label="First Oscillator")

        # Second oscillator — blue markers
        ax.plot(Frequency_val2, lifetime2_values,
                color="blue", linewidth=0.5, alpha=0.5, zorder=1)
        alpha2 = np.clip(df_results["fraction_RMS2[m/s]"].values, None, 1.0)
        rgba2 = np.zeros((len(alpha2), 4))
        rgba2[:, 2] = 1.0   # blue channel
        rgba2[:, 3] = alpha2
        ax.scatter(Frequency_val2, lifetime2_values,
                   facecolors=rgba2, edgecolors="none",
                   marker="s", s=20, zorder=3, label="Second Oscillator")

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


def plot_lifetimes_colored(Frequency_val: np.ndarray,
                            Frequency_val2: np.ndarray,
                            lifetime0_values: np.ndarray,
                            lifetime2_values: np.ndarray,
                            df_results: pd.DataFrame,
                            timespan_obs: float,
                            name: str,
                            xlim_full: tuple = (0.0, 1.0),
                            xlim_zoom: tuple = (0.0, 0.1)) -> None:
    """Plot GP kernel lifetimes using a rainbow colourmap for RMS fraction,
    at full range and zoomed in."""
    for xlim, suffix in [(xlim_full, ""), (xlim_zoom, "_zoomed")]:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

        ax.plot(Frequency_val, lifetime0_values,
                color="gray", linewidth=0.5, alpha=0.5, zorder=2)
        sc1 = ax.scatter(
            Frequency_val, lifetime0_values,
            c=df_results["fraction_RMS1[m/s]"],
            cmap="rainbow", s=10, edgecolor="none", zorder=4,
            label="First Oscillator",
        )
        cbar = fig.colorbar(sc1, ax=ax)
        cbar.set_label("Fraction of RMS — first oscillator")

        ax.plot(Frequency_val2, lifetime2_values,
                color="gray", linewidth=0.5, alpha=0.5, zorder=1)
        ax.scatter(
            Frequency_val2, lifetime2_values,
            c=df_results["fraction_RMS2[m/s]"],
            cmap="rainbow", s=10, marker="s", edgecolor="none", zorder=3,
            label="Second Oscillator",
        )

        ax.axhline(timespan_obs, color="black", linestyle="--",
                   label="Timespan_obs")
        ax.set_xlabel("Frequency of oscillator [1/d]")
        ax.set_ylabel("Lifetime [d]")
        ax.set_xlim(xlim)
        ax.set_yscale("log")
        ax.grid(which="major", linestyle="-")
        ax.minorticks_on()
        ax.grid(which="minor", color="#DDDDDD", linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"lifetimes_{name}_colored{suffix}.png")
        plt.close(fig)


def _plot_oscillator_panel(Frequency_axis: np.ndarray,
                            delta_log_like: np.ndarray,
                            Q_values: np.ndarray,
                            S_values: np.ndarray,
                            lifetime_values: np.ndarray,
                            jitter_values: np.ndarray,
                            xlabel_freq: str,
                            title_suffix: str,
                            filename: str,
                            null: bool = False) -> None:
    """Generic 5-panel diagnostic plot for one oscillator."""
    ylabel_ll = "Null Log Likelihood" if null else "Δ lnL"
    fig, axs = plt.subplots(5, 1, figsize=(20, 12), sharex=True, dpi=300)

    axs[0].plot(Frequency_axis, delta_log_like, "r-", marker=".")
    axs[0].set_ylabel(ylabel_ll)
    axs[0].grid()

    axs[1].plot(Frequency_axis, Q_values, "g-", marker=".")
    axs[1].set_ylabel(f"Q ({title_suffix})")
    axs[1].set_yscale("log")
    axs[1].grid()

    axs[2].plot(Frequency_axis, S_values, "b-", marker=".")
    axs[2].set_ylabel(f"S0 [(m/s)²] ({title_suffix})")
    axs[2].set_yscale("log")
    axs[2].grid()

    axs[3].plot(Frequency_axis, lifetime_values, "m-", marker=".")
    axs[3].set_ylabel("Lifetime [d]")
    axs[3].set_yscale("log")
    axs[3].grid()

    axs[4].plot(Frequency_axis, jitter_values, "c-", marker=".")
    axs[4].set_xlabel(xlabel_freq)
    axs[4].set_ylabel("Jitter [m/s]")
    axs[4].set_yscale("log")
    axs[4].grid()

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_oscillator_diagnostics(Frequency_val: np.ndarray,
                                 Frequency_val2: np.ndarray,
                                 delta_log_like: np.ndarray,
                                 Q0_values: np.ndarray,
                                 S0_values: np.ndarray,
                                 lifetime0_values: np.ndarray,
                                 Q2_values: np.ndarray,
                                 S2_values: np.ndarray,
                                 lifetime2_values: np.ndarray,
                                 jitter_values: np.ndarray,
                                 null_log_like_arr: np.ndarray,
                                 null_Q0_values: np.ndarray,
                                 null_S0_values: np.ndarray,
                                 null_lifetime0_values: np.ndarray,
                                 null_Q2_values: np.ndarray,
                                 null_S2_values: np.ndarray,
                                 null_lifetime2_values: np.ndarray,
                                 null_jitter_values: np.ndarray,
                                 name: str) -> None:
    """Produce all 5-panel diagnostic plots for first and second oscillators,
    both for the GP model and the null model."""
    # GP — first oscillator
    _plot_oscillator_panel(
        Frequency_val, delta_log_like, Q0_values, S0_values,
        lifetime0_values, jitter_values,
        xlabel_freq="Frequency [1/d]",
        title_suffix="first oscillator",
        filename=f"GP_diagnostics_{name}_oscillator1.png",
    )
    # GP — second oscillator
    _plot_oscillator_panel(
        Frequency_val2, delta_log_like, Q2_values, S2_values,
        lifetime2_values, jitter_values,
        xlabel_freq="Frequency [1/d]",
        title_suffix="second oscillator",
        filename=f"GP_diagnostics_{name}_oscillator2.png",
    )
    # Null — first oscillator
    _plot_oscillator_panel(
        Frequency_val, null_log_like_arr, null_Q0_values, null_S0_values,
        null_lifetime0_values, null_jitter_values,
        xlabel_freq="Frequency [1/d]",
        title_suffix="first oscillator",
        filename=f"Null_GP_diagnostics_{name}_oscillator1.png",
        null=True,
    )
    # Null — second oscillator
    _plot_oscillator_panel(
        Frequency_val2, null_log_like_arr, null_Q2_values, null_S2_values,
        null_lifetime2_values, null_jitter_values,
        xlabel_freq="Frequency [1/d]",
        title_suffix="second oscillator",
        filename=f"Null_GP_diagnostics_{name}_oscillator2.png",
        null=True,
    )


def plot_gp_prediction(t: np.ndarray,
                       y: np.ndarray,
                       yerr: np.ndarray,
                       t_pred: np.ndarray,
                       mu: np.ndarray,
                       std: np.ndarray,
                       residuals: np.ndarray,
                       name: str) -> None:
    """Plot the GP prediction over the data and the residuals below it."""
    fig = plt.figure(figsize=(10, 6), dpi=200)
    gs = gridspec.GridSpec(3, 1)

    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0, label="Data")
    ax1.plot(t_pred, mu, "r-", label="GP prediction")
    ax1.fill_between(t_pred, mu - std, mu + std,
                     color="r", alpha=0.2, label="1σ uncertainty")
    ax1.set_xlabel("Time [d]")
    ax1.set_ylabel("RV [m/s]")
    ax1.legend()

    ax2 = fig.add_subplot(gs[2, 0])
    gp_rms = np.sqrt(np.mean(residuals ** 2))
    ax2.scatter(t, residuals, c="k", s=10,
                label=f"GP residual RMS = {gp_rms:.2f} m/s")
    ax2.axhline(0, color="r", linestyle="--")
    ax2.set_xlabel("Time [d]")
    ax2.set_ylabel("Residuals [m/s]")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(f"{name}_gp_prediction_residuals.png")
    plt.close(fig)

def compute_initial_params_guess(w0_list: np.ndarray,
                                  rms_scatter: float) -> pd.DataFrame:
    """Build a DataFrame of analytical initial parameters for the double-SHO
    GP, one row per frequency, without running a grid-search.

    Initial values
    --------------
    jitter  = rms_scatter / 2
    Q       = 10  (both oscillators)
    S       = (2π / w0) * rms_scatter² * π  = 2π² * rms_scatter² / w0
    """
    log_Q = np.log(10.0)
    log_sigma = np.log(rms_scatter / 2.0)

    records = []
    for w0 in w0_list:
        S = 2.0 * np.pi ** 2 * rms_scatter ** 2 / w0
        log_S0 = np.log(S)
        records.append({
            "w0":       w0,
            "log_S0_1": log_S0,
            "log_Q1":   log_Q,
            "log_S0_2": log_S0,
            "log_Q2":   log_Q,
            "log_sigma": log_sigma,
        })
    return pd.DataFrame(records)
