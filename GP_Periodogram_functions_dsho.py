"""
GP_Periodogram_functions_dsho.py
=================================
Dependent double-SHO (dSHO) kernel variant of the GP Periodogram pipeline.

The dSHO kernel is a double-SHO where the quality factor of the second
oscillator is constrained to be exactly twice that of the first:

    Q_2 = 2 * Q_1

This means both oscillators share the same damping lifetime:

    tau = 2*Q_1 / w0 = 2*Q_2 / (2*w0)

The kernel therefore has one fewer free hyper-parameter than the unconstrained
double-SHO.  Free parameters per frequency:
    log_S0_1, log_Q1, log_S0_2, log_sigma  (4 total)

Exposes the same public interface as GP_Periodogram_functions.py so that
GP_Periodogram_main.py can import from any of the three kernel modules
transparently:

    gridsearch_initial_params
    compute_initial_params_guess
    run_gp_periodogram
    build_results_dataframe
    gp_predict_best
    plot_gp_periodogram
    plot_lifetimes_transparent

All shared functions (null model, bootstrap FAP, GLS/timeseries plots, GP
prediction plot) live in GP_Periodogram_functions and are imported there.

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
import mlp
import celerite
from celerite import terms
from scipy.optimize import minimize
from joblib import Parallel, delayed
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _neg_log_like(params: np.ndarray, y: np.ndarray, gp) -> float:
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)


def _neg_log_like_dsho(params4: np.ndarray, y: np.ndarray, gp) -> float:
    """Wrapper for dSHO optimisation.

    Accepts 4 free parameters [log_S0_1, log_Q1, log_S0_2, log_sigma],
    derives log_Q2 = log_Q1 + ln(2), assembles the full 5-parameter vector,
    and evaluates the GP log-likelihood.
    """
    log_S0_1, log_Q1, log_S0_2, log_sigma = params4
    log_Q2 = log_Q1 + np.log(2.0)
    full_params = np.array([log_S0_1, log_Q1, log_S0_2, log_Q2, log_sigma])
    gp.set_parameter_vector(full_params)
    return -gp.log_likelihood(y)


# ---------------------------------------------------------------------------
# Grid-search for GP initial parameters — dSHO
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
    """Evaluate the dSHO GP log-likelihood on a parameter grid for a fixed
    angular frequency *w0*.

    The grid has 4 axes (log_S0_1, log_Q1, log_S0_2, log_sigma); log_Q2 is
    derived at each point as log_Q2 = log_Q1 + ln(2).

    Returns a single-row DataFrame with the parameter combination that
    maximises Δ lnL.
    """
    # Build the kernel with both omegas frozen; Q2 is managed via full vector
    term1 = terms.SHOTerm(log_S0=0, log_Q=0, log_omega0=np.log(w0))
    term1.freeze_parameter("log_omega0")
    term2 = terms.SHOTerm(log_S0=0, log_Q=0, log_omega0=np.log(2.0 * w0))
    term2.freeze_parameter("log_omega0")
    jitter_term = terms.JitterTerm(log_sigma=-2)
    kernel = term1 + term2 + jitter_term

    gp = celerite.GP(kernel, mean=weighted_mean)
    gp.compute(t, yerr)

    best_delta_ll = -np.inf
    best_record = None

    for log_S0_1 in log_S0_grid:
        for log_Q1 in log_Q_grid:
            log_Q2 = log_Q1 + np.log(2.0)
            for log_S0_2 in log_S0_grid:
                for log_sigma in log_sigma_grid:
                    gp.set_parameter_vector(
                        [log_S0_1, log_Q1, log_S0_2, log_Q2, log_sigma]
                    )
                    try:
                        log_like = gp.log_likelihood(y)
                    except Exception:
                        continue
                    delta_ll = log_like - null_log_likelihood
                    if delta_ll > best_delta_ll:
                        best_delta_ll = delta_ll
                        best_record = {
                            "w0":        w0,
                            "log_S0_1":  log_S0_1,
                            "log_Q1":    log_Q1,
                            "log_S0_2":  log_S0_2,
                            "log_sigma": log_sigma,
                            "delta_ll":  delta_ll,
                        }

    if best_record is None:
        # Fallback: first grid point
        best_record = {
            "w0":        w0,
            "log_S0_1":  log_S0_grid[0],
            "log_Q1":    log_Q_grid[0],
            "log_S0_2":  log_S0_grid[0],
            "log_sigma": log_sigma_grid[0],
            "delta_ll":  -np.inf,
        }

    return pd.DataFrame([best_record])


def gridsearch_initial_params(t: np.ndarray,
                               y: np.ndarray,
                               yerr: np.ndarray,
                               w0_list: np.ndarray,
                               weighted_mean: float,
                               null_log_likelihood: float,
                               name: str,
                               rms_scatter: float,
                               n_points: int = 7) -> pd.DataFrame:
    """Grid-search the initial dSHO kernel parameters over a frequency grid.

    For each angular frequency in *w0_list*, searches over a 4-D grid of
    (log_S0_1, log_Q1, log_S0_2, log_sigma) with log_Q2 = log_Q1 + ln(2)
    enforced at every grid point.

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
        Used to derive the S0 grid range.
    n_points : int
        Number of grid points per parameter axis.

    Returns
    -------
    optimal_df : pd.DataFrame
        One row per entry in *w0_list* with the best-fit initial parameters.
    """
    s0guessMax=1 
    log_S0_grid   = np.linspace(-15, 15, n_points)
    log_Q_grid    = np.linspace(np.log(0.502), 15, 7)
    log_sigma_grid = np.linspace(np.log(rms_scatter/10), np.log(rms_scatter), 7)

    all_results = Parallel(n_jobs=-2)(
        delayed(_compute_for_w0_gridsearch)(
            w0, t, y, yerr, weighted_mean, null_log_likelihood,
            log_S0_grid, log_Q_grid, log_sigma_grid,
        )
        for w0 in tqdm(w0_list, desc="Grid-search initial parameters (dSHO)")
    )

    optimal_df = pd.concat(all_results, ignore_index=True)
    optimal_df.to_csv(f"optimal_parameters {name}.csv", index=False)

    # Diagnostic: Q1 vs frequency
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(
        optimal_df["w0"] / (2 * np.pi),
        np.exp(optimal_df["log_Q1"]),
        "o", label="Q1 (Q2 = 2×Q1)",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [1/d]")
    ax.set_ylabel("log Q")
    ax.legend()
    ax.set_title("Optimal Q1 vs frequency (dSHO: Q2 = 2×Q1)")
    fig.savefig(f"Q_vs_frequency {name}.png", dpi=300)
    plt.close(fig)

    # Diagnostic: S0 and S1 vs frequency
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(
        optimal_df["w0"] / (2 * np.pi),
        np.exp(optimal_df["log_S0_1"]),
        "o", label="log_S0",
    )
    ax.plot(
        optimal_df["w0"] / (2 * np.pi),
        np.exp(optimal_df["log_S0_2"]),
        ".", label="log_S1",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [1/d]")
    ax.set_ylabel("log_S")
    ax.legend()
    ax.set_title("Optimal log_S0 log_S1 vs frequency")
    fig.savefig(f"logS0_log_S1_vs_frequency {name}.png", dpi=300)
    plt.close(fig)

    # Diagnostic: Q1 vs frequency
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(
        optimal_df["w0"] / (2 * np.pi),
        np.exp(optimal_df["log_sigma"]),
        "o", label="log Jitter",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [1/d]")
    ax.set_ylabel("log Jitter")
    ax.legend()
    ax.set_title("Optimal log Jitter vs frequency")
    fig.savefig(f"log Jitter_vs_frequency {name}.png", dpi=300)
    plt.close(fig)

    return optimal_df


# ---------------------------------------------------------------------------
# GP periodogram — per-frequency optimisation — dSHO
# ---------------------------------------------------------------------------

def _process_single_w0(w0: float,
                        y: np.ndarray,
                        t: np.ndarray,
                        yerr: np.ndarray,
                        weighted_mean: float,
                        null_log_likelihood: float,
                        df_initial: pd.DataFrame) -> list:
    """Optimise the dSHO GP kernel at a fixed angular frequency *w0*.

    Uses the pre-computed grid-search result (``df_initial``) to warm-start
    the L-BFGS-B optimisation over 4 free parameters:
        [log_S0_1, log_Q1, log_S0_2, log_sigma]

    log_Q2 is derived internally as log_Q1 + ln(2) at every function
    evaluation.

    Returns
    -------
    list
        [S0_1, S0_2, Q_1, Q_2, delta_log_like, w0, w_2, jitter]
    """
    log_Q_lowest = np.log(0.501)

    # -- Retrieve warm-start parameters from the grid-search table ----------
    closest_idx = (df_initial["w0"] - w0).abs().idxmin()
    log_S0_1  = float(df_initial.loc[closest_idx, "log_S0_1"])
    log_Q1    = float(df_initial.loc[closest_idx, "log_Q1"])
    log_S0_2  = float(df_initial.loc[closest_idx, "log_S0_2"])
    log_sigma = float(df_initial.loc[closest_idx, "log_sigma"])

    # -- Build the kernel (Q2 is not frozen; managed via full param vector) -
    log_Q2 = log_Q1 + np.log(2.0)

    bounds_sho = {
        "log_S0":    (-15, 15),
        "log_Q":     (log_Q_lowest, 15),
        "log_omega0": (-15, 15),
    }
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

    # -- Optimise over 4 free params with Q2 = 2*Q1 enforced in wrapper ----
    # Bounds: [log_S0_1, log_Q1, log_S0_2, log_sigma]
    bounds4 = [(-15, 15), (log_Q_lowest, 15), (-15, 15), (-15, 15)]
    x0 = [log_S0_1, log_Q1, log_S0_2, log_sigma]

    r = minimize(
        _neg_log_like_dsho,
        x0,
        method="L-BFGS-B",
        bounds=bounds4,
        args=(y, gp),
    )

    log_S0_1_opt, log_Q1_opt, log_S0_2_opt, log_sigma_opt = r.x
    log_Q2_opt = log_Q1_opt + np.log(2.0)

    # Set final parameter vector so gp.log_likelihood returns the optimum
    gp.set_parameter_vector(
        [log_S0_1_opt, log_Q1_opt, log_S0_2_opt, log_Q2_opt, log_sigma_opt]
    )

    S0_1   = np.exp(log_S0_1_opt)
    Q_1    = np.exp(log_Q1_opt)
    S0_2   = np.exp(log_S0_2_opt)
    Q_2    = np.exp(log_Q2_opt)          # = 2 * Q_1
    w_2    = 2.0 * w0
    jitter = np.exp(log_sigma_opt)

    delta_log_like = gp.log_likelihood(y) - null_log_likelihood

    return [S0_1, S0_2, Q_1, Q_2, delta_log_like, w0, w_2, jitter]


def run_gp_periodogram(t: np.ndarray,
                       y: np.ndarray,
                       yerr: np.ndarray,
                       w0_list: np.ndarray,
                       weighted_mean: float,
                       null_log_likelihood: float,
                       df_initial: pd.DataFrame) -> list:
    """Run the dSHO GP periodogram over the full frequency grid in parallel.

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
        for w0 in tqdm(w0_list, desc="GP periodogram (dSHO)")
    )
    return result


# ---------------------------------------------------------------------------
# Derived quantities from the raw periodogram result — dSHO
# ---------------------------------------------------------------------------

def build_results_dataframe(result: list, rms_scatter: float) -> pd.DataFrame:
    """Convert the raw list returned by :func:`run_gp_periodogram` into a
    tidy DataFrame with all derived quantities.

    Because Q_2 = 2*Q_1 and omega_2 = 2*omega_1, the two oscillators share
    a common lifetime: tau = 2*Q_1/w0 = 2*Q_2/w_2.

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

    df["Frequency_1"] = df["w0"]  / (2.0 * np.pi)
    df["Frequency_2"] = df["w_2"] / (2.0 * np.pi)
    df["Period_1"]    = 2.0 * np.pi / df["w0"]
    df["Period_2"]    = 2.0 * np.pi / df["w_2"]

    # Shared lifetime (identical for both oscillators by construction)
    df["lifetime_1"] = 2.0 * df["Q_1"] / df["w0"]
    df["lifetime_2"] = 2.0 * df["Q_2"] / df["w_2"]   # == lifetime_1

    df["Amplitude_1[m/s]"] = np.sqrt(2.0 * df["Q_1"] * df["w0"]  * df["S0_1"])
    df["Amplitude_2[m/s]"] = np.sqrt(2.0 * df["Q_2"] * df["w_2"] * df["S0_2"])

    norm_factor = math.sqrt(2.0) * rms_scatter
    df["fraction_RMS1[m/s]"] = df["Amplitude_1[m/s]"] / norm_factor
    df["fraction_RMS2[m/s]"] = df["Amplitude_2[m/s]"] / norm_factor

    return df


# ---------------------------------------------------------------------------
# GP prediction at the best-fit frequency — dSHO
# ---------------------------------------------------------------------------

def gp_predict_best(t: np.ndarray,
                    y: np.ndarray,
                    yerr: np.ndarray,
                    df_results: pd.DataFrame,
                    weighted_mean: float) -> tuple:
    """Reconstruct the dSHO GP at the highest-Δ lnL frequency and return
    predictions on a dense grid.

    Returns
    -------
    t_pred : np.ndarray
    mu : np.ndarray
    std : np.ndarray
    residuals : np.ndarray  (evaluated at the original time stamps)
    """
    best = df_results.loc[df_results["delta_log_like"].idxmax()]

    S0_1   = float(best["S0_1"])
    Q_1    = float(best["Q_1"])
    S0_2   = float(best["S0_2"])
    Q_2    = float(best["Q_2"])   # == 2 * Q_1
    w0     = float(best["w0"])
    w_2    = float(best["w_2"])
    jitter = float(best["jitter"])

    log_Q_lowest = np.log(0.501)
    bounds = {
        "log_S0":     (-15, 15),
        "log_Q":      (log_Q_lowest, 15),
        "log_omega0": (-15, 15),
    }

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
# Analytical initial parameter guess — dSHO
# ---------------------------------------------------------------------------

def compute_initial_params_guess(w0_list: np.ndarray,
                                  rms_scatter: float) -> pd.DataFrame:
    """Build a DataFrame of analytical initial parameters for the dSHO GP,
    one row per frequency, without running a grid-search.

    Initial values
    --------------
    jitter  = rms_scatter / 2
    Q       = 10  (first oscillator; second is 2×)
    S       = 2π² * rms_scatter² / w0
    """
    log_Q    = np.log(10.0)
    log_sigma = np.log(rms_scatter / 2.0)

    records = []
    for w0 in w0_list:
        S       = 2.0 * np.pi ** 2 * rms_scatter ** 2 / w0
        log_S0  = np.log(S)
        records.append({
            "w0":        w0,
            "log_S0_1":  log_S0,
            "log_Q1":    log_Q,
            "log_S0_2":  log_S0,
            "log_sigma": log_sigma,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting — dSHO
# ---------------------------------------------------------------------------

def plot_gp_periodogram(Frequency_val: np.ndarray,
                        delta_log_like: np.ndarray,
                        df_results: pd.DataFrame,
                        name: str,
                        xlim_full: tuple = (0.0, 1.0),
                        xlim_zoom: tuple = (0.0, 0.1),
                        fap_1pct: float | None = None,
                        fap_10pct: float | None = None) -> None:
    """Plot the dSHO GP periodogram coloured by the RMS fraction of the first
    oscillator, at full range and zoomed in."""
    for xlim, suffix in [(xlim_full, ""), (xlim_zoom, "_zoomed")]:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.plot(Frequency_val, delta_log_like, color="gray",
                linewidth=0.5, alpha=0.5, zorder=1)
        sc = ax.scatter(
            Frequency_val, delta_log_like,
            c=df_results["fraction_RMS1[m/s]"],
            cmap="rainbow", s=10, edgecolor="none", zorder=2,
            vmin=0, vmax=1.1,
        )
        cbar = fig.colorbar(
            sc,
            cax=plt.gca().inset_axes([0.30, 0.96, 0.4, 0.035]),
            orientation="horizontal",
        )
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
        #ax.set_ylim([-5,120])

        #ax.grid(which="major", linestyle="-")
        #ax.minorticks_on()
        #ax.grid(which="minor", color="#DDDDDD", linestyle="--", alpha=0.6)
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
    """Plot the common dSHO kernel lifetime using a rainbow colourmap for the
    RMS fraction fraction_RMS1, at full range and
    zoomed in.

    Because Q_2 = 2*Q_1 and omega_2 = 2*omega_1, both oscillators share
    exactly the same lifetime, so only one curve is plotted.
    """
    #RMS fraction as colour metric
    frac_total = (
        df_results["fraction_RMS1[m/s]"].values
    )

    for xlim, suffix in [(xlim_full, ""), (xlim_zoom, "_zoomed")]:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

        ax.plot(Frequency_val, lifetime0_values,
                color="gray", linewidth=0.5, alpha=0.5, zorder=1)
        sc = ax.scatter(
            Frequency_val, lifetime0_values,
            c=frac_total,
            cmap="rainbow", s=10, edgecolor="none", zorder=3,
            vmin=0, vmax=1.1,
        )
        cbar = fig.colorbar(
            sc,
            cax=ax.inset_axes([0.30, 0.96, 0.4, 0.035]),
            orientation="horizontal",
        )
        cbar.set_label("RMS fraction of first oscillator", fontsize=9)
        cbar.ax.tick_params(labelsize=9)

        ax.axhline(timespan_obs, color="black", linestyle="--",
                   label="Timespan_obs")
        ax.set_xlabel("Frequency of first oscillator [1/d]")
        ax.set_ylabel("Lifetime [d]")
        ax.set_xlim(xlim)
        ax.set_yscale("log")
        #ax.grid(which="major", linestyle="-")
        #ax.minorticks_on()
        #ax.grid(which="minor", color="#DDDDDD", linestyle="--", alpha=0.6)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(f"lifetimes_{name}_transparent{suffix}.png")
        plt.close(fig)
