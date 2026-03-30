"""
GP_main.py
==========
Entry-point for the GP Periodogram pipeline.

Usage
-----
    python GP_main.py

All configuration lives in the CONFIG block below.  The script
1. Loads the RV time series.
2. Computes diagnostic plots (dt, uncertainty, timeseries).
3. Fits the null model (constant + jitter).
4. Runs the GLS periodogram (Astropy and mlp/Zechmeister).
5. Runs a bootstrap FAP estimation for the GLS.
6. Runs the grid-search for GP initial parameters (or loads an existing
   CSV if RUN_GRIDSEARCH is False).
7. Runs the double-SHO GP periodogram.
8. Produces all GP periodogram plots.
9. Builds the best-fit GP prediction and saves residuals.

Reference
---------
Daspute et al. (2026) — Interpretable Gaussian Process Periodogram.
"""

import os
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

import mlp  # local GLS implementation — required as-is



# =============================================================================
# CONFIG — edit this block to point at your data and tune the run
# =============================================================================

NAME = "activity RV asymmetric two spot region configuration double sho"
#NAME = "timeseries single SHO kernel"

#Kernel model: "double_sho" uses two SHO terms; "single_sho" uses one.
KERNEL = "double_sho"  # "single_sho" | "double_sho"

# Path to input time series CSV (must contain columns 't', 'y', 'yerr')
TIMESERIES_CSV = "timeseries activity RV asymmetric TWO spot region configuration.csv" # "timeseries single SHO kernel.csv"

#path to store girdsearch results. 
GRIDSEARCH_CSV = f"optimal_parameters {NAME}.csv"

RUN_GRIDSEARCH = True  # Run gridsearch on intial parameters if True. It is slow and reliable. False flag uses same initial parameters for all frequencies for faster execution. 

# Number of bootstrap iterations for the GLS FAP estimation
N_BOOT_GLS = 100

# Number of grid points per parameter axis in the gridsearch
GRIDSEARCH_N_POINTS = 7

# Frequency grid limits
# w0min is derived from the data timespan; w0max is set so that
# frequency ≤ 1/day (aliasing starts above the Nyquist of daily sampling).
W0MAX_PERIOD_DAYS = 1.0  # minimum period to probe [days]

# Frequency axis limits for zoomed plots
XLIM_FULL = (0.0, 1.0)
XLIM_ZOOM = (0.0, 0.1)

#Oversampling of frequency grid by a factor of 10 compared to resolution of periodogram is recommended for precision and correct inference. 
OVERSAMPLING_FACTOR = 10
# Output CSV for GP periodogram results
GP_RESULTS_CSV = f"{NAME}_optimized_1D_GP_periodogram.csv"

# Residuals CSV
RESIDUALS_CSV = f"{NAME}_residuals_double_SHO.csv"

# =============================================================================
# END CONFIG
# =============================================================================


#Conditional import block

from GP_Periodogram_functions import (
    compute_null_model,
    bootstrap_fap,
    plot_timeseries,
    plot_gls_dlnl,
    plot_gls_diagnostics,
    plot_gp_prediction,
)

if KERNEL == "double_sho":
    from GP_Periodogram_functions import (
        gridsearch_initial_params,
        compute_initial_params_guess,   # ADD THIS
        run_gp_periodogram,
        build_results_dataframe,
        gp_predict_best,
        plot_gp_periodogram,
        plot_lifetimes_transparent,
    )
elif KERNEL == "single_sho":
    from GP_Periodogram_functions_sho import (
        gridsearch_initial_params,
        compute_initial_params_guess,   # ADD THIS
        run_gp_periodogram,
        build_results_dataframe,
        gp_predict_best,
        plot_gp_periodogram,
        plot_lifetimes_transparent,
    )
else:
    raise ValueError(f"Unknown KERNEL: '{KERNEL}'. Choose 'single_sho' or 'double_sho'.")


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"[1/9] Loading data from '{TIMESERIES_CSV}'")
    df_raw = pd.read_csv(TIMESERIES_CSV)
    t = df_raw["t"].values
    y = df_raw["y"].values
    yerr = df_raw["yerr"].values

    #dt = pd.Series(t).diff().dropna()
    #median_dt = float(np.median(dt))
    rms_scatter = float(np.sqrt(np.mean((y - np.mean(y)) ** 2)))

    print(f"  N obs         : {len(t)}")
    #print(f"  Median dt     : {median_dt:.3f} d")
    print(f"  Median yerr   : {np.median(yerr):.3f} m/s")
    print(f"  RMS scatter   : {rms_scatter:.3f} m/s")

    # ------------------------------------------------------------------
    # 2. Diagnostic plots
    # ------------------------------------------------------------------
    print("[2/9] Diagnostic plots")
    #plot_dt(dt, NAME)
    #plot_uncertainty(yerr, NAME)
    plot_timeseries(t, y, yerr, rms_scatter, NAME)

    # ------------------------------------------------------------------
    # 3. Null model (constant + jitter)
    # ------------------------------------------------------------------
    print("[3/9] Computing null model")
    null_log_likelihood, weighted_mean, optimal_jitter = compute_null_model(y, yerr)
    print(f"  Null lnL      : {null_log_likelihood:.4f}")
    print(f"  Weighted mean : {weighted_mean:.4f}")
    print(f"  Optimal jitter: {optimal_jitter:.4f} m/s")

    # ------------------------------------------------------------------
    # 4. Frequency grid
    # ------------------------------------------------------------------
    timespan_obs = float(np.max(t) - np.min(t))
    w0min = 2*np.pi / (timespan_obs)
    w0max = 2.0 * np.pi / W0MAX_PERIOD_DAYS
    w0_list = np.arange(w0min, w0max, w0min)
    frequency = w0_list / (2.0 * np.pi)
    print(f"  Timespan      : {timespan_obs:.1f} d")
    print(f"  Freq grid     : {len(w0_list)} points  "
          f"[{frequency[0]:.5f}, {frequency[-1]:.4f}] 1/d")

    # ------------------------------------------------------------------
    # 5. GLS periodogram
    # ------------------------------------------------------------------
    #print("[4/9] GLS periodogram (Astropy)")
    #ls = LombScargle(t, y, yerr)
    #power = ls.power(frequency)
    #power_fap_1pct = float(ls.false_alarm_level(0.01))
    #power_fap_10pct = float(ls.false_alarm_level(0.10))
    #plot_gls_astropy(frequency, power, power_fap_1pct, power_fap_10pct, NAME)

    print("[4/9] GLS periodogram (mlp/Zechmeister, Δ lnL normalisation)")
    data_tuple = (t, y, yerr)
    gls = mlp.Gls(
        [data_tuple],
        fbeg=frequency[0],
        fend=frequency[-1],
        norm="dlnL",
        freq=frequency,
        verbose=True,
    )
    plot_gls_dlnl(gls.freq, gls.power, NAME,
                  xlim_full=XLIM_FULL, xlim_zoom=XLIM_ZOOM)
    plot_gls_diagnostics(gls, NAME)

    # ------------------------------------------------------------------
    # 6. Bootstrap FAP for GLS
    # ------------------------------------------------------------------
    print("[5/9] Bootstrap FAP for GLS")
    fap_1pct_gls, fap_10pct_gls = bootstrap_fap(
        t, y, yerr, frequency, jitter=optimal_jitter, n_boot=N_BOOT_GLS
    )
    print(f"  GLS FAP 1%  threshold  : {fap_1pct_gls:.3f} Δ lnL")
    print(f"  GLS FAP 10% threshold  : {fap_10pct_gls:.3f} Δ lnL")


    # ------------------------------------------------------------------
    # 8. Grid-search for GP initial parameters
    # ------------------------------------------------------------------
    if RUN_GRIDSEARCH:
        print("[7/9] Grid-search for GP initial parameters")
        df_initial = gridsearch_initial_params(
            t, y, yerr, w0_list, weighted_mean, null_log_likelihood,
            NAME, rms_scatter, n_points=GRIDSEARCH_N_POINTS,
        )
    else:
        print("[7/9] Using analytical initial parameter guess (no grid-search)")
        df_initial = compute_initial_params_guess(w0_list, rms_scatter)
        print(f"  Built {len(df_initial)} rows from analytical guess")

    # ------------------------------------------------------------------
    # 9. GP periodogram
    # ------------------------------------------------------------------
    print("[8/9] Running double-SHO GP periodogram")
    w0_list = np.arange(w0min, w0max, w0min/OVERSAMPLING_FACTOR)
    
    result = run_gp_periodogram(
        t, y, yerr, w0_list, weighted_mean, null_log_likelihood, df_initial
    )

    df_results = build_results_dataframe(result, rms_scatter)
    df_results.to_csv(GP_RESULTS_CSV, index=False)
    print(f"  Saved GP results to '{GP_RESULTS_CSV}'")

    # Convenience arrays used repeatedly in plotting
    Frequency_val       = df_results["Frequency_1"].values
    delta_log_like      = df_results["delta_log_like"].values
    Q0_values           = df_results["Q_1"].values
    S0_values           = df_results["S0_1"].values
    lifetime0_values    = df_results["lifetime_1"].values
    jitter_values       = df_results["jitter"].values
    null_log_like_arr   = df_results["null_log_like"].values
    null_Q0_values      = df_results["null_Q_1"].values
    null_S0_values      = df_results["null_S0_1"].values
    null_lifetime0_values = 2.0 * df_results["null_Q_1"].values / df_results["w0"].values
    null_jitter_values  = df_results["null_jitter"].values

    if KERNEL == "double_sho":
        Frequency_val2        = df_results["Frequency_2"].values
        Q2_values             = df_results["Q_2"].values
        S2_values             = df_results["S0_2"].values
        lifetime2_values      = df_results["lifetime_2"].values
        null_Q2_values        = df_results["null_Q_2"].values
        null_S2_values        = df_results["null_S0_2"].values
        null_lifetime2_values = 2.0 * df_results["null_Q_2"].values / df_results["w_2"].values

    best_idx = int(df_results["delta_log_like"].idxmax())
    best_freq = Frequency_val[best_idx]
    best_period = 1.0 / best_freq
    best_dlnl = delta_log_like[best_idx]
    print(f"  Best frequency : {best_freq:.5f} 1/d  "
          f"(P = {best_period:.2f} d,  Δ lnL = {best_dlnl:.2f})")

    # ------------------------------------------------------------------
    # 10. Plots
    # ------------------------------------------------------------------
    print("[9/9] Generating plots")

    plot_gp_periodogram(
        Frequency_val, delta_log_like, df_results, NAME,
        xlim_full=XLIM_FULL, xlim_zoom=XLIM_ZOOM,
    )

    if KERNEL == "double_sho":
        plot_lifetimes_transparent(
            Frequency_val, Frequency_val2,
            lifetime0_values, lifetime2_values,
            df_results, timespan_obs, NAME,
            xlim_full=XLIM_FULL, xlim_zoom=XLIM_ZOOM,
        )
    else:
        plot_lifetimes_transparent(
            Frequency_val,
            lifetime0_values,
            df_results, timespan_obs, NAME,
            xlim_full=XLIM_FULL, xlim_zoom=XLIM_ZOOM,
        )

    #plot_lifetimes_colored(
    #    Frequency_val, Frequency_val2,
    #    lifetime0_values, lifetime2_values,
    #    df_results, timespan_obs, NAME,
    #    xlim_full=XLIM_FULL, xlim_zoom=XLIM_ZOOM,
    #)

    #plot_oscillator_diagnostics(
    #    Frequency_val, Frequency_val2,
    #    delta_log_like,
    #    Q0_values, S0_values, lifetime0_values,
    #    Q2_values, S2_values, lifetime2_values,
    #    jitter_values,
    #    null_log_like_arr,
    #    null_Q0_values, null_S0_values, null_lifetime0_values,
    #    null_Q2_values, null_S2_values, null_lifetime2_values,
    #    null_jitter_values,
    #    NAME,
    #)

    # ------------------------------------------------------------------
    # 11. GP prediction and residuals at best-fit frequency
    # ------------------------------------------------------------------
    print("    Building GP prediction at best-fit frequency")
    t_pred, mu, std, residuals = gp_predict_best(
        t, y, yerr, df_results, weighted_mean
    )
    plot_gp_prediction(t, y, yerr, t_pred, mu, std, residuals, NAME)

    pd.DataFrame({
        "t[d]": t,
        "Residual[m/s]": residuals,
        "E_residuals[m/s]": yerr,
    }).to_csv(RESIDUALS_CSV, index=False)
    print(f"  Saved residuals to '{RESIDUALS_CSV}'")

    print("\nDone.")


if __name__ == "__main__":
    main()
