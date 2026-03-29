# Gaussian Process Periodogram

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

An open-source Gaussian Process (GP) periodogram for identifying and differentiating **periodic** and **quasi-periodic** signals in unevenly sampled, noisy, and scarce time series data.

Developed by M. Daspute, L. Tal-Or, M. Perger, P. Chaturvedi, I. Ribas, and Sreenivasan K. (Daspute et al. 2026, Astronomy & Astrophysics to be submitted).

---


## Installation and Usage

Clone the repository and install dependencies:

```bash
git clone https://github.com/mangeshdaspute/Gaussian-Process-Periodogram.git
cd Gaussian-Process-Periodogram
pip install -r requirements.txt
```
The default configuration is designed for detecting rotational and Keplerian signals in radial velocity timeseries.
Open GP_Periodogram_main.py and edit the configuration code block, which contains - Name of the simulation, Kernel, CSV timeseries filename, Oversampling factor, etc.  

Run code with:
```bash
python3 GP_Periodogram_main.py
```

---

## Overview

Standard periodogram tools such as the Generalized Lomb-Scargle (GLS) periodogram fit a strictly sinusoidal model to the data. They are well-suited for detecting stable, coherent periodic signals (e.g., Keplerian orbits), but perform poorly on signals that evolve in phase and amplitude over time, such as rotationally modulated stellar activity.

This code implements a GP periodogram using a **double SHO (Simple Harmonic Oscillator) kernel** from the [`celerite`](https://github.com/dfm/celerite) package. Rather than fitting a sine wave, it fits a covariance model to the data at each trial frequency, quantifying how well a quasi-periodic or periodic signal of that frequency explains the observed time series. The figure of merit is the **delta log-likelihood (ΔlnL)**: the difference between the GP log-likelihood and the log-likelihood of a constant null model.

### Why a double SHO kernel?

The double SHO kernel consists of two coupled SHO terms. The second oscillator is fixed at twice the frequency of the first. This is physically motivated by the observation that rotationally modulated stellar activity often produces power at both the rotational frequency and its first harmonic. The independent quality factors (Q) of the two oscillators allow the periodogram to distinguish:

- **Keplerian (planetary) signals**: stable, long-lived, captured by one oscillator; the other shuts down.
- **Quasi-periodic stellar activity signals**: split between both oscillators, with finite lifetimes on the order of a few multiples of the rotational period.

The kernel covariance function in the limit of large Q is:

```
k(τ) ≈ S₀ ω₀ Q₀ exp(−ω₀τ / 2Q₀) cos(ω₀τ)
      + S₁ ω₁ Q₁ exp(−ω₁τ / 2Q₁) cos(ω₁τ)
```

where `ω` is angular frequency, `S` is the power spectral density amplitude, `Q` is the quality factor, and `τ = |tᵢ − tⱼ|` is the time lag between observations.

---

## Key Features

- **Detects quasi-periodic signals** that change phase and amplitude over time, where GLS fails or detects harmonics instead of the true period.
- **Identifies the true rotational period** rather than its harmonic more reliably than GLS.
- **Differentiates Keplerian from activity signals** using the optimised kernel parameters: lifetime, quality factor, and amplitude of each oscillator.
- **Color-coded periodogram output**: the RMS fraction of the first oscillator encodes signal origin — red indicates the first oscillator captures most of the variance (Keplerian), green/blue indicates shared contribution between oscillators (activity).
- **Lifetime periodogram**: a companion plot showing the signal decay lifetime at each trial frequency for both oscillators.
- **Parallel execution** by default using all available CPU cores via `joblib`.
- Outputs CSV files with full results per frequency and diagnostic PNG plots.
- Single SHO kernel mode also available for general quasi-periodic signal detection beyond astrophysics.

---

## Application Example: Stellar Radial Velocity

The primary application demonstrated in the accompanying paper is identifying the **rotational period of stars** and **Keplerian exoplanetary signals** in stellar radial velocity (RV) time series. Rotationally modulated star spots produce quasi-periodic RV variations that can mimic planetary signals, leading to false positives or obscuring true exoplanet detections.

---

## Functions

#### `process_w0(w0, y, t, yerr, timespan_obs, weighted_mean_value, null_log_like, df)`

Core function of the GP periodogram. Builds and optimises the double SHO GP model at a single trial angular frequency `w0`, using initial parameters loaded from the grid search CSV file. Runs `neg_log_like` minimisation via L-BFGS-B. Computes and returns ΔlnL, optimised kernel parameters, RV amplitudes, lifetimes, quality factors, jitter, and RMS fractions for both oscillators.

| Argument | Type | Description |
|---|---|---|
| `w0` | `float` | Trial angular frequency [rad/day]. |
| `y` | `np.ndarray` | Observed data values. |
| `t` | `np.ndarray` | Observation timestamps. |
| `yerr` | `np.ndarray` | Per-observation measurement uncertainties. |
| `timespan_obs` | `float` | Total observation timespan in days. |
| `weighted_mean_value` | `float` | Weighted mean of the data used as GP mean. |
| `null_log_like` | `float` | Null model log-likelihood for ΔlnL computation. |
| `df` | `pd.DataFrame` | Grid search results DataFrame; supplies initial kernel parameters. |

**Returns:** `dict` or `pd.Series` containing ΔlnL, `log_S0_1`, `log_Q1`, lifetime₁, amplitude₁, `log_S0_2`, `log_Q2`, lifetime₂, amplitude₂, jitter, RMS fraction₁, RMS fraction₂.

---

#### `compute_for_w0(w0, t, y, yerr, weighted_mean_value, null_log_likelihood_value, log_S0_grid, log_Q_grid, log_sigma_grid)`

Performs the full nested grid search over kernel parameters at a single trial angular frequency `w0`. Constructs a double SHO kernel with `term1` at `w0` and `term2` at `2×w0` (both with frozen `log_omega0`), plus a `JitterTerm`. Evaluates ΔlnL at every combination of grid points and returns a single-row DataFrame with the parameter combination that maximises ΔlnL.

| Argument | Type | Description |
|---|---|---|
| `w0` | `float` | Trial angular frequency [rad/day] of the first oscillator. |
| `t` | `np.ndarray` | Observation timestamps. |
| `y` | `np.ndarray` | Observed data values. |
| `yerr` | `np.ndarray` | Per-observation measurement uncertainties. |
| `weighted_mean_value` | `float` | Weighted mean of `y` used as the GP mean. |
| `null_log_likelihood_value` | `float` | Null model log-likelihood for computing ΔlnL. |
| `log_S0_grid` | `np.ndarray` | Grid of `log_S0` values to search (applied to both oscillators). |
| `log_Q_grid` | `np.ndarray` | Grid of `log_Q` values to search (applied to both oscillators). |
| `log_sigma_grid` | `np.ndarray` | Grid of `log_sigma` (jitter) values to search. |

**Returns:** `pd.DataFrame` — single row with columns `w0`, `log_S0_1`, `log_Q1`, `log_S0_2`, `log_Q2`, `log_sigma`, `delta_ll`.

---

## Interpreting the Output

### GP Periodogram Plot

- **X-axis**: Frequency of the first oscillator [1/d].
- **Y-axis**: ΔlnL — log-likelihood of the GP model relative to the null model.
- **Color**: RMS fraction of the first oscillator relative to the total data RMS.
  - **Red**: First oscillator captures most of the variance → Keplerian/periodic signal at the trial frequency.
  - **Blue/Purple**: Second oscillator captures most of the variance → the real signal is at twice the trial frequency; this peak is an alias/duplicate.

### Lifetime Plot

- **X-axis**: Frequency of each oscillator [1/d].
- **Y-axis**: Signal lifetime [days] = `QP/π`.
- **Red circles**: Lifetime of the first oscillator.
- **Blue squares**: Lifetime of the second oscillator.
- **Marker opacity**: Proportional to RMS fraction. Opaque markers indicate an oscillator that captures significant variance.
- **Black dashed line**: Observation timespan. Lifetimes exceeding this by a factor of 2 indicate a stable, coherent signal (consistent with a Keplerian orbit). Lifetimes comparable to a few oscillator periods indicate a quasi-periodic signal (consistent with stellar activity).

### Signal Classification Heuristics

| Property | Circular Planet / Periodic | Stellar Activity / Quasi-periodic | Eccentric Planet |
|---|---|---|---|
| First oscillator lifetime | > 10 * timespan | period of oscillator to 10 * timespan | > 10 * timespan |
| Second oscillator lifetime | < period of oscillator | period of oscillator to 10 * timespan | > 10 * timespan |
| RMS fraction of second oscillator | ≈ 0 | Significant |  Significant |
| Which Oscillator has higher RMS fraction | First | Second |  First |






