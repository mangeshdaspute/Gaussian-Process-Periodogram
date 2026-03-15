# Gaussian Process Periodogram

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

An open-source Gaussian Process (GP) periodogram for identifying and differentiating **periodic** and **quasi-periodic** signals in unevenly sampled, noisy, and scarce time series data.

Developed by M. Daspute, L. Tal-Or, M. Perger, P. Chaturvedi, I. Ribas, and Sreenivasan K. (Daspute et al. 2026, *Astronomy & Astrophysics*, submitted).

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
