# coding: utf-8
"""
Improved KW (Kitsul-Wright) skewness estimation.

MC-validated best pipeline (see kw_mc_validation.py ablation study):
  1. Gaussian kernel (replaces Epanechnikov — smoother, no boundary kinks)
  2. LSCV bandwidth (leave-one-out cross-validation, adapts per date)
  3. Flat-vol tail extrapolation (smooth IV amplifies noise — validated by MC)
  4. Finer grid (200 -> 1600) + PCHIP spline quantile extraction
  5. Residual-based noise estimation for bootstrap
  6. No edge-band shutdown (Z_raw diagnostic instead)
  7. Asymmetry profile (multi-depth quantile decomposition)

Reads results/call_curves.pkl, computes improved KW Bowley skewness,
and compares with the original pipeline from options_implied_inflation_pdf.py.

Usage:
    python kw_improved_skewness.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import norm
from scipy.integrate import trapezoid, simpson, cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR    = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE_DIR, "results")
FIG_DIR     = os.path.join(BASE_DIR, "fig")
os.makedirs(FIG_DIR, exist_ok=True)

CALL_CURVES_PKL = os.path.join(DATA_DIR, "call_curves.pkl")
RESULTS_CSV     = os.path.join(DATA_DIR, "updated_results.csv")
OUTPUT_CSV      = os.path.join(DATA_DIR, "kw_improved_skewness.csv")

AREA_FOCUS = "EU"
RUN_BOOTSTRAP = True
N_BOOT = 50

# ============================================================
# NUMERICAL INTEGRATION
# ============================================================
def integrate(x, y):
    """Simpson's rule with trapezoid fallback."""
    if len(x) < 2:
        return np.nan
    if len(x) >= 3:
        try:
            return simpson(y, x=x)
        except Exception:
            pass
    return trapezoid(y, x)


# ============================================================
# IMPROVEMENT 1: GAUSSIAN KERNEL (replaces Epanechnikov)
# ============================================================
def gaussian_kernel(u):
    """Gaussian kernel: infinitely smooth, no boundary kinks."""
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)


def epanechnikov(u):
    """Original Epanechnikov kernel (for comparison)."""
    u = np.asarray(u, float)
    out = np.zeros_like(u)
    m = np.abs(u) <= 1.0
    out[m] = 0.75 * (1.0 - u[m]**2)
    return out


# ============================================================
# IMPROVEMENT 2: SMOOTH IV SPLINE TAIL EXTRAPOLATION
# ============================================================
def black76_call(K, F, sigma, B, T=1.0):
    """Black-76 call price."""
    K = np.asarray(K, float)
    sigma = float(sigma)
    if sigma <= 0:
        return np.maximum(B * (F - K), 0.0)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return B * (F * norm.cdf(d1) - K * norm.cdf(d2))


def implied_vol_from_call(C_market, K, F, B, T=1.0, vol_bounds=(0.001, 5.0)):
    """Invert Black-76 for implied vol."""
    C_market, K = float(C_market), float(K)
    intrinsic = max(B * (F - K), 0.0)
    if C_market <= intrinsic + 1e-12:
        return np.nan
    def obj(sigma):
        return float(black76_call(K, F, sigma, B, T)) - C_market
    try:
        return optimize.brentq(obj, vol_bounds[0], vol_bounds[1])
    except (ValueError, RuntimeError):
        return np.nan


def extrapolate_call_prices_flat(k_obs, C_obs, B, ypi,
                                  k_left=-0.20, k_right=0.20, n_extrap=200):
    """Original flat-vol tail extrapolation (for baseline comparison)."""
    k_obs = np.asarray(k_obs, float)
    C_obs = np.asarray(C_obs, float)
    idx = np.argsort(k_obs)
    k_obs, C_obs = k_obs[idx], C_obs[idx]

    F = 1.0 + ypi
    K_obs = 1.0 + k_obs

    sigma_left = implied_vol_from_call(C_obs[0], K_obs[0], F, B)
    sigma_right = implied_vol_from_call(C_obs[-1], K_obs[-1], F, B)

    if not np.isfinite(sigma_left):
        for i in range(1, len(C_obs)):
            sigma_left = implied_vol_from_call(C_obs[i], K_obs[i], F, B)
            if np.isfinite(sigma_left):
                break
    if not np.isfinite(sigma_right):
        for i in range(len(C_obs) - 2, -1, -1):
            sigma_right = implied_vol_from_call(C_obs[i], K_obs[i], F, B)
            if np.isfinite(sigma_right):
                break

    if not np.isfinite(sigma_left) or not np.isfinite(sigma_right):
        return k_obs, C_obs

    k_left_grid = np.linspace(k_left, k_obs[0], n_extrap, endpoint=False)
    C_left = black76_call(1.0 + k_left_grid, F, sigma_left, B)

    k_right_grid = np.linspace(k_obs[-1], k_right, n_extrap)[1:]
    C_right = black76_call(1.0 + k_right_grid, F, sigma_right, B)

    return (np.concatenate([k_left_grid, k_obs, k_right_grid]),
            np.concatenate([C_left, C_obs, C_right]))


def extrapolate_call_prices_smooth(k_obs, C_obs, B, ypi,
                                    k_left=-0.20, k_right=0.20, n_extrap=200):
    """
    Smooth IV spline tail extrapolation (Improvement 2).

    Instead of flat vol at boundaries, fit a quadratic to the last 3
    observed implied vols and extrapolate the smile smoothly.
    """
    k_obs = np.asarray(k_obs, float)
    C_obs = np.asarray(C_obs, float)
    idx = np.argsort(k_obs)
    k_obs, C_obs = k_obs[idx], C_obs[idx]

    F = 1.0 + ypi
    K_obs = 1.0 + k_obs

    # Compute IV at each observed strike
    ivs = np.array([implied_vol_from_call(C_obs[i], K_obs[i], F, B)
                     for i in range(len(k_obs))])

    # Find valid IVs
    valid = np.isfinite(ivs)
    if valid.sum() < 3:
        # Fallback to flat extrapolation
        return extrapolate_call_prices_flat(k_obs, C_obs, B, ypi,
                                            k_left, k_right, n_extrap)

    # Fit quadratic to left 3 and right 3 valid IVs
    valid_idx = np.where(valid)[0]
    n_fit = min(3, len(valid_idx))

    left_idx = valid_idx[:n_fit]
    right_idx = valid_idx[-n_fit:]

    poly_left = np.polyfit(k_obs[left_idx], ivs[left_idx], min(2, n_fit - 1))
    poly_right = np.polyfit(k_obs[right_idx], ivs[right_idx], min(2, n_fit - 1))

    # Left wing
    k_left_grid = np.linspace(k_left, k_obs[0], n_extrap, endpoint=False)
    sigma_left_ext = np.polyval(poly_left, k_left_grid)
    sigma_left_ext = np.maximum(sigma_left_ext, 0.002)  # floor

    C_left = np.array([float(black76_call(1.0 + ki, F, si, B))
                        for ki, si in zip(k_left_grid, sigma_left_ext)])

    # Right wing
    k_right_grid = np.linspace(k_obs[-1], k_right, n_extrap)[1:]
    sigma_right_ext = np.polyval(poly_right, k_right_grid)
    sigma_right_ext = np.maximum(sigma_right_ext, 0.002)

    C_right = np.array([float(black76_call(1.0 + ki, F, si, B))
                         for ki, si in zip(k_right_grid, sigma_right_ext)])

    # Ensure monotonicity (call prices must be non-increasing)
    k_ext = np.concatenate([k_left_grid, k_obs, k_right_grid])
    C_ext = np.concatenate([C_left, C_obs, C_right])

    # Enforce non-increasing by backward pass
    for i in range(1, len(C_ext)):
        if C_ext[i] > C_ext[i - 1]:
            C_ext[i] = C_ext[i - 1]
    C_ext = np.maximum(C_ext, 0.0)

    return k_ext, C_ext


# ============================================================
# ORIGINAL KW (baseline for comparison)
# ============================================================
def select_bandwidth_original(k):
    """Original bandwidth selector (ceiling-bound problem)."""
    k = np.asarray(k, float)
    n = len(k)
    if n < 5:
        return np.nan
    std = float(np.std(k, ddof=1))
    spac = np.diff(np.sort(k))
    med_spac = float(np.median(spac)) if len(spac) > 0 else 0.0
    h1 = 1.06 * std * (n ** (-1 / 5))
    h2 = 1.5 * med_spac
    return max(h1, h2, 1e-4)


def local_poly_derivative_epan(k_obs, y_obs, grid, h):
    """Original local polynomial with Epanechnikov kernel."""
    k_obs = np.asarray(k_obs, float)
    y_obs = np.asarray(y_obs, float)
    grid = np.asarray(grid, float)
    P  = np.full_like(grid, np.nan, dtype=float)
    P1 = np.full_like(grid, np.nan, dtype=float)
    P2 = np.full_like(grid, np.nan, dtype=float)
    for j, k0 in enumerate(grid):
        u = (k_obs - k0) / h
        w = epanechnikov(u)
        if np.sum(w) < 1e-8:
            continue
        x = k_obs - k0
        X = np.column_stack([np.ones_like(x), x, 0.5 * x**2])
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = y_obs * sw
        try:
            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        except Exception:
            continue
        P[j], P1[j], P2[j] = beta[0], beta[1], beta[2]
    return P, P1, P2


def kw_density_original(k, C, B, ypi):
    """Original KW density: Epanechnikov kernel, flat-vol tails, 200-pt grid, edge shutdown."""
    k = np.asarray(k, float)
    C = np.asarray(C, float)

    # No tail extension in the original options_implied_inflation_pdf.py
    kmin, kmax = float(np.min(k)), float(np.max(k))
    grid = np.linspace(kmin, kmax, 200)

    h = select_bandwidth_original(k)
    if not np.isfinite(h) or h <= 0:
        return None

    P, P1, P2 = local_poly_derivative_epan(k, C, grid, h)
    if np.all(~np.isfinite(P2)):
        return None

    f = (1.0 / B) * P2
    f = np.where(np.isfinite(f), f, 0.0)
    f = np.maximum(f, 0.0)

    Z = integrate(grid, f)
    if not np.isfinite(Z) or Z <= 0:
        return None
    f = f / Z

    # Edge-band shutdown (original logic)
    L, U = -0.01, 0.05
    eps = 0.003
    thr = 0.07
    mask_L = (grid >= L) & (grid <= L + eps)
    mask_U = (grid > U - eps) & (grid <= U)
    p_edge_L = integrate(grid[mask_L], f[mask_L]) if np.any(mask_L) else 0.0
    p_edge_U = integrate(grid[mask_U], f[mask_U]) if np.any(mask_U) else 0.0
    if not np.isfinite(p_edge_L):
        p_edge_L = 0.0
    if not np.isfinite(p_edge_U):
        p_edge_U = 0.0
    edge_ok = (p_edge_L <= thr) and (p_edge_U <= thr)

    return grid, f, h, Z, edge_ok


# ============================================================
# IMPROVEMENT 3: LSCV BANDWIDTH SELECTION
# ============================================================
def local_poly_derivative_gauss(k_obs, y_obs, grid, h):
    """
    Local polynomial regression with Gaussian kernel (Improvement 1).

    Vectorized: precompute all weights as a (n_grid, n_obs) matrix,
    then solve each WLS problem. Truncates Gaussian at 5*h for speed.
    """
    k_obs = np.asarray(k_obs, float)
    y_obs = np.asarray(y_obs, float)
    grid = np.asarray(grid, float)

    n_grid = len(grid)
    n_obs = len(k_obs)
    P  = np.full(n_grid, np.nan, dtype=float)
    P1 = np.full(n_grid, np.nan, dtype=float)
    P2 = np.full(n_grid, np.nan, dtype=float)

    # Vectorize: compute u matrix (n_grid x n_obs)
    U = (k_obs[None, :] - grid[:, None]) / h  # (n_grid, n_obs)
    W = gaussian_kernel(U)  # (n_grid, n_obs)

    # For each grid point, solve weighted least squares
    for j in range(n_grid):
        w = W[j]
        if np.sum(w) < 1e-8:
            continue
        x = k_obs - grid[j]
        X = np.column_stack([np.ones(n_obs), x, 0.5 * x**2])
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = y_obs * sw
        try:
            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        except Exception:
            continue
        P[j], P1[j], P2[j] = beta[0], beta[1], beta[2]
    return P, P1, P2


def _local_poly_predict_one(k_obs, y_obs, k_eval, h):
    """Predict call price at a single point using Gaussian local poly."""
    u = (k_obs - k_eval) / h
    w = gaussian_kernel(u)
    if np.sum(w) < 1e-8:
        return np.nan
    x = k_obs - k_eval
    X = np.column_stack([np.ones_like(x), x, 0.5 * x**2])
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y_obs * sw
    try:
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    except Exception:
        return np.nan
    return beta[0]


def lscv_bandwidth(k_obs, C_obs, k_ext, C_ext):
    """
    Select bandwidth by leave-one-out cross-validation (Improvement 3).

    Uses continuous optimization (golden-section) on the LOO score,
    evaluated on the 7 observed strikes. This ensures bandwidth varies
    smoothly across dates.
    """
    k_obs = np.asarray(k_obs, float)
    C_obs = np.asarray(C_obs, float)

    h_base = select_bandwidth_original(k_ext)
    if not np.isfinite(h_base):
        return h_base

    # Precompute LOO data (observed strike removed, rest of extrapolated kept)
    obs_in_ext = np.isin(k_ext, k_obs)
    k_ext_only = k_ext[~obs_in_ext]
    C_ext_only = C_ext[~obs_in_ext]

    def loo_score(h):
        total_err = 0.0
        n_valid = 0
        for i in range(len(k_obs)):
            k_loo = np.concatenate([np.delete(k_obs, i), k_ext_only])
            C_loo = np.concatenate([np.delete(C_obs, i), C_ext_only])
            idx = np.argsort(k_loo)
            k_loo, C_loo = k_loo[idx], C_loo[idx]
            C_pred = _local_poly_predict_one(k_loo, C_loo, k_obs[i], h)
            if np.isfinite(C_pred):
                total_err += (C_obs[i] - C_pred)**2
                n_valid += 1
        return total_err / max(n_valid, 1) if n_valid > 0 else 1e10

    # Search in [0.3*h_base, 2.5*h_base]
    h_lo = 0.3 * h_base
    h_hi = 2.5 * h_base

    res = optimize.minimize_scalar(loo_score, bounds=(h_lo, h_hi),
                                    method='bounded',
                                    options={'xatol': h_base * 0.01})
    return float(res.x)


# ============================================================
# IMPROVEMENT 4: SPLINE-BASED QUANTILE EXTRACTION
# ============================================================
def quantile_spline(p_val, F_spline, x_lo, x_hi):
    """Invert CDF spline to find quantile using Brent's method."""
    try:
        return brentq(lambda xi: float(F_spline(xi)) - p_val, x_lo, x_hi,
                      xtol=1e-10, rtol=1e-10)
    except (ValueError, RuntimeError):
        return np.nan


def compute_quantiles_spline(x, f, quantile_probs):
    """
    Compute quantiles using PCHIP monotone spline on CDF (Improvement 4).

    Much more precise than searchsorted + linear interpolation,
    especially for the Bowley numerator (difference of near-equal quantities).
    """
    F = np.zeros_like(x)
    F[1:] = cumulative_trapezoid(f, x)
    # Ensure monotonicity
    F = np.maximum.accumulate(F)
    F = F / F[-1]  # normalize

    # Build monotone spline
    # Remove duplicate F values for interpolation
    mask = np.diff(F, prepend=-1) > 0
    if mask.sum() < 4:
        # Too few unique CDF values, fall back to linear
        results = {}
        for p in quantile_probs:
            idx = np.searchsorted(F, p, side='left')
            idx = np.clip(idx, 1, len(x) - 1)
            dF = F[idx] - F[idx - 1]
            frac = (p - F[idx - 1]) / max(dF, 1e-15)
            results[p] = x[idx - 1] + frac * (x[idx] - x[idx - 1])
        return results

    F_spline = PchipInterpolator(x[mask], F[mask])

    results = {}
    for p in quantile_probs:
        results[p] = quantile_spline(p, F_spline, x[mask][0], x[mask][-1])
    return results


# ============================================================
# IMPROVEMENT 5: RESIDUAL-BASED NOISE ESTIMATION
# ============================================================
def estimate_noise_scale(k_obs, C_obs, k_ext, C_ext, h):
    """
    Estimate price noise from local polynomial fit residuals (Improvement 5).

    Uses the local polynomial smooth to predict call prices at observed
    strikes, then computes relative residuals. Only uses core strikes
    (above 5th percentile of price) to avoid noise from near-zero tail prices.
    """
    k_obs = np.asarray(k_obs, float)
    C_obs = np.asarray(C_obs, float)

    # Predict at observed strikes using the local polynomial
    C_fit = np.array([_local_poly_predict_one(k_ext, C_ext, ki, h)
                      for ki in k_obs])

    # Only use strikes with prices above 5% of the max (core strikes)
    price_thr = 0.05 * np.max(C_obs)
    valid = np.isfinite(C_fit) & (C_obs > price_thr)
    if valid.sum() < 3:
        return 0.02  # fallback

    residuals = (C_obs[valid] - C_fit[valid]) / C_obs[valid]
    noise = float(np.std(np.abs(residuals)))
    # Clamp to reasonable range
    return np.clip(noise, 0.005, 0.10)


# ============================================================
# IMPROVED KW DENSITY (all improvements combined)
# ============================================================
def kw_density_improved(k, C, B, ypi, grid_n=1600):
    """
    MC-validated best KW density estimation pipeline:
      - Gaussian kernel (smooth, no boundary kinks)
      - LSCV bandwidth (adapts per date, LOO on observed strikes)
      - Flat-vol tail extrapolation (MC showed smooth IV amplifies noise)
      - Finer grid (1600 points)
      - No edge-band shutdown (Z_raw diagnostic instead)
    """
    k = np.asarray(k, float)
    C = np.asarray(C, float)

    # Flat-vol tail extrapolation (MC-validated: smooth IV amplifies noise)
    k_fit, C_fit = extrapolate_call_prices_flat(k, C, B, ypi)

    # LSCV bandwidth: LOO on 7 observed strikes (fast), using full data (Improvement 3)
    h = lscv_bandwidth(k, C, k_fit, C_fit)
    if not np.isfinite(h) or h <= 0:
        return None

    # Finer grid (Improvement 1)
    kmin, kmax = float(np.min(k_fit)), float(np.max(k_fit))
    grid = np.linspace(kmin, kmax, grid_n)

    # Gaussian kernel local polynomial (Improvement 1)
    P, P1, P2 = local_poly_derivative_gauss(k_fit, C_fit, grid, h)
    if np.all(~np.isfinite(P2)):
        return None

    f = (1.0 / B) * P2
    f = np.where(np.isfinite(f), f, 0.0)
    f = np.maximum(f, 0.0)

    Z_raw = integrate(grid, f)
    if not np.isfinite(Z_raw) or Z_raw <= 0:
        return None
    f = f / Z_raw

    return grid, f, h, Z_raw


# ============================================================
# MOMENTS AND ASYMMETRY MEASURES
# ============================================================
def moments_from_density(x, f):
    """Return (mean, var, skew, kurt) from continuous density."""
    Z = integrate(x, f)
    if not np.isfinite(Z) or Z <= 0:
        return (np.nan, np.nan, np.nan, np.nan)
    f = f / Z
    mu  = integrate(x, x * f)
    var = integrate(x, (x - mu)**2 * f)
    if var <= 0 or not np.isfinite(var):
        return (mu, np.nan, np.nan, np.nan)
    sd = np.sqrt(var)
    m3 = integrate(x, (x - mu)**3 * f)
    m4 = integrate(x, (x - mu)**4 * f)
    return (mu, var, m3 / sd**3, m4 / sd**4)


def bowley_skewness(x, f):
    """Bowley (Galton) quartile skewness using spline quantiles."""
    Z = integrate(x, f)
    if not np.isfinite(Z) or Z <= 0:
        return np.nan
    f_norm = f / Z
    qs = compute_quantiles_spline(x, f_norm, [0.25, 0.50, 0.75])
    Q1, Q2, Q3 = qs[0.25], qs[0.50], qs[0.75]
    if any(np.isnan(v) for v in [Q1, Q2, Q3]):
        return np.nan
    iqr = Q3 - Q1
    if iqr < 1e-15:
        return np.nan
    return (Q3 + Q1 - 2 * Q2) / iqr


def bowley_skewness_linear(x, f):
    """Bowley with original linear interpolation quantiles (for comparison)."""
    Z = integrate(x, f)
    if not np.isfinite(Z) or Z <= 0:
        return np.nan
    f_norm = f / Z
    F = np.zeros_like(x)
    F[1:] = cumulative_trapezoid(f_norm, x)

    def quantile(p_val):
        idx = np.searchsorted(F, p_val, side='left')
        idx = np.clip(idx, 1, len(x) - 1)
        dF = F[idx] - F[idx - 1]
        frac = (p_val - F[idx - 1]) / max(dF, 1e-15)
        return x[idx - 1] + frac * (x[idx] - x[idx - 1])

    Q1, Q2, Q3 = quantile(0.25), quantile(0.50), quantile(0.75)
    iqr = Q3 - Q1
    if iqr < 1e-15:
        return np.nan
    return (Q3 + Q1 - 2 * Q2) / iqr


# ============================================================
# IMPROVEMENT 7: ASYMMETRY PROFILE
# ============================================================
def asymmetry_profile(x, f, quantile_pairs=None):
    """
    Bowley-type skewness at multiple quantile depths.

    Tells you whether asymmetry is driven by the tails (10/90)
    or the core (40/60).
    """
    if quantile_pairs is None:
        quantile_pairs = [(0.1, 0.9), (0.2, 0.8), (0.25, 0.75), (0.4, 0.6)]

    Z = integrate(x, f)
    if not np.isfinite(Z) or Z <= 0:
        return {}
    f_norm = f / Z

    all_probs = set()
    all_probs.add(0.5)
    for (p_lo, p_hi) in quantile_pairs:
        all_probs.add(p_lo)
        all_probs.add(p_hi)

    qs = compute_quantiles_spline(x, f_norm, sorted(all_probs))
    q_med = qs.get(0.5, np.nan)

    profile = {}
    for (p_lo, p_hi) in quantile_pairs:
        q_lo = qs.get(p_lo, np.nan)
        q_hi = qs.get(p_hi, np.nan)
        if any(np.isnan(v) for v in [q_lo, q_hi, q_med]):
            profile[f'asym_{int(p_lo*100)}_{int(p_hi*100)}'] = np.nan
            continue
        iqr = q_hi - q_lo
        if iqr > 1e-15:
            profile[f'asym_{int(p_lo*100)}_{int(p_hi*100)}'] = (q_hi + q_lo - 2 * q_med) / iqr
        else:
            profile[f'asym_{int(p_lo*100)}_{int(p_hi*100)}'] = np.nan
    return profile


# ============================================================
# BOOTSTRAP WITH RESIDUAL-BASED NOISE
# ============================================================
def bootstrap_bowley_kw(k, C, B, ypi, noise_scale, h_fixed, n_boot=100):
    """
    Bootstrap for improved KW Bowley skewness with residual-based noise.

    Uses a fixed bandwidth (from the point estimate) to avoid re-running
    LSCV on each bootstrap replicate, making it ~50x faster.
    """
    k = np.asarray(k, float)
    C = np.asarray(C, float)
    bowleys = []
    rng = np.random.default_rng(42)

    for _ in range(n_boot):
        eps = rng.normal(0, noise_scale, size=len(C))
        C_boot = C * (1.0 + eps)
        C_boot = np.maximum(C_boot, 0.0)
        # Enforce monotonicity
        idx = np.argsort(k)
        C_sorted = C_boot[idx]
        for i in range(1, len(C_sorted)):
            if C_sorted[i] > C_sorted[i - 1]:
                C_sorted[i] = C_sorted[i - 1]
        C_boot[idx] = C_sorted

        # Use flat-vol extrapolation + fixed bandwidth (skip LSCV for speed)
        k_fit, C_fit = extrapolate_call_prices_flat(k, C_boot, B, ypi)
        kmin, kmax = float(np.min(k_fit)), float(np.max(k_fit))
        grid = np.linspace(kmin, kmax, 800)
        _P, _P1, P2 = local_poly_derivative_gauss(k_fit, C_fit, grid, h_fixed)
        if np.all(~np.isfinite(P2)):
            continue
        f = (1.0 / B) * P2
        f = np.where(np.isfinite(f), f, 0.0)
        f = np.maximum(f, 0.0)
        Z = integrate(grid, f)
        if not np.isfinite(Z) or Z <= 0:
            continue
        f = f / Z
        bw = bowley_skewness(grid, f)
        if np.isfinite(bw):
            bowleys.append(bw)

    if len(bowleys) < 10:
        return np.nan, np.nan, np.nan
    a = np.array(bowleys)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), float(np.std(a))


# ============================================================
# MAIN ANALYSIS
# ============================================================
def main():
    print("=" * 70)
    print("IMPROVED KW SKEWNESS ESTIMATION")
    print("=" * 70)

    print("\nLoading call curves...")
    with open(CALL_CURVES_PKL, "rb") as fh:
        call_curves = pickle.load(fh)

    # Load original results for comparison
    res_old = pd.read_csv(RESULTS_CSV)
    res_old["date"] = pd.to_datetime(res_old["date"])

    print(f"Processing {len(call_curves)} date/area combinations...\n")

    rows = []
    total = len(call_curves)
    h_values_original = []
    h_values_improved = []

    for i, ((date_str, area), cc) in enumerate(sorted(call_curves.items())):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{total}] {date_str} {area}")

        k   = cc['k']
        C   = cc['C']
        B   = cc['B']
        ypi = cc['ypi']

        row = {'date': date_str, 'area': area, 'B': B, 'ypi': ypi}

        # ----- ORIGINAL KW (baseline) -----
        kw_orig = kw_density_original(k, C, B, ypi)
        if kw_orig is not None:
            grid_o, f_o, h_o, Z_o, edge_ok_o = kw_orig
            mu_o, var_o, skew_o, kurt_o = moments_from_density(grid_o, f_o)
            bowley_o = bowley_skewness_linear(grid_o, f_o)  # original linear quantiles
            h_values_original.append(h_o)

            if not edge_ok_o:
                skew_o = np.nan
                bowley_o = np.nan
        else:
            mu_o = var_o = skew_o = kurt_o = np.nan
            bowley_o = np.nan
            h_o = Z_o = np.nan

        row.update({
            'mean_KW_orig': mu_o, 'var_KW_orig': var_o,
            'skew_KW_orig': skew_o, 'kurt_KW_orig': kurt_o,
            'bowley_KW_orig': bowley_o,
            'h_KW_orig': h_o, 'Z_raw_orig': Z_o,
        })

        # ----- IMPROVED KW -----
        kw_imp = kw_density_improved(k, C, B, ypi, grid_n=1600)
        if kw_imp is not None:
            grid_i, f_i, h_i, Z_i = kw_imp
            mu_i, var_i, skew_i, kurt_i = moments_from_density(grid_i, f_i)
            bowley_i = bowley_skewness(grid_i, f_i)
            h_values_improved.append(h_i)

            # Asymmetry profile (Improvement 7)
            prof = asymmetry_profile(grid_i, f_i)

            # Noise estimation (Improvement 5) - use extrapolated data
            k_ext, C_ext = extrapolate_call_prices_flat(k, C, B, ypi)
            noise_est = estimate_noise_scale(k, C, k_ext, C_ext, h_i)
        else:
            mu_i = var_i = skew_i = kurt_i = np.nan
            bowley_i = np.nan
            h_i = Z_i = np.nan
            noise_est = np.nan
            prof = {}

        row.update({
            'mean_KW_imp': mu_i, 'var_KW_imp': var_i,
            'skew_KW_imp': skew_i, 'kurt_KW_imp': kurt_i,
            'bowley_KW_imp': bowley_i,
            'h_KW_imp': h_i, 'Z_raw_imp': Z_i,
            'noise_scale_est': noise_est,
        })
        row.update(prof)

        # ----- BOOTSTRAP (improved, with residual-based noise) -----
        if RUN_BOOTSTRAP and kw_imp is not None and np.isfinite(h_i):
            ci_lo, ci_hi, ci_std = bootstrap_bowley_kw(
                k, C, B, ypi, noise_est, h_fixed=h_i, n_boot=N_BOOT)
        else:
            ci_lo = ci_hi = ci_std = np.nan

        row.update({
            'bowley_ci_lo': ci_lo, 'bowley_ci_hi': ci_hi,
            'bowley_boot_std': ci_std,
        })

        rows.append(row)

    # --- Save results ---
    df = pd.DataFrame(rows).sort_values(["area", "date"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

    # ============================================================
    # COMPARISON AND DIAGNOSTICS
    # ============================================================
    d = df[df["area"] == AREA_FOCUS].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")

    if len(d) == 0:
        print(f"No data for area={AREA_FOCUS}.")
        return

    # --- Helper functions ---
    def autocorr(series, lag=1):
        s = series.dropna()
        if len(s) < lag + 5:
            return np.nan
        return float(s.autocorr(lag=lag))

    def sign_flip_rate(series):
        s = series.dropna().values
        if len(s) < 2:
            return np.nan
        signs = np.sign(s)
        return float((np.diff(signs) != 0).mean())

    # --- Compute metrics ---
    metrics = {}

    # Bandwidth
    h_orig = d["h_KW_orig"].dropna()
    h_imp = d["h_KW_imp"].dropna()
    metrics['h_unique_orig'] = h_orig.nunique()
    metrics['h_unique_imp'] = h_imp.nunique()
    metrics['h_mean_orig'] = float(h_orig.mean())
    metrics['h_mean_imp'] = float(h_imp.mean())
    metrics['h_cv_orig'] = float(h_orig.std() / h_orig.mean()) if h_orig.mean() > 0 else 0
    metrics['h_cv_imp'] = float(h_imp.std() / h_imp.mean()) if h_imp.mean() > 0 else 0

    # Coverage
    metrics['cov_orig'] = int(d["bowley_KW_orig"].notna().sum())
    metrics['cov_imp'] = int(d["bowley_KW_imp"].notna().sum())
    metrics['n_dates'] = len(d)

    # Persistence
    metrics['ac1_orig'] = autocorr(d["bowley_KW_orig"], 1)
    metrics['ac1_imp'] = autocorr(d["bowley_KW_imp"], 1)
    metrics['ac3_orig'] = autocorr(d["bowley_KW_orig"], 3)
    metrics['ac3_imp'] = autocorr(d["bowley_KW_imp"], 3)

    # Sign-flip rate
    metrics['flip_orig'] = sign_flip_rate(d["bowley_KW_orig"])
    metrics['flip_imp'] = sign_flip_rate(d["bowley_KW_imp"])

    # Bowley descriptive stats
    bw_orig = d["bowley_KW_orig"].dropna()
    bw_imp = d["bowley_KW_imp"].dropna()
    metrics['bowley_mean_orig'] = float(bw_orig.mean())
    metrics['bowley_mean_imp'] = float(bw_imp.mean())
    metrics['bowley_std_orig'] = float(bw_orig.std())
    metrics['bowley_std_imp'] = float(bw_imp.std())
    metrics['bowley_neg_pct_orig'] = float((bw_orig < 0).mean() * 100)
    metrics['bowley_neg_pct_imp'] = float((bw_imp < 0).mean() * 100)

    # Skew vs Bowley sign agreement
    bv_imp = d[["skew_KW_imp", "bowley_KW_imp"]].dropna()
    metrics['sign_agree_imp'] = float((np.sign(bv_imp["skew_KW_imp"]) ==
                                        np.sign(bv_imp["bowley_KW_imp"])).mean()) if len(bv_imp) > 0 else np.nan
    bv_orig = d[["skew_KW_orig", "bowley_KW_orig"]].dropna()
    metrics['sign_agree_orig'] = float((np.sign(bv_orig["skew_KW_orig"]) ==
                                         np.sign(bv_orig["bowley_KW_orig"])).mean()) if len(bv_orig) > 0 else np.nan

    # Bootstrap CI
    ci_width = (d["bowley_ci_hi"] - d["bowley_ci_lo"]).dropna()
    metrics['ci_mean_width'] = float(ci_width.mean()) if len(ci_width) > 0 else np.nan
    metrics['noise_mean'] = float(d['noise_scale_est'].dropna().mean())
    metrics['bowley_abs_mean_imp'] = float(bw_imp.abs().mean())
    metrics['ci_signal_ratio'] = (metrics['ci_mean_width'] / metrics['bowley_abs_mean_imp']
                                   if metrics['bowley_abs_mean_imp'] > 0 else np.inf)

    # Asymmetry profile
    asym_cols = [c for c in d.columns if c.startswith("asym_")]
    if len(asym_cols) >= 2:
        asym_corr = d[asym_cols].dropna().corr()
        metrics['asym_min_corr'] = float(asym_corr.values[np.triu_indices_from(asym_corr.values, k=1)].min())
    else:
        metrics['asym_min_corr'] = np.nan

    # --- Print comprehensive comparison table ---
    print("\n" + "=" * 70)
    print("SIGNAL QUALITY COMPARISON: ORIGINAL vs IMPROVED KW")
    print("=" * 70)

    table_rows = [
        ("Bandwidth: unique h values", f"{metrics['h_unique_orig']}", f"{metrics['h_unique_imp']}"),
        ("Bandwidth: mean h", f"{metrics['h_mean_orig']:.5f}", f"{metrics['h_mean_imp']:.5f}"),
        ("Bandwidth: CV(h)", f"{metrics['h_cv_orig']:.4f}", f"{metrics['h_cv_imp']:.4f}"),
        ("Bowley coverage", f"{metrics['cov_orig']}/{metrics['n_dates']}", f"{metrics['cov_imp']}/{metrics['n_dates']}"),
        ("Bowley mean", f"{metrics['bowley_mean_orig']:+.4f}", f"{metrics['bowley_mean_imp']:+.4f}"),
        ("Bowley std", f"{metrics['bowley_std_orig']:.4f}", f"{metrics['bowley_std_imp']:.4f}"),
        ("Bowley % negative", f"{metrics['bowley_neg_pct_orig']:.1f}%", f"{metrics['bowley_neg_pct_imp']:.1f}%"),
        ("AC(1) persistence", f"{metrics['ac1_orig']:.3f}", f"{metrics['ac1_imp']:.3f}"),
        ("AC(3) persistence", f"{metrics['ac3_orig']:.3f}", f"{metrics['ac3_imp']:.3f}"),
        ("Sign-flip rate", f"{metrics['flip_orig']:.3f}", f"{metrics['flip_imp']:.3f}"),
        ("Skew-Bowley sign agree", f"{100*metrics['sign_agree_orig']:.1f}%", f"{100*metrics['sign_agree_imp']:.1f}%"),
        ("Bootstrap CI width", "N/A", f"{metrics['ci_mean_width']:.4f}"),
        ("CI/signal ratio", "N/A", f"{metrics['ci_signal_ratio']:.2f}x"),
        ("Noise scale (est.)", "5.00% (fixed)", f"{100*metrics['noise_mean']:.2f}% (adaptive)"),
        ("Asym. profile min corr", "N/A", f"{metrics['asym_min_corr']:.3f}"),
    ]

    print(f"\n  {'Metric':<30s}  {'Original KW':<20s}  {'Improved KW':<20s}")
    print(f"  {'─'*30}  {'─'*20}  {'─'*20}")
    for label, val_o, val_i in table_rows:
        print(f"  {label:<30s}  {val_o:<20s}  {val_i:<20s}")

    # --- Pass/Fail summary ---
    tests = {
        'Bandwidth adapts': metrics['h_unique_imp'] > metrics['h_unique_orig'],
        'Coverage >= original': metrics['cov_imp'] >= metrics['cov_orig'],
        'AC(3) persistence': metrics['ac3_imp'] > metrics['ac3_orig'],
        'Fewer sign flips': metrics['flip_imp'] < metrics['flip_orig'],
        'Skew-Bowley agreement': metrics['sign_agree_imp'] >= metrics['sign_agree_orig'],
        'CI is informative': metrics['ci_signal_ratio'] < 10,
        'Asym profile consistent': metrics['asym_min_corr'] > 0.3 if np.isfinite(metrics['asym_min_corr']) else False,
    }

    n_pass = sum(tests.values())
    print(f"\n  PASS/FAIL:")
    for name, passed in tests.items():
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\n  OVERALL: {n_pass}/{len(tests)} passed")

    # ============================================================
    # PLOTS
    # ============================================================

    # --- Plot 1: Main comparison — Bowley time series ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})

    ax = axes[0]
    ax.plot(d["date"], d["bowley_KW_orig"],
            label="Original KW (Epanechnikov, Silverman h, 200pt)",
            alpha=0.7, linewidth=1, color="C0")
    ax.plot(d["date"], d["bowley_KW_imp"],
            label="Improved KW (Gaussian, LSCV h, 1600pt, flat tails)",
            linewidth=1.5, color="C3")
    if RUN_BOOTSTRAP and "bowley_ci_lo" in d.columns:
        ax.fill_between(d["date"], d["bowley_ci_lo"], d["bowley_ci_hi"],
                        alpha=0.15, color="C3", label="95% bootstrap CI")
    ax.axhline(0, color='gray', linewidth=0.5, zorder=0)
    ax.set_ylabel("Bowley skewness")
    ax.set_title(f"KW Bowley Skewness: Original vs MC-Validated Improved ({AREA_FOCUS})")
    ax.set_ylim(-1, 1)
    ax.legend(loc="lower left", fontsize=8)

    # Bandwidth subplot
    ax2 = axes[1]
    ax2.plot(d["date"], d["h_KW_orig"], label=f"Original h (mean={metrics['h_mean_orig']:.4f})",
             alpha=0.7, linewidth=1, color="C0")
    ax2.plot(d["date"], d["h_KW_imp"], label=f"LSCV h (mean={metrics['h_mean_imp']:.4f})",
             linewidth=1, color="C3")
    ax2.set_ylabel("Bandwidth h")
    ax2.set_xlabel("Date")
    ax2.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_kw_bowley_orig_vs_improved.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: Scatter — original vs improved Bowley ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 2a: Orig vs Improved Bowley
    ax = axes[0]
    both = d[["bowley_KW_orig", "bowley_KW_imp"]].dropna()
    ax.scatter(both["bowley_KW_orig"], both["bowley_KW_imp"], s=15, alpha=0.6,
               edgecolors='none')
    lims = [-0.6, 0.6]
    ax.plot(lims, lims, 'k--', linewidth=0.5)
    ax.axhline(0, color='gray', linewidth=0.3)
    ax.axvline(0, color='gray', linewidth=0.3)
    ax.set_xlabel("Original Bowley")
    ax.set_ylabel("Improved Bowley")
    ax.set_title("Bowley: Original vs Improved")
    ax.set_xlim(lims); ax.set_ylim(lims)
    if len(both) > 2:
        corr = both["bowley_KW_orig"].corr(both["bowley_KW_imp"])
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=9, va='top')

    # 2b: Histograms
    ax = axes[1]
    bins = np.linspace(-0.6, 0.6, 30)
    ax.hist(bw_orig, bins=bins, alpha=0.5, label="Original", color="C0", density=True)
    ax.hist(bw_imp, bins=bins, alpha=0.5, label="Improved", color="C3", density=True)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel("Bowley skewness")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Bowley Estimates")
    ax.legend(fontsize=8)

    # 2c: Bandwidth distributions
    ax = axes[2]
    ax.hist(h_orig, bins=30, alpha=0.5, label="Original (Silverman)", color="C0")
    ax.hist(h_imp, bins=30, alpha=0.5, label="Improved (LSCV)", color="C3")
    ax.set_xlabel("Bandwidth h")
    ax.set_ylabel("Count")
    ax.set_title("Bandwidth Distribution")
    ax.legend(fontsize=8)

    fig.suptitle(f"Diagnostic Scatter and Distributions ({AREA_FOCUS})", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_kw_diagnostics.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    # --- Plot 3: Asymmetry profile over time ---
    asym_cols = [c for c in d.columns if c.startswith("asym_")]
    if len(asym_cols) > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        colors_asym = ['C0', 'C1', 'C2', 'C4']
        for col, clr in zip(asym_cols, colors_asym):
            label = col.replace("asym_", "Q").replace("_", "/Q")
            ax.plot(d["date"], d[col], label=label, alpha=0.8, color=clr, linewidth=1)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("Quantile-pair skewness")
        ax.set_title(f"Asymmetry Profile Over Time — Improved KW ({AREA_FOCUS})")
        ax.legend(loc="best", fontsize=8)
        ax.set_ylim(-1, 1)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig_kw_asymmetry_profile.pdf"),
                    bbox_inches="tight")
        plt.close(fig)

    # --- Plot 4: Standard skewness comparison ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(d["date"], d["skew_KW_orig"], label="Original mu3/sigma3",
            alpha=0.7, linewidth=1, color="C0")
    ax.plot(d["date"], d["skew_KW_imp"], label="Improved mu3/sigma3",
            linewidth=1.5, color="C3")
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Skewness (mu3/sigma3)")
    ax.set_title(f"KW Standard Skewness: Original vs Improved ({AREA_FOCUS})")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(-4, 4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_kw_skewness_orig_vs_improved.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    # --- Plot 5: Noise scale estimates ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(d["date"], d["noise_scale_est"], marker='o', markersize=3,
            linewidth=1, color="C3")
    ax.axhline(0.05, color='red', linestyle='--', alpha=0.6,
               label="Old fixed noise (5%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Estimated noise scale")
    ax.set_title(f"Residual-Based Noise Estimation ({AREA_FOCUS})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_kw_noise_estimation.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    # --- Plot 6: Summary comparison table as figure ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    cell_text = []
    cell_colors = []
    for label, val_o, val_i in table_rows:
        cell_text.append([label, val_o, val_i])
        cell_colors.append(['#f5f5f5', 'white', 'white'])

    # Add pass/fail rows
    for name, passed in tests.items():
        status = 'PASS' if passed else 'FAIL'
        color = '#c8e6c9' if passed else '#ffcdd2'
        cell_text.append([name, '', status])
        cell_colors.append(['#f5f5f5', 'white', color])

    table = ax.table(cellText=cell_text,
                     colLabels=['Metric', 'Original KW', 'Improved KW'],
                     cellColours=cell_colors,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    # Bold header
    for j in range(3):
        table[0, j].set_text_props(fontweight='bold')
        table[0, j].set_facecolor('#e0e0e0')

    fig.suptitle(f"KW Skewness: Original vs Improved — Summary ({AREA_FOCUS})",
                 fontsize=12, y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_kw_summary_table.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots saved to {FIG_DIR}/")
    print(f"  - fig_kw_bowley_orig_vs_improved.pdf  (main time series)")
    print(f"  - fig_kw_diagnostics.pdf              (scatter, histograms)")
    print(f"  - fig_kw_asymmetry_profile.pdf         (multi-depth asymmetry)")
    print(f"  - fig_kw_skewness_orig_vs_improved.pdf (standard skewness)")
    print(f"  - fig_kw_noise_estimation.pdf          (adaptive noise)")
    print(f"  - fig_kw_summary_table.pdf             (comparison table)")


if __name__ == "__main__":
    main()
