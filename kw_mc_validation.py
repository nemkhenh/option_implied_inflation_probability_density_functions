# coding: utf-8
"""
Monte Carlo validation of KW skewness recovery from option prices.

Generates synthetic call prices from KNOWN distributions with controlled
skewness, runs both original and improved KW on them, and measures bias,
RMSE, and sign agreement. An ablation study isolates which improvement
component causes the negative-skewness bias observed in the improved pipeline.

Usage:
    python kw_mc_validation.py           # full run (~30 min)
    python kw_mc_validation.py --quick   # sanity check (~2 min)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, skewnorm
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

# Import component functions from the improved KW module
from kw_improved_skewness import (
    gaussian_kernel, epanechnikov,
    local_poly_derivative_gauss, local_poly_derivative_epan,
    select_bandwidth_original, lscv_bandwidth,
    extrapolate_call_prices_smooth, extrapolate_call_prices_flat,
    bowley_skewness, bowley_skewness_linear,
    moments_from_density, integrate,
    compute_quantiles_spline,
    black76_call, implied_vol_from_call,
)

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(__file__)
FIG_DIR  = os.path.join(BASE_DIR, "fig")
DATA_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)

OUTPUT_REPS_CSV    = os.path.join(DATA_DIR, "mc_validation_replicates.csv")
OUTPUT_SUMMARY_CSV = os.path.join(DATA_DIR, "mc_validation_summary.csv")

STRIKES = np.array([-0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
B_DEFAULT = 0.987
FINE_GRID = np.linspace(-0.20, 0.20, 5000)  # for integration


# ============================================================
# STEP 1: GROUND-TRUTH DISTRIBUTIONS
# ============================================================
def make_normal_density(mu, sigma, grid):
    """Normal distribution with known skewness = 0."""
    f = norm.pdf(grid, loc=mu, scale=sigma)
    true_bowley = 0.0  # symmetric
    true_skewness = 0.0
    return f, true_bowley, true_skewness


def make_skewnormal_density(alpha, target_mean, omega, grid):
    """
    Skew-normal distribution with controlled skewness.

    Parameters
    ----------
    alpha : float  -- shape parameter (alpha > 0 => positive skew)
    target_mean : float -- desired mean of the distribution
    omega : float -- scale parameter
    grid : np.ndarray

    Returns f, true_bowley, true_skewness
    """
    # Compute location parameter xi so that mean = target_mean
    delta = alpha / np.sqrt(1 + alpha**2)
    mean_offset = omega * delta * np.sqrt(2 / np.pi)
    xi = target_mean - mean_offset

    f = skewnorm.pdf(grid, a=alpha, loc=xi, scale=omega)

    # True Bowley from analytical CDF inversion
    Q1 = skewnorm.ppf(0.25, a=alpha, loc=xi, scale=omega)
    Q2 = skewnorm.ppf(0.50, a=alpha, loc=xi, scale=omega)
    Q3 = skewnorm.ppf(0.75, a=alpha, loc=xi, scale=omega)
    iqr = Q3 - Q1
    true_bowley = (Q3 + Q1 - 2 * Q2) / iqr if iqr > 1e-15 else 0.0

    # True standard skewness (analytical formula)
    mu_z = delta * np.sqrt(2 / np.pi)
    sigma_z = np.sqrt(1 - 2 * delta**2 / np.pi)
    true_skewness = ((4 - np.pi) / 2) * (mu_z**3) / (sigma_z**3)

    return f, float(true_bowley), float(true_skewness)


def make_mixture_density(mu1, sigma1, mu2, sigma2, w1, grid):
    """
    Mixture of two normals: f(x) = w1*N(mu1,sigma1) + (1-w1)*N(mu2,sigma2).
    """
    w2 = 1.0 - w1
    f = w1 * norm.pdf(grid, mu1, sigma1) + w2 * norm.pdf(grid, mu2, sigma2)

    # True moments
    mean = w1 * mu1 + w2 * mu2
    var = (w1 * (sigma1**2 + mu1**2) + w2 * (sigma2**2 + mu2**2)) - mean**2
    sd = np.sqrt(var)
    m3 = (w1 * ((mu1 - mean)**3 + 3 * (mu1 - mean) * sigma1**2) +
          w2 * ((mu2 - mean)**3 + 3 * (mu2 - mean) * sigma2**2))
    true_skewness = m3 / sd**3

    # True Bowley via numerical CDF inversion
    dx = grid[1] - grid[0]
    F = np.cumsum(f) * dx
    F = F / F[-1]

    def cdf_val(x_val):
        return w1 * norm.cdf(x_val, mu1, sigma1) + w2 * norm.cdf(x_val, mu2, sigma2)

    try:
        Q1 = brentq(lambda x: cdf_val(x) - 0.25, grid[0], grid[-1])
        Q2 = brentq(lambda x: cdf_val(x) - 0.50, grid[0], grid[-1])
        Q3 = brentq(lambda x: cdf_val(x) - 0.75, grid[0], grid[-1])
        iqr = Q3 - Q1
        true_bowley = (Q3 + Q1 - 2 * Q2) / iqr if iqr > 1e-15 else 0.0
    except (ValueError, RuntimeError):
        true_bowley = np.nan

    return f, float(true_bowley), float(true_skewness)


# ============================================================
# STEP 2: SYNTHETIC CALL PRICE GENERATION
# ============================================================
def synthetic_call_prices(grid, f, strikes_k, B):
    """
    Generate no-arbitrage call prices from a known density.

    C(k) = B * integral_k^inf (x - k) * f(x) dx
    """
    f_norm = f / integrate(grid, f)  # ensure normalized
    C = np.zeros(len(strikes_k))
    for i, ki in enumerate(strikes_k):
        mask = grid >= ki
        if mask.sum() < 2:
            C[i] = 0.0
            continue
        integrand = (grid[mask] - ki) * f_norm[mask]
        C[i] = B * integrate(grid[mask], integrand)
    C = np.maximum(C, 0.0)
    return C


def add_noise_and_project(C, k, noise_level, rng):
    """Add multiplicative noise and enforce monotonicity."""
    if noise_level <= 0:
        return C.copy()
    eps = rng.normal(0, noise_level, size=len(C))
    C_noisy = C * (1.0 + eps)
    C_noisy = np.maximum(C_noisy, 0.0)
    # Enforce non-increasing
    idx = np.argsort(k)
    C_sorted = C_noisy[idx].copy()
    for i in range(1, len(C_sorted)):
        if C_sorted[i] > C_sorted[i - 1]:
            C_sorted[i] = C_sorted[i - 1]
    C_noisy[idx] = C_sorted
    return C_noisy


# ============================================================
# STEP 3: ABLATION METHOD CONFIGURATIONS
# ============================================================
METHOD_CONFIGS = {
    "original": {
        "kernel": "epanechnikov",
        "tail_extrap": "flat",
        "bandwidth": "original",
        "grid_n": 200,
        "quantile_method": "linear",
        "edge_shutdown": True,
        "label": "Original (Epan+flat+200pt)",
    },
    "improved_all": {
        "kernel": "gaussian",
        "tail_extrap": "smooth_iv",
        "bandwidth": "lscv",
        "grid_n": 1600,
        "quantile_method": "spline",
        "edge_shutdown": False,
        "label": "Improved (all 7 changes)",
    },
    "ablation_kernel": {
        "kernel": "gaussian",
        "tail_extrap": "flat",
        "bandwidth": "original",
        "grid_n": 200,
        "quantile_method": "linear",
        "edge_shutdown": True,
        "label": "Abl: +Gaussian kernel",
    },
    "ablation_tails": {
        "kernel": "epanechnikov",
        "tail_extrap": "smooth_iv",
        "bandwidth": "original",
        "grid_n": 200,
        "quantile_method": "linear",
        "edge_shutdown": True,
        "label": "Abl: +smooth IV tails",
    },
    "ablation_grid": {
        "kernel": "epanechnikov",
        "tail_extrap": "flat",
        "bandwidth": "original",
        "grid_n": 1600,
        "quantile_method": "spline",
        "edge_shutdown": True,
        "label": "Abl: +fine grid+spline",
    },
    "ablation_bw": {
        "kernel": "epanechnikov",
        "tail_extrap": "flat",
        "bandwidth": "lscv",
        "grid_n": 200,
        "quantile_method": "linear",
        "edge_shutdown": True,
        "label": "Abl: +LSCV bandwidth",
    },
    "improved_flat_tails": {
        "kernel": "gaussian",
        "tail_extrap": "flat",
        "bandwidth": "lscv",
        "grid_n": 1600,
        "quantile_method": "spline",
        "edge_shutdown": False,
        "label": "Improved+flat tails (fix)",
    },
}


def kw_density_configurable(k_obs, C_obs, B, ypi, config):
    """
    Run KW density estimation with a specific configuration.

    Dispatches to the correct component functions based on config dict.
    Returns dict with density results, or None on failure.
    """
    k_obs = np.asarray(k_obs, float)
    C_obs = np.asarray(C_obs, float)

    # 1. Tail extrapolation
    if config["tail_extrap"] == "smooth_iv":
        k_ext, C_ext = extrapolate_call_prices_smooth(k_obs, C_obs, B, ypi)
    else:
        k_ext, C_ext = extrapolate_call_prices_flat(k_obs, C_obs, B, ypi)

    # 2. Bandwidth selection
    if config["bandwidth"] == "lscv":
        h = lscv_bandwidth(k_obs, C_obs, k_ext, C_ext)
    else:
        h = select_bandwidth_original(k_ext)

    if not np.isfinite(h) or h <= 0:
        return None

    # 3. Grid
    grid_n = config["grid_n"]
    kmin, kmax = float(np.min(k_ext)), float(np.max(k_ext))
    grid = np.linspace(kmin, kmax, grid_n)

    # 4. Local polynomial with chosen kernel
    if config["kernel"] == "gaussian":
        P, P1, P2 = local_poly_derivative_gauss(k_ext, C_ext, grid, h)
    else:
        P, P1, P2 = local_poly_derivative_epan(k_ext, C_ext, grid, h)

    if np.all(~np.isfinite(P2)):
        return None

    # 5. Density extraction
    f = (1.0 / B) * P2
    f = np.where(np.isfinite(f), f, 0.0)
    f = np.maximum(f, 0.0)

    Z_raw = integrate(grid, f)
    if not np.isfinite(Z_raw) or Z_raw <= 0:
        return None
    f = f / Z_raw

    # 6. Edge-band shutdown
    edge_ok = True
    if config["edge_shutdown"]:
        L, U = -0.01, 0.05
        eps_e = 0.003
        thr = 0.07
        mask_L = (grid >= L) & (grid <= L + eps_e)
        mask_U = (grid > U - eps_e) & (grid <= U)
        p_edge_L = integrate(grid[mask_L], f[mask_L]) if np.any(mask_L) else 0.0
        p_edge_U = integrate(grid[mask_U], f[mask_U]) if np.any(mask_U) else 0.0
        if not np.isfinite(p_edge_L):
            p_edge_L = 0.0
        if not np.isfinite(p_edge_U):
            p_edge_U = 0.0
        edge_ok = (p_edge_L <= thr) and (p_edge_U <= thr)

    # 7. Compute Bowley and moments
    if config["quantile_method"] == "spline":
        bw = bowley_skewness(grid, f)
    else:
        bw = bowley_skewness_linear(grid, f)

    mu, var, skew_m3, kurt = moments_from_density(grid, f)

    if not edge_ok:
        bw = np.nan
        skew_m3 = np.nan

    return {
        'grid': grid, 'density': f,
        'h': h, 'Z_raw': Z_raw,
        'bowley': bw, 'skewness_m3': skew_m3,
        'mean': mu, 'variance': var,
        'edge_ok': edge_ok,
    }


# ============================================================
# STEP 4: DISTRIBUTION DEFINITIONS
# ============================================================
DISTRIBUTIONS = [
    {
        "name": "normal_zero_skew",
        "type": "normal",
        "params": {"mu": 0.015, "sigma": 0.012},
        "desc": "Normal (skew=0)",
    },
    {
        "name": "skewnorm_neg2",
        "type": "skewnormal",
        "params": {"alpha": -2, "target_mean": 0.015, "omega": 0.012},
        "desc": "SkewNorm a=-2",
    },
    {
        "name": "skewnorm_neg1",
        "type": "skewnormal",
        "params": {"alpha": -1, "target_mean": 0.015, "omega": 0.012},
        "desc": "SkewNorm a=-1",
    },
    {
        "name": "skewnorm_pos1",
        "type": "skewnormal",
        "params": {"alpha": 1, "target_mean": 0.015, "omega": 0.012},
        "desc": "SkewNorm a=+1",
    },
    {
        "name": "skewnorm_pos2",
        "type": "skewnormal",
        "params": {"alpha": 2, "target_mean": 0.015, "omega": 0.012},
        "desc": "SkewNorm a=+2",
    },
    {
        "name": "mixture_left_skew",
        "type": "mixture",
        "params": {"mu1": 0.005, "sigma1": 0.015, "mu2": 0.020,
                   "sigma2": 0.006, "w1": 0.3},
        "desc": "Mixture (left-skew)",
    },
]


def generate_density(dist_def, grid):
    """Generate density on grid from distribution definition."""
    dtype = dist_def["type"]
    p = dist_def["params"]
    if dtype == "normal":
        return make_normal_density(p["mu"], p["sigma"], grid)
    elif dtype == "skewnormal":
        return make_skewnormal_density(p["alpha"], p["target_mean"], p["omega"], grid)
    elif dtype == "mixture":
        return make_mixture_density(p["mu1"], p["sigma1"], p["mu2"], p["sigma2"],
                                    p["w1"], grid)
    else:
        raise ValueError(f"Unknown distribution type: {dtype}")


# ============================================================
# STEP 5: MC ENGINE
# ============================================================
def run_mc_validation(n_reps=200, noise_levels=None, seed=42):
    """
    Full Monte Carlo experiment.

    Returns
    -------
    df_reps : pd.DataFrame -- one row per (replicate, method)
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.03, 0.05]

    rng = np.random.default_rng(seed)
    rows = []
    total_combos = len(DISTRIBUTIONS) * len(noise_levels)
    combo_idx = 0

    for dist_def in DISTRIBUTIONS:
        # Generate ground truth density (once per distribution)
        f_true, true_bowley, true_skew = generate_density(dist_def, FINE_GRID)
        f_norm = f_true / integrate(FINE_GRID, f_true)
        true_mean = integrate(FINE_GRID, FINE_GRID * f_norm)

        # Generate clean call prices (once per distribution)
        C_clean = synthetic_call_prices(FINE_GRID, f_true, STRIKES, B_DEFAULT)

        # Verify call prices are valid
        if np.any(C_clean < 0) or np.any(np.diff(C_clean) > 1e-10):
            print(f"  WARNING: Call prices for {dist_def['name']} violate no-arb constraints")

        for noise_level in noise_levels:
            combo_idx += 1
            actual_reps = 1 if noise_level == 0.0 else n_reps

            print(f"  [{combo_idx}/{total_combos}] {dist_def['desc']}, "
                  f"noise={noise_level}, reps={actual_reps}")

            for rep in range(actual_reps):
                # Add noise
                C_noisy = add_noise_and_project(C_clean, STRIKES, noise_level, rng)

                # Run each method
                for method_name, config in METHOD_CONFIGS.items():
                    ypi = float(true_mean)
                    result = kw_density_configurable(
                        STRIKES, C_noisy, B_DEFAULT, ypi, config)

                    if result is not None:
                        est_bowley = result['bowley']
                        est_skew = result['skewness_m3']
                        est_mean = result['mean']
                        est_var = result['variance']
                        h_val = result['h']
                        z_raw = result['Z_raw']
                        success = True
                    else:
                        est_bowley = est_skew = est_mean = est_var = np.nan
                        h_val = z_raw = np.nan
                        success = False

                    rows.append({
                        'rep': rep,
                        'dist': dist_def['name'],
                        'dist_desc': dist_def['desc'],
                        'true_bowley': true_bowley,
                        'true_skewness': true_skew,
                        'true_mean': true_mean,
                        'noise': noise_level,
                        'method': method_name,
                        'label': config['label'],
                        'est_bowley': est_bowley,
                        'est_skewness': est_skew,
                        'est_mean': est_mean,
                        'est_variance': est_var,
                        'h': h_val,
                        'Z_raw': z_raw,
                        'success': success,
                    })

    return pd.DataFrame(rows)


# ============================================================
# STEP 6: METRICS AND PASS/FAIL
# ============================================================
def compute_summary(df_reps):
    """Aggregate per-replicate results into summary statistics."""
    groups = df_reps.groupby(['dist', 'dist_desc', 'noise', 'method', 'label'])
    summary_rows = []

    for (dist, desc, noise, method, label), grp in groups:
        true_bw = grp['true_bowley'].iloc[0]
        true_sk = grp['true_skewness'].iloc[0]

        # Filter to finite estimates
        bw_vals = grp['est_bowley'].dropna()
        sk_vals = grp['est_skewness'].dropna()

        n_total = len(grp)
        n_success = grp['success'].sum()

        if len(bw_vals) > 0:
            bias_bw = float((bw_vals - true_bw).mean())
            rmse_bw = float(np.sqrt(((bw_vals - true_bw)**2).mean()))

            if abs(true_bw) > 0.005:  # nonzero true skew
                sign_agree = float((np.sign(bw_vals) == np.sign(true_bw)).mean())
            else:
                sign_agree = np.nan  # not meaningful for zero-skew
        else:
            bias_bw = rmse_bw = sign_agree = np.nan

        if len(sk_vals) > 0:
            bias_sk = float((sk_vals - true_sk).mean())
            rmse_sk = float(np.sqrt(((sk_vals - true_sk)**2).mean()))
        else:
            bias_sk = rmse_sk = np.nan

        summary_rows.append({
            'dist': dist, 'dist_desc': desc,
            'noise': noise, 'method': method, 'label': label,
            'true_bowley': true_bw, 'true_skewness': true_sk,
            'bias_bowley': bias_bw, 'rmse_bowley': rmse_bw,
            'bias_skewness': bias_sk, 'rmse_skewness': rmse_sk,
            'sign_agreement': sign_agree,
            'success_rate': n_success / max(n_total, 1),
            'n_reps': n_total,
            'mean_h': grp['h'].mean(),
        })

    return pd.DataFrame(summary_rows)


def determine_pass_fail(df_summary):
    """Apply pass/fail criteria to each method."""
    results = []
    methods = df_summary['method'].unique()

    for method in methods:
        ms = df_summary[df_summary['method'] == method]
        label = ms['label'].iloc[0]

        # Criterion 1: Normal bias < 0.02
        normal_clean = ms[(ms['dist'] == 'normal_zero_skew') & (ms['noise'] == 0.0)]
        if len(normal_clean) > 0:
            bias_normal = abs(normal_clean['bias_bowley'].iloc[0])
            results.append({
                'method': method, 'label': label,
                'criterion': 'BIAS_NORMAL',
                'value': bias_normal,
                'threshold': 0.02,
                'passed': bias_normal < 0.02,
            })

        # Criterion 2: Skew-normal bias < 0.05
        skewnorm_clean = ms[(ms['dist'].str.startswith('skewnorm')) & (ms['noise'] == 0.0)]
        if len(skewnorm_clean) > 0:
            max_bias = skewnorm_clean['bias_bowley'].abs().max()
            results.append({
                'method': method, 'label': label,
                'criterion': 'BIAS_SKEW',
                'value': max_bias,
                'threshold': 0.05,
                'passed': max_bias < 0.05,
            })

        # Criterion 3: Sign agreement >= 0.90 (clean skew-normal)
        skewnorm_sign = skewnorm_clean.dropna(subset=['sign_agreement'])
        if len(skewnorm_sign) > 0:
            min_sign = skewnorm_sign['sign_agreement'].min()
            results.append({
                'method': method, 'label': label,
                'criterion': 'SIGN_RATE',
                'value': min_sign,
                'threshold': 0.90,
                'passed': min_sign >= 0.90,
            })

        # Criterion 4: Success rate >= 0.95
        min_success = ms['success_rate'].min()
        results.append({
            'method': method, 'label': label,
            'criterion': 'SUCCESS',
            'value': min_success,
            'threshold': 0.95,
            'passed': min_success >= 0.95,
        })

        # Criterion 5: RMSE at noise=0.03 < 0.15
        noisy = ms[ms['noise'] == 0.03]
        if len(noisy) > 0:
            max_rmse = noisy['rmse_bowley'].max()
            results.append({
                'method': method, 'label': label,
                'criterion': 'RMSE_NOISE',
                'value': max_rmse,
                'threshold': 0.15,
                'passed': max_rmse < 0.15 if np.isfinite(max_rmse) else False,
            })

    return pd.DataFrame(results)


# ============================================================
# STEP 7: PLOTS
# ============================================================
def plot_ablation_bias(df_summary, output_dir):
    """Bar chart: Bowley bias by method, panel per distribution (noise=0)."""
    clean = df_summary[df_summary['noise'] == 0.0].copy()
    dists = clean['dist_desc'].unique()
    methods = clean['method'].unique()
    n_dists = len(dists)

    fig, axes = plt.subplots(1, n_dists, figsize=(3.5 * n_dists, 5), sharey=True)
    if n_dists == 1:
        axes = [axes]

    for ax, dist_desc in zip(axes, dists):
        sub = clean[clean['dist_desc'] == dist_desc].set_index('method')
        biases = [sub.loc[m, 'bias_bowley'] if m in sub.index else np.nan
                  for m in methods]
        labels = [sub.loc[m, 'label'] if m in sub.index else m
                  for m in methods]
        colors = ['C0' if m == 'original' else
                  'C3' if m == 'improved_all' else
                  'C2' if m == 'improved_flat_tails' else
                  'C7' for m in methods]

        bars = ax.barh(range(len(methods)), biases, color=colors, alpha=0.8)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.axvline(-0.02, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axvline(0.02, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_xlabel("Bowley bias")
        ax.set_title(dist_desc, fontsize=9)

    fig.suptitle("Ablation Study: Bowley Bias (noise=0)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_mc_bias_ablation.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def plot_rmse_vs_noise(df_summary, output_dir):
    """Line chart: RMSE vs noise level, one line per method."""
    dists = df_summary['dist_desc'].unique()
    n_dists = len(dists)

    fig, axes = plt.subplots(1, n_dists, figsize=(3.5 * n_dists, 4), sharey=True)
    if n_dists == 1:
        axes = [axes]

    for ax, dist_desc in zip(axes, dists):
        sub = df_summary[df_summary['dist_desc'] == dist_desc]
        for method in sub['method'].unique():
            ms = sub[sub['method'] == method].sort_values('noise')
            label_short = ms['label'].iloc[0]
            lw = 2 if method in ('original', 'improved_all', 'improved_flat_tails') else 1
            ls = '-' if method in ('original', 'improved_all', 'improved_flat_tails') else '--'
            ax.plot(ms['noise'], ms['rmse_bowley'], marker='o', markersize=3,
                    linewidth=lw, linestyle=ls, label=label_short)
        ax.set_xlabel("Noise level")
        ax.set_title(dist_desc, fontsize=9)

    axes[0].set_ylabel("RMSE (Bowley)")
    axes[-1].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1.02, 1))
    fig.suptitle("RMSE vs Noise Level", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_mc_rmse_vs_noise.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def plot_scatter_est_vs_true(df_reps, output_dir):
    """Scatter: estimated vs true Bowley, panel per method (noise=0)."""
    clean = df_reps[df_reps['noise'] == 0.0].copy()
    methods_to_show = ['original', 'improved_all', 'improved_flat_tails',
                       'ablation_tails', 'ablation_kernel']
    methods_present = [m for m in methods_to_show if m in clean['method'].unique()]
    n_m = len(methods_present)

    fig, axes = plt.subplots(1, n_m, figsize=(3.5 * n_m, 3.5), sharey=True)
    if n_m == 1:
        axes = [axes]

    for ax, method in zip(axes, methods_present):
        sub = clean[clean['method'] == method]
        ax.scatter(sub['true_bowley'], sub['est_bowley'], s=30, alpha=0.7,
                   c=[{'normal_zero_skew': 'C0', 'skewnorm_neg2': 'C3',
                       'skewnorm_neg1': 'C1', 'skewnorm_pos1': 'C2',
                       'skewnorm_pos2': 'C4', 'mixture_left_skew': 'C5'
                       }.get(d, 'gray') for d in sub['dist']])
        lims = [-0.15, 0.15]
        ax.plot(lims, lims, 'k--', linewidth=0.5)
        ax.axhline(0, color='gray', linewidth=0.3)
        ax.axvline(0, color='gray', linewidth=0.3)
        ax.set_xlim(lims)
        ax.set_ylim([-0.5, 0.5])
        ax.set_xlabel("True Bowley")
        ax.set_title(sub['label'].iloc[0], fontsize=8)

    axes[0].set_ylabel("Estimated Bowley")
    fig.suptitle("Estimated vs True Bowley (noise=0)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_mc_scatter_est_vs_true.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def plot_pass_fail(df_pf, output_dir):
    """Summary pass/fail table as figure."""
    methods = df_pf['method'].unique()
    criteria = df_pf['criterion'].unique()

    fig, ax = plt.subplots(figsize=(10, 0.4 * len(methods) + 2))
    ax.axis('off')

    col_labels = ['Method'] + list(criteria)
    cell_text = []
    cell_colors = []

    for method in methods:
        ms = df_pf[df_pf['method'] == method]
        label = ms['label'].iloc[0]
        row_text = [label]
        row_colors = ['white']
        for crit in criteria:
            cr = ms[ms['criterion'] == crit]
            if len(cr) > 0:
                passed = cr['passed'].iloc[0]
                val = cr['value'].iloc[0]
                thr = cr['threshold'].iloc[0]
                txt = f"{'PASS' if passed else 'FAIL'}\n{val:.3f}"
                row_text.append(txt)
                row_colors.append('#c8e6c9' if passed else '#ffcdd2')
            else:
                row_text.append('N/A')
                row_colors.append('#e0e0e0')
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellColours=cell_colors, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)

    fig.suptitle("MC Validation Pass/Fail Summary", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_mc_pass_fail.pdf"),
                bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    quick = "--quick" in sys.argv

    if quick:
        n_reps = 10
        noise_levels = [0.0, 0.03]
        print("=" * 60)
        print("MC VALIDATION (QUICK MODE)")
        print("=" * 60)
    else:
        n_reps = 200
        noise_levels = [0.0, 0.01, 0.03, 0.05]
        print("=" * 60)
        print("MC VALIDATION (FULL MODE)")
        print("=" * 60)

    print(f"\nDistributions: {len(DISTRIBUTIONS)}")
    print(f"Noise levels: {noise_levels}")
    print(f"Replicates (noisy): {n_reps}")
    print(f"Methods: {len(METHOD_CONFIGS)}")
    est_fits = len(DISTRIBUTIONS) * (1 + len([n for n in noise_levels if n > 0]) * n_reps) * len(METHOD_CONFIGS)
    print(f"Estimated density fits: {est_fits}")
    print()

    t0 = time.time()
    df_reps = run_mc_validation(n_reps=n_reps, noise_levels=noise_levels)
    elapsed = time.time() - t0
    print(f"\nMC completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save replicates
    df_reps.to_csv(OUTPUT_REPS_CSV, index=False)
    print(f"Replicates saved to {OUTPUT_REPS_CSV}")

    # Compute summary
    df_summary = compute_summary(df_reps)
    df_summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    print(f"Summary saved to {OUTPUT_SUMMARY_CSV}")

    # Pass/fail
    df_pf = determine_pass_fail(df_summary)

    # Print key results
    print("\n" + "=" * 60)
    print("KEY RESULTS: BOWLEY BIAS AT noise=0 (pure pipeline bias)")
    print("=" * 60)
    clean_summary = df_summary[df_summary['noise'] == 0.0]
    for method in METHOD_CONFIGS:
        ms = clean_summary[clean_summary['method'] == method]
        if len(ms) == 0:
            continue
        label = ms['label'].iloc[0]
        biases = ms[['dist_desc', 'bias_bowley']].set_index('dist_desc')
        print(f"\n  {label}:")
        for dist_desc, row in biases.iterrows():
            bias_val = row['bias_bowley']
            flag = " *** BIAS" if abs(bias_val) > 0.02 else ""
            print(f"    {dist_desc:25s}  bias = {bias_val:+.4f}{flag}")

    # Print pass/fail
    print("\n" + "=" * 60)
    print("PASS/FAIL SUMMARY")
    print("=" * 60)
    for method in METHOD_CONFIGS:
        pf = df_pf[df_pf['method'] == method]
        if len(pf) == 0:
            continue
        label = pf['label'].iloc[0]
        n_pass = pf['passed'].sum()
        n_total = len(pf)
        print(f"\n  {label}: {n_pass}/{n_total} passed")
        for _, row in pf.iterrows():
            status = "PASS" if row['passed'] else "FAIL"
            print(f"    [{status}] {row['criterion']}: "
                  f"{row['value']:.4f} (threshold: {row['threshold']:.4f})")

    # Plots
    print("\nGenerating plots...")
    plot_ablation_bias(df_summary, FIG_DIR)
    plot_rmse_vs_noise(df_summary, FIG_DIR)
    plot_scatter_est_vs_true(df_reps, FIG_DIR)
    plot_pass_fail(df_pf, FIG_DIR)

    print(f"Plots saved to {FIG_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
