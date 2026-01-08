# -*- coding: utf-8 -*-
"""
Master-level (simple) script for the project:
- Build 1Y inflation call price curve from caps + floors (via parity)
- Method 1: BL (Breeden–Litzenberger) => density => moments
- Method 2: "BKM-style" moments directly from option integrals (on G = 1+pi)

How to use:
1) Put the 4 CSV files in one folder:
   - cleaned_caps_quotes_1y.csv
   - cleaned_floors_quotes_1y.csv
   - cleaned_swaps_curves_1y.csv
   - manifest_coverage_1y.csv
2) Change DATA_DIR below.
3) Run:  python familyname1_familyname2_option_prices.py

Output:
- familyname1_familyname2_option_prices.csv (saved inside DATA_DIR)

Notes:
- price_per_1 is in "percent of notional" in these files (e.g. 0.3222 = 0.3222%).
  We convert to decimals by dividing by 100.
- BL uses finite differences, so it is sensitive to noise. We apply a simple
  monotonicity fix to stabilize it (call prices should be non-increasing in strike).
"""

import os
import numpy as np
import pandas as pd

# =========================
# 1) EDIT THIS ONLY
# =========================
DATA_DIR = "/mnt/data"   # <-- change to your folder path
OUT_NAME = "familyname1_familyname2_option_prices.csv"

# =========================
# 2) Small helper functions
# =========================
def trapz(x, y):
    """Safe trapezoid integral."""
    if len(x) < 2:
        return np.nan
    return np.trapz(y, x)

def moments_from_density(x, f):
    """Return mean, var, skew, kurt for a density f(x) on grid x."""
    Z = trapz(x, f)
    if not np.isfinite(Z) or Z <= 0:
        return (np.nan, np.nan, np.nan, np.nan)
    f = f / Z
    mu = trapz(x, x * f)
    var = trapz(x, (x - mu) ** 2 * f)
    if var <= 0 or not np.isfinite(var):
        return (mu, np.nan, np.nan, np.nan)
    sd = np.sqrt(var)
    m3 = trapz(x, (x - mu) ** 3 * f)
    m4 = trapz(x, (x - mu) ** 4 * f)
    skew = m3 / (sd ** 3)
    kurt = m4 / (sd ** 4)
    return (mu, var, skew, kurt)

def build_call_curve_one_date(caps_df, floors_df, B, ypi):
    """
    Build call prices C(k) at observed strikes k for one (date, area).
    - Caps are calls directly.
    - Floors are puts; convert to calls using parity:
        C(k) = P(k) + B*(E[pi] - k), with E[pi] ~ ypi_n (1Y swap rate)
    Returns: DataFrame columns [k, C] (C in decimals of notional)
    """
    # Convert % to decimal premium
    caps = caps_df.copy()
    floors = floors_df.copy()
    caps["C"] = caps["price_per_1"] / 100.0
    floors["P"] = floors["price_per_1"] / 100.0

    # Calls from floors via parity
    floors["C"] = floors["P"] + B * (ypi - floors["k"])

    # Stack calls from caps + floors
    cc = pd.concat(
        [
            caps[["k", "C"]],
            floors[["k", "C"]],
        ],
        ignore_index=True
    )

    # Average if duplicates
    cc = cc.groupby("k", as_index=False)["C"].mean().sort_values("k").reset_index(drop=True)

    # Basic no-arbitrage stabilization: calls should be non-increasing in strike
    # (higher strike => lower call price)
    C = cc["C"].to_numpy()
    for i in range(1, len(C)):
        if C[i] > C[i-1]:
            C[i] = C[i-1]
    cc["C"] = np.maximum(C, 0.0)  # calls can't be negative

    return cc

def bl_density_from_call(k, C, B):
    """
    BL density: f(k) = (1/B) * d^2 C / dk^2
    Use finite differences on a regular grid (linear interpolation).
    """
    if len(k) < 5:
        return None

    # Regular grid on strike
    kmin, kmax = float(np.min(k)), float(np.max(k))
    grid = np.linspace(kmin, kmax, 200)

    # Linear interpolation of call prices
    Cg = np.interp(grid, k, C)

    # Second derivative with finite differences
    d1 = np.gradient(Cg, grid)
    d2 = np.gradient(d1, grid)
    f = (1.0 / B) * d2

    # Post-process: clip negatives and renormalize (on the grid only)
    f = np.maximum(f, 0.0)
    Z = trapz(grid, f)
    if not np.isfinite(Z) or Z <= 0:
        return None
    f = f / Z

    return grid, f

def bkm_style_moments_from_calls(k, C, B, ypi):
    """
    "BKM-style" direct moments for G = 1 + pi (positive):
      Price of call on G with strike K is: C(K) = B * E[(G - K)^+]

    Known identity (for m >= 2):
      E[G^m] = m(m-1) * ∫_0^∞ K^(m-2) * E[(G-K)^+] dK
            = m(m-1)/B * ∫_0^∞ K^(m-2) * C(K) dK

    We approximate the integral on available strikes, and:
      - add K=0 point: C(0) = B*E[G] = B*(1+ypi)
      - add K=Kmax point with C=0 beyond observed range (conservative)
    """
    if len(k) < 3:
        return (np.nan, np.nan, np.nan, np.nan)

    # convert to gross strike K = 1 + k
    K = 1.0 + np.asarray(k, dtype=float)
    C = np.asarray(C, dtype=float)

    # add K=0 and K=Kmax_tail
    Kmax = float(np.max(K))
    K_tail = Kmax + 0.02  # small extension to anchor to zero
    K_aug = np.concatenate(([0.0], K, [K_tail]))
    C_aug = np.concatenate(([B * (1.0 + ypi)], C, [0.0]))

    # sort
    idx = np.argsort(K_aug)
    K_aug = K_aug[idx]
    C_aug = C_aug[idx]

    # helper for raw moments of G
    def EG_m(m):
        if m == 1:
            return 1.0 + ypi
        coeff = m * (m - 1) / B
        integrand = (K_aug ** (m - 2)) * C_aug
        return coeff * trapz(K_aug, integrand)

    EG1 = EG_m(1)
    EG2 = EG_m(2)
    EG3 = EG_m(3)
    EG4 = EG_m(4)

    # central moments of G (same as pi up to a shift)
    muG = EG1
    varG = EG2 - EG1 ** 2
    if not np.isfinite(varG) or varG <= 0:
        return (muG - 1.0, np.nan, np.nan, np.nan)

    # central 3rd and 4th from raw moments
    mu3 = EG3 - 3 * muG * EG2 + 2 * (muG ** 3)
    mu4 = EG4 - 4 * muG * EG3 + 6 * (muG ** 2) * EG2 - 3 * (muG ** 4)

    sd = np.sqrt(varG)
    skew = mu3 / (sd ** 3)
    kurt = mu4 / (sd ** 4)

    # moments of pi = G - 1: mean shifts by -1, others unchanged
    mu_pi = muG - 1.0
    return (mu_pi, varG, skew, kurt)

# =========================
# 3) Load data
# =========================
caps = pd.read_csv(os.path.join(DATA_DIR, "cleaned_caps_quotes_1y.csv"))
floors = pd.read_csv(os.path.join(DATA_DIR, "cleaned_floors_quotes_1y.csv"))
swaps = pd.read_csv(os.path.join(DATA_DIR, "cleaned_swaps_curves_1y.csv"))
manifest = pd.read_csv(os.path.join(DATA_DIR, "manifest_coverage_1y.csv"))

# ensure types
for df in (caps, floors, swaps, manifest):
    df["date"] = pd.to_datetime(df["date"])

# quick filters: n=1 only (should already be true)
caps = caps[caps["n"] == 1].copy()
floors = floors[floors["n"] == 1].copy()
swaps = swaps[swaps["n"] == 1].copy()
manifest = manifest[manifest["n"] == 1].copy()

# =========================
# 4) Loop on (date, area)
# =========================
rows = []
keys = swaps[["date", "area"]].drop_duplicates().sort_values(["area", "date"])

for (dt, area) in keys.itertuples(index=False):
    sw = swaps[(swaps["date"] == dt) & (swaps["area"] == area)]
    if sw.empty:
        continue
    B = float(sw["B"].iloc[0])
    ypi = float(sw["ypi_n"].iloc[0])

    caps_g = caps[(caps["date"] == dt) & (caps["area"] == area)]
    floors_g = floors[(floors["date"] == dt) & (floors["area"] == area)]
    if caps_g.empty and floors_g.empty:
        continue

    call_curve = build_call_curve_one_date(caps_g, floors_g, B=B, ypi=ypi)
    k = call_curve["k"].to_numpy()
    C = call_curve["C"].to_numpy()

    # ---- Method 1: BL moments
    bl = bl_density_from_call(k, C, B)
    if bl is None:
        mu_bl = var_bl = skew_bl = kurt_bl = np.nan
        p_defl = p_hi = np.nan
    else:
        kgrid, f = bl
        mu_bl, var_bl, skew_bl, kurt_bl = moments_from_density(kgrid, f)
        # tail probs like Kitsul-Wright: P(pi<0), P(pi>4%)
        p_defl = trapz(kgrid[kgrid < 0], f[kgrid < 0]) if np.any(kgrid < 0) else 0.0
        p_hi = trapz(kgrid[kgrid > 0.04], f[kgrid > 0.04]) if np.any(kgrid > 0.04) else 0.0

    # ---- Method 2: BKM-style moments
    mu_bkm, var_bkm, skew_bkm, kurt_bkm = bkm_style_moments_from_calls(k, C, B, ypi)

    # ---- coverage diagnostics
    man = manifest[(manifest["date"] == dt) & (manifest["area"] == area)]
    if not man.empty:
        Kmin = float(man["K_min_obs"].iloc[0])
        Kmax = float(man["K_max_obs"].iloc[0])
        Kwidth = Kmax - Kmin
    else:
        Kmin = float(np.min(1.0 + k))
        Kmax = float(np.max(1.0 + k))
        Kwidth = Kmax - Kmin

    rows.append({
        "date": dt.date().isoformat(),
        "area": area,
        "B": B,
        "ypi_n": ypi,
        "K_min_obs": Kmin,
        "K_max_obs": Kmax,
        "K_width": Kwidth,

        "mean_BL": mu_bl,
        "var_BL": var_bl,
        "skew_BL": skew_bl,
        "kurt_BL": kurt_bl,
        "p_deflation_BL": p_defl,
        "p_highinfl_BL": p_hi,

        "mean_BKM": mu_bkm,
        "var_BKM": var_bkm,
        "skew_BKM": skew_bkm,
        "kurt_BKM": kurt_bkm,
    })

out = pd.DataFrame(rows).sort_values(["area", "date"])
out_path = os.path.join(DATA_DIR, OUT_NAME)
out.to_csv(out_path, index=False)

print("Saved:", out_path)
print(out.head(10).to_string(index=False))
