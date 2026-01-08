#!/usr/bin/env python3
"""
Skeleton: Inflation option-implied moments (1Y) using
(1) Breeden–Litzenberger (BL): prices -> pdf -> moments
(2) "BKM-style" model-free moments: prices -> moments (via option integrals)

Designed for the provided project CSVs:
- cleaned_caps_quotes_1y.csv
- cleaned_floors_quotes_1y.csv
- cleaned_swaps_curves_1y.csv
- manifest_coverage_1y.csv

Usage:
  python familyname1_familyname2_option_prices.py --data_dir /path/to/csvs --out_csv results.csv

Notes:
- The input column `price_per_1` appears to be in percent of notional. We convert to decimal by /100.
- All computations are per (date, area) with n=1.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------
def _to_decimal_price(df: pd.DataFrame) -> pd.DataFrame:
    """Convert `price_per_1` from percent-of-notional to decimal-of-notional."""
    out = df.copy()
    out["price"] = out["price_per_1"].astype(float) / 100.0
    return out


def load_inputs(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load the 4 project CSVs."""
    caps = pd.read_csv(data_dir / "cleaned_caps_quotes_1y.csv")
    floors = pd.read_csv(data_dir / "cleaned_floors_quotes_1y.csv")
    swaps = pd.read_csv(data_dir / "cleaned_swaps_curves_1y.csv")
    manifest = pd.read_csv(data_dir / "manifest_coverage_1y.csv")

    # Basic typing
    for df in (caps, floors, swaps, manifest):
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["area"] = df["area"].astype(str)
        df["n"] = df["n"].astype(int)

    # Filter to n=1 (project is 1Y)
    caps = caps[caps["n"] == 1].copy()
    floors = floors[floors["n"] == 1].copy()
    swaps = swaps[swaps["n"] == 1].copy()
    manifest = manifest[manifest["n"] == 1].copy()

    caps = _to_decimal_price(caps)
    floors = _to_decimal_price(floors)

    return {"caps": caps, "floors": floors, "swaps": swaps, "manifest": manifest}


@dataclass
class SliceInputs:
    """All inputs for a (date, area) slice."""
    date: object
    area: str
    B: float
    ypi: float
    K_star: float
    call_df: pd.DataFrame  # columns: k, K, C (decimal), source
    K_min_obs: Optional[float] = None
    K_max_obs: Optional[float] = None


def build_call_curve_for_slice(
    date: object,
    area: str,
    caps: pd.DataFrame,
    floors: pd.DataFrame,
    swaps: pd.DataFrame,
    manifest: pd.DataFrame,
) -> Optional[SliceInputs]:
    """
    Build a unified call price curve C(k) for a given (date, area).
    Uses:
      - caps as direct calls
      - floors converted to calls via put-call parity:
          C(k) = P_floor(k) + B*(ypi - k)
        where ypi is the 1Y inflation swap rate (forward mean of inflation).
    """
    sw = swaps[(swaps["date"] == date) & (swaps["area"] == area)]
    if sw.empty:
        return None
    B = float(sw.iloc[0]["B"])
    ypi = float(sw.iloc[0]["ypi_n"])
    K_star = float(sw.iloc[0].get("K_star", 1.0 + ypi))

    cap_slice = caps[(caps["date"] == date) & (caps["area"] == area)].copy()
    floor_slice = floors[(floors["date"] == date) & (floors["area"] == area)].copy()
    if cap_slice.empty and floor_slice.empty:
        return None

    # Caps: direct call prices
    cap_calls = cap_slice[["k", "K", "price"]].copy()
    cap_calls.rename(columns={"price": "C"}, inplace=True)
    cap_calls["source_call"] = "cap_direct"

    # Floors -> calls via parity
    # C_from_floor = P_floor + B*(ypi - k)
    floor_calls = floor_slice[["k", "K", "price"]].copy()
    floor_calls["C"] = floor_calls["price"] + B * (ypi - floor_calls["k"].astype(float))
    floor_calls = floor_calls[["k", "K", "C"]]
    floor_calls["source_call"] = "floor_parity"

    # Merge & average if duplicate strikes exist
    call_df = pd.concat([cap_calls[["k", "K", "C", "source_call"]],
                         floor_calls[["k", "K", "C", "source_call"]]], ignore_index=True)
    call_df["k"] = call_df["k"].astype(float)
    call_df["K"] = call_df["K"].astype(float)
    call_df["C"] = call_df["C"].astype(float)

    call_df = (
        call_df.groupby(["k", "K"], as_index=False)
        .agg(C=("C", "mean"), n_sources=("source_call", "count"))
        .sort_values("k")
        .reset_index(drop=True)
    )

    # Attach coverage diagnostics if present
    man = manifest[(manifest["date"] == date) & (manifest["area"] == area)]
    K_min_obs = float(man.iloc[0]["K_min_obs"]) if not man.empty else None
    K_max_obs = float(man.iloc[0]["K_max_obs"]) if not man.empty else None

    return SliceInputs(
        date=date, area=area, B=B, ypi=ypi, K_star=K_star,
        call_df=call_df, K_min_obs=K_min_obs, K_max_obs=K_max_obs
    )


# -----------------------------
# No-arbitrage projection (optional but recommended)
# -----------------------------
def project_call_noarb(k: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Project call prices onto the closest (L2) set satisfying basic no-arbitrage:
      - C >= 0
      - C is non-increasing in strike
      - C is convex in strike (second finite differences >= 0)
    Uses CVXPY if available; otherwise returns original C.
    """
    try:
        import cvxpy as cp  # type: ignore
    except Exception:
        return C

    n = len(C)
    x = cp.Variable(n)

    constraints = [x >= 0]
    # monotone decreasing in strike:
    for i in range(n - 1):
        constraints += [x[i + 1] <= x[i]]

    # convexity: second differences >= 0 (assumes k sorted)
    # Works best for near-uniform spacing; still acceptable for small irregularities
    for i in range(1, n - 1):
        constraints += [x[i + 1] - 2 * x[i] + x[i - 1] >= 0]

    obj = cp.Minimize(cp.sum_squares(x - C))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    if x.value is None:
        return C
    return np.asarray(x.value).reshape(-1)


# -----------------------------
# BL density: prices -> pdf -> moments
# -----------------------------
def bl_density_from_calls(
    k: np.ndarray,
    C: np.ndarray,
    B: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discrete BL density approximation:
      f(k_i) ~ (1/B) * d2C/dk2
    We estimate d2C with second finite differences.
    Returns:
      k_grid (same as input)
      pdf (same length, endpoints set to 0)
    """
    # Ensure sorted
    order = np.argsort(k)
    k = k[order]
    C = C[order]

    # Second derivative via non-uniform finite differences
    pdf = np.zeros_like(C, dtype=float)
    for i in range(1, len(k) - 1):
        k0, k1, k2 = k[i - 1], k[i], k[i + 1]
        C0, C1, C2 = C[i - 1], C[i], C[i + 1]

        # Non-uniform second derivative formula (quadratic fit through 3 points)
        denom = (k0 - k1) * (k0 - k2) * (k1 - k2)
        # second derivative of interpolating parabola at k1:
        d2 = 2 * (
            C0 * (k1 - k2)
            + C1 * (k2 - k0)
            + C2 * (k0 - k1)
        ) / denom
        pdf[i] = (1.0 / B) * d2

    # Clean up: clip negatives and renormalize
    pdf = np.maximum(pdf, 0.0)
    area = np.trapz(pdf, k)
    if area > 0:
        pdf = pdf / area
    return k, pdf


def moments_from_pdf(k: np.ndarray, pdf: np.ndarray) -> Dict[str, float]:
    """Compute mean/var/skew/kurtosis from a (k, pdf) grid."""
    k = np.asarray(k, dtype=float)
    pdf = np.asarray(pdf, dtype=float)
    # Normalize defensively
    area = np.trapz(pdf, k)
    if area <= 0:
        return {"mean": np.nan, "var": np.nan, "skew": np.nan, "kurt": np.nan}
    pdf = pdf / area

    mean = float(np.trapz(k * pdf, k))
    m2 = float(np.trapz(((k - mean) ** 2) * pdf, k))
    m3 = float(np.trapz(((k - mean) ** 3) * pdf, k))
    m4 = float(np.trapz(((k - mean) ** 4) * pdf, k))

    if m2 <= 0:
        return {"mean": mean, "var": 0.0, "skew": np.nan, "kurt": np.nan}

    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2)  # (non-excess) kurtosis
    return {"mean": mean, "var": m2, "skew": float(skew), "kurt": float(kurt)}


def tail_probs_from_pdf(k: np.ndarray, pdf: np.ndarray) -> Dict[str, float]:
    """Example tail probs: P(pi<0), P(pi>0.04)."""
    k = np.asarray(k, float)
    pdf = np.asarray(pdf, float)
    area = np.trapz(pdf, k)
    if area <= 0:
        return {"p_deflation": np.nan, "p_highinfl": np.nan}
    pdf = pdf / area

    # Integrate on subsets
    def _integrate(mask: np.ndarray) -> float:
        kk = k[mask]
        pp = pdf[mask]
        if len(kk) < 2:
            return 0.0
        return float(np.trapz(pp, kk))

    p_defl = _integrate(k < 0.0)
    p_hi = _integrate(k > 0.04)
    return {"p_deflation": p_defl, "p_highinfl": p_hi}


# -----------------------------
# BKM-style moments: prices -> moments (via option integrals)
# -----------------------------
def bkm_style_moments_from_calls(
    k: np.ndarray,
    C: np.ndarray,
    B: float,
    ypi: float,
    K_star: float,
    max_order: int = 4,
) -> Dict[str, float]:
    """
    "BKM-style" model-free moments for inflation, adapted to 1Y inflation options.

    We work with G = 1 + pi (gross inflation factor), which is > 0.
    Observed strikes are K = 1 + k. We build call prices C(K) and use:
        E[(G - K)^+] = C(K)/B

    For m >= 2 and nonnegative G, one identity is:
        E[G^m] = m(m-1) * ∫_0^∞ K^{m-2} E[(G-K)^+] dK
              = m(m-1) * ∫_0^∞ K^{m-2} (C(K)/B) dK
    With finite strike coverage, we approximate integral on [0, K_max_obs] and assume C(K)=0 beyond K_max_obs.

    This tends to UNDERSTATE higher moments if the right tail is not well covered.
    """
    k = np.asarray(k, float)
    C = np.asarray(C, float)
    K = 1.0 + k

    # Add an anchor at K=0 with call price C(0)=B*E[G]=B*K_star
    # (since (G-0)^+ = G)
    K_ext = np.concatenate([[0.0], K])
    C_ext = np.concatenate([[B * K_star], C])

    # Sort by K
    order = np.argsort(K_ext)
    K_ext = K_ext[order]
    C_ext = C_ext[order]

    # Ensure non-negative call prices
    C_ext = np.maximum(C_ext, 0.0)

    # Compute raw moments of G up to max_order
    EG = {0: 1.0, 1: K_star}  # E[G^0]=1, E[G^1]=E[G]=1+ypi
    for m in range(2, max_order + 1):
        integrand = (K_ext ** (m - 2)) * (C_ext / B)
        integral = float(np.trapz(integrand, K_ext))
        EG[m] = m * (m - 1) * integral

    # Convert to raw moments of pi = G - 1 via binomial expansion:
    # E[(G-1)^r] = Σ_{j=0..r} C(r,j) E[G^j] (-1)^{r-j}
    Epi = {0: 1.0, 1: ypi}
    for r in range(2, max_order + 1):
        s = 0.0
        for j in range(0, r + 1):
            comb = math.comb(r, j)
            s += comb * EG.get(j, np.nan) * ((-1.0) ** (r - j))
        Epi[r] = float(s)

    # Central moments of pi
    mu = Epi[1]
    m2 = Epi[2] - mu ** 2
    m3 = Epi[3] - 3 * mu * Epi[2] + 2 * mu ** 3
    m4 = Epi[4] - 4 * mu * Epi[3] + 6 * (mu ** 2) * Epi[2] - 3 * mu ** 4

    if m2 <= 0:
        return {"mean": float(mu), "var": float(max(m2, 0.0)), "skew": np.nan, "kurt": np.nan}

    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2)
    return {"mean": float(mu), "var": float(m2), "skew": float(skew), "kurt": float(kurt)}


# -----------------------------
# Main loop
# -----------------------------
def compute_all(data_dir: Path) -> pd.DataFrame:
    dfs = load_inputs(data_dir)
    caps, floors, swaps, manifest = dfs["caps"], dfs["floors"], dfs["swaps"], dfs["manifest"]

    out_rows = []
    # Iterate over all (date, area) present in swaps (your canonical month-end grid)
    for (date, area), _ in swaps.groupby(["date", "area"]):
        sl = build_call_curve_for_slice(date, area, caps, floors, swaps, manifest)
        if sl is None:
            continue

        k = sl.call_df["k"].to_numpy(float)
        C = sl.call_df["C"].to_numpy(float)

        # Optional no-arbitrage projection
        C_adj = project_call_noarb(k, C)

        # --- BL: prices -> pdf -> moments
        k_grid, pdf = bl_density_from_calls(k, C_adj, sl.B)
        m_bl = moments_from_pdf(k_grid, pdf)
        tails = tail_probs_from_pdf(k_grid, pdf)

        # --- BKM-style: prices -> moments
        m_bkm = bkm_style_moments_from_calls(k, C_adj, sl.B, sl.ypi, sl.K_star, max_order=4)

        out_rows.append({
            "date": sl.date,
            "area": sl.area,
            "B": sl.B,
            "ypi_n": sl.ypi,
            "K_star": sl.K_star,
            "K_min_obs": sl.K_min_obs,
            "K_max_obs": sl.K_max_obs,
            "n_strikes": int(len(k)),
            # BL moments
            "mean_BL": m_bl["mean"],
            "var_BL": m_bl["var"],
            "skew_BL": m_bl["skew"],
            "kurt_BL": m_bl["kurt"],
            # Tail probs from BL density
            "p_deflation_BL": tails["p_deflation"],
            "p_highinfl_BL": tails["p_highinfl"],
            # BKM-style moments
            "mean_BKM": m_bkm["mean"],
            "var_BKM": m_bkm["var"],
            "skew_BKM": m_bkm["skew"],
            "kurt_BKM": m_bkm["kurt"],
        })

    res = pd.DataFrame(out_rows).sort_values(["area", "date"]).reset_index(drop=True)
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder containing the 4 input CSVs.")
    parser.add_argument("--out_csv", type=str, default="moments_1y_bl_bkm.csv", help="Output CSV filename.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out = compute_all(data_dir)
    out_path = Path(args.out_csv)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()
