# Plan: KW-Focused Improvements for Bowley Skewness

## Context

From the previous round, KW + Bowley skewness emerged as the best-performing combination: stable, bounded [-1,1], and consistent with Pearson median skewness (r=0.91 correlation). The standard mu3/sigma3 disagrees in sign 19% of the time — too tail-sensitive.

This plan brainstorms concrete improvements to the KW density estimation pipeline specifically for better asymmetry measurement via Bowley skewness.

**File to modify**: `skewness_analysis.py`
**Data**: 191 EU date observations, 7 strikes per date (k from -0.01 to 0.05), h_KW ~0.04

---

## Problem 1: Bandwidth Is Stuck at a Ceiling (91% of dates)

**Current**: `select_bandwidth` uses `h = max(1.06*std*n^(-1/5), 1.5*median_spacing, 1e-4)`. With 7 strikes spaced ~0.01 apart, `1.5 * 0.01 = 0.015` always wins over Silverman's rule, giving h ≈ 0.0155 on observed strikes. But after tail extrapolation adds ~400 points, the bandwidth jumps to h ≈ 0.04 and stays there.

**Why it matters**: A single bandwidth for ALL dates means the smoother cannot adapt to dates with more or less curvature. Over-smoothing flattens asymmetry; under-smoothing amplifies noise.

**Fix: Least-Squares Cross-Validation (LSCV) for the second derivative**

```python
def lscv_bandwidth(k, C, B, h_candidates=None):
    """Select bandwidth by minimizing integrated squared error of f''."""
    if h_candidates is None:
        h_base = select_bandwidth(k)
        h_candidates = h_base * np.array([0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0])

    scores = []
    for h in h_candidates:
        # Leave-one-out CV: for each point i, fit without it, evaluate at i
        loo_errors = []
        for i in range(len(k)):
            k_loo = np.delete(k, i)
            C_loo = np.delete(C, i)
            P, P1, P2 = local_poly_derivative(k_loo, C_loo, k[i:i+1], h)
            C_pred = P[0]
            if np.isfinite(C_pred):
                loo_errors.append((C[i] - C_pred)**2)
        scores.append(np.mean(loo_errors) if loo_errors else np.inf)

    return h_candidates[np.argmin(scores)]
```

**Alternative**: plug-in bandwidth (Sheather-Jones) which estimates the optimal h directly from the data curvature.

---

## Problem 2: Tail Extrapolation Creates Kinks

**Current**: `extrapolate_call_prices` uses flat implied vol at boundaries. The implied vol surface σ(K) has a discontinuity at the boundary between observed and extrapolated strikes. When KW's local polynomial smooths across this kink, it creates artifacts in f''(k).

**Why it matters**: Bowley uses Q1 and Q3, which often fall in the transition zone between observed and extrapolated data.

**Fix: Smooth IV spline continuation**

Instead of flat vol, fit a quadratic (or cubic) to the last 3-4 observed implied vols and extrapolate the smile smoothly:

```python
def extrapolate_call_prices_smooth(k_obs, C_obs, B, ypi, ...):
    # Compute IV at each observed strike
    ivs = [implied_vol_from_call(C_obs[i], 1+k_obs[i], 1+ypi, B) for i in range(len(k_obs))]

    # Fit quadratic to left/right 3 strikes
    # Left: ivs[:3], Right: ivs[-3:]
    poly_left = np.polyfit(k_obs[:3], ivs[:3], 2)
    poly_right = np.polyfit(k_obs[-3:], ivs[-3:], 2)

    # Extrapolate IV smoothly, then price with Black-76
    sigma_left_ext = np.polyval(poly_left, k_left_grid)
    sigma_right_ext = np.polyval(poly_right, k_right_grid)
    # Floor at some minimum vol to prevent degenerate prices
    sigma_left_ext = np.maximum(sigma_left_ext, 0.002)
    ...
```

This eliminates the σ discontinuity and produces smoother f'' in the tails.

---

## Problem 3: Epanechnikov Kernel Has Discontinuous 2nd Derivative

**Current**: `epanechnikov(u) = 0.75*(1-u^2)` for |u|<=1, 0 otherwise. Its second derivative jumps at u=±1. Since KW extracts density via f = (1/B)*P2 where P2 comes from local polynomial regression with Epanechnikov weights, the hard cutoff creates small but visible kinks in the density.

**Why it matters**: Kinks in the density affect the CDF, which affects quantile extraction for Bowley.

**Options**:
- **Gaussian kernel**: `K(u) = exp(-u^2/2)/sqrt(2π)` — infinitely smooth, no boundary effects. Slower convergence but more stable for small n.
- **Quartic (biweight) kernel**: `K(u) = (15/16)*(1-u^2)^2` — compact support like Epanechnikov but smoother at boundary.
- **Higher polynomial order**: Use cubic (p=3) instead of quadratic (p=2) local polynomial. This estimates f, f', f'', f''' and the extra degree of freedom absorbs boundary artifacts.

**Recommendation**: Switch to Gaussian kernel. With only 7 (or ~407 after extrapolation) data points, the efficiency loss vs Epanechnikov is negligible, and smoothness matters more.

```python
def gaussian_kernel(u):
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
```

---

## Problem 4: Quantile Extraction Is Coarse

**Current**: CDF computed on 400-point grid (spacing ~0.001 = 0.1% inflation). Quantiles found by `searchsorted` + linear interpolation. With 0.1% spacing, the CDF step between grid points is ~0.0025, so quantile placement has ~0.05% inflation error.

**Why it matters**: Bowley = (Q3+Q1-2*Q2)/(Q3-Q1). If Q1 and Q3 are poorly located, the numerator (which is a DIFFERENCE of near-equal quantities) is dominated by discretization error.

**Fixes (pick any or combine)**:
1. **Finer grid**: Increase grid_n from 400 to 1600 (spacing 0.025% inflation). Simple, effective.
2. **Local refinement**: After finding approximate quantile location on coarse grid, refine with bisection on a 50-point sub-grid around the candidate. More efficient than global refinement.
3. **Spline CDF**: Fit a monotone cubic spline to (x, F(x)), then invert analytically. Most accurate.

```python
# Option 3: Spline-based quantile
from scipy.interpolate import PchipInterpolator
F_spline = PchipInterpolator(x, F)  # monotone cubic
# Invert: find x such that F_spline(x) = p
from scipy.optimize import brentq
def quantile_spline(p, F_spline, x):
    return brentq(lambda xi: F_spline(xi) - p, x[0], x[-1])
```

---

## Problem 5: Bootstrap Noise Level Is Assumed, Not Estimated

**Current**: `noise_scale=0.05` (5% multiplicative noise). This is arbitrary. If actual option price noise is 1-2%, the bootstrap CIs are 2.5-5x too wide, making inference useless (CI width is currently ~21x the point estimate).

**Why it matters**: Wide CIs mean we can never reject "skewness = 0", even when the density is visibly asymmetric.

**Fixes**:
1. **Estimate noise from time-series**: Compute std of month-over-month price changes at each strike, normalize by price level → empirical noise_scale. Requires access to the raw time series.
2. **Residual-based noise**: After fitting the KW smooth, compute residuals `e_i = C_obs[i] - C_fit(k_i)`. Use `std(e) / mean(C)` as noise_scale.
3. **Calibrate to desired CI width**: If 95% CI should be ~±0.3 (covering the Bowley range), back out the noise level that produces this.

**Recommendation**: Option 2 (residual-based) is the most principled and doesn't require external data:

```python
def estimate_noise_scale(k, C, B, ypi):
    """Estimate price noise from KW fit residuals."""
    kw = kw_density_from_call(k, C, B, extend_tails=True, ypi=ypi)
    if kw is None:
        return 0.05  # fallback
    grid, f, h = kw
    # Reconstruct call prices from density: C(k) = B * int_k^inf (x-k)*f(x) dx
    C_fit = np.array([B * integrate(grid[grid >= ki],
                       (grid[grid >= ki] - ki) * f[grid >= ki])
                      for ki in k])
    residuals = (C - C_fit) / np.maximum(C, 1e-8)
    return float(np.std(residuals))
```

---

## Problem 6: Edge-Band Shutdown Is Arbitrary

**Current**: If >7% probability mass in the edge band [L, L+0.003] or [U-0.003, U], all moments are set to NaN. Thresholds are hard-coded (L=-0.01, U=0.05, eps=0.003, thr=0.07).

**Why it matters**: Only affects 1.6% of dates currently, but with improved tail extrapolation the density extends well beyond [-1%, 5%], so edge-band detection at those bounds becomes less relevant.

**Options**:
1. **Remove it**: With tail extrapolation, the density now extends to [-20%, +20%]. Edge mass at [-1%, 5%] is no longer a truncation artifact — it's real density.
2. **Replace with goodness-of-fit test**: Use a KS test or chi-squared test between the KW density and the MaxEnt density. Flag dates where they disagree significantly.
3. **Widen the edge bands**: Move L, U to the extrapolated boundaries (e.g., L=-0.15, U=0.15) and test there.

**Recommendation**: Option 1 (remove) with a diagnostic column `Z_raw_KW` (the unnormalized integral before rescaling) as a quality flag. If Z_raw is far from a sensible value, that's a better indicator than edge mass.

---

## Problem 7: No Asymmetry Decomposition

**Current**: Bowley gives a single number. There's no decomposition of WHERE the asymmetry comes from (left tail heavier? right tail thinner? mode shifted?).

**Enhancement: Octile-based skewness profile**

Compute Bowley-like measures at multiple quantile pairs to create an "asymmetry profile":

```python
def asymmetry_profile(x, f, quantile_pairs=[(0.1,0.9), (0.2,0.8), (0.25,0.75), (0.4,0.6)]):
    """Bowley-type skewness at multiple quantile depths."""
    # ... compute CDF, quantile function ...
    profile = {}
    for (p_lo, p_hi) in quantile_pairs:
        q_lo = quantile(p_lo)
        q_hi = quantile(p_hi)
        q_med = quantile(0.5)
        iqr = q_hi - q_lo
        if iqr > 1e-15:
            profile[f'asym_{int(p_lo*100)}_{int(p_hi*100)}'] = (q_hi + q_lo - 2*q_med) / iqr
    return profile
```

This tells you: is asymmetry driven by the tails (10/90 pair) or the core (40/60 pair)?

---

## Implementation Order

1. **Finer grid** (grid_n=400→1600) + spline quantile extraction — immediate precision gain, ~5 lines
2. **Gaussian kernel** — swap `epanechnikov` for `gaussian_kernel`, ~3 lines
3. **LSCV bandwidth** — new function, integrate into `kw_density_from_call`, ~30 lines
4. **Smooth IV spline extrapolation** — modify `extrapolate_call_prices`, ~20 lines
5. **Residual-based noise estimation** for bootstrap — new function, ~15 lines
6. **Remove edge-band shutdown** + add Z_raw diagnostic — ~5 lines
7. **Asymmetry profile** — new function + CSV columns + plot, ~40 lines

## Verification

1. Run `skewness_analysis.py` before/after each change
2. Compare Bowley time series stability (should be smoother with LSCV + finer grid)
3. Compare bootstrap CI width (should shrink with residual-based noise)
4. Check that LSCV bandwidth varies across dates (no longer ceiling-bound)
5. Inspect asymmetry profiles for economic interpretability
