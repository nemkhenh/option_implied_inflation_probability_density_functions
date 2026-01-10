#coding: utf-8
"""
Project steps:
- Build 1Y inflation call price curve from caps + floors (via parity)
- Method 1: BL (Breeden-Litzenberger) - call prices => density => moments
- Method 2: BKM (Bakshi, Kapadia, and Madan 2003) - options integrals => moments (on G = 1 + pi)

How to use:
1) Put the 4 CSV files in one folder:
    - cleaned_caps_quotes_1y.csv
    - cleaned_floors_quotes_1y.csv
    - cleaned_swaps_curves_1y.csv
    - manifest_coverage_1y.csv
2) Change DATA_DIR below.
3) Run: python NGUYEN_MIEUZET_MORENO_option_prices.py

Output:
- output_prices.csv (name changeable via OUT_NAME)

Notes:
- price_per_1 is in "percent of notional" in these "cleaned" files (e.g. 0.3222 = 0.3222%).
    We convert to decimals by dividing by 100.
- BL uses finite differences, so it is sensitive to noise. 
    We apply a simple monotonicity fix to stabilize it (call prices should be non-increasing in strike).
"""

import os
import numpy as np
import pandas as pd
import cvxpy as cp

#EDIT THIS ONLY
DATA_DIR=r"C:\Users\MG132LG\Desktop\Cours\uni\QMF\data" #<-- change to your folder path
OUT_NAME= "output_prices.csv" #<-- change on your preference in order to not mess up with other group :>

#----------------------Data------------------------------------------
caps = pd.read_csv(os.path.join(DATA_DIR, "cleaned_caps_quotes_1y.csv"))
floors = pd.read_csv(os.path.join(DATA_DIR, "cleaned_floors_quotes_1y.csv"))
swaps = pd.read_csv(os.path.join(DATA_DIR, "cleaned_swaps_curves_1y.csv"))
manifest = pd.read_csv(os.path.join(DATA_DIR, "manifest_coverage_1y.csv"))

#ensuring types and maturity for inflation (n=1)
for df in (caps, floors, swaps, manifest):
    df["date"]=pd.to_datetime(df["date"])

caps = caps[caps["n"] == 1].copy()
floors = floors[floors["n"] == 1].copy()
swaps = swaps[swaps["n"] == 1].copy()
manifest = manifest[manifest["n"] == 1].copy()

#-----------------Helper functions-----------------------------------
def trapz(x,y):
    """Safe trapezoid integral."""
    if len(x)<2:
        return np.nan
    return np.trapz(y,x)

def moments_from_density(x,f):
    """Return mean, skew, var, skew, kurt for a density f(x) on grid x."""
    Z=trapz(x,f)

    if not np.isfinite(Z) or Z<=0:
        return (np.nan,np.nan,np.nan,np.nan)
    
    f=f/Z
    mu=trapz(x,x*f)
    var=trapz(x,(x-mu)**2*f)
    if var <=0 or not np.isfinite(var):
        return(mu,np.nan,np.nan,np.nan)
    sd=np.sqrt(var)
    m3=trapz(x,(x-mu)**3*f)
    m4=trapz(x,(x-mu)**4*f)

    skew=m3/(sd**3)
    kurt=m4/(sd**4)

    return(mu,var,skew,kurt)

#Build call curve function for a date
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

#For BL
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

#For BKM
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

#For Max Entropy
def moments_discrete(x,p):
    """Mean/Variance/Skew/Kurt for discrete distribution."""
    p=np.asarray(p,float)
    x=np.asarray(x,float)
    s=p.sum()

    if not np.isfinite(s) or s <=0:
        return(np.nan,np.nan,np.nan,np.nan)
    
    p=p/s
    mu=(p*x).sum()
    var=(p*(x-mu)**2).sum()

    if not np.isfinite(var) or var<=0:
        return(mu,np.nan,np.nan,np.nan)
    
    sd=np.sqrt(var)
    skew=(p*(x-mu)**3).sum()/(sd**3)
    kurt=(p*(x-mu)**4).sum()/(sd**4)
    return (mu, var, skew, kurt)

def maxent_fit(pi_grid,strikes_k,call_prices_C,B, lam=1e4):
    """
    """
    pi=np.asarray(pi_grid,float)
    k=np.asarray(strikes_k,float)
    C=np.asarray(call_prices_C,float)

    Pay=np.maximum(pi[None,:]-k[:,None],0.0) #(J,I)
    A=B*Pay

    I=len(pi)
    p=cp.Variable(I)

    #convex entroy term: sum p log p (via entropy)
    entropy_convex=-cp.sum(cp.entr(p))
    price_misfit=cp.sum_squares(A @ p-C)

    prob=cp.Problem(
        cp.Minimize(entropy_convex+lam*price_misfit),
        [p>=0,cp.sum(p)==1]
    )
    prob.solve(solver=cp.ECOS, verbose=False)

    if p.value is None:
        return None
    return np.maximum(np.array(p.value).reshape(-1),0.0)

#-----------------------------main-------------------------------------

rows=[]
keys=swaps[["date","area"]].drop_duplicates().sort_values(["area",'date'])

for (dt,area) in keys.itertuples(index=False):
    #swaps
    sw=swaps[(swaps["date"]==dt)&(swaps["area"]==area)]
    if sw.empty:
        continue

    B=float(sw["B"].iloc[0])
    ypi=float(sw["ypi_n"].iloc[0])

    #caps,floors
    caps_g = caps[(caps["date"] == dt) & (caps["area"] == area)]
    floors_g = floors[(floors["date"] == dt) & (floors["area"] == area)]
    if caps_g.empty and floors_g.empty:
        continue

    call_curve=build_call_curve_one_date(caps_g,floors_g,B=B,ypi=ypi)
    k=call_curve["k"].to_numpy()
    C=call_curve["C"].to_numpy()

    #Method BL
    bl=bl_density_from_call(k,C,B)
    if bl is None:
        mu_bl = var_bl = skew_bl = kurt_bl = np.nan
        p_defl = p_hi = np.nan
    else:
        kgrid, f = bl
        mu_bl, var_bl, skew_bl, kurt_bl = moments_from_density(kgrid, f)
        # tail probs like Kitsul-Wright: P(pi<0), P(pi>4%)
        p_defl = trapz(kgrid[kgrid < 0], f[kgrid < 0]) if np.any(kgrid < 0) else 0.0
        p_hi = trapz(kgrid[kgrid > 0.04], f[kgrid > 0.04]) if np.any(kgrid > 0.04) else 0.0
    

    #method BKM
    mu_bkm, var_bkm, skew_bkm, kurt_bkm = bkm_style_moments_from_calls(k, C, B, ypi)

    #method Max-entropy
    #Choose a pi grid slightly wider than observed strikes (but keep pi>-1)
    kmin,kmax=float(np.min(k)),float(np.max(k))
    pi_min=max(-0.05, kmin-0.02)
    pi_max=min(0.15, kmax+0.05)
    pi_grid=np.linspace(pi_min,pi_max,401)

    p=maxent_fit(pi_grid,k,C,B,lam=1e4)

    mu_maxent, var_maxent, skew_maxent, kurt_maxent = moments_discrete(pi_grid,p)
    Pay=np.maximum(pi_grid[None,:]-k[:,None],0.0)
    C_fit=B*(Pay @ p)
    rmse_maxent=float(np.sqrt(np.mean((C_fit-C)**2)))


    #coverage diagnostics
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

        "mean_maxent":mu_maxent,
        "var_maxent": var_maxent,
        "skew_maxent": skew_maxent,
        "kurt_maxent": kurt_maxent,
        "rmse_maxent":rmse_maxent
    })

out=pd.DataFrame(rows).sort_values(["area","date"])
out_path=os.path.join(DATA_DIR,OUT_NAME)
out.to_csv(out_path,index=False)

print("Saved:",out_path)
print(out.head(10).to_string(index=False))