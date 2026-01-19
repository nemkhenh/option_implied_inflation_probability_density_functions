#coding: utf-8
"""
How to use:
1) Put the 4 CSV files in one folder:
    - cleaned_caps_quotes_1y.csv
    - cleaned_floors_quotes_1y.csv
    - cleaned_swaps_curves_1y.csv
    - manifest_coverage_1y.csv
2) Change DATA_DIR below.
3) pip install cvxpy ecos scs (if not already installed)
3) Run: python file

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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages  # ADDED (for multi-page PDF)

#EDIT THIS ONLY
DATA_DIR=r"/content" #<-- change to your folder path with the data files
RESULTS_CSV= "results.csv"

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
    return np.trapezoid(y,x)

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

#Build call curve function for a date
def build_call_curve_one_date(caps_df, floors_df, B, ypi):
    """
    Build call prices C(k) under forward measure at observed strikes k for one (date, area).
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
    C = cc["C"].to_numpy()
    for i in range(1, len(C)):
        if C[i] > C[i-1]:
            C[i] = C[i-1]
    cc["C"] = np.maximum(C, 0.0)

    return cc

#For BL (finite differences)
def bl_density_from_call(k, C, B):
    if len(k) < 5:
        return None

    kmin, kmax = float(np.min(k)), float(np.max(k))
    grid = np.linspace(kmin, kmax, 200)

    Cg = np.interp(grid, k, C)

    d1 = np.gradient(Cg, grid)
    d2 = np.gradient(d1, grid)
    f = (1.0 / B) * d2

    f = np.maximum(f, 0.0)
    Z = trapz(grid, f)
    if not np.isfinite(Z) or Z <= 0:
        return None
    f = f / Z

    return grid, f

#For BKM (diagnostic)
def bkm_style_moments_from_calls(k, C, B, ypi):
    if len(k) < 3:
        return (np.nan, np.nan, np.nan, np.nan)

    K = 1.0 + np.asarray(k, dtype=float)
    C = np.asarray(C, dtype=float)

    Kmax = float(np.max(K))
    K_tail = Kmax + 0.02
    K_aug = np.concatenate(([0.0], K, [K_tail]))
    C_aug = np.concatenate(([B * (1.0 + ypi)], C, [0.0]))

    idx = np.argsort(K_aug)
    K_aug = K_aug[idx]
    C_aug = C_aug[idx]

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

    muG = EG1
    varG = EG2 - EG1 ** 2
    if not np.isfinite(varG) or varG <= 0:
        return (muG - 1.0, np.nan, np.nan, np.nan)

    mu3 = EG3 - 3 * muG * EG2 + 2 * (muG ** 3)
    mu4 = EG4 - 4 * muG * EG3 + 6 * (muG ** 2) * EG2 - 3 * (muG ** 4)

    sd = np.sqrt(varG)
    skew = mu3 / (sd ** 3)
    kurt = mu4 / (sd ** 4)

    return (muG - 1.0, varG, skew, kurt)

#For Max Entropy
def maxent_fit(pi_grid,strikes_k,call_prices_C,B, lam=1e4):
    pi=np.asarray(pi_grid,float)
    k=np.asarray(strikes_k,float)
    C=np.asarray(call_prices_C,float)

    Pay=np.maximum(pi[None,:]-k[:,None],0.0)
    A=B*Pay

    I=len(pi)
    p=cp.Variable(I)

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

#For Kitsul-Wright
def epanechnikov(u):
    u=np.asarray(u,float)
    out=np.zeros_like(u)
    m = np.abs(u) <= 1.0
    out[m] = 0.75 * (1.0 - u[m] ** 2)
    return out

def select_bandwidth(k):
    k=np.asarray(k,float)
    n=len(k)
    if n<5:
        return np.nan
    std=float(np.std(k,ddof=1))
    spac=np.diff(np.sort(k))
    med_spac=float(np.median(spac)) if len(spac)>0 else 0.0
    h1=1.06*std*(n**(-1/5))
    h2=1.5*med_spac
    return max(h1,h2,1e-4)

def local_poly_derivative(k_obs,y_obs,grid,h):
    k_obs=np.asarray(k_obs,float)
    y_obs=np.asarray(y_obs,float)
    grid=np.asarray(grid,float)

    P = np.full_like(grid, np.nan, dtype=float)
    P1 = np.full_like(grid, np.nan, dtype=float)
    P2 = np.full_like(grid, np.nan, dtype=float)

    for j,k0 in enumerate(grid):
        u=(k_obs - k0) / h
        w=epanechnikov(u)
        if np.sum(w) < 1e-8:
            continue
        x=k_obs - k0
        X=np.column_stack([np.ones_like(x), x, 0.5*x**2])

        sw=np.sqrt(w)
        Xw=X*sw[:,None]
        yw=y_obs*sw

        try:
            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        except Exception:
            continue

        P[j], P1[j], P2[j] = beta[0], beta[1], beta[2]

    return P, P1, P2

def kw_density_from_call(k, C, B, grid_n=200, h=None):
    k=np.asarray(k,float)
    C=np.asarray(C,float)

    kmin, kmax = float(np.min(k)), float(np.max(k))
    grid = np.linspace(kmin, kmax, grid_n)

    if h is None:
        h=select_bandwidth(k)

    if not np.isfinite(h) or h <=0:
        return None

    P,P1,P2=local_poly_derivative(k,C,grid,h)
    if np.all(~np.isfinite(P2)):
        return None

    f=(1.0/B)*P2
    f=np.where(np.isfinite(f),f,0.0)
    f=np.maximum(f,0.0)

    Z=trapz(grid,f)
    if not np.isfinite(Z) or Z <=0:
        return None
    f=f/Z
    return grid,f,h

#-----------------------------main-------------------------------------

rows=[]
keys=swaps[["date","area"]].drop_duplicates().sort_values(["area",'date'])

for (dt,area) in keys.itertuples(index=False):
    sw=swaps[(swaps["date"]==dt)&(swaps["area"]==area)]
    if sw.empty:
        continue

    B=float(sw["B"].iloc[0])
    ypi=float(sw["ypi_n"].iloc[0])

    caps_g = caps[(caps["date"] == dt) & (caps["area"] == area)]
    floors_g = floors[(floors["date"] == dt) & (floors["area"] == area)]
    if caps_g.empty and floors_g.empty:
        continue

    call_curve=build_call_curve_one_date(caps_g,floors_g,B=B,ypi=ypi)
    k=call_curve["k"].to_numpy()
    C=call_curve["C"].to_numpy()

    bl=bl_density_from_call(k,C,B)
    if bl is None:
        mu_bl = var_bl = skew_bl = kurt_bl = np.nan
        p_defl = p_hi = np.nan
    else:
        kgrid, f = bl
        mu_bl, var_bl, skew_bl, kurt_bl = moments_from_density(kgrid, f)
        p_defl = trapz(kgrid[kgrid < 0], f[kgrid < 0]) if np.any(kgrid < 0) else 0.0
        p_hi = trapz(kgrid[kgrid > 0.04], f[kgrid > 0.04]) if np.any(kgrid > 0.04) else 0.0

    mu_bkm, var_bkm, skew_bkm, kurt_bkm = bkm_style_moments_from_calls(k, C, B, ypi)

    kmin,kmax=float(np.min(k)),float(np.max(k))
    pi_min=max(-0.05, kmin-0.02)
    pi_max=min(0.15, kmax+0.05)
    pi_grid=np.linspace(pi_min,pi_max,401)

    p=maxent_fit(pi_grid,k,C,B,lam=1e4)
    if p is None:
        mu_maxent = var_maxent = skew_maxent = kurt_maxent = np.nan
        rmse_maxent = np.nan
        p_defl_maxent = np.nan
        p_hi_maxent = np.nan
    else:
        mu_maxent, var_maxent, skew_maxent, kurt_maxent = moments_discrete(pi_grid, p)

        Pay = np.maximum(pi_grid[None,:] - k[:,None], 0.0)
        C_fit = B * (Pay @ p)
        rmse_maxent = float(np.sqrt(np.mean((C_fit - C) ** 2)))

        p_defl_maxent = float(p[pi_grid < 0].sum()) if np.any(pi_grid < 0) else 0.0
        p_hi_maxent   = float(p[pi_grid > 0.04].sum()) if np.any(pi_grid > 0.04) else 0.0

    kw=kw_density_from_call(k,C,B)
    if kw is None:
        mu_kw=var_kw=skew_kw=kurt_kw=np.nan
        p_defl_kw=p_hi_kw=np.nan
        h_kw=np.nan
    else:
        kgrid_kw,f_kw,h_kw=kw
        mu_kw, var_kw, skew_kw, kurt_kw = moments_from_density(kgrid_kw,f_kw)
        p_defl_kw = trapz(kgrid_kw[kgrid_kw < 0], f_kw[kgrid_kw < 0]) if np.any(kgrid_kw < 0) else 0.0
        p_hi_kw = trapz(kgrid_kw[kgrid_kw > 0.04], f_kw[kgrid_kw > 0.04]) if np.any(kgrid_kw > 0.04) else 0.0

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
        "rmse_maxent":rmse_maxent,
        "p_deflation_maxent": p_defl_maxent,
        "p_highinfl_maxent": p_hi_maxent,

        "mean_KW": mu_kw,
        "var_KW": var_kw,
        "skew_KW": skew_kw,
        "kurt_KW": kurt_kw,
        "p_deflation_KW": p_defl_kw,
        "p_highinfl_KW": p_hi_kw,
        "h_KW": h_kw,
    })

out=pd.DataFrame(rows).sort_values(["area","date"])
out_path=os.path.join(DATA_DIR,RESULTS_CSV)
out.to_csv(out_path,index=False)

#---------------------Analysis of output-------------------------------

AREA = "EU"

res = pd.read_csv(os.path.join(DATA_DIR, RESULTS_CSV))
res["date"] = pd.to_datetime(res["date"])
d = res[res["area"] == AREA].sort_values("date")

plt.figure()
plt.plot(d["date"], d["mean_BL"], label="BL")
plt.plot(d["date"], d["mean_KW"], label="KW")
plt.plot(d["date"], d["mean_maxent"], label="MaxEnt")
plt.plot(d["date"], d["mean_BKM"], label="BKM")
plt.legend()
plt.xlabel("Date"); plt.ylabel("Implied mean inflation")
plt.title(f"Mean comparison ({AREA})")
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "fig_mean_comparison.pdf"), bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(d["date"], d["var_BL"], label="BL")
plt.plot(d["date"], d["var_KW"], label="KW")
plt.plot(d["date"], d["var_maxent"], label="MaxEnt")
plt.plot(d["date"], d["var_BKM"], label="BKM")
plt.legend()
plt.xlabel("Date"); plt.ylabel("Implied variance")
plt.title(f"Variance comparison ({AREA})")
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "fig_variance_comparison.pdf"), bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(d["date"], d["skew_BL"], label="BL")
plt.plot(d["date"], d["skew_KW"], label="KW")
plt.plot(d["date"], d["skew_maxent"], label="MaxEnt")
plt.legend()
plt.xlabel("Date"); plt.ylabel("Implied skewness")
plt.title(f"Skewness comparison ({AREA})")
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "fig_skewness_comparison.pdf"), bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(d["date"], d["kurt_BL"], label="BL")
plt.plot(d["date"], d["kurt_KW"], label="KW")
plt.plot(d["date"], d["kurt_maxent"], label="MaxEnt")
plt.legend()
plt.xlabel("Date"); plt.ylabel("Implied kurtosis")
plt.title(f"Kurtosis comparison ({AREA})")
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "fig_kurtosis_comparison.pdf"), bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(d["date"], d["p_deflation_BL"], label="BL")
plt.plot(d["date"], d["p_deflation_KW"], label="KW")
plt.plot(d["date"], d["p_deflation_maxent"], label="MaxEnt")
plt.legend()
plt.xlabel("Date"); plt.ylabel("P(deflation) = P(pi < 0)")
plt.title(f"Deflation probability ({AREA})")
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "fig_deflation_probability.pdf"), bbox_inches="tight")
plt.close()

# --------------------- ADDED: PDF comparison on selected dates ---------------------

# Select three representative dates in AREA by strike coverage width (min/median/max)
d_sel = d.dropna(subset=["K_width"]).copy()
d_sel = d_sel.sort_values("K_width")
if len(d_sel) > 0:
    idx_min = d_sel.index[0]
    idx_max = d_sel.index[-1]
    idx_med = d_sel.index[len(d_sel)//2]
    selected_dates = [
        pd.to_datetime(d.loc[idx_min, "date"]),
        pd.to_datetime(d.loc[idx_med, "date"]),
        pd.to_datetime(d.loc[idx_max, "date"]),
    ]
else:
    selected_dates = []

pdf_out = os.path.join(DATA_DIR, f"fig_pdf_comparison_{AREA}.pdf")
if len(selected_dates) > 0:
    with PdfPages(pdf_out) as pdf:
        for dt in selected_dates:
            sw = swaps[(swaps["date"] == dt) & (swaps["area"] == AREA)]
            if sw.empty:
                continue
            B = float(sw["B"].iloc[0])
            ypi = float(sw["ypi_n"].iloc[0])

            caps_g = caps[(caps["date"] == dt) & (caps["area"] == AREA)]
            floors_g = floors[(floors["date"] == dt) & (floors["area"] == AREA)]
            if caps_g.empty and floors_g.empty:
                continue

            call_curve = build_call_curve_one_date(caps_g, floors_g, B=B, ypi=ypi)
            k = call_curve["k"].to_numpy()
            C = call_curve["C"].to_numpy()

            # BL density
            bl = bl_density_from_call(k, C, B)
            # KW density
            kw = kw_density_from_call(k, C, B)

            # MaxEnt discrete density
            kmin, kmax = float(np.min(k)), float(np.max(k))
            pi_min = max(-0.05, kmin - 0.02)
            pi_max = min(0.15, kmax + 0.05)
            pi_grid = np.linspace(pi_min, pi_max, 401)
            p_me = maxent_fit(pi_grid, k, C, B, lam=1e4)

            plt.figure()
            if bl is not None:
                kgrid_bl, f_bl = bl
                plt.plot(kgrid_bl, f_bl, label="BL")
            if kw is not None:
                kgrid_kw, f_kw, _h = kw
                plt.plot(kgrid_kw, f_kw, label="KW")
            if p_me is not None:
                p_me = np.asarray(p_me, float)
                s = p_me.sum()
                if np.isfinite(s) and s > 0:
                    p_me = p_me / s
                    dx = float(pi_grid[1] - pi_grid[0]) if len(pi_grid) > 1 else 1.0
                    f_me = p_me / dx
                    plt.plot(pi_grid, f_me, label="MaxEnt")

            plt.legend()
            plt.xlabel("Inflation strike / inflation grid")
            plt.ylabel("Implied density")
            plt.title(f"Implied densities comparison ({AREA}) — {dt.date().isoformat()}")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

# ------------------ Table - results (time-average across dates) ------------------

AREA_TABLE = "EU"
res = pd.read_csv(os.path.join(DATA_DIR, RESULTS_CSV))
res["date"] = pd.to_datetime(res["date"])
dtab = res[res["area"] == AREA_TABLE].sort_values("date")

# Columns per method
table_cols = {
    "BL":     ["mean_BL", "var_BL", "skew_BL", "kurt_BL"],
    "KW":     ["mean_KW", "var_KW", "skew_KW", "kurt_KW"],
    "BKM":    ["mean_BKM", "var_BKM", "skew_BKM", "kurt_BKM"],
    "MaxEnt": ["mean_maxent", "var_maxent", "skew_maxent", "kurt_maxent"],
}

# Compute time-average across dates
rows = []
for method, cols in table_cols.items():
    vals = dtab[cols].astype(float).mean(skipna=True)
    rows.append([method, vals[cols[0]], vals[cols[1]], vals[cols[2]], vals[cols[3]]])

avg_tbl = pd.DataFrame(rows, columns=["Method", "Mean", "Variance", "Skewness", "Kurtosis"])

print(avg_tbl)
