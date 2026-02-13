#coding: utf-8
"""
How to use:
1) Put the 4 CSV files in one folder:
    - cleaned_caps_quotes_1y.csv
    - cleaned_floors_quotes_1y.csv
    - cleaned_swaps_curves_1y.csv
    - manifest_coverage_1y.csv
    - ICP.M.U2.Y.000000.3.INX_20260125163340.csv (for comparison purpose on plots)
2) Change DATA_DIR below.
3) pip install cvxpy ecos scs (if not already installed)
3) Run: python file

(unfortunately, we couldn't put it on github as data sources came from Bloomberg)

@author: nemkhenh
"""

import os
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages  # ADDED (for multi-page PDF)
from scipy import optimize
from scipy.stats import norm, t as student_t, skewnorm, beta as beta_dist
from scipy.integrate import trapezoid


#EDIT THIS ONLY
DATA_DIR=r"C:\Users\MG132LG\Desktop\Cours\uni\QMF\data" #<-- change to your folder path
OUT_DIR=r"C:\Users\MG132LG\Desktop\Cours\uni\QMF\outputs" #<-- change to your folder path
RESULTS_CSV= "updated_results.csv"


# ============================================================
# DATA
# ============================================================
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

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def trapz(x,y):
    """Safe trapezoid integral."""
    if len(x)<2:
        return np.nan
    return trapezoid(y,x)

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


def project_call_curve_monotone_convex(k, C, B, enforce_slope_bounds=True):
    """
    QP projection of observed call prices C(k) onto no-arbitrage constraints (n=1):
      (i)   C(k) >= 0
      (ii)  C(k) is non-increasing in k
      (iii) C(k) is convex in k  (slopes non-decreasing)
      (iv)  slope bounds: -B <= dC/dk <= 0  (optional, recommended)

    Uses non-uniform finite-difference slopes:
        slope_i = (C_{i+1}-C_i)/(k_{i+1}-k_i)

    Returns
    -------
    C_adj : np.ndarray
        Adjusted call prices aligned with original input order.
    info : dict
        Diagnostics (solver status).
    """
    k = np.asarray(k, float)
    C = np.asarray(C, float)

    if len(k) < 3:
        return np.maximum(C, 0.0), {"status": "skipped_small_n"}

    # sort by strike
    idx = np.argsort(k)
    k_s = k[idx]
    C_s = C[idx]

    dk = np.diff(k_s)
    if np.any(dk <= 0):
        # duplicates / non-strict: safest fallback is isotonic (monotone) + clip
        C_fb = C_s.copy()
        for i in range(1, len(C_fb)):
            if C_fb[i] > C_fb[i-1]:
                C_fb[i] = C_fb[i-1]
        C_fb = np.maximum(C_fb, 0.0)
        C_out = np.empty_like(C_fb)
        C_out[idx] = C_fb
        return C_out, {"status": "fallback_nonunique_k"}

    n = len(k_s)
    x = cp.Variable(n)

    # slopes length n-1
    slopes = (x[1:] - x[:-1]) / dk

    cons = []
    cons += [x >= 0]                 # non-negativity
    cons += [x[1:] <= x[:-1]]        # decreasing
    cons += [slopes[1:] >= slopes[:-1]]  # convexity (non-decreasing slopes)

    if enforce_slope_bounds:
        # -B <= slope <= 0
        Bf = float(B)
        if not np.isfinite(Bf) or Bf <= 0:
            raise ValueError("B must be positive/finite to enforce slope bounds.")
        cons += [slopes <= 0.0]
        cons += [slopes >= -Bf]

    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - C_s)), cons)

    # OSQP is robust for QPs; ECOS fallback
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        prob.solve(solver=cp.ECOS, verbose=False)

    if x.value is None:
        # fallback: your previous monotone fix
        C_fb = C_s.copy()
        for i in range(1, len(C_fb)):
            if C_fb[i] > C_fb[i-1]:
                C_fb[i] = C_fb[i-1]
        C_fb = np.maximum(C_fb, 0.0)
        C_out = np.empty_like(C_fb)
        C_out[idx] = C_fb
        return C_out, {"status": "solver_failed_fallback"}

    C_adj_s = np.asarray(x.value, float).reshape(-1)
    C_out = np.empty_like(C_adj_s)
    C_out[idx] = C_adj_s
    return C_out, {"status": prob.status}


def build_call_curve_one_date(caps_df, floors_df, B, ypi):
    """
    Build call prices C(k) under forward measure at observed strikes k for one (date, area).
    - Caps are calls directly.
    - Floors are puts; convert to calls using parity:
        C(k) = P(k) + B*(E[pi] - k), with E[pi] ~ ypi_n (1Y swap rate)
    Returns: DataFrame columns [k, C] (C in decimals of notional)
    """
    caps = caps_df.copy()
    floors = floors_df.copy()
    caps["C"] = caps["price_per_1"] / 100.0
    floors["P"] = floors["price_per_1"] / 100.0

    floors["C"] = floors["P"] + B * (ypi - floors["k"])

    cc = pd.concat([caps[["k", "C"]], floors[["k", "C"]]], ignore_index=True)

    # Average if duplicates
    cc = cc.groupby("k", as_index=False)["C"].mean().sort_values("k").reset_index(drop=True)

    # --- NEW: QP projection enforcing (C>=0, decreasing, convex, -B<=slope<=0) ---
    k_arr = cc["k"].to_numpy()
    C_arr = cc["C"].to_numpy()

    C_adj, _info = project_call_curve_monotone_convex(
        k_arr, C_arr, B=B, enforce_slope_bounds=True
    )
    cc["C"] = C_adj

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

# ============================================================
# MAIN
# ============================================================

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

    #Breeden-Litzenberger
    bl=bl_density_from_call(k,C,B)
    if bl is None:
        mu_bl = var_bl = skew_bl = kurt_bl = np.nan
        p_defl = p_hi = np.nan
    else:
        kgrid, f = bl
        mu_bl, var_bl, skew_bl, kurt_bl = moments_from_density(kgrid, f)
        p_defl = trapz(kgrid[kgrid < 0], f[kgrid < 0]) if np.any(kgrid < 0) else 0.0
        p_hi = trapz(kgrid[kgrid > 0.04], f[kgrid > 0.04]) if np.any(kgrid > 0.04) else 0.0

    #BKM
    mu_bkm, var_bkm, skew_bkm, kurt_bkm = bkm_style_moments_from_calls(k, C, B, ypi)

    kmin,kmax=float(np.min(k)),float(np.max(k))
    pi_min=max(-0.05, kmin-0.02)
    pi_max=min(0.15, kmax+0.05)
    pi_grid=np.linspace(pi_min,pi_max,401)

    #MaxEnt
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

    #Kistul Wright
    kw=kw_density_from_call(k,C,B)
    if kw is None:
        mu_kw=var_kw=skew_kw=kurt_kw=np.nan
        p_defl_kw=p_hi_kw=np.nan
        h_kw=np.nan
    else:
        kgrid_kw,f_kw,h_kw=kw

        mu_kw, var_kw, skew_kw, kurt_kw = moments_from_density(kgrid_kw,f_kw)
        p_defl_kw=trapz(kgrid_kw[kgrid_kw <0], f_kw[kgrid_kw <0]) if np.any(kgrid_kw<0) else 0.0
        p_hi_kw=trapz(kgrid_kw[kgrid_kw >0.04], f_kw[kgrid_kw >0.04]) if np.any(kgrid_kw>0.04) else 0.0


        #KW shutdown signal near adge
        L,U=-0.01,0.05
        eps=0.003 # edge band
        thr=0.07 #threshold 7% mass in edge band triggers shutdown

        p_edge_L = trapz(kgrid_kw[(kgrid_kw >=L)&(kgrid_kw<=L+eps)],
                          f_kw[(kgrid_kw >=L)&(kgrid_kw<=L+eps)]) if np.any((kgrid_kw >=L)&(kgrid_kw<=L+eps)) else 0.0
        p_edge_U = trapz(kgrid_kw[(kgrid_kw >U-eps)&(kgrid_kw <=U)],
                        f_kw[(kgrid_kw >U-eps)&(kgrid_kw <=U)]) if np.any((kgrid_kw >U-eps)&(kgrid_kw <=U)) else 0.0
        
        kw_ok=(p_edge_L<=thr)and (p_edge_U<=thr)
        if not kw_ok:
            mu_kw=var_kw= skew_kw= kurt_kw= np.nan
            p_defl_kw=p_hi_kw=np.nan


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
out_path=os.path.join(OUT_DIR,RESULTS_CSV)
out.to_csv(out_path,index=False)

# ============================================================
# EMPIRICAL ANALYSIS
# ============================================================

AREA = "EU"

res = pd.read_csv(os.path.join(OUT_DIR, RESULTS_CSV))
res["date"] = pd.to_datetime(res["date"])
d = res[res["area"] == AREA].sort_values("date")

plt.figure()
plt.plot(d["date"], 100 * d["mean_BL"], label="BL")
plt.plot(d["date"], 100 * d["mean_KW"], label="KW")
plt.plot(d["date"], 100 * d["mean_maxent"], label="MaxEnt")
plt.plot(d["date"], 100 * d["mean_BKM"], label="BKM")
plt.legend()
plt.xlabel("Date"); plt.ylabel("Implied mean inflation")
plt.title(f"Mean comparison ({AREA})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_mean_comparison.pdf"), bbox_inches="tight")
plt.close()

# --- Add Euro Area HICP inflation to the existing plot ---

# 1) Load HICP index (monthly index) from ECB CSV in ./data
HICP_FILE = "ICP.M.U2.Y.000000.3.INX_20260125163340.csv"
hicp_path = os.path.join(DATA_DIR,HICP_FILE)

hicp_raw = pd.read_csv(hicp_path)

# Columns in the file:
# - "DATE"
# - "TIME PERIOD"
# - "HICP - Overall index (ICP.M.U2.Y.000000.3.INX)"
HICP_COL = "HICP - Overall index (ICP.M.U2.Y.000000.3.INX)"

hicp = hicp_raw[["DATE", HICP_COL]].rename(columns={"DATE": "date", HICP_COL: "hicp_index"}).copy()
hicp["date"] = pd.to_datetime(hicp["date"])
hicp["hicp_index"] = pd.to_numeric(hicp["hicp_index"], errors="coerce")
hicp = hicp.dropna(subset=["hicp_index"]).sort_values("date")

# YoY inflation in percent
hicp["hicp_yoy_pct"] = 100.0 * (hicp["hicp_index"] / hicp["hicp_index"].shift(12) - 1.0)

# Align to the implied-mean plot window
hicp_plot = hicp[(hicp["date"] >= d["date"].min()) & (hicp["date"] <= d["date"].max())].copy()


# 4) Plot: implied mean inflation (left axis) + HICP YoY (right axis)
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(d["date"], 100 * d["mean_BL"], label="BL")
ax.plot(d["date"], 100 * d["mean_KW"], label="KW")
ax.plot(d["date"], 100 * d["mean_maxent"], label="MaxEnt")
ax.plot(d["date"], 100 * d["mean_BKM"], label="BKM")

ax.set_xlabel("Date")
ax.set_ylabel("Implied mean inflation")
ax.set_title(f"Mean comparison ({AREA})")

ax2 = ax.twinx()
ax2.plot(hicp_plot["date"], hicp_plot["hicp_yoy_pct"], linestyle="--", label="HICP YoY (Euro Area)")
ax2.set_ylabel("HICP inflation (YoY, %)")

# One combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_mean_comparison_with_hicp.pdf"), bbox_inches="tight")
plt.close(fig)


# --- Variance comparison with BKM on secondary y-axis ---

fig, ax = plt.subplots()

ax.plot(d["date"], d["var_BL"], label="BL")
ax.plot(d["date"], d["var_KW"], label="KW")
ax.set_xlabel("Date")
ax.set_ylabel("Implied variance")

# Combined legend
ax2 = ax.twinx()
ax2.plot(hicp_plot["date"], hicp_plot["hicp_yoy_pct"], linestyle="--", label="HICP YoY (Euro Area)")
ax2.set_ylabel("HICP inflation (YoY, %)")

# One combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

ax.set_title(f"Variance comparison ({AREA})")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_variance_comparison.pdf"), bbox_inches="tight")
plt.close(fig)


# --- Variance comparison with BKM on secondary y-axis ---

fig, ax = plt.subplots()

ax.plot(d["date"], d["var_BL"], label="BL")
ax.plot(d["date"], d["var_KW"], label="KW")
ax.plot(d["date"], d["var_maxent"], label="MaxEnt")
ax.set_xlabel("Date")
ax.set_ylabel("Implied variance")

ax2 = ax.twinx()
ax2.plot(d["date"], d["var_BKM"], label="BKM", linestyle="--", color="black")
ax2.set_ylabel("Implied variance (BKM)")

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

ax.set_title(f"Variance comparison ({AREA})")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_variance_comparison_all.pdf"), bbox_inches="tight")
plt.close(fig)


plt.figure()
plt.plot(d["date"], d["skew_BL"], label="BL")
plt.plot(d["date"], d["skew_KW"], label="KW")
plt.plot(d["date"], d["skew_maxent"], label="MaxEnt")
plt.legend()
plt.xlabel("Date"); plt.ylabel("Implied skewness")
plt.title(f"Skewness comparison ({AREA})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_skewness_comparison.pdf"), bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(d["date"], d["kurt_BL"], label="BL")
plt.plot(d["date"], d["kurt_KW"], label="KW")
plt.plot(d["date"], d["kurt_maxent"], label="MaxEnt")
plt.legend()
plt.xlabel("Date"); plt.ylabel("Implied kurtosis")
plt.title(f"Kurtosis comparison ({AREA})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_kurtosis_comparison.pdf"), bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(d["date"], d["p_deflation_BL"], label="BL")
plt.plot(d["date"], d["p_deflation_KW"], label="KW")
plt.plot(d["date"], d["p_deflation_maxent"], label="MaxEnt")
plt.legend()
plt.xlabel("Date"); plt.ylabel("P(deflation) = P(pi < 0)")
plt.title(f"Deflation probability ({AREA})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_deflation_probability.pdf"), bbox_inches="tight")
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

pdf_out = os.path.join(OUT_DIR, f"fig_pdf_comparison_{AREA}.pdf")
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
res = pd.read_csv(os.path.join(OUT_DIR, RESULTS_CSV))
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

# ============================================================
# KEEP ONLY SKEW-t
# 1) Single chosen date: plot KW + Skew-t, legend shows mean/var (no AIC)
# 2) Several dates: plot Skew-t pdfs across dates, legend shows date + mean/var
# ============================================================

from numpy.polynomial.hermite import hermgauss

# -----------------------
# Helpers: weighted ll + GH skew-t logpdf
# -----------------------
def _wll_from_logpdf(logpdf_vals, w, N_eff):
    return N_eff * float(np.sum(w * logpdf_vals))

def _skewt_logpdf_gh(x, df, alpha, mu, sd, n_gh=40):
    df = float(df)
    sd = float(sd)
    if df <= 2.0 or sd <= 0:
        return -np.inf * np.ones_like(x, dtype=float)

    delta = alpha / np.sqrt(1.0 + alpha**2)   # in (-1,1)
    xz = (x - mu) / sd

    nodes, weights = hermgauss(n_gh)
    u = np.sqrt(2.0) * np.abs(nodes)          # half-normal via symmetry trick
    pref = 1.0 / np.sqrt(np.pi)
    s_u = np.sqrt((delta * u) ** 2 + (1.0 - delta**2))

    z_over_s = xz[:, None] / s_u[None, :]
    comp = student_t.pdf(z_over_s, df=df) / s_u[None, :]

    fz = pref * (comp * weights[None, :]).sum(axis=1)
    fy = np.maximum(fz / sd, 1e-300)
    return np.log(fy)

def fit_skewt_to_kw_points(x_pts, f_pts, mu0=None, sd0=None, n_gh=40):
    """
    Fit skew-t to 7 KW-evaluated points by weighted MLE.
    Inputs:
      x_pts: grid of 7 points
      f_pts: KW density evaluated at x_pts (not necessarily normalized)
    Returns: dict with params (df, alpha, mu, sd)
    """
    x_pts = np.asarray(x_pts, float)
    f_pts = np.asarray(f_pts, float)
    f_pts = np.maximum(f_pts, 1e-14)
    w = f_pts / f_pts.sum()
    N_eff = len(x_pts)

    if mu0 is None:
        mu0 = float(np.sum(w * x_pts))
    if sd0 is None:
        v0 = float(np.sum(w * (x_pts - mu0) ** 2))
        sd0 = float(np.sqrt(v0)) if np.isfinite(v0) and v0 > 0 else 0.01

    def nll(theta):
        log_dfm2, alpha, mu, log_sd = theta
        df = 2.0 + np.exp(log_dfm2)
        sd = np.exp(log_sd)
        lp = _skewt_logpdf_gh(x_pts, df=df, alpha=alpha, mu=mu, sd=sd, n_gh=n_gh)
        return -_wll_from_logpdf(lp, w, N_eff)

    x0 = np.array([np.log(10.0), 0.0, float(mu0), np.log(max(float(sd0), 1e-4))])
    res = optimize.minimize(nll, x0=x0, method="L-BFGS-B")

    log_dfm2, alpha, mu, log_sd = res.x
    df = float(2.0 + np.exp(log_dfm2))
    sd = float(np.exp(log_sd))
    return {
        "ok": bool(res.success),
        "params": (df, float(alpha), float(mu), sd),
        "message": str(res.message),
    }

def pdf_skewt(x, params, n_gh=60):
    df, alpha, mu, sd = params
    return np.exp(_skewt_logpdf_gh(np.asarray(x, float), df=df, alpha=alpha, mu=mu, sd=sd, n_gh=n_gh))

def moments_from_pdf_grid(x, f):
    f = np.asarray(f, float)
    x = np.asarray(x, float)
    f = np.where(np.isfinite(f), f, 0.0)
    f = np.maximum(f, 0.0)
    Z = trapz(x, f)
    if (not np.isfinite(Z)) or (Z <= 0):
        return (np.nan, np.nan)
    f = f / Z
    mu = trapz(x, x * f)
    var = trapz(x, (x - mu) ** 2 * f)
    return (float(mu), float(var))

# ============================================================
# 1) SINGLE DATE PLOT: KW vs Skew-t (legend shows mean/var only)
# ============================================================

AREA_PLOT = "EU"
DATE_TARGET = pd.to_datetime("2017-07-01")

sw_area = swaps[swaps["area"] == AREA_PLOT].copy()
if sw_area.empty:
    raise ValueError(f"No swaps for AREA={AREA_PLOT}.")

avail_dates = pd.to_datetime(sw_area["date"]).dropna().unique()
avail_dates = np.sort(avail_dates.astype("datetime64[ns]"))
target64 = np.datetime64(DATE_TARGET.to_datetime64())
idx = int(np.argmin(np.abs(avail_dates - target64)))
DATE_PLOT = pd.to_datetime(avail_dates[idx])

sw = swaps[(swaps["date"] == DATE_PLOT) & (swaps["area"] == AREA_PLOT)]
B = float(sw["B"].iloc[0])
ypi = float(sw["ypi_n"].iloc[0])

caps_g = caps[(caps["date"] == DATE_PLOT) & (caps["area"] == AREA_PLOT)]
floors_g = floors[(floors["date"] == DATE_PLOT) & (floors["area"] == AREA_PLOT)]
if caps_g.empty and floors_g.empty:
    raise ValueError(f"No caps/floors for AREA={AREA_PLOT}, DATE={DATE_PLOT.date()}.")

call_curve = build_call_curve_one_date(caps_g, floors_g, B=B, ypi=ypi)
k = call_curve["k"].to_numpy()
C = call_curve["C"].to_numpy()

kw = kw_density_from_call(k, C, B)
if kw is None:
    raise ValueError("KW density failed for this date.")
x_kw, f_kw, h_kw = kw

# 7 evaluation points + weights
x_pts = np.array([-0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05], dtype=float)
f_pts = np.interp(x_pts, x_kw, f_kw, left=0.0, right=0.0)
f_pts = np.maximum(f_pts, 1e-14)

# initial guess from KW moments
mu_kw, var_kw, *_ = moments_from_density(x_kw, f_kw)
sd0 = float(np.sqrt(var_kw)) if np.isfinite(var_kw) and var_kw > 0 else 0.01
mu0 = float(mu_kw) if np.isfinite(mu_kw) else float(np.sum((f_pts / f_pts.sum()) * x_pts))

# fit skew-t
fit = fit_skewt_to_kw_points(x_pts, f_pts, mu0=mu0, sd0=sd0, n_gh=40)
params_st = fit["params"]

# --- define percent formatter once ---
from matplotlib.ticker import FuncFormatter
pct_fmt = FuncFormatter(lambda x, pos: f"{100*x:.0f}%")

# --- plot grid ---
x_plot = np.linspace(-0.02, 0.06, 900)

f_st = pdf_skewt(x_plot, params_st, n_gh=60)
f_st = np.where(np.isfinite(f_st), f_st, 0.0)
f_st = np.maximum(f_st, 0.0)
Z = trapz(x_plot, f_st)
if np.isfinite(Z) and Z > 0:
    f_st = f_st / Z

# moments for legend
mu_st, var_st = moments_from_pdf_grid(x_plot, f_st)
mu_kw2, var_kw2 = moments_from_pdf_grid(x_kw, f_kw)

plt.figure(figsize=(10, 5))
plt.plot(
    x_kw, f_kw,
    label=f"KW (mean={100*mu_kw2:.2f}%, var={var_kw2:.6g})"
)
plt.plot(
    x_plot, f_st, linestyle="--",
    label=f"Skew-t (mean={100*mu_st:.2f}%, var={var_st:.6g})"
)
plt.scatter(x_pts, f_pts, marker="o", s=25, label="KW @ 7 points")

# --- x-axis as inflation in percent ---
plt.xlabel("Inflation")
plt.xlim(-0.02, 0.06)
plt.xticks(np.arange(-0.02, 0.061, 0.01))
plt.gca().xaxis.set_major_formatter(pct_fmt)

plt.ylabel("Density")
plt.title(f"KW vs Skew-t — {AREA_PLOT} — {DATE_PLOT.date().isoformat()}")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, f"fig_kw_vs_skewt_{AREA_PLOT}_{DATE_PLOT.date().isoformat()}.pdf"),
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 2) MULTI-DATE PLOT: Skew-t pdfs across dates (legend shows date + mean/var)
# ============================================================

AREA_MULTI = "EU"

# choose dates to plot:
# - option A: explicit list (recommended)
DATES_TO_PLOT = [
    "2017-01-01",
    #"2021-01-01",
    "2021-07-15",
    "2025-01-01",
    "2025-08-01",
]
DATES_TO_PLOT = ["2017-08-01",
"2021-07-15",
"2025-01-01",
"2025-05-01"]
DATES_TO_PLOT = [pd.to_datetime(d) for d in DATES_TO_PLOT]

# map each requested date to the nearest available (in swaps, same logic as above)
sw_area = swaps[swaps["area"] == AREA_MULTI].copy()
if sw_area.empty:
    raise ValueError(f"No swaps for AREA={AREA_MULTI}.")

avail_dates = pd.to_datetime(sw_area["date"]).dropna().unique()
avail_dates = np.sort(avail_dates.astype("datetime64[ns]"))

def nearest_available_date(ts):
    t64 = np.datetime64(ts.to_datetime64())
    j = int(np.argmin(np.abs(avail_dates - t64)))
    return pd.to_datetime(avail_dates[j])

dates_used = []
for dt_req in DATES_TO_PLOT:
    dt_use = nearest_available_date(dt_req)
    if dt_use not in dates_used:
        dates_used.append(dt_use)

# fixed plot grid for comparability (restricted to [-2%, +6%])
x_plot = np.linspace(-0.02, 0.06, 1200)

plt.figure(figsize=(10, 5))

for dt_use in dates_used:
    sw = swaps[(swaps["date"] == dt_use) & (swaps["area"] == AREA_MULTI)]
    if sw.empty:
        continue
    B = float(sw["B"].iloc[0])
    ypi = float(sw["ypi_n"].iloc[0])

    caps_g = caps[(caps["date"] == dt_use) & (caps["area"] == AREA_MULTI)]
    floors_g = floors[(floors["date"] == dt_use) & (floors["area"] == AREA_MULTI)]
    if caps_g.empty and floors_g.empty:
        continue

    call_curve = build_call_curve_one_date(caps_g, floors_g, B=B, ypi=ypi)
    k = call_curve["k"].to_numpy()
    C = call_curve["C"].to_numpy()

    kw = kw_density_from_call(k, C, B)
    if kw is None:
        continue
    x_kw, f_kw, _h = kw

    # 7 pts + weights
    x_pts = np.array([-0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05], dtype=float)
    f_pts = np.interp(x_pts, x_kw, f_kw, left=0.0, right=0.0)
    f_pts = np.maximum(f_pts, 1e-14)

    mu_kw, var_kw, *_ = moments_from_density(x_kw, f_kw)
    sd0 = float(np.sqrt(var_kw)) if np.isfinite(var_kw) and var_kw > 0 else 0.01
    mu0 = float(mu_kw) if np.isfinite(mu_kw) else float(np.sum((f_pts / f_pts.sum()) * x_pts))

    fit = fit_skewt_to_kw_points(x_pts, f_pts, mu0=mu0, sd0=sd0, n_gh=40)
    if not fit["ok"]:
        continue
    params_st = fit["params"]

    f_st = pdf_skewt(x_plot, params_st, n_gh=60)
    f_st = np.where(np.isfinite(f_st), f_st, 0.0)
    f_st = np.maximum(f_st, 0.0)
    Z = trapz(x_plot, f_st)
    if np.isfinite(Z) and Z > 0:
        f_st = f_st / Z

    mu_st, var_st = moments_from_pdf_grid(x_plot, f_st)

    plt.plot(
        x_plot, f_st,
        label=f"{dt_use.date().isoformat()} (mean={100*mu_st:.2f}%, var={var_st:.6g})"
    )

# --- x-axis as inflation in percent ---
plt.xlabel("Inflation")
plt.xlim(-0.02, 0.06)
plt.xticks(np.arange(-0.02, 0.061, 0.01))
plt.gca().xaxis.set_major_formatter(pct_fmt)

plt.ylabel("Density")
plt.title(f"Skew-t implied pdfs over time — {AREA_MULTI}")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"fig_skewt_pdfs_over_time_{AREA_MULTI}.pdf"),
            bbox_inches="tight")
plt.show()
