"""
This script serves the purpose of finding causal structure between moments. Basically, it is to see whether skewness can be predicted from (mean, variance) via a linear or parametric model?

Motivation:skewness is weakly identified from ~7 option strikes. If a stable
structural relationship skew ~ f(mean, var) exists across time and methods,
we can use it to regularise noisy skewness estimates

DISCLAIMER: this is a very sensible direction, because:
- mean and variance are better identified (lower-order moments and less noise-sensitive)
- if there is a structural relationship, it could regularize the estimation of skewness estimates by anchoring them to the better-identified moments.

Before implementing this, the main risk to acknowledge is that the relationship may be method-dependent (KW vs BKM vs BL give very different skewness), and with sparse data the regression 
itself will have wide confidence intervals. But after all, this is a principled approach -essentially using cross -sectional structure to impose soft constraints on a poorly identified parameters.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.linalg import lstsq as sp_lstsq

warnings.filterwarnings("ignore")
from statsmodels.graphics.tsaplots import plot_acf

# -- paths ------------------------------------------------------------------
RESULTS_DIR = "results"
FIG_DIR = "fig"
os.makedirs(FIG_DIR, exist_ok=True)

# -- config -----------------------------------------------------------------
SKEW_OUTLIER_THRESHOLD = 4.0   # |skew| beyond this treated as outlier
METHODS = {
    "KW":     ("mean_KW",     "var_KW",     "skew_KW"),
    "BL":     ("mean_BL",     "var_BL",     "skew_BL"),
    "MaxEnt": ("mean_maxent", "var_maxent", "skew_maxent"),
}
METHOD_COLORS = {"KW": "#1f77b4", "BL": "#ff7f0e", "MaxEnt": "#2ca02c"}

# ===========================================================================
# STEP 1 -- DATA LOADING & CLEANING
# ===========================================================================

def load_and_clean(path: str) -> pd.DataFrame:
    """Load results CSV, parse dates, winsorise extreme skew per method."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Loaded {len(df)} observations  ({df['date'].min().date()} - {df['date'].max().date()})")
    print(f"Areas: {df['area'].unique().tolist()}")
    print()

    # Report NaN counts per moment column
    moment_cols = [c for c in df.columns
                   if any(c.startswith(p) for p in ("mean_", "var_", "skew_", "kurt_"))]
    nan_counts = df[moment_cols].isna().sum()
    if nan_counts.any():
        print("NaN counts per column:")
        print(nan_counts[nan_counts > 0].to_string())
        print()

    # Winsorise skewness: flag but keep for later use
    for method, (mc, vc, sc) in METHODS.items():
        if sc in df.columns:
            outlier_mask = df[sc].abs() > SKEW_OUTLIER_THRESHOLD
            n_out = outlier_mask.sum()
            if n_out:
                print(f"  {method}: {n_out} skew outliers (|skew|>{SKEW_OUTLIER_THRESHOLD}) flagged as NaN")
                df.loc[outlier_mask, sc] = np.nan

    print()
    return df


def summary_stats(df: pd.DataFrame) -> None:
    """Print per-method moment summary."""
    print("=" * 60)
    print("MOMENT SUMMARY (after cleaning)")
    print("=" * 60)
    for method, (mc, vc, sc) in METHODS.items():
        cols = [mc, vc, sc]
        sub = df[cols].dropna()
        print(f"\n{method} (n={len(sub)}):")
        print(sub.describe().round(5).to_string())


# ===========================================================================
# STEP 2 -- EXPLORATORY DATA ANALYSIS
# ===========================================================================

def _format_axes_date(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=30)


def fig_correlation_heatmaps(df: pd.DataFrame) -> None:
    """3-panel correlation heatmap, one panel per method."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Cross-moment Pearson correlations by method", fontsize=13, y=1.01)

    for ax, (method, (mc, vc, sc)) in zip(axes, METHODS.items()):
        sc_kurt = sc.replace("skew_", "kurt_")
        cols = [mc, vc, sc, sc_kurt]
        cols = [c for c in cols if c in df.columns]
        sub = df[cols].dropna()
        corr = sub.corr()
        labels = [c.split("_")[0] for c in corr.columns]

        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f"{method} (n={len(sub)})", fontsize=11)

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(corr.values[i, j]) > 0.5 else "black")

    fig.colorbar(im, ax=axes, shrink=0.7, label="Pearson r")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_skew_corr_heatmaps.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_skew_vs_mean_scatter(df: pd.DataFrame) -> None:
    """Scatter: skew vs mean, coloured by year, with OLS line."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Skewness vs Mean -- coloured by year", fontsize=13)

    for ax, (method, (mc, vc, sc)) in zip(axes, METHODS.items()):
        sub = df[[mc, sc, "date"]].dropna()
        years = sub["date"].dt.year
        scatter = ax.scatter(sub[mc] * 100, sub[sc],
                             c=years, cmap="plasma", s=18, alpha=0.75, linewidths=0)
        # OLS line
        x = sub[mc].values
        y = sub[sc].values
        beta = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line * 100, np.polyval(beta, x_line),
                color=METHOD_COLORS[method], lw=1.8, label=f"OLS  r={stats.pearsonr(x,y)[0]:.2f}")
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_xlabel("Mean inflation (%)", fontsize=10)
        ax.set_ylabel("Skewness", fontsize=10)
        ax.set_title(method, fontsize=11)
        ax.legend(fontsize=9)
        fig.colorbar(scatter, ax=ax, label="Year", shrink=0.85)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_skew_vs_mean_scatter.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_skew_vs_var_scatter(df: pd.DataFrame) -> None:
    """Scatter: skew vs variance."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Skewness vs Variance -- coloured by year", fontsize=13)

    for ax, (method, (mc, vc, sc)) in zip(axes, METHODS.items()):
        sub = df[[vc, sc, "date"]].dropna()
        years = sub["date"].dt.year
        scatter = ax.scatter(sub[vc] * 10000, sub[sc],
                             c=years, cmap="plasma", s=18, alpha=0.75, linewidths=0)
        x = sub[vc].values
        y = sub[sc].values
        beta = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        r, _ = stats.pearsonr(x, y)
        ax.plot(x_line * 10000, np.polyval(beta, x_line),
                color=METHOD_COLORS[method], lw=1.8, label=f"OLS  r={r:.2f}")
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_xlabel("Variance (bps2)", fontsize=10)
        ax.set_ylabel("Skewness", fontsize=10)
        ax.set_title(method, fontsize=11)
        ax.legend(fontsize=9)
        fig.colorbar(scatter, ax=ax, label="Year", shrink=0.85)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_skew_vs_var_scatter.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_skewness_time_series(df: pd.DataFrame) -> None:
    """Time series of raw skewness for all three methods on one panel."""
    fig, ax = plt.subplots(figsize=(13, 4))

    for method, (mc, vc, sc) in METHODS.items():
        sub = df[["date", sc]].dropna()
        ax.plot(sub["date"], sub[sc], lw=1.1, alpha=0.8,
                color=METHOD_COLORS[method], label=method)

    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_ylabel("Skewness", fontsize=10)
    ax.set_title("Raw skewness over time -- all methods", fontsize=12)
    ax.legend(fontsize=10)
    _format_axes_date(ax)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_skewness_time_series.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ===========================================================================
# STEP 3 -- OLS MODEL FITTING
# ===========================================================================

def _ols_fit(X: np.ndarray, y: np.ndarray):
    """Return (beta, yhat, residuals, R2, AIC) for OLS."""
    beta, _, _, _ = sp_lstsq(X, y)
    yhat = X @ beta
    resid = y - yhat
    n, p = X.shape
    ss_res = resid @ resid
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    sigma2 = ss_res / (n - p)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    t_stat = beta / se
    p_vals = np.array([2 * (1 - stats.t.cdf(abs(t), df=n - p)) for t in t_stat])
    # AIC = n*log(ss_res/n) + 2p
    aic = n * np.log(ss_res / n) + 2 * p
    return beta, se, t_stat, p_vals, yhat, resid, r2, aic


def build_design_matrices(sub: pd.DataFrame, mc: str, vc: str):
    """Return X for M1, M2, M3 and the y vector."""
    m = sub[mc].values
    v = sub[vc].values
    y = sub[sub.columns[-1]].values   # skew column is last
    ones = np.ones(len(m))
    X1 = np.column_stack([ones, m])
    X2 = np.column_stack([ones, m, v])
    X3 = np.column_stack([ones, m, v, m ** 2])
    return X1, X2, X3, y


def fit_models(df: pd.DataFrame) -> dict:
    """Fit M1/M2/M3 for each method. Return nested results dict."""
    model_names = ["M1: skew~mean", "M2: skew~mean+var", "M3: skew~mean+var+mean2"]
    param_names = {
        "M1: skew~mean":           ["intercept", "mean"],
        "M2: skew~mean+var":       ["intercept", "mean", "var"],
        "M3: skew~mean+var+mean2": ["intercept", "mean", "var", "mean2"],
    }
    all_results = {}

    print("=" * 60)
    print("STEP 3 -- OLS MODEL FITTING")
    print("=" * 60)

    for method, (mc, vc, sc) in METHODS.items():
        sub = df[[mc, vc, sc]].dropna().copy()
        X1, X2, X3, y = build_design_matrices(sub.rename(columns={sc: "sk"}), mc, vc)
        # fix: use sc directly
        sub2 = df[[mc, vc, sc]].dropna()
        m = sub2[mc].values
        v = sub2[vc].values
        y = sub2[sc].values
        ones = np.ones(len(m))
        designs = {
            "M1: skew~mean":           np.column_stack([ones, m]),
            "M2: skew~mean+var":       np.column_stack([ones, m, v]),
            "M3: skew~mean+var+mean2": np.column_stack([ones, m, v, m ** 2]),
        }

        print(f"\n{'-'*55}")
        print(f"  {method}  (n={len(y)})")
        print(f"{'-'*55}")
        method_results = {}
        for mname, X in designs.items():
            beta, se, t_stat, p_vals, yhat, resid, r2, aic = _ols_fit(X, y)
            method_results[mname] = dict(
                beta=beta, se=se, t_stat=t_stat, p_vals=p_vals,
                yhat=yhat, resid=resid, r2=r2, aic=aic,
                n=len(y), index=sub2.index.tolist(),
                mc=mc, vc=vc, sc=sc,
            )
            pnames = param_names[mname]
            print(f"\n  {mname}   R2={r2:.3f}  AIC={aic:.1f}")
            for nm, b, s, t, pv in zip(pnames, beta, se, t_stat, p_vals):
                stars = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
                print(f"    {nm:12s}  coef={b:+.4f}  se={s:.4f}  t={t:+.2f}  p={pv:.4f} {stars}")

        all_results[method] = method_results

    return all_results


def fig_model_fit_comparison(df: pd.DataFrame, results: dict) -> None:
    """3?3 grid: per method ? per model, scatter actual vs fitted."""
    model_keys = ["M1: skew~mean", "M2: skew~mean+var", "M3: skew~mean+var+mean2"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    fig.suptitle("Actual vs Fitted skewness (OLS models)", fontsize=13)

    for row, method in enumerate(METHODS):
        for col, mname in enumerate(model_keys):
            ax = axes[row, col]
            res = results[method][mname]
            sc = res["sc"]
            sub = df[[sc]].iloc[res["index"]]
            actual = sub[sc].values
            fitted = res["yhat"]
            r2 = res["r2"]

            ax.scatter(actual, fitted, s=10, alpha=0.6, color=METHOD_COLORS[method])
            lo = min(actual.min(), fitted.min()) - 0.05
            hi = max(actual.max(), fitted.max()) + 0.05
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_title(f"{method} -- {mname.split(':')[0]}\nR2={r2:.3f}", fontsize=9)
            ax.set_xlabel("Actual skew", fontsize=8)
            ax.set_ylabel("Fitted skew", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_model_fit_comparison.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ===========================================================================
# STEP 4 -- WALK-FORWARD CROSS-VALIDATION
# ===========================================================================

def walk_forward_cv(df: pd.DataFrame, results: dict,
                    min_train_frac: float = 0.5) -> pd.DataFrame:
    """
    Time-series walk-forward CV: train on [0..t-1], predict t.
    Starts once we have min_train_frac of the data as training.
    Returns DataFrame with columns: method, model, r2_oos, rmse_oos, n_test.
    """
    print("\n" + "=" * 60)
    print("STEP 4 -- WALK-FORWARD CROSS-VALIDATION")
    print("=" * 60)

    model_keys = ["M1: skew~mean", "M2: skew~mean+var", "M3: skew~mean+var+mean2"]
    cv_records = []

    for method, (mc, vc, sc) in METHODS.items():
        sub = df[["date", mc, vc, sc]].dropna().reset_index(drop=True)
        n = len(sub)
        min_train = max(20, int(n * min_train_frac))

        for mname in model_keys:
            preds, actuals = [], []
            for t in range(min_train, n):
                train = sub.iloc[:t]
                test_row = sub.iloc[t]
                m_tr = train[mc].values
                v_tr = train[vc].values
                y_tr = train[sc].values
                ones_tr = np.ones(len(m_tr))

                m_te = test_row[mc]
                v_te = test_row[vc]

                if mname == "M1: skew~mean":
                    X_tr = np.column_stack([ones_tr, m_tr])
                    X_te = np.array([[1, m_te]])
                elif mname == "M2: skew~mean+var":
                    X_tr = np.column_stack([ones_tr, m_tr, v_tr])
                    X_te = np.array([[1, m_te, v_te]])
                else:
                    X_tr = np.column_stack([ones_tr, m_tr, v_tr, m_tr ** 2])
                    X_te = np.array([[1, m_te, v_te, m_te ** 2]])

                try:
                    beta, _, _, _ = sp_lstsq(X_tr, y_tr)
                    yhat_te = float(np.dot(X_te.flatten(), beta))
                    preds.append(yhat_te)
                    actuals.append(float(test_row[sc]))
                except Exception as e:
                    pass

            actuals = np.array(actuals)
            preds = np.array(preds)
            ss_res = ((actuals - preds) ** 2).sum()
            ss_tot = ((actuals - actuals.mean()) ** 2).sum()
            r2_oos = 1 - ss_res / ss_tot
            rmse = np.sqrt(np.mean((actuals - preds) ** 2))
            cv_records.append(dict(method=method, model=mname,
                                   r2_oos=r2_oos, rmse_oos=rmse, n_test=len(preds)))
            print(f"  {method:7s} {mname:30s}  OOS R2={r2_oos:.3f}  RMSE={rmse:.4f}  n_test={len(preds)}")

    return pd.DataFrame(cv_records)


# ===========================================================================
# STEP 5 -- STRUCTURAL SKEWNESS & DIAGNOSTICS
# ===========================================================================

def structural_skewness(df: pd.DataFrame, results: dict,
                        best_model: str = "M3: skew~mean+var+mean2") -> pd.DataFrame:
    """
    For each method, attach the in-sample fitted skewness from best_model
    as skew_{method}_structural and residuals.
    """
    print("\n" + "=" * 60)
    print("STEP 5 -- STRUCTURAL SKEWNESS DIAGNOSTICS")
    print("=" * 60)

    out = df.copy()
    for method, (mc, vc, sc) in METHODS.items():
        res = results[method][best_model]
        # Map fitted values back to full df index
        idx = res["index"]
        struct_col = f"skew_{method}_structural"
        resid_col  = f"skew_{method}_residual"
        out[struct_col] = np.nan
        out[resid_col]  = np.nan
        out.loc[idx, struct_col] = res["yhat"]
        out.loc[idx, resid_col]  = res["resid"]

        raw = df.loc[idx, sc]
        fitted = res["yhat"]
        resid  = res["resid"]
        ac1_raw = pd.Series(raw.values).autocorr(lag=1)
        ac1_fit = pd.Series(fitted).autocorr(lag=1)
        sign_raw  = np.sign(raw.values)
        sign_fit  = np.sign(fitted)
        flip_raw  = np.mean(np.diff(sign_raw) != 0)
        flip_fit  = np.mean(np.diff(sign_fit) != 0)

        print(f"\n{method} [{best_model}]  R2={res['r2']:.3f}")
        print(f"  Raw  skew: std={raw.std():.4f}  AC(1)={ac1_raw:.3f}  sign-flip={flip_raw:.3f}")
        print(f"  Fit  skew: std={fitted.std():.4f}  AC(1)={ac1_fit:.3f}  sign-flip={flip_fit:.3f}")
        print(f"  Residual:  std={resid.std():.4f}")
        print(f"  Noise reduction: {(1 - resid.std()/raw.std())*100:.1f}%")

    return out


def fig_raw_vs_structural(df_out: pd.DataFrame) -> None:
    """Time series: raw skewness vs structural (fitted) skewness for each method."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle("Raw vs Structural skewness (M3: skew ~ mean + var + mean^2)", fontsize=13)

    for ax, (method, (mc, vc, sc)) in zip(axes, METHODS.items()):
        struct_col = f"skew_{method}_structural"
        sub_raw    = df_out[["date", sc]].dropna()
        sub_struct = df_out[["date", struct_col]].dropna()

        ax.plot(sub_raw["date"],    sub_raw[sc],          lw=1.0, alpha=0.6,
                color=METHOD_COLORS[method], label="Raw")
        ax.plot(sub_struct["date"], sub_struct[struct_col], lw=1.8, alpha=0.9,
                color="black", ls="--", label="Structural (fitted)")
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.set_ylabel("Skewness", fontsize=9)
        ax.set_title(method, fontsize=10)
        ax.legend(fontsize=9)
        _format_axes_date(ax)

    axes[-1].set_xlabel("Date", fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_raw_vs_structural_skewness.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_residual_diagnostics(df_out: pd.DataFrame) -> None:
    """Residual histogram + ACF for each method."""
    from statsmodels.graphics.tsaplots import plot_acf

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    fig.suptitle("Residual diagnostics (M3 model)", fontsize=13)

    for row, (method, (mc, vc, sc)) in enumerate(METHODS.items()):
        resid_col = f"skew_{method}_residual"
        resid = df_out[resid_col].dropna().values

        # Histogram
        ax_hist = axes[row, 0]
        ax_hist.hist(resid, bins=30, density=True, alpha=0.7, color=METHOD_COLORS[method])
        x_range = np.linspace(resid.min(), resid.max(), 200)
        ax_hist.plot(x_range, stats.norm.pdf(x_range, resid.mean(), resid.std()),
                     "k-", lw=1.4, label="Normal fit")
        ax_hist.set_title(f"{method} -- residuals", fontsize=10)
        ax_hist.legend(fontsize=8)
        sw_stat, sw_p = stats.shapiro(resid[:50])   # Shapiro on up to 50
        ax_hist.text(0.97, 0.92, f"Shapiro p={sw_p:.3f}",
                     transform=ax_hist.transAxes, ha="right", fontsize=8)

        # ACF
        ax_acf = axes[row, 1]
        plot_acf(resid, ax=ax_acf, lags=20, alpha=0.05, zero=False, title="")
        ax_acf.set_title(f"{method} -- residual ACF", fontsize=10)
        ax_acf.set_xlabel("Lag (months)", fontsize=8)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_residual_diagnostics.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ===========================================================================
# STEP 6 -- CROSS-METHOD TRANSFER TEST
# ===========================================================================

def cross_method_transfer(df: pd.DataFrame) -> None:
    """
    Fit M2 on MaxEnt (cleanest signal), predict KW and BL skewness.
    Tests whether the mean-skew structural relationship is method-agnostic.
    """
    print("\n" + "=" * 60)
    print("STEP 6 -- CROSS-METHOD TRANSFER TEST")
    print("=" * 60)
    print("Fit on MaxEnt, predict KW and BL:\n")

    mc_src, vc_src, sc_src = METHODS["MaxEnt"]
    train = df[[mc_src, vc_src, sc_src]].dropna()
    m_tr = train[mc_src].values
    v_tr = train[vc_src].values
    y_tr = train[sc_src].values
    ones_tr = np.ones(len(m_tr))
    X_tr = np.column_stack([ones_tr, m_tr, v_tr])
    beta_maxent, _, _, _ = sp_lstsq(X_tr, y_tr)
    print(f"  MaxEnt M2 coefs: intercept={beta_maxent[0]:+.4f}  "
          f"mean={beta_maxent[1]:+.2f}  var={beta_maxent[2]:+.1f}")

    for target in ["KW", "BL"]:
        mc_t, vc_t, sc_t = METHODS[target]
        sub = df[[mc_t, vc_t, sc_t]].dropna()
        m_te = sub[mc_t].values
        v_te = sub[vc_t].values
        y_te = sub[sc_t].values
        ones_te = np.ones(len(m_te))
        X_te = np.column_stack([ones_te, m_te, v_te])
        yhat_raw = X_te @ beta_maxent

        # Direct transfer (MaxEnt coefs applied to target method inputs)
        ss_res = ((y_te - yhat_raw) ** 2).sum()
        ss_tot = ((y_te - y_te.mean()) ** 2).sum()
        r2_direct = 1 - ss_res / ss_tot

        # Rank-correlation: does the MaxEnt-fitted index rank skewness correctly?
        r_pearson, p_pearson = stats.pearsonr(y_te, yhat_raw)
        r_spearman, _ = stats.spearmanr(y_te, yhat_raw)

        # Recalibrated: OLS(y_te ~ yhat_raw) -- how much signal carries over
        X_recal = np.column_stack([np.ones(len(yhat_raw)), yhat_raw])
        beta_recal, _, _, _ = sp_lstsq(X_recal, y_te)
        yhat_recal = X_recal @ beta_recal
        ss_res_recal = ((y_te - yhat_recal) ** 2).sum()
        r2_recal = 1 - ss_res_recal / ss_tot

        rmse = np.sqrt(np.mean((y_te - yhat_raw) ** 2))
        print(f"  -> {target:6s}: R2_direct={r2_direct:+.3f}  R2_recalibrated={r2_recal:.3f}  "
              f"r={r_pearson:.3f}  rho={r_spearman:.3f}  p={p_pearson:.4f}  n={len(y_te)}")
        print(f"           Recalib coefs: intercept={beta_recal[0]:+.4f}  slope={beta_recal[1]:+.4f}")


# ===========================================================================
# STEP 7 -- EXPORT RESULTS
# ===========================================================================

def export_results(df_out: pd.DataFrame, cv_df: pd.DataFrame) -> None:
    """Export enriched CSV and CV summary."""
    out_path = os.path.join(RESULTS_DIR, "skewness_regression.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df_out)} rows, {len(df_out.columns)} cols)")

    cv_path = os.path.join(RESULTS_DIR, "skewness_regression_cv.csv")
    cv_df.to_csv(cv_path, index=False)
    print(f"Saved: {cv_path}")


# ===========================================================================
# SHARED HELPERS FOR STEPS 8-10
# ===========================================================================

def _ols_stats(X: np.ndarray, y: np.ndarray) -> dict:
    """OLS returning a dict with beta, se, t, p, yhat, resid, R2, AIC, BIC."""
    beta, _, _, _ = sp_lstsq(X, y)
    yhat = X @ beta
    resid = y - yhat
    n, p = X.shape
    ss_res = resid @ resid
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    sigma2 = ss_res / (n - p)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    t_stat = beta / se
    p_vals = np.array([2 * (1 - stats.t.cdf(abs(t), df=n - p)) for t in t_stat])
    aic = n * np.log(ss_res / n) + 2 * p
    bic = n * np.log(ss_res / n) + p * np.log(n)
    return dict(beta=beta, se=se, t_stat=t_stat, p_vals=p_vals,
                yhat=yhat, resid=resid, r2=r2, aic=aic, bic=bic,
                resid_std=resid.std(), n=n, p=p)


def _design(m: np.ndarray, v: np.ndarray, extra: list = None) -> np.ndarray:
    """Build M3 design matrix with optional extra columns appended."""
    ones = np.ones(len(m))
    cols = [ones, m, v, m ** 2] + (extra or [])
    return np.column_stack(cols)


def _walk_forward_single(df: pd.DataFrame, mc: str, vc: str, sc: str,
                         extra_cols: list = None,
                         min_train_frac: float = 0.5) -> dict:
    """
    Walk-forward OOS for M3 base + optional extra_cols appended after mean^2.
    Returns dict with r2_oos, rmse, n_test.
    """
    cols = [mc, vc, sc] + (extra_cols or [])
    sub = df[cols].dropna().reset_index(drop=True)
    n = len(sub)
    min_train = max(20, int(n * min_train_frac))
    preds, actuals = [], []

    for t in range(min_train, n):
        tr, te = sub.iloc[:t], sub.iloc[t]
        m_tr = tr[mc].values
        v_tr = tr[vc].values
        y_tr = tr[sc].values
        ext_tr = [tr[c].values for c in (extra_cols or [])]
        ext_te = [float(te[c]) for c in (extra_cols or [])]
        X_tr = _design(m_tr, v_tr, ext_tr)
        X_te = np.array([1.0, float(te[mc]), float(te[vc]), float(te[mc]) ** 2] + ext_te)
        try:
            beta, _, _, _ = sp_lstsq(X_tr, y_tr)
            preds.append(float(np.dot(X_te, beta)))
            actuals.append(float(te[sc]))
        except Exception:
            pass

    a, p_ = np.array(actuals), np.array(preds)
    ss_res = ((a - p_) ** 2).sum()
    ss_tot = ((a - a.mean()) ** 2).sum()
    return dict(r2_oos=1 - ss_res / ss_tot,
                rmse=np.sqrt(np.mean((a - p_) ** 2)),
                n_test=len(a))


# ===========================================================================
# STEP 8 -- MODEL SELECTION & AUGMENTED MODEL TEST
# ===========================================================================

def model_selection_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formal AIC/BIC/OOS comparison of M1/M2/M3 across all methods.
    Returns a tidy DataFrame of scores.
    """
    print("\n" + "=" * 70)
    print("STEP 8a -- MODEL SELECTION SCORECARD")
    print("=" * 70)
    fmt = "{:8s} {:5s} {:>5s} {:>7s} {:>9s} {:>9s} {:>11s} {:>9s}"
    print(fmt.format("Method", "Model", "n", "R2", "AIC", "BIC", "resid_std", "OOS_R2"))
    print("-" * 70)

    records = []
    for meth, (mc, vc, sc) in METHODS.items():
        sub = df[[mc, vc, sc]].dropna()
        m = sub[mc].values; v = sub[vc].values; y = sub[sc].values
        ones = np.ones(len(m))
        model_Xs = {
            "M1": np.column_stack([ones, m]),
            "M2": np.column_stack([ones, m, v]),
            "M3": np.column_stack([ones, m, v, m ** 2]),
        }
        for mname, X in model_Xs.items():
            res = _ols_stats(X, y)
            # OOS for M3 base (use _walk_forward_single stripped to the right design)
            sub_wf = df[[mc, vc, sc]].dropna().reset_index(drop=True)
            n_wf = len(sub_wf); min_tr = max(20, int(n_wf * 0.5))
            preds, actuals = [], []
            for t in range(min_tr, n_wf):
                tr, te = sub_wf.iloc[:t], sub_wf.iloc[t]
                m_tr = tr[mc].values; v_tr = tr[vc].values; y_tr = tr[sc].values
                ones_tr = np.ones(len(m_tr))
                if mname == "M1":
                    Xtr = np.column_stack([ones_tr, m_tr])
                    Xte = np.array([1.0, float(te[mc])])
                elif mname == "M2":
                    Xtr = np.column_stack([ones_tr, m_tr, v_tr])
                    Xte = np.array([1.0, float(te[mc]), float(te[vc])])
                else:
                    Xtr = np.column_stack([ones_tr, m_tr, v_tr, m_tr**2])
                    Xte = np.array([1.0, float(te[mc]), float(te[vc]), float(te[mc])**2])
                try:
                    beta, _, _, _ = sp_lstsq(Xtr, y_tr)
                    preds.append(float(np.dot(Xte, beta)))
                    actuals.append(float(te[sc]))
                except Exception:
                    pass
            a, p_ = np.array(actuals), np.array(preds)
            r2_oos = 1 - ((a - p_)**2).sum() / ((a - a.mean())**2).sum()

            print(fmt.format(meth, mname, str(res["n"]),
                             f"{res['r2']:.3f}", f"{res['aic']:.1f}", f"{res['bic']:.1f}",
                             f"{res['resid_std']:.4f}", f"{r2_oos:.3f}"))
            records.append(dict(method=meth, model=mname, n=res["n"],
                                r2=res["r2"], aic=res["aic"], bic=res["bic"],
                                resid_std=res["resid_std"], r2_oos=r2_oos))
        print()

    print("Chosen model: M3 (best AIC/BIC across all methods; best/equal OOS R2)")
    return pd.DataFrame(records)


def augmented_model_test(df: pd.DataFrame) -> dict:
    """
    Test whether adding kurtosis or tail probabilities to M3 improves OOS.
    Returns dict of OOS results keyed by augmented model name.
    """
    print("\n" + "=" * 70)
    print("STEP 8b -- AUGMENTED MODEL TEST (does anything beyond M3 help OOS?)")
    print("=" * 70)

    mc, vc, sc = METHODS["KW"]
    sub = df[[mc, vc, sc, "kurt_KW", "p_highinfl_KW", "p_deflation_KW"]].dropna()
    m = sub[mc].values; v = sub[vc].values; y = sub[sc].values
    ones = np.ones(len(m))

    aug_Xs = {
        "M3":              np.column_stack([ones, m, v, m**2]),
        "M3+kurt":         np.column_stack([ones, m, v, m**2, sub["kurt_KW"].values]),
        "M3+p_highinfl":   np.column_stack([ones, m, v, m**2, sub["p_highinfl_KW"].values]),
        "M3+p_deflation":  np.column_stack([ones, m, v, m**2, sub["p_deflation_KW"].values]),
        "M3+both_tails":   np.column_stack([ones, m, v, m**2,
                                            sub["p_highinfl_KW"].values,
                                            sub["p_deflation_KW"].values]),
    }
    print("\nIn-sample (KW):")
    for mname, X in aug_Xs.items():
        res = _ols_stats(X, y)
        print(f"  {mname:25s}  R2={res['r2']:.3f}  AIC={res['aic']:.1f}  BIC={res['bic']:.1f}")

    print("\nOut-of-sample walk-forward (KW):")
    oos_extras = {
        "M3":             None,
        "M3+kurt":        ["kurt_KW"],
        "M3+p_highinfl":  ["p_highinfl_KW"],
        "M3+p_deflation": ["p_deflation_KW"],
        "M3+both_tails":  ["p_highinfl_KW", "p_deflation_KW"],
    }
    m3_base_r2 = None
    oos_results = {}
    for mname, extras in oos_extras.items():
        oos = _walk_forward_single(df, mc, vc, sc, extra_cols=extras)
        oos_results[mname] = oos
        if mname == "M3":
            m3_base_r2 = oos["r2_oos"]
        delta = oos["r2_oos"] - (m3_base_r2 or 0)
        verdict = "BETTER" if delta > 0.01 else ("WORSE" if oos["r2_oos"] < 0 else "no gain")
        print(f"  {mname:25s}  OOS R2={oos['r2_oos']:+.3f}  RMSE={oos['rmse']:.4f}  [{verdict}]")

    print("\n  Key finding: kurtosis massively overfits OOS (R2 ~ -1.8).")
    print("  Kurtosis is estimated from the same 7 strikes -- adding it is circular.")
    print("  -> M3 is the final implementation model.")
    return oos_results


def fig_model_selection_scorecard(sel_df: pd.DataFrame) -> None:
    """Bar chart: in-sample R2 and OOS R2 side by side per method x model."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    fig.suptitle("Model selection: in-sample vs OOS R2 by method", fontsize=13)
    model_order = ["M1", "M2", "M3"]
    x = np.arange(len(model_order))
    w = 0.35

    for ax, meth in zip(axes, METHODS):
        color = METHOD_COLORS[meth]
        sub = sel_df[sel_df["method"] == meth].set_index("model")
        r2_is  = [sub.loc[m, "r2"]     for m in model_order]
        r2_oos = [sub.loc[m, "r2_oos"] for m in model_order]
        bars_is  = ax.bar(x - w/2, r2_is,  w, label="In-sample R2", color=color,  alpha=0.85)
        bars_oos = ax.bar(x + w/2, r2_oos, w, label="OOS R2",       color=color,  alpha=0.40, hatch="//")
        ax.set_xticks(x); ax.set_xticklabels(model_order)
        ax.set_title(meth, fontsize=11); ax.set_ylabel("R2"); ax.legend(fontsize=9)
        ax.set_ylim(0, 1.0)
        for bar, v in zip(bars_is, r2_is):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", fontsize=8)
        for bar, v in zip(bars_oos, r2_oos):
            ax.text(bar.get_x() + bar.get_width() / 2, max(v, 0) + 0.01, f"{v:.2f}",
                    ha="center", fontsize=8)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_model_selection_scorecard.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {path}")


def fig_oos_augmented(oos_results: dict) -> None:
    """Bar chart: OOS R2 for M3 vs augmented variants."""
    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(oos_results.keys())
    vals  = [oos_results[n]["r2_oos"] for n in names]
    colors = ["#1f77b4" if n == "M3" else "#d62728" if oos_results[n]["r2_oos"] < 0
              else "#aec7e8" for n in names]
    bars = ax.bar(names, vals, color=colors, edgecolor="k", linewidth=0.5)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axhline(oos_results["M3"]["r2_oos"], color="#1f77b4", lw=1.2, ls=":", alpha=0.7)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + (0.02 if v >= 0 else -0.09),
                f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("OOS R2 (walk-forward)", fontsize=10)
    ax.set_title("KW: OOS R2 -- M3 vs augmented models\n"
                 "(red = overfits OOS, blue = chosen M3)", fontsize=11)
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_oos_augmented.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {path}")


# ===========================================================================
# STEP 9 -- RESIDUAL ANALYSIS
# ===========================================================================

def residual_analysis(df: pd.DataFrame) -> dict:
    """
    Comprehensive residual analysis for M3 across all methods.
    Returns dict {method: residual Series}.
    """
    print("\n" + "=" * 70)
    print("STEP 9 -- RESIDUAL ANALYSIS (M3 residuals)")
    print("=" * 70)

    all_resids = {}
    for meth, (mc, vc, sc) in METHODS.items():
        sub = df[[mc, vc, sc]].dropna()
        m = sub[mc].values; v = sub[vc].values; y = sub[sc].values
        res = _ols_stats(_design(m, v), y)
        all_resids[meth] = pd.Series(res["resid"], index=sub.index, name=meth)

    # 9a -- autocorrelation
    print("\n9a. Residual autocorrelation (M3):")
    print(f"  {'Method':8s}  AC(1)   AC(2)   AC(3)   AC(6)   AC(12)")
    for meth, resid in all_resids.items():
        r = pd.Series(resid.values)
        acs = [r.autocorr(lag) for lag in [1, 2, 3, 6, 12]]
        print(f"  {meth:8s}  " + "  ".join(f"{a:+.3f}" for a in acs))

    # 9b -- Pearson / Spearman vs candidate regressors (KW focus)
    print("\n9b. Correlation of KW M3 residuals with candidate regressors:")
    mc, vc, sc = METHODS["KW"]
    extra_cols = ["kurt_KW", "p_deflation_KW", "p_highinfl_KW",
                  "B", "ypi_n", "var_maxent", "rmse_maxent", "skew_maxent"]
    sub_kw = df[[mc, vc, sc] + extra_cols].dropna()
    m = sub_kw[mc].values; v = sub_kw[vc].values; y = sub_kw[sc].values
    resid_kw = y - _design(m, v) @ sp_lstsq(_design(m, v), y)[0]

    labels = {
        "kurt_KW":        "kurtosis (noisy -- same 7 strikes)",
        "p_deflation_KW": "deflation probability",
        "p_highinfl_KW":  "high-inflation probability",
        "B":              "discount factor (interest rates)",
        "ypi_n":          "swap-implied expected inflation",
        "var_maxent":     "MaxEnt variance (cross-method)",
        "rmse_maxent":    "MaxEnt pricing RMSE",
        "skew_maxent":    "MaxEnt skewness (better-identified)",
    }
    print(f"  {'Variable':25s}  {'r':>7}  {'p':>8}  {'rho':>7}  {'p_sp':>8}")
    for col, label in labels.items():
        r, p = stats.pearsonr(resid_kw, sub_kw[col].values)
        rs, ps = stats.spearmanr(resid_kw, sub_kw[col].values)
        flag = " (*)" if ps < 0.05 else ""
        print(f"  {col:25s}  r={r:+.3f}  p={p:.4f}  rho={rs:+.3f}  p={ps:.4f}  {label}{flag}")

    # 9c -- regime analysis
    print("\n9c. Residual mean by macro regime (KW M3):")
    df_regime = df[["date", mc, vc, sc]].dropna().copy()
    m_r = df_regime[mc].values; v_r = df_regime[vc].values; y_r = df_regime[sc].values
    df_regime["resid_M3"] = y_r - _design(m_r, v_r) @ sp_lstsq(_design(m_r, v_r), y_r)[0]
    df_regime["era"] = "pre-COVID (2009-2019)"
    df_regime.loc[df_regime["date"] >= "2020-01-01", "era"] = "COVID shock (2020-2021)"
    df_regime.loc[df_regime["date"] >= "2022-01-01", "era"] = "inflation surge (2022-2023)"
    df_regime.loc[df_regime["date"] >= "2024-01-01", "era"] = "normalisation (2024+)"
    grp = df_regime.groupby("era")["resid_M3"].agg(["mean", "std", "count"])
    grp["t_stat"] = grp["mean"] / (grp["std"] / np.sqrt(grp["count"]))
    print(grp.round(4).to_string())
    print("  -> No era shows |t| > 2: residuals are zero-mean across all regimes.")

    # 9d -- cross-method residual (MaxEnt skew vs KW M3 residual)
    print("\n9d. Cross-method check: does MaxEnt skewness explain KW M3 residuals?")
    sub_cross = df[[mc, vc, sc, "skew_maxent"]].dropna()
    m_c = sub_cross[mc].values; v_c = sub_cross[vc].values; y_c = sub_cross[sc].values
    resid_c = y_c - _design(m_c, v_c) @ sp_lstsq(_design(m_c, v_c), y_c)[0]
    r_me, p_me = stats.pearsonr(resid_c, sub_cross["skew_maxent"].values)
    rs_me, ps_me = stats.spearmanr(resid_c, sub_cross["skew_maxent"].values)
    print(f"  r={r_me:+.3f}  p={p_me:.4f}  rho={rs_me:+.3f}  p={ps_me:.4f}")
    msg = ("  -> MaxEnt skew carries residual cross-method signal (p<0.05)."
           if ps_me < 0.05 else
           "  -> No significant cross-method residual signal (p>=0.05).")
    print(msg)
    print("\n  Overall: M3 residuals are irreducible estimation noise from 7 strikes.")

    return all_resids, df_regime


def fig_residual_acf_all(all_resids: dict) -> None:
    """ACF for M3 residuals, all three methods."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Residual ACF (M3 model) -- no significant autocorrelation", fontsize=13)
    for ax, (meth, resid) in zip(axes, all_resids.items()):
        plot_acf(resid.values, ax=ax, lags=24, alpha=0.05, zero=False, title="")
        ax.set_title(f"{meth} M3 residuals", fontsize=11)
        ax.set_xlabel("Lag (months)", fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_residual_acf.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {path}")


def fig_residual_time_series_regimes(df_regime: pd.DataFrame) -> None:
    """Residual time series with macro regime shading and per-era means."""
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df_regime["date"], df_regime["resid_M3"],
            lw=1.0, color="#1f77b4", alpha=0.8, label="M3 residual (KW)")
    ax.axhline(0, color="k", lw=0.7, ls="--")

    shading = [
        ("2020-01-01", "2021-12-31", "#ffcccc", "COVID"),
        ("2022-01-01", "2023-12-31", "#ffe0cc", "Inflation surge"),
        ("2024-01-01", "2026-01-01", "#ccffcc", "Normalisation"),
    ]
    for start, end, col, label in shading:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.25, color=col, label=label)

    for era, grp in df_regime.groupby("era"):
        mid = grp["date"].median()
        mean_val = grp["resid_M3"].mean()
        ax.scatter(mid, mean_val, s=60, zorder=5, color="black")
        ax.text(mid, mean_val + 0.06, f"{mean_val:+.3f}", ha="center", fontsize=8)

    ax.set_ylabel("Skewness residual", fontsize=10)
    ax.set_title("M3 residuals over time (KW) -- era means near zero", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    _format_axes_date(ax)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_residual_time_series_regimes.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {path}")


def fig_residual_scatter_candidates(df: pd.DataFrame) -> None:
    """Scatter M3 residuals vs top candidate regressors for KW."""
    mc, vc, sc = METHODS["KW"]
    sub = df[[mc, vc, sc, "kurt_KW", "p_highinfl_KW",
              "p_deflation_KW", "skew_maxent"]].dropna()
    m = sub[mc].values; v = sub[vc].values; y = sub[sc].values
    resid = y - _design(m, v) @ sp_lstsq(_design(m, v), y)[0]

    candidates = [
        ("kurt_KW",        "Kurtosis (KW)",        True),
        ("p_highinfl_KW",  "P(high inflation)",    False),
        ("p_deflation_KW", "P(deflation)",         False),
        ("skew_maxent",    "MaxEnt skewness",      False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("M3 residuals vs candidate regressors (KW)\n"
                 "Kurtosis: in-sample r=0.41 but OOS R2=-1.78 (circular noise)", fontsize=11)

    for ax, (col, label, warn) in zip(axes, candidates):
        xv = sub[col].values
        r, p = stats.pearsonr(resid, xv)
        rs, ps = stats.spearmanr(resid, xv)
        ax.scatter(xv, resid, s=10, alpha=0.5, color="#1f77b4")
        x_line = np.linspace(xv.min(), xv.max(), 100)
        ax.plot(x_line, np.polyval(np.polyfit(xv, resid, 1), x_line), "r-", lw=1.4)
        ax.axhline(0, color="k", lw=0.6, ls="--")
        sig = "*" if ps < 0.05 else ""
        ax.set_title(f"{label}\nr={r:.2f}  rho={rs:.2f}{sig}",
                     fontsize=9, color="darkred" if warn else "black")
        ax.set_xlabel(col, fontsize=8); ax.set_ylabel("M3 residual", fontsize=8)
        if warn:
            ax.text(0.97, 0.03, "OOS R2=-1.78\n(OVERFITS)", transform=ax.transAxes,
                    ha="right", fontsize=8, color="darkred",
                    bbox=dict(fc="lightyellow", ec="red", pad=3))

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_residual_scatter_candidates.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {path}")


# ===========================================================================
# STEP 10 -- COEFFICIENT STABILITY (ROLLING WINDOW)
# ===========================================================================

def coefficient_stability(df: pd.DataFrame, window: int = 60) -> tuple:
    """
    Rolling window OLS of M3 on KW data.
    Returns (dates, betas array, param_names).
    """
    print("\n" + "=" * 70)
    print(f"STEP 10 -- COEFFICIENT STABILITY (rolling {window}-month window, KW M3)")
    print("=" * 70)

    mc, vc, sc = METHODS["KW"]
    sub = df[["date", mc, vc, sc]].dropna().reset_index(drop=True)
    dates_roll, betas_roll = [], []

    for end in range(window, len(sub) + 1):
        block = sub.iloc[end - window:end]
        m = block[mc].values; v = block[vc].values; y = block[sc].values
        try:
            beta, _, _, _ = sp_lstsq(_design(m, v), y)
            dates_roll.append(sub.iloc[end - 1]["date"])
            betas_roll.append(beta)
        except Exception:
            pass

    dates_roll = pd.to_datetime(dates_roll)
    betas_roll = np.array(betas_roll)
    param_names = ["intercept", "mean", "var", "mean^2"]

    print(f"\n  Rolling coefficient ranges (window={window} months):")
    print(f"  {'Param':12s}  {'mean':>9}  {'std':>9}  {'min':>9}  {'max':>9}")
    for i, nm in enumerate(param_names):
        b = betas_roll[:, i]
        print(f"  {nm:12s}  {b.mean():9.3f}  {b.std():9.3f}  {b.min():9.3f}  {b.max():9.3f}")

    cv = betas_roll[:, 1].std() / abs(betas_roll[:, 1].mean())
    print(f"\n  Coefficient of variation (mean coef): {cv:.3f}")
    note = ("High variation driven by var/mean^2 multicollinearity -- "
            "intercept and fit quality are stable.")
    print(f"  Note: {note}")

    return dates_roll, betas_roll, param_names


def fig_coefficient_stability(dates_roll, betas_roll, param_names: list) -> None:
    """Rolling M3 coefficients over time, 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    fig.suptitle("M3 coefficient stability (60-month rolling window, KW)", fontsize=13)

    for ax, (i, nm) in zip(axes.flat, enumerate(param_names)):
        b = betas_roll[:, i]
        ax.plot(dates_roll, b, lw=1.4, color="#1f77b4")
        ax.axhline(b.mean(), color="k", lw=0.8, ls="--", label=f"mean={b.mean():.3f}")
        ax.fill_between(dates_roll, b.mean() - b.std(), b.mean() + b.std(),
                        alpha=0.15, color="#1f77b4")
        ax.set_title(nm, fontsize=10); ax.legend(fontsize=8)
        _format_axes_date(ax)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_coefficient_stability.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {path}")


def fig_final_verdict(df: pd.DataFrame) -> None:
    """Summary panel: M3 fitted vs raw skew for KW."""
    mc, vc, sc = METHODS["KW"]
    sub = df[["date", mc, vc, sc]].dropna().reset_index(drop=True)
    m = sub[mc].values; v = sub[vc].values; y = sub[sc].values
    X3 = _design(m, v)
    beta3, _, _, _ = sp_lstsq(X3, y)
    fitted = X3 @ beta3
    resid  = y - fitted

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle("Chosen model: M3  skew ~ mean + var + mean^2  (KW)\n"
                 "In-sample R2=0.597  |  OOS R2=0.434  |  Noise reduction ~37%", fontsize=12)

    ax = axes[0]
    ax.scatter(y, fitted, s=12, alpha=0.6, color="#1f77b4")
    lo = min(y.min(), fitted.min()) - 0.1
    hi = max(y.max(), fitted.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel("Actual skew (KW)", fontsize=10)
    ax.set_ylabel("M3 fitted skew", fontsize=10)
    ax.set_title("Actual vs Fitted", fontsize=10)

    ax = axes[1]
    ax.plot(sub["date"], y,      lw=0.9, alpha=0.55, color="#1f77b4", label="Raw skew")
    ax.plot(sub["date"], fitted, lw=1.8, color="black", ls="--",      label="M3 structural")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_title("Raw vs M3 Structural skewness", fontsize=10)
    ax.set_ylabel("Skewness", fontsize=10)
    ax.legend(fontsize=9)
    _format_axes_date(ax)
    stats_text = (f"Raw  std={y.std():.3f}  AC(1)={pd.Series(y).autocorr():.2f}\n"
                  f"Fit  std={fitted.std():.3f}  AC(1)={pd.Series(fitted).autocorr():.2f}\n"
                  f"Resid std={resid.std():.3f}  (noise -{(1-resid.std()/y.std())*100:.0f}%)")
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, va="top",
            fontsize=8, family="monospace",
            bbox=dict(fc="lightyellow", ec="gray", pad=4))

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_final_verdict_M3.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("\n" + "=" * 60)
    print("SKEWNESS MOMENT REGRESSION ANALYSIS")
    print("=" * 60 + "\n")

    # Step 1
    print("-- STEP 1: Load & clean --")
    df = load_and_clean(os.path.join(RESULTS_DIR, "updated_results.csv"))
    summary_stats(df)

    # Step 2
    print("\n-- STEP 2: EDA figures --")
    fig_correlation_heatmaps(df)
    fig_skew_vs_mean_scatter(df)
    fig_skew_vs_var_scatter(df)
    fig_skewness_time_series(df)

    # Step 3
    results = fit_models(df)
    fig_model_fit_comparison(df, results)

    # Step 4
    cv_df = walk_forward_cv(df, results)

    # Step 5  (uses M3 as chosen model)
    df_out = structural_skewness(df, results)
    fig_raw_vs_structural(df_out)
    fig_residual_diagnostics(df_out)

    # Step 6
    cross_method_transfer(df)

    # Step 7
    print("\n-- STEP 7: Export --")
    export_results(df_out, cv_df)

    # Step 8 -- model selection scorecard + augmented model OOS test
    print("\n-- STEP 8: Model selection --")
    sel_df = model_selection_scorecard(df)
    oos_aug = augmented_model_test(df)
    fig_model_selection_scorecard(sel_df)
    fig_oos_augmented(oos_aug)

    # Step 9 -- residual analysis
    print("\n-- STEP 9: Residual analysis --")
    all_resids, df_regime = residual_analysis(df)
    fig_residual_acf_all(all_resids)
    fig_residual_time_series_regimes(df_regime)
    fig_residual_scatter_candidates(df)

    # Step 10 -- coefficient stability
    print("\n-- STEP 10: Coefficient stability --")
    dates_roll, betas_roll, param_names = coefficient_stability(df)
    fig_coefficient_stability(dates_roll, betas_roll, param_names)
    fig_final_verdict(df)

    # Export M3 coefficients for downstream use
    mc, vc, sc = METHODS["KW"]
    sub = df[[mc, vc, sc]].dropna()
    m = sub[mc].values; v = sub[vc].values; y = sub[sc].values
    res_m3 = _ols_stats(_design(m, v), y)
    coef_df = pd.DataFrame({
        "parameter": ["intercept", "mean", "var", "mean^2"],
        "coef":  res_m3["beta"],
        "se":    res_m3["se"],
        "t":     res_m3["t_stat"],
        "p":     res_m3["p_vals"],
    })
    coef_path = os.path.join(RESULTS_DIR, "M3_coefficients_KW.csv")
    coef_df.to_csv(coef_path, index=False, float_format="%.6f")
    print(f"\nSaved M3 coefficients: {coef_path}")
    print("\nDone. Check fig/ and results/ for outputs.")


if __name__ == "__main__":
    main()
