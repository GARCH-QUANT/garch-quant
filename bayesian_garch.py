#!/usr/bin/env python3.12
"""
Bayesian GARCH(1,1) with Student-t Errors
Gibbs + Metropolis-Hastings Sampling for Posterior Inference
Produces: posterior distributions, vol forecasts, VaR probability predictions
"""

import warnings, os, sys, io, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist
from scipy.special import gammaln
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ── Config ─────────────────────────────────────────────────
OUTPUT_DIR = "/home/agentuser/.hermes/papers/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLR_BG   = "#0D1117"
CLR_TEXT = "#E6EDF3"
CLR_MUTED= "#8B949E"
CLR_CHAIN1= "#58A6FF"
CLR_CHAIN2= "#3FB950"
CLR_CHAIN3= "#F78166"
CLR_PRIOR = "#6E7681"
CLR_CRED  = "#388BFD26"

plt.rcParams.update({
    "axes.facecolor"   : CLR_BG,
    "figure.facecolor" : CLR_BG,
    "axes.edgecolor"   : "#30363D",
    "axes.labelcolor"  : CLR_TEXT,
    "text.color"       : CLR_TEXT,
    "xtick.color"      : CLR_MUTED,
    "ytick.color"      : CLR_MUTED,
    "grid.color"       : "#21262D",
    "grid.alpha"       : 0.5,
    "font.family"       : "sans-serif",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

import os
TWELVE_DATA_KEY = os.environ.get("TWELVE_DATA_KEY", "")
TUSHARE_TOKEN   = os.environ.get("TUSHARE_TOKEN", "")

# ── Data Fetch ─────────────────────────────────────────────
def get_tushare_data(ts_code, start, end):
    try:
        import tushare as ts
        tushare_token = os.environ.get("TUSHARE_TOKEN", "")
        ts.set_token(tushare_token)
        pro = ts.pro_api(tushare_token)
        df = pro.index_daily(ts_code=ts_code, start_date=start.replace("-", ""),
                             end_date=end.replace("-", ""))
        if df is not None and len(df) > 0:
            df = df.rename(columns={"trade_date":"date","close":"close"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")["close"]
            return df
    except:
        pass
    return None

def get_twelve_data(symbol, start, end):
    import requests
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {"symbol": symbol, "interval": "1day", "outputsize": 500,
                  "format": "JSON", "apikey": TWELVE_DATA_KEY}
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if "values" not in data:
            return None
        vals = data["values"]
        records = [(v["datetime"], float(v["close"])) for v in reversed(vals)]
        idx = pd.to_datetime([r[0] for r in records])
        return pd.Series([r[1] for r in records], index=idx, name="close")
    except:
        return None

def fetch_data(ticker, name):
    end   = (datetime.today()).strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
    pass  # quiet
    if name == "CSI 300":
        s = get_tushare_data("000300.SH", start, end)
        if s is not None and len(s) > 100:
            return s.dropna()
    s = get_twelve_data(ticker, start, end)
    if s is not None and len(s) > 100:
        return s.dropna()
    return None

# ── Student-t Log-PDF (GARCH context) ─────────────────────
def student_t_logpdf(x, nu):
    """Log pdf of standard Student-t with nu degrees of freedom"""
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2)
            - 0.5 * np.log(np.pi * nu)
            - ((nu + 1) / 2) * np.log1p(x**2 / nu))

def garch_log_likelihood(params, r, h, nu):
    """GARCH(1,1) log-likelihood with Student-t errors"""
    omega, alpha, beta = params
    T = len(r)
    ll = 0.0
    for t in range(1, T):
        ll += student_t_logpdf(r[t] / np.sqrt(h[t]), nu)
        ll -= 0.5 * np.log(h[t])
    return ll

# ── Prior Specification ─────────────────────────────────────
# Prior: omega ~ Gamma(a_omega, b_omega), alpha, beta ~ Beta(a, b) truncated
# Standard uninformative: a_omega=1e-4, b_omega=1e-4, a=0.05, b=0.95

PRIOR = {
    "omega_a": 1e-4, "omega_b": 1e-4,
    "alpha_a": 2.0,  "alpha_b": 5.0,
    "beta_a":  2.0,  "beta_b":  5.0,
    "nu_min":  3.0,  "nu_max":  30.0,
}

def log_prior(params, nu):
    omega, alpha, beta = params
    # Truncated support: omega>0, alpha>=0, beta>=0, alpha+beta<1
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return -np.inf

    lp = 0.0
    # Gamma prior on omega
    a_o, b_o = PRIOR["omega_a"], PRIOR["omega_b"]
    lp += (a_o - 1) * np.log(omega) - b_o * omega
    # Beta prior on alpha
    a_a, b_a = PRIOR["alpha_a"], PRIOR["alpha_b"]
    lp += (a_a - 1) * np.log(alpha + 1e-10) + (b_a - 1) * np.log(1 - alpha + 1e-10)
    # Beta prior on beta
    a_b, b_b = PRIOR["beta_a"], PRIOR["beta_b"]
    lp += (a_b - 1) * np.log(beta + 1e-10) + (b_b - 1) * np.log(1 - beta + 1e-10)
    # Uniform prior on nu
    if nu < PRIOR["nu_min"] or nu > PRIOR["nu_max"]:
        return -np.inf
    return lp

# ── MH Proposal ────────────────────────────────────────────
def mh_step(current, proposal_sd):
    """Gaussian random walk Metropolis-Hastings"""
    proposed = current + np.random.randn() * proposal_sd
    return proposed

# ── Gibbs Sampler ──────────────────────────────────────────
def sample_garch_posterior(r, n_burn=3000, n_sample=3000, n_chains=3):
    """
    Corrected Gibbs + MH for Bayesian GARCH(1,1) with Student-t errors.
    Uses data augmentation (latent h_t) for closed-form Gibbs updates,
    and proper MH with Jacobian for log-omega/alpha/beta proposals.
    """
    T = len(r)
    r = np.asarray(r, dtype=float)

    results = []

    for chain in range(n_chains):
        # ── Storage ────────────────────────────────────────

        # ── Initialize parameters ──────────────────────────
        omega = np.random.exponential(0.02)
        alpha = np.random.beta(2, 5)
        beta  = np.random.beta(2, 5)
        nu    = np.random.uniform(4, 15)

        # ── Initialize conditional variances ────────────────
        h = np.zeros(T)
        h[0] = np.var(r) / 252
        for t in range(1, T):
            h[t] = omega + alpha * r[t-1]**2 + beta * h[t-1]
        h = np.clip(h, 1e-8, None)

        # ── Storage ────────────────────────────────────────
        samples_omega = []
        samples_alpha = []
        samples_beta  = []
        samples_nu    = []
        samples_h     = []

        # Proposal sds (tuned)
        prop_sd = {"omega": 0.02, "alpha": 0.05, "beta": 0.05, "nu": 0.5}

        # Acceptance counters
        acc_omega = 0; acc_alpha = 0; acc_beta = 0; acc_nu = 0

        for i in range(n_burn + n_sample):
            # ── 1. Update latent variances h (Gibbs) ────────
            # Full-conditional for h_t is scaled-inverse-chi-square
            # We use the t-distribution mixture representation:
            # r_t | h_t, nu ~ t_nu(0, h_t)  →  mixture of normals
            for t in range(1, T):
                # Draw from conditional: IG(scale=sigma_t^2, df=nu)
                sigma2_t = (omega + alpha * r[t-1]**2 + beta * h[t-1])
                sigma2_t = max(sigma2_t, 1e-10)
                # Inverse-gamma: h_t ~ IG(nu/2, nu*sigma2_t/2)
                shape = nu / 2.0
                scale = nu * sigma2_t / 2.0
                # Sample from IG via gamma: IG(shape, scale) = Gamma(shape, 1/scale)^-1
                g = np.random.gamma(shape, 1.0 / scale)
                h[t] = 1.0 / g
            h = np.clip(h, 1e-8, None)

            # ── 2. Update omega (MH in log-space) ─────────────
            log_omega = np.log(omega)
            log_omega_prop = log_omega + np.random.randn() * prop_sd["omega"]
            omega_prop = np.exp(log_omega_prop)
            if omega_prop > 1e-8:
                # Full conditional of omega ~ Gamma(a, b) where
                # a = 1e-4 + T/2, b = 1e-4 + sum(1/h_t)
                a_o = PRIOR["omega_a"] + T / 2.0
                b_o = PRIOR["omega_b"] + np.sum(1.0 / h[1:])
                lp_curr = (PRIOR["omega_a"] - 1) * np.log(omega) - PRIOR["omega_b"] * omega
                lp_prop = (PRIOR["omega_a"] - 1) * np.log(omega_prop) - PRIOR["omega_b"] * omega_prop
                # Add log- Jacobian for log-space proposal
                lp_prop += log_omega_prop
                lp_curr += log_omega
                # Likelihood ratio
                ll_diff = 0.0
                for t in range(1, T):
                    ll_diff += (student_t_logpdf(r[t] / np.sqrt(h[t]), nu)
                              - student_t_logpdf(r[t] / np.sqrt(h[t]), nu))  # same h, skip
                # Metropolis ratio
                acc = (lp_prop - lp_curr)
                if np.log(np.random.rand()) < acc:
                    omega = omega_prop
                    acc_omega += 1

            # ── 3. Update alpha (MH with logit transform) ─────
            logit_alpha = np.log(alpha / (1 - alpha))
            logit_prop = logit_alpha + np.random.randn() * prop_sd["alpha"]
            alpha_prop = 1.0 / (1.0 + np.exp(-logit_prop))
            alpha_prop = np.clip(alpha_prop, 1e-4, 1 - 1e-4)
            if alpha_prop + beta < 0.999:
                # Full conditional via prior + likelihood
                lp_curr = ((PRIOR["alpha_a"] - 1) * np.log(alpha + 1e-10)
                         + (PRIOR["alpha_b"] - 1) * np.log(1 - alpha + 1e-10))
                lp_prop = ((PRIOR["alpha_a"] - 1) * np.log(alpha_prop + 1e-10)
                         + (PRIOR["alpha_b"] - 1) * np.log(1 - alpha_prop + 1e-10))
                # Logit Jacobian: dAlpha/dLogit = alpha*(1-alpha)
                lp_prop += np.log(alpha_prop * (1 - alpha_prop))
                lp_curr += np.log(alpha * (1 - alpha))
                # Recompute h with new alpha
                h_new = omega + alpha_prop * r[:-1]**2 + beta * h[:-1]
                h_new = np.concatenate([[h_new[0]], h_new])  # pad
                h_new = np.clip(h_new, 1e-8, None)
                # Likelihood
                ll_curr = sum(student_t_logpdf(r[t] / np.sqrt(h[t]), nu)
                              - 0.5 * np.log(h[t]) for t in range(1, T))
                ll_prop = sum(student_t_logpdf(r[t] / np.sqrt(h_new[t]), nu)
                              - 0.5 * np.log(h_new[t]) for t in range(1, T))
                acc = (lp_prop + ll_prop) - (lp_curr + ll_curr)
                if np.log(np.random.rand()) < acc:
                    alpha = alpha_prop
                    h = h_new
                    acc_alpha += 1

            # ── 4. Update beta (MH with logit transform) ─────
            logit_beta = np.log(beta / (1 - beta))
            logit_prop = logit_beta + np.random.randn() * prop_sd["beta"]
            beta_prop = 1.0 / (1.0 + np.exp(-logit_prop))
            beta_prop = np.clip(beta_prop, 1e-4, 1 - 1e-4)
            if alpha + beta_prop < 0.999:
                lp_curr = ((PRIOR["beta_a"] - 1) * np.log(beta + 1e-10)
                         + (PRIOR["beta_b"] - 1) * np.log(1 - beta + 1e-10))
                lp_prop = ((PRIOR["beta_a"] - 1) * np.log(beta_prop + 1e-10)
                         + (PRIOR["beta_b"] - 1) * np.log(1 - beta_prop + 1e-10))
                lp_prop += np.log(beta_prop * (1 - beta_prop))
                lp_curr += np.log(beta * (1 - beta))
                # Recompute h with new beta
                h_new = omega + alpha * r[:-1]**2 + beta_prop * h[:-1]
                h_new = np.concatenate([[h_new[0]], h_new])
                h_new = np.clip(h_new, 1e-8, None)
                ll_curr = sum(student_t_logpdf(r[t] / np.sqrt(h[t]), nu)
                              - 0.5 * np.log(h[t]) for t in range(1, T))
                ll_prop = sum(student_t_logpdf(r[t] / np.sqrt(h_new[t]), nu)
                              - 0.5 * np.log(h_new[t]) for t in range(1, T))
                acc = (lp_prop + ll_prop) - (lp_curr + ll_curr)
                if np.log(np.random.rand()) < acc:
                    beta = beta_prop
                    h = h_new
                    acc_beta += 1

            # ── 5. Update nu (MH) ─────────────────────────────
            nu_prop = nu + np.random.randn() * prop_sd["nu"]
            nu_prop = np.clip(nu_prop, PRIOR["nu_min"], PRIOR["nu_max"])
            lp_curr = log_prior((omega, alpha, beta), nu)
            lp_prop = log_prior((omega, alpha, beta), nu_prop)
            ll_curr = sum(student_t_logpdf(r[t] / np.sqrt(h[t]), nu)
                          - 0.5 * np.log(h[t]) for t in range(1, T))
            ll_prop = sum(student_t_logpdf(r[t] / np.sqrt(h[t]), nu_prop)
                          - 0.5 * np.log(h[t]) for t in range(1, T))
            acc = (lp_prop + ll_prop) - (lp_curr + ll_curr)
            if np.log(np.random.rand()) < acc:
                nu = nu_prop
                acc_nu += 1

            # ── Adapt proposal sds ──────────────────────────
            if i > 0 and i % 500 == 0:
                n_iters = i - (n_burn if i > n_burn else 0)
                if n_iters > 50:
                    for k, acc_cnt in [("omega", acc_omega), ("alpha", acc_alpha),
                                        ("beta", acc_beta), ("nu", acc_nu)]:
                        rate = acc_cnt / n_iters
                        if rate < 0.15:
                            prop_sd[k] *= 0.8
                        elif rate > 0.45:
                            prop_sd[k] *= 1.2
                    if i <= n_burn:
                        acc_omega = acc_alpha = acc_beta = acc_nu = 0

            # Store after burn-in
            if i >= n_burn:
                samples_omega.append(omega)
                samples_alpha.append(alpha)
                samples_beta.append(beta)
                samples_nu.append(nu)
                samples_h.append(h[-1])

        samples = {
            "omega": np.array(samples_omega),
            "alpha": np.array(samples_alpha),
            "beta":  np.array(samples_beta),
            "nu":    np.array(samples_nu),
            "h_last": np.array(samples_h),
        }
        results.append(samples)

    return results, r

# ── Forecasting ─────────────────────────────────────────────
def forecast_garch(samples, r_last, h_last, n_ahead=5, n_draws=1000):
    """
    Simulate future volatility paths from posterior samples.
    Returns: mean path, credible intervals, VaR probabilities.
    """
    omega  = samples["omega"]
    alpha  = samples["alpha"]
    beta   = samples["beta"]
    nu     = samples["nu"]

    n_samples = len(omega)
    idx = np.random.choice(n_samples, size=n_draws, replace=True)

    paths = np.zeros((n_draws, n_ahead))
    h_cur = np.array([h_last] * n_draws)
    r_cur = np.array([r_last] * n_draws)

    for t in range(n_ahead):
        # Sample from t-distribution using posterior mean params
        omega_s = omega[idx]
        alpha_s = alpha[idx]
        beta_s  = beta[idx]
        nu_s    = nu[idx]

        # Simulate r_{t+1} | h_t
        r_draws = r_cur * 0  # placeholder
        for i in range(n_draws):
            # Student-t: sqrt(h) * t / sqrt(nu/(nu-2))
            scale = np.sqrt(h_cur[i]) * np.sqrt(nu_s[i] / (nu_s[i] - 2))
            r_draws[i] = np.random.standard_t(nu_s[i]) * scale

        # Update variance
        h_next = omega_s + alpha_s * r_cur**2 + beta_s * h_cur
        paths[:, t] = np.sqrt(h_next) * np.sqrt(252)  # annualized vol
        h_cur = h_next
        r_cur = r_draws

    # Credible intervals
    mean_path = np.mean(paths, axis=0)
    q05 = np.percentile(paths, 5, axis=0)
    q95 = np.percentile(paths, 95, axis=0)
    q25 = np.percentile(paths, 25, axis=0)
    q75 = np.percentile(paths, 75, axis=0)

    return mean_path, q05, q95, q25, q75, paths

def compute_var_prob(samples, r_last, h_last, confidence=0.95, n_ahead=1, n_draws=5000):
    """Probability that VaR is breached n_ahead days ahead."""
    omega = samples["omega"]
    alpha = samples["alpha"]
    beta  = samples["beta"]
    nu    = samples["nu"]

    idx = np.random.choice(len(omega), size=n_draws, replace=True)

    h_cur = np.full(n_draws, h_last)
    r_cur = np.full(n_draws, r_last)

    breaches = np.zeros((n_draws, n_ahead))
    for t in range(n_ahead):
        omega_s = omega[idx]
        alpha_s = alpha[idx]
        beta_s  = beta[idx]
        nu_s    = nu[idx]

        r_draws = np.zeros(n_draws)
        for i in range(n_draws):
            scale = np.sqrt(h_cur[i]) * np.sqrt(nu_s[i] / (nu_s[i] - 2))
            r_draws[i] = np.random.standard_t(nu_s[i]) * scale

        # VaR: quantile of simulated returns
        var_threshold = np.percentile(r_draws, (1 - confidence) * 100)
        breaches[:, t] = r_draws < var_threshold

        h_next = omega_s + alpha_s * r_cur**2 + beta_s * h_cur
        h_cur = h_next
        r_cur = r_draws

    breach_prob = np.mean(np.any(breaches, axis=1))
    return breach_prob

# ── Plotting ────────────────────────────────────────────────
def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=CLR_BG, edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_posteriors(results, asset_name):
    """Plot posterior distributions of GARCH parameters"""
    n_params = 4
    fig, axes = plt.subplots(1, n_params, figsize=(16, 3.5))
    param_names = ["omega", "alpha", "beta", "nu"]
    true_vals = [None, None, None, None]  # No frequentist comparsion here

    colors = [CLR_CHAIN1, CLR_CHAIN2, CLR_CHAIN3]

    for j, (pn, ax) in enumerate(zip(param_names, axes)):
        for i, samples in enumerate(results):
            vals = samples[pn]
            ax.hist(vals, bins=60, density=True, alpha=0.4, color=colors[i],
                    label=f"Chain {i+1}", edgecolor='none')

        # Overlay posterior mean line
        all_vals = np.concatenate([s[pn] for s in results])
        mean_val = np.mean(all_vals)
        std_val  = np.std(all_vals)
        cred_lo  = np.percentile(all_vals, 5)
        cred_hi  = np.percentile(all_vals, 95)
        ax.axvline(mean_val, color="white", lw=1.5, linestyle="-", label=f"Mean={mean_val:.4g}")
        ax.axvline(cred_lo,  color="white", lw=1,   linestyle="--", alpha=0.6)
        ax.axvline(cred_hi,  color="white", lw=1,   linestyle="--", alpha=0.6)

        ax.set_title(pn.upper(), fontsize=12, fontweight='bold')
        ax.set_xlabel(param_names[j] if j > 0 else "", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor(CLR_BG)

        # Stats annotation
        stats_text = f"Mean={mean_val:.4g}\nStd={std_val:.4g}\n95% CI: [{cred_lo:.4g}, {cred_hi:.4g}]"
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                fontsize=8, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor=CLR_BG, alpha=0.7, edgecolor='#30363D'))

    axes[0].legend(fontsize=8, loc='upper right')
    fig.suptitle(f"{asset_name} — Bayesian GARCH(1,1) Posterior Distributions",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig_to_buf(fig)

def plot_vol_forecast(results, dates, r, asset_name, n_ahead=5):
    """Plot historical vol + posterior forecast with credible bands"""
    # Compute historical conditional vol (using posterior mean params)
    omega  = np.mean(np.concatenate([s["omega"] for s in results]))
    alpha  = np.mean(np.concatenate([s["alpha"] for s in results]))
    beta   = np.mean(np.concatenate([s["beta"]  for s in results]))
    nu     = np.mean(np.concatenate([s["nu"]    for s in results]))

    T = len(r)
    h_hist = np.zeros(T)
    h_hist[0] = np.var(r) / 252
    for t in range(1, T):
        h_hist[t] = omega + alpha * r[t-1]**2 + beta * h_hist[t-1]
    vol_hist = np.sqrt(h_hist) * np.sqrt(252)

    # Forecast
    mean_f, q05_f, q95_f, q25_f, q75_f, paths = forecast_garch(
        {k: np.concatenate([s[k] for s in results]) for k in ["omega","alpha","beta","nu"]},
        r[-1], h_hist[-1], n_ahead=n_ahead
    )

    # Dates for forecast
    last_date = dates[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_ahead)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Historical vol
    ax.plot(dates[-200:], vol_hist[-200:], color=CLR_CHAIN1, lw=1.2, label="Historical Vol (Posterior Mean)")

    # Forecast with credible band
    ax.plot(forecast_dates, mean_f, color=CLR_CHAIN2, lw=2, marker='o', markersize=5,
            label="Forecast Mean")
    ax.fill_between(forecast_dates, q05_f, q95_f, color=CLR_CRED,
                    label="90% Credible Interval", alpha=0.8)
    ax.fill_between(forecast_dates, q25_f, q75_f, color=CLR_CHAIN2,
                    label="50% Credible Interval", alpha=0.4)

    ax.axhline(vol_hist[-1], color=CLR_MUTED, lw=1, linestyle=":", alpha=0.7)
    ax.set_ylabel("Annualized Volatility (%)", fontsize=11)
    ax.set_title(f"{asset_name} — Volatility Forecast (Bayesian GARCH, {n_ahead}-Day Ahead)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(CLR_BG)

    # Annotate forecast values
    for i, (fd, m, lo, hi) in enumerate(zip(forecast_dates, mean_f, q05_f, q95_f)):
        ax.annotate(f"{m:.1f}%\n[{lo:.1f},{hi:.1f}]", (fd, m),
                    textcoords="offset points", xytext=(0, -40),
                    fontsize=8, ha='center',
                    bbox=dict(boxstyle='round', facecolor=CLR_BG, alpha=0.7, edgecolor='#30363D'))

    plt.tight_layout()
    return fig_to_buf(fig)

def plot_var_table(results, r_last, h_last, asset_name):
    """Compute and display VaR breach probabilities"""
    confidence_levels = [0.90, 0.95, 0.99]
    horizons = [1, 5]

    data = []
    for conf in confidence_levels:
        for h in horizons:
            bp = compute_var_prob(
                {k: np.concatenate([s[k] for s in results]) for k in ["omega","alpha","beta","nu"]},
                r_last, h_last, confidence=conf, n_ahead=h, n_draws=3000
            )
            expected_loss = bp * (1 - conf) * 100
            data.append({
                "Confidence": f"{int(conf*100)}%",
                "Horizon": f"{h}d",
                "Breach Prob": f"{bp*100:.2f}%",
                "Expected Loss": f"{expected_loss:.3f}%",
            })

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    ax.set_facecolor(CLR_BG)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1E3A5F")
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor("#1A1A2E")
            cell.set_text_props(color=CLR_TEXT)
        else:
            cell.set_facecolor("#16213E")
            cell.set_text_props(color=CLR_TEXT)
        cell.set_edgecolor("#30363D")

    fig.suptitle(f"{asset_name} — VaR Breach Probability (Bayesian GARCH)",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig_to_buf(fig), df

def plot_trace(results, asset_name):
    """Trace plots to assess MCMC convergence"""
    n_params = 4
    fig, axes = plt.subplots(4, 2, figsize=(14, 10))
    param_names = ["omega", "alpha", "beta", "nu"]
    colors = [CLR_CHAIN1, CLR_CHAIN2, CLR_CHAIN3]

    for j, pn in enumerate(param_names):
        # Trace plot
        ax = axes[j, 0]
        for i, samples in enumerate(results):
            ax.plot(samples[pn][::5], alpha=0.7, color=colors[i], lw=0.5)
        ax.set_ylabel(pn.upper(), fontsize=10)
        ax.set_facecolor(CLR_BG)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_title("Trace Plot (every 5th sample)", fontsize=11)

        # Autocorrelation
        ax = axes[j, 1]
        for i, samples in enumerate(results):
            vals = samples[pn][::5]
            T = len(vals)
            max_lag = min(100, T // 2 - 1)
            acf = np.array([np.corrcoef(vals[:-lag], vals[lag:])[0,1]
                           if lag < T and len(vals[:-lag]) > 1 and len(vals[lag:]) > 1 else 0
                           for lag in range(max_lag)])
            ax.plot(acf, alpha=0.7, color=colors[i], lw=1)
        ax.set_ylabel(pn.upper(), fontsize=10)
        ax.set_facecolor(CLR_BG)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_title("Autocorrelation", fontsize=11)

    fig.suptitle(f"{asset_name} — MCMC Diagnostics", fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig_to_buf(fig)

# ── Main ───────────────────────────────────────────────────
def main():
    # Assets to analyze
    ASSETS = [
        ("CSI 300", "HSI:IND"),
        ("AAPL",    "AAPL"),
        ("TSLA",    "TSLA"),
    ]

    all_results = {}

    for name, symbol in ASSETS:
        series = fetch_data(symbol, name)
        if series is None:
            continue

        rets = series.pct_change().dropna()
        rets = rets[abs(rets) < 0.25]
        r = rets.values
        dates = rets.index

        results, r_arr = sample_garch_posterior(r)

        combined_results = {k: np.concatenate([s[k] for s in results]) for k in ["omega","alpha","beta","nu"]}

        post_buf = plot_posteriors(results, name)
        post_path = f"{OUTPUT_DIR}/bayesian_posterior_{name.replace(' ','_')}.png"
        with open(post_path, 'wb') as f:
            f.write(post_buf.getvalue())

        trace_buf = plot_trace(results, name)
        trace_path = f"{OUTPUT_DIR}/bayesian_trace_{name.replace(' ','_')}.png"
        with open(trace_path, 'wb') as f:
            f.write(trace_buf.getvalue())

        vol_buf = plot_vol_forecast(results, dates, r_arr, name, n_ahead=5)
        vol_path = f"{OUTPUT_DIR}/bayesian_vol_forecast_{name.replace(' ','_')}.png"
        with open(vol_path, 'wb') as f:
            f.write(vol_buf.getvalue())

        omega_mean = np.mean(combined_results["omega"])
        alpha_mean = np.mean(combined_results["alpha"])
        beta_mean = np.mean(combined_results["beta"])
        h_init = np.var(r_arr) / 252
        h_last = h_init
        for _ in range(10):
            h_last = omega_mean + alpha_mean * r_arr[-1]**2 + beta_mean * h_last

        var_buf, var_df = plot_var_table(results, r_arr[-1], h_last, name)
        var_path = f"{OUTPUT_DIR}/bayesian_var_{name.replace(' ','_')}.png"
        with open(var_path, 'wb') as f:
            f.write(var_buf.getvalue())

        all_results[name] = {
            "results": results,
            "r": r_arr,
            "dates": dates,
            "var_buf": var_buf,
            "var_df": var_df,
            "h_last": h_last,
        }

    for name, data in all_results.items():
        results = data["results"]
        r_arr = data["r"]

        all_omega = np.concatenate([s["omega"] for s in results])
        all_alpha = np.concatenate([s["alpha"] for s in results])
        all_beta  = np.concatenate([s["beta"]  for s in results])
        all_nu    = np.concatenate([s["nu"]    for s in results])

        omega_m, alpha_m, beta_m, nu_m = (np.mean(all_omega), np.mean(all_alpha),
                                           np.mean(all_beta), np.mean(all_nu))

        lr_vol = np.sqrt(omega_m / (1 - alpha_m - beta_m)) * np.sqrt(252)

        fp_mean, fp_lo, fp_hi, *_ = forecast_garch(
            {k: np.concatenate([s[k] for s in results]) for k in ["omega","alpha","beta","nu"]},
            r_arr[-1], data["h_last"], n_ahead=5
        )[:3]

        var_95_1d = compute_var_prob(
            {k: np.concatenate([s[k] for s in results]) for k in ["omega","alpha","beta","nu"]},
            r_arr[-1], data["h_last"], confidence=0.95, n_ahead=1, n_draws=3000
        )

    return all_results

if __name__ == "__main__":
    main()
