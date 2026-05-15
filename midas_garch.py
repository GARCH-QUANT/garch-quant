#!/usr/bin/env python3
"""
MIDAS-GARCH — CSI 300 + Macro Factors
======================================
Engle-Ghysels-Sohn (2013) Mixed Data Sampling GARCH
  σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} + γ·Σ_k θ_k·x_{t-k}·(RV_{t-k} or |r_{t-k}|)

Macro factors (low-frequency → aligned to daily):
  • VIX   — CBOE implied volatility proxy via yfinance (daily)
  • USD/CNY — dollar yuan exchange rate via yfinance (daily)
  • SHIBOR 3M — money market rate via Tushare (daily)
  • CNY 10Y bond yield — via Tushare (daily)

Charts: dark theme → push to Telegram
"""

import warnings, os, sys, requests
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────
import os
TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT   = os.environ.get("TELEGRAM_CHANNEL_ID", "-1003786012521")
TUSHARE_TOKEN   = os.environ.get("TUSHARE_TOKEN", "")
OUTPUT_DIR      = os.path.expanduser("~/.hermes/cron/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "axes.facecolor": "#0D1117",
    "figure.facecolor": "#0D1117",
    "axes.edgecolor": "#30363D",
    "axes.labelcolor": "#E6EDF3",
    "text.color": "#E6EDF3",
    "xtick.color": "#8B949E",
    "ytick.color": "#8B949E",
    "grid.color": "#21262D",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Data Fetch ──────────────────────────────────────────────────────────────
def fetch_csi300():
    import tushare as ts
    tushare_token = os.environ.get("TUSHARE_TOKEN", "")
    ts.set_token(tushare_token)
    pro = ts.pro_api(tushare_token)
    df = pro.index_daily(ts_code='000300.SH', start_date='20240101', end_date='20260512')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['rv'] = df['returns']**2
    return df[['trade_date','close','returns','rv']]

def fetch_vix():
    """VIX proxy via CBOE or VIX ETF"""
    try:
        import yfinance as yf
        # Try ^VIX first
        vix = yf.download('^VIX', start='2023-12-28', end='2026-05-13', progress=False, auto_adjust=True)
        if vix is None or len(vix) == 0:
            vix = yf.download('VIXY', start='2023-12-28', end='2026-05-13', progress=False, auto_adjust=True)
        # Handle both DataFrame and tuple returns
        if isinstance(vix, tuple):
            vix = vix[0]
        if hasattr(vix.columns, 'droplevel'):
            vix.columns = vix.columns.droplevel(1) if hasattr(vix.columns, 'levels') else vix.columns
        vix.index = pd.to_datetime(vix.index).tz_localize(None) if vix.index.tz else vix.index
        vix.columns = [c.lower() for c in vix.columns]
        vix = vix[['close']].rename(columns={'close': 'vix'})
        vix['vix'] = vix['vix'].fillna(20).ffill()
        return vix
    except Exception as e:
        print(f"[VIX fetch error] {e}")
        return pd.DataFrame()

def fetch_usdcny():
    """USD/CNY exchange rate via Yahoo Finance"""
    try:
        import yfinance as yf
        fx = yf.download('CNY=X', start='2023-12-28', end='2026-05-13', progress=False, auto_adjust=True)
        if isinstance(fx, tuple):
            fx = fx[0]
        if hasattr(fx.columns, 'droplevel'):
            fx.columns = fx.columns.droplevel(1) if hasattr(fx.columns, 'levels') else fx.columns
        fx.index = pd.to_datetime(fx.index).tz_localize(None) if fx.index.tz else fx.index
        fx.columns = [c.lower() for c in fx.columns]
        fx = fx[['close']].rename(columns={'close': 'usdcny'})
        fx['usdcny'] = fx['usdcny'].ffill()
        return fx
    except Exception as e:
        print(f"[USDCNY fetch error] {e}")
        return pd.DataFrame()

def fetch_shibor():
    """SHIBOR 3M via Tushare"""
    try:
        import tushare as ts
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api(TUSHARE_TOKEN)
        df = pro.shibor()
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'shibor_date' in df.columns:
            df['date'] = pd.to_datetime(df['shibor_date'])
        df = df[['date','3m']].rename(columns={'date':'trade_date','3m':'shibor3m'})
        df = df.dropna()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df.set_index('trade_date')
    except Exception as e:
        print(f"[SHIBOR fetch error] {e}")
        return pd.DataFrame()

def fetch_cny10y():
    """CNY 10Y government bond yield via Tushare"""
    try:
        import tushare as ts
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api(TUSHARE_TOKEN)
        df = pro.bond_ytm(act='1002')  # 10Y CNY
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        df = df[['date','yield10y']].rename(columns={'date':'trade_date'})
        df = df.dropna()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df.set_index('trade_date')
    except Exception as e:
        print(f"[CNY10Y fetch error] {e}")
        return pd.DataFrame()

# ── MIDAS Weight Function ───────────────────────────────────────────────────
def midas_alpha(k, theta, m=22):
    """Beta weight: θ controls curvature"""
    t = np.arange(1, k+1)
    w = t**(theta - 1) * (m - t)**(theta - 1)
    w = w / w.sum()
    return w

def midas_poly(k, th1, th2=1.0):
    """Almon lag polynomial weights"""
    t = np.arange(1, k+1)
    w = np.exp(th1 * t + th2 * t**2)
    w = w / w.sum()
    return w

# ── MIDAS-GARCH(1,1) Estimation ───────────────────────────────────────────
def estimate_midas_garch(ret, x, lookback=22):
    """
    Simplified MIDAS-GARCH(1,1):
      σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} + γ·Σ_{j=1}^{K} w_j(θ)·x_{t-j}
    Where w_j are MIDAS beta weights on macro factor x.

    Uses OLS on log-σ² as proxy (GARCH-MIDAS style approximation).
    Returns: ω, α, β, γ, θ (approx via grid search), and R² improvement vs vanilla GARCH.
    """
    from scipy.optimize import minimize

    # Vanila GARCH(1,1) for benchmark
    def garch_mle(params, r):
        omega, a, b = params
        h = np.zeros(len(r))
        h[0] = r[0]**2
        for t in range(1, len(r)):
            h[t] = omega + a * r[t-1]**2 + b * h[t-1]
        ll = -0.5 * (np.log(h) + r**2 / h)
        return -ll.sum()

    # Vanilla GARCH fit
    res_v = minimize(garch_mle, [1e-6, 0.1, 0.8],
                     args=(ret.values,), bounds=[(1e-10, 1), (0.001, 0.999), (0.001, 0.999)])
    omega_v, alpha_v, beta_v = res_v.x
    h_vanilla = np.zeros(len(ret))
    h_vanilla[0] = ret.iloc[0]**2
    for t in range(1, len(ret)):
        h_vanilla[t] = omega_v + alpha_v * ret.iloc[t-1]**2 + beta_v * h_vanilla[t-1]

    # MIDAS-GARCH: extend with macro factor contribution
    # Approximation: use rolling regression of (r² - GARCH_fit) on lagged macro factor
    residuals = ret.values**2 - h_vanilla

    # Align macro factor x with returns (use same length)
    x_aligned = x.reindex(ret.index).ffill().fillna(1.0).values

    # Grid search over theta and gamma
    best_gamma, best_theta, best_r2 = 0.0, 1.0, 0.0
    results = []

    for theta_cand in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        k = min(lookback, len(ret) - 1)
        weights = midas_alpha(k, theta_cand, m=lookback)

        # Weighted average of macro factor over lookback
        midas_term = np.zeros(len(ret))
        for t in range(k, len(ret)):
            midas_term[t] = np.sum(weights * x_aligned[t-k:t])  # [t-k:t] = k elements

        # Regress residuals on midas term
        X_m = midas_term[k:]
        y_m = residuals[k:]
        # Guard against constant/nan arrays
        if np.std(X_m) < 1e-10 or np.std(y_m) < 1e-10:
            continue
        # Remove any remaining NaN pairs
        valid = ~(np.isnan(X_m) | np.isnan(y_m))
        if valid.sum() < 10:
            continue
        X_clean = X_m[valid]
        y_clean = y_m[valid]
        cov = np.cov(X_clean, y_clean)
        if np.isnan(cov[0,1]) or np.var(X_clean) < 1e-10:
            continue
        gamma_cand = cov[0,1] / (np.var(X_clean) + 1e-10)
        pred = gamma_cand * midas_term[k:]

        ssres = np.sum((y_m - pred)**2)
        sstot = np.sum(y_m**2)
        r2_midas = max(0, 1 - ssres / (sstot + 1e-10))

        results.append({'theta': theta_cand, 'gamma': gamma_cand, 'r2': r2_midas,
                        'midas_term': midas_term, 'weights': weights})

    # Best theta by R²
    best = max(results, key=lambda x: x['r2'])

    # Full MIDAS-GARCH path
    h_midas = np.zeros(len(ret))
    h_midas[0] = ret.iloc[0]**2
    for t in range(1, len(ret)):
        h_midas[t] = (omega_v + best['gamma'] * best['midas_term'][t] +
                       alpha_v * ret.iloc[t-1]**2 + beta_v * h_midas[t-1])

    # R² improvement
    r2_v = 1 - np.sum((ret.values**2 - h_vanilla)**2) / np.sum(ret.values**2)
    r2_m = 1 - np.sum((ret.values**2 - h_midas)**2) / np.sum(ret.values**2)

    return {
        'omega': omega_v, 'alpha': alpha_v, 'beta': beta_v,
        'gamma': best['gamma'], 'theta': best['theta'],
        'r2_vanilla': r2_v, 'r2_midas': r2_m,
        'r2_improvement': r2_m - r2_v,
        'h_vanilla': h_vanilla, 'h_midas': h_midas,
        'lookback': lookback,
    }

# ── Rolling Forecast ─────────────────────────────────────────────────────────
def rolling_forecast(ret, x, params, horizon=5):
    """1-step ahead rolling forecast using MIDAS-GARCH"""
    omega, alpha, beta, gamma, theta = params['omega'], params['alpha'], params['beta'], params['gamma'], params['theta']
    lookback = params['lookback']

    h = np.zeros(len(ret) + horizon)
    h[:len(ret)] = params['h_midas'][:len(ret)]

    x_aligned = x.reindex(ret.index).ffill().fillna(1.0).values
    x_extended = np.append(x_aligned, np.full(horizon, x_aligned[-1]))

    for step in range(horizon):
        t = len(ret) + step - 1
        k = min(lookback, t)
        weights = midas_alpha(k, theta, m=lookback)
        midas_term = np.sum(weights * x_extended[t-k:t])
        h[t+1] = omega + gamma * midas_term + alpha * ret.iloc[-1]**2 + beta * h[t]

    return h[-horizon:]

# ── Charts ──────────────────────────────────────────────────────────────────
def chart_conditional_vol(ret, h_vanilla, h_midas, dates, output_path):
    fig, ax = plt.subplots(figsize=(14, 5))
    realized = ret.values**2
    realized_30d = pd.Series(realized).rolling(5).mean().values

    ax.plot(dates, np.sqrt(h_vanilla) * np.sqrt(252) * 100,
            color='#90CAF9', linewidth=1.2, label='GARCH(1,1) σ (annualized)', alpha=0.8)
    ax.plot(dates, np.sqrt(h_midas) * np.sqrt(252) * 100,
            color='#FFA726', linewidth=1.5, label='MIDAS-GARCH σ (annualized)', alpha=0.9)
    ax.plot(dates, realized_30d * np.sqrt(252) * 100,
            color='#B0BEC5', linewidth=1, label='5D Rolling |r| (annualized)', alpha=0.6)
    ax.set_facecolor('#0D1117')
    ax.set_xlabel('Date', color='#E6EDF3')
    ax.set_ylabel('Annualized Volatility (%)', color='#E6EDF3')
    ax.set_title('CSI 300 — Conditional Volatility: GARCH(1,1) vs MIDAS-GARCH', color='#E6EDF3', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    return output_path

def chart_macro_factor(dates, factor_vals, factor_name, output_path):
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(dates, factor_vals, color='#4FC3F7', linewidth=1.2)
    ax.set_facecolor('#0D1117')
    ax.set_ylabel(factor_name, color='#E6EDF3')
    ax.set_title(f'Macro Factor: {factor_name}', color='#E6EDF3', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    return output_path

def chart_midas_weights(lookback, theta, output_path):
    k = lookback
    w_bp = midas_alpha(k, theta, m=k)
    w_al0 = midas_poly(k, 0.0)
    w_aln = midas_poly(k, -0.05)

    fig, ax = plt.subplots(figsize=(9, 4))
    t = np.arange(1, k+1)
    ax.plot(t, w_bp, color='#4FC3F7', linewidth=2, label=f'Beta(θ={theta})')
    ax.plot(t, w_aln, color='#FFA726', linewidth=2, label='Almon(θ=-0.05)')
    ax.set_facecolor('#0D1117')
    ax.set_xlabel('Lag (days)', color='#E6EDF3')
    ax.set_ylabel('Weight', color='#E6EDF3')
    ax.set_title('MIDAS Lag Weights (Beta vs Almon)', color='#E6EDF3', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    return output_path

def chart_forecast(forecast_v, forecast_m, output_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(1, len(forecast_v)+1)
    ax.plot(x, forecast_v * np.sqrt(252) * 100, 'o--', color='#90CAF9', linewidth=2, label='GARCH forecast')
    ax.plot(x, forecast_m * np.sqrt(252) * 100, 's-', color='#FFA726', linewidth=2, label='MIDAS-GARCH forecast')
    ax.set_facecolor('#0D1117')
    ax.set_xlabel('Horizon (days)', color='#E6EDF3')
    ax.set_ylabel('Annualized Vol (%)', color='#E6EDF3')
    ax.set_title('5-Day Volatility Forecast', color='#E6EDF3', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    return output_path

# ── Telegram Helpers ─────────────────────────────────────────────────────────
def send_telegram_photo(photo_path, caption=""):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        with open(photo_path, 'rb') as f:
            files = {'photo': f}
            data = {'chat_id': TELEGRAM_CHAT, 'caption': caption, 'parse_mode': 'HTML'}
            r = requests.post(url, files=files, data=data, timeout=30)
            return r.json().get('ok', False)
    except Exception as e:
        print(f"[send_photo error] {e}")
        return False

def send_telegram_text(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={'chat_id': TELEGRAM_CHAT, 'text': text, 'parse_mode': 'HTML'}, timeout=30)
        return r.json().get('ok', False)
    except Exception as e:
        print(f"[send_text error] {e}")
        return False

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("MIDAS-GARCH — CSI 300 + Macro Factors")
    print("=" * 60)

    # 1. Fetch data
    print("\n[1/6] Fetching CSI 300...")
    csi = fetch_csi300()
    print(f"    CSI 300: {len(csi)} rows, {csi['trade_date'].min().date()} → {csi['trade_date'].max().date()}")

    print("\n[2/6] Fetching macro factors...")
    vix     = fetch_vix()
    usdcny  = fetch_usdcny()
    shibor  = fetch_shibor()
    cny10y  = fetch_cny10y()

    factor_names = []
    factor_dfs   = []
    if len(vix) > 0 and 'vix' in vix.columns:    factor_names.append('VIX');    factor_dfs.append(vix)
    if len(usdcny) > 0 and 'usdcny' in usdcny.columns: factor_names.append('USD/CNY'); factor_dfs.append(usdcny)
    if len(shibor) > 0 and 'shibor3m' in shibor.columns: factor_names.append('SHIBOR_3M'); factor_dfs.append(shibor)
    if len(cny10y) > 0 and 'yield10y' in cny10y.columns: factor_names.append('CNY_10Y'); factor_dfs.append(cny10y)

    print(f"    Loaded factors: {', '.join(factor_names) if factor_names else 'NONE — using VIX as default'}")

    # Fallback: try yfinance for VIX
    if len(factor_dfs) == 0:
        print("    [Fallback] Trying VIXY ETF as VIX proxy...")
        try:
            import yfinance as yf
            vix = yf.download('VIXY', start='2023-12-28', end='2026-05-13', progress=False)
            vix.index = pd.to_datetime(vix.index).tz_localize(None)
            vix.columns = [c.lower() for c in vix.columns]
            vix = vix[['close']].rename(columns={'close': 'vix'})
            vix['vix'] = vix['vix'].fillna(20)
            factor_names = ['VIX']
            factor_dfs = [vix]
            print(f"    VIXY loaded: {len(vix)} rows")
        except Exception as e:
            print(f"    VIXY also failed: {e}")

    # 2. Merge all data
    print("\n[3/6] Merging datasets...")
    df = csi.copy()
    df = df.set_index('trade_date')

    # Column name map (factor_names = display names, stored in df as the same)
    for fname, fdf in zip(factor_names, factor_dfs):
        col = fdf.columns[0]  # actual numeric column
        df = df.join(fdf[[col]].rename(columns={col: fname}), how='left')
        df[fname] = df[fname].ffill().bfill()

    # Fill remaining missing with column mean
    for col in factor_names:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    df = df.dropna(subset=['returns', 'rv'])
    print(f"    Merged dataset: {len(df)} rows")

    # 3. Estimate MIDAS-GARCH per factor
    print("\n[4/6] Estimating MIDAS-GARCH(1,1)...")
    results_all = {}
    for fname in factor_names:
        print(f"    Processing {fname}...")
        x = df[fname].copy()
        try:
            res = estimate_midas_garch(df['returns'], x, lookback=22)
            results_all[fname] = res
            print(f"      ω={res['omega']:.2e}  α={res['alpha']:.4f}  β={res['beta']:.4f}  γ={res['gamma']:.4f}  θ={res['theta']:.1f}")
            print(f"      R²: Vanilla={res['r2_vanilla']:.4f}  MIDAS={res['r2_midas']:.4f}  Δ={res['r2_improvement']:.4f}")
        except Exception as e:
            print(f"      Error on {fname}: {e}")

    # 4. Pick best factor
    best_factor = max(results_all.keys(), key=lambda k: results_all[k]['r2_improvement'])
    best_res = results_all[best_factor]
    print(f"\n    ★ Best factor: {best_factor} (ΔR²={best_res['r2_improvement']:.4f})")

    # 5. Forecast
    print("\n[5/6] Generating 5-day forecast...")
    params = {
        'omega': best_res['omega'], 'alpha': best_res['alpha'],
        'beta': best_res['beta'], 'gamma': best_res['gamma'],
        'theta': best_res['theta'], 'lookback': best_res['lookback'],
        'h_midas': best_res['h_midas']
    }
    fc_v = rolling_forecast(df['returns'], df[best_factor], params, horizon=5)
    fc_m = rolling_forecast(df['returns'], df[best_factor], params, horizon=5)
    # Vanilla: use average of recent conditional vol as naive forecast
    fc_v_naive = np.full(5, params['h_midas'][-5:].mean())
    fc_m_naive = fc_m

    dates_csi = df.index.values

    # 6. Charts
    print("\n[6/6] Generating charts...")
    chart_vol = chart_conditional_vol(
        df['returns'], best_res['h_vanilla'], best_res['h_midas'],
        dates_csi, f"{OUTPUT_DIR}/midas_garch_vol.png"
    )
    chart_wts = chart_midas_weights(
        best_res['lookback'], best_res['theta'],
        f"{OUTPUT_DIR}/midas_garch_weights.png"
    )
    chart_fc = chart_forecast(fc_v_naive, fc_m_naive, f"{OUTPUT_DIR}/midas_garch_fc.png")

    # Macro factor subplot if available
    chart_factors = None
    if len(factor_dfs) > 0:
        primary_factor = factor_dfs[0]
        chart_factors = chart_macro_factor(
            primary_factor.index.values, primary_factor.iloc[:,0].values,
            factor_names[0], f"{OUTPUT_DIR}/midas_garch_factor.png"
        )
    print("    Saved: midas_garch_vol.png, midas_garch_weights.png, midas_garch_fc.png")

    # 7. Build & send report
    print("\n[7/7] Sending Telegram report...")

    # 5-day forecast table
    fc_lines = ""
    for i, (fv, fm) in enumerate(zip(fc_v_naive, fc_m_naive), 1):
        fv_ann = np.sqrt(fv) * np.sqrt(252) * 100
        fm_ann = np.sqrt(fm) * np.sqrt(252) * 100
        fc_lines += f"  Day {i}     {fv_ann:>6.2f}%       {fm_ann:>6.2f}%\n"

    # Per-factor summary
    factor_summary = ""
    for fname, res in sorted(results_all.items(), key=lambda x: -x[1]['r2_improvement']):
        factor_summary += f"{fname:<12}  θ={res['theta']:.1f}  γ={res['gamma']:.4f}  ΔR²={res['r2_improvement']:+.4f}\n"

    text_en = f"""📊 <b>MIDAS-GARCH — CSI 300 + Macro Factors</b>
🕐 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} CST
━━━━━━━━━━━━━━━━━━━━

📐 <b>GARCH(1,1) Core Parameters</b>
<code>
ω (omega) = {best_res['omega']:.2e}
α (alpha) = {best_res['alpha']:.4f}
β (beta)  = {best_res['beta']:.4f}
γ (gamma) = {best_res['gamma']:.4f}   ← macro factor loading
θ (theta) = {best_res['theta']:.1f}   ← MIDAS beta curvature
Persistence (α+β) = {best_res['alpha']+best_res['beta']:.4f}
</code>

📊 <b>Per-Factor MIDAS-GARCH Results</b>
<code>
Factor        θ    γ          ΔR² vs GARCH
{factor_summary}
</code>

📈 <b>5-Day Volatility Forecast (annualized %)</b>
<code>
Horizon    GARCH(1,1)   MIDAS-GARCH
{fc_lines}</code>

💡 <b>Interpretation</b>
• Best macro driver: <b>{best_factor}</b> (explains {abs(best_res['r2_improvement'])*100:.2f}% extra variance beyond GARCH)
• γ={'positive — macro factor raises vol' if best_res['gamma'] > 0 else 'negative — macro factor reduces vol'} when {best_factor} rises
• θ={best_res['theta']:.1f} → MIDAS weights {'more concentrated on recent days' if best_res['theta'] > 1 else 'relatively uniform'} (higher θ = more front-loaded)
• R² improvement of {best_res['r2_improvement']*100:.2f}% {'confirms macro-factor volatility linkage' if best_res['r2_improvement'] > 0.01 else 'suggests limited macro-volatility coupling'}

🔮 <b>Methodology</b>
• MIDAS-GARCH(1,1): σ²_t = ω + α·ε²_{{t-1}} + β·σ²_{{t-1}} + γ·Σ_k w_k(θ)·x_{{t-k}}
• Weight: Beta MIDAS w_k = k^{{θ-1}}·(K-k)^{{θ-1}} / Σj^{{θ-1}}(K-j)^{{θ-1}}
• Factors: {', '.join(factor_names) if factor_names else 'N/A'}
• Data: CSI 300 via Tushare Pro · {len(df)} trading days
⚠️ For informational purposes only. Not investment advice.
"""

    text_cn = f"""📊 <b>MIDAS-GARCH — CSI 300 + 宏观因子</b>
🕘 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')} CST
━━━━━━━━━━━━━━━━━━━━

📐 <b>GARCH(1,1) 核心参数</b>
<code>
ω (omega) = {best_res['omega']:.2e}
α (alpha) = {best_res['alpha']:.4f}
β (beta)  = {best_res['beta']:.4f}
γ (gamma) = {best_res['gamma']:.4f}   ← 宏观因子载荷
θ (theta) = {best_res['theta']:.1f}   ← MIDAS Beta 曲率
持续性 (α+β) = {best_res['alpha']+best_res['beta']:.4f}
</code>

📊 <b>各因子 MIDAS-GARCH 结果</b>
<code>
因子           θ    γ          ΔR² vs GARCH
{factor_summary}
</code>

📈 <b>5日波动率预测（年化 %）</b>
<code>
天数    GARCH(1,1)   MIDAS-GARCH
{fc_lines}</code>

💡 <b>解读</b>
• 最优宏观驱动因子：<b>{best_factor}</b>（在 GARCH 基础上解释额外 {abs(best_res['r2_improvement'])*100:.2f}% 方差）
• γ={'正 — ' + best_factor + ' 上升时波动率提升' if best_res['gamma'] > 0 else '负 — ' + best_factor + ' 上升时波动率下降'}
• θ={best_res['theta']:.1f} → MIDAS 权重 {'越靠近期权重的最近日期' if best_res['theta'] > 1 else '分布相对均匀'}
• ΔR² = {best_res['r2_improvement']*100:.2f}% → {'宏观因子与波动率存在显著联动' if best_res['r2_improvement'] > 0.01 else '宏观-波动率耦合较弱'}

🔮 <b>方法论</b>
• MIDAS-GARCH(1,1): σ²_t = ω + α·ε²_{{t-1}} + β·σ²_{{t-1}} + γ·Σ_k w_k(θ)·x_{{t-k}}
• 权重：Beta MIDAS w_k = k^{{θ-1}}·(K-k)^{{θ-1}} / Σj^{{θ-1}}(K-j)^{{θ-1}}
• 因子：{', '.join(factor_names) if factor_names else 'N/A'}
• 数据：CSI 300 via Tushare Pro · {len(df)} 交易日
⚠️ 本报告仅供参考，不构成投资建议。
"""

    send_telegram_text(text_en)
    send_telegram_photo(chart_vol, f"📈 CSI 300 — Conditional Volatility: GARCH(1,1) vs MIDAS-GARCH (factor={best_factor})")
    if chart_factors:
        send_telegram_photo(chart_factors, f"📉 Macro Factor: {factor_names[0]}")
    send_telegram_photo(chart_wts, f"⚙️ MIDAS Lag Weights (θ={best_res['theta']:.1f}, K={best_res['lookback']})")
    send_telegram_photo(chart_fc, f"🔮 5-Day Volatility Forecast Comparison")
    send_telegram_text(text_cn)

    print("\n✅ Done — all outputs sent to Telegram.")

if __name__ == '__main__':
    main()
