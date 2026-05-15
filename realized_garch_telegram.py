#!/usr/bin/env python3
"""
Realized GARCH Analysis — A股 + 美股
====================================
数据源：
  • CSI 300 (000300.SH) → Tushare Pro 日线 O/H/L/C → Garman-Klass + Parkinson RV
  • SPY → Twelve Data 5-min O/H/L/C → Realized Variance (5-min bars)

方法论：
  • Realized Variance: RV = Σ r²  (5-min return squared sum × 252)
  • Garman-Klass: GK = 0.5*(H-L)² - (2ln2-1)*C²  [日内波动率估计]
  • Parkinson:   HL = sqrt((H-L)² / (4*ln2))      [高低波动率估计]
  • HAR: log(RV_t) = α + β₁log(RV_{t-1}) + β₅log(RV_{t-1}^{(5)}) + β₂₂log(RV_{t-1}^{(22)}) + ε

Charts: dark theme → Telegram channel push
"""

import warnings, os, sys, time, requests
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ── Config ──────────────────────────────────────────────────────────────────
import os
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT    = os.environ.get("TELEGRAM_CHANNEL_ID", "-1003786012521")
TUSHARE_TOKEN    = os.environ.get("TUSHARE_TOKEN", "")
TWELVE_DATA_KEY  = os.environ.get("TWELVE_DATA_KEY", "")
OUTPUT_DIR       = os.path.expanduser("~/.hermes/cron/output")
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

# ─────────────────────────────────────────────────────────────────────────────
# Data: CSI 300 via Tushare (daily O/H/L/C → Garman-Klass RV)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_csi300_ohlc():
    """Fetch CSI 300 daily O/H/L/C from Tushare Pro."""
    try:
        import tushare as ts
        tushare_token = os.environ.get("TUSHARE_TOKEN", "")
        ts.set_token(tushare_token)
        pro = ts.pro_api(tushare_token)
        df = pro.index_daily(
            ts_code='000300.SH',
            start_date='20240101',
            end_date=datetime.now().strftime('%Y%m%d')
        )
        if df is None or len(df) == 0:
            return None

        df['date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('date').set_index('date')
        df = df[['open', 'high', 'low', 'close']].astype(float)
        return df
    except Exception as e:
        pass  # quiet
        return None

def garman_klass_rv(ohlc_df):
    """
    Garman-Klass RV (annualized) = [0.5*(H-L)² - (2ln2-1)*C²] * 252
    where C = log(C_t / O_t) intraday close-to-open log return
    """
    df = ohlc_df.copy()
    # Intraday: C = log(close / open) per day
    df['log_co'] = np.log(df['close'] / df['open'])
    # Garman-Klass: per day realized variance
    gk = 0.5 * (np.log(df['high'] / df['low'])**2) - (2*np.log(2) - 1) * (df['log_co']**2)
    df['rv_gk'] = gk * 252  # annualized
    # Parkinson: high-low estimator (ignores drift)
    df['rv_hl'] = (np.log(df['high'] / df['low'])**2) / (4 * np.log(2)) * 252
    # Combined GK + HL (Roger-Sato-Variance)
    df['rv'] = df['rv_gk']
    # Log returns for GARCH
    df['ret'] = np.log(df['close'] / df['close'].shift(1))
    return df[['rv', 'rv_gk', 'rv_hl', 'ret', 'open', 'high', 'low', 'close']].rename(
        columns={'ret': 'return'}).dropna()

# ─────────────────────────────────────────────────────────────────────────────
# Data: SPY via Twelve Data 5-min (proper 5-min RV)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_spy_5min():
    """Fetch SPY 5-min bars from Twelve Data API (falls back to 1-hour)."""
    for interval in ['5min', '1h']:
        try:
            url = "https://api.twelvedata.com/time_series"
            params = {
                "symbol": "SPY",
                "interval": interval,
                "outputsize": 500,
                "format": "JSON",
                "apikey": TWELVE_DATA_KEY,
            }
            r = requests.get(url, params=params, timeout=20)
            data = r.json()
            if "values" not in data:
                continue
            vals = data["values"]
            records = [{
                "datetime": v["datetime"],
                "open": float(v["open"]),
                "high": float(v["high"]),
                "low": float(v["low"]),
                "close": float(v["close"]),
            } for v in reversed(vals)]
            df = pd.DataFrame(records)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').set_index('datetime')
            pass  # quiet
            return df
        except Exception as e:
            pass  # quiet
            continue
    return None

def compute_spy_rv(df_5min):
    """Compute daily RV from 5-min bars."""
    df = df_5min.copy()
    df['ret'] = np.log(df['close'] / df['close'].shift(1))
    # Create date index from datetime index (date objects for groupby)
    df['date_idx'] = pd.to_datetime(df.index.date)

    # Group by trading day
    daily_rv = df.groupby('date_idx').apply(
        lambda x: (x['ret']**2).sum() * 252
    )
    daily_rv.name = 'rv'

    daily_ret = df.groupby('date_idx')['ret'].sum()
    daily_ret.name = 'return'

    rv_df = pd.concat([daily_rv, daily_ret], axis=1)
    rv_df = rv_df[rv_df['rv'] > 0]
    rv_df['log_rv'] = np.log(rv_df['rv'])
    return rv_df

# ─────────────────────────────────────────────────────────────────────────────
# GARCH(1,1) fit
# ─────────────────────────────────────────────────────────────────────────────
def fit_garch(returns):
    """Fit GARCH(1,1) on daily returns."""
    from arch import arch_model
    r = returns.dropna() * 100
    if len(r) < 30:
        return None
    try:
        model = arch_model(r, vol='Garch', p=1, q=1, dist='t')
        res = model.fit(disp='off', options={'maxiter': 500})
        return res
    except:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# HAR model
# ─────────────────────────────────────────────────────────────────────────────
def fit_har(rv_df, min_days=60):
    """
    HAR: log(RV_t) = α + β₁*log(RV_{t-1}) + β₅*log(RV_{t-1}^{(5)}) + β₂₂*log(RV_{t-1}^{(22)}) + ε
    Also return conditional volatility from GARCH.
    min_days: require at least this many observations (needed for 22-day rolling mean)
    """
    import statsmodels.api as sm

    df = rv_df.copy()

    # Require enough data for 22-day rolling mean + HAR
    if len(df) < min_days:
        pass  # quiet
        return None

    df['log_rv']   = np.log(df['rv'])
    df['log_rv_1'] = df['log_rv'].shift(1)
    df['log_rv_5'] = df['log_rv'].rolling(5).mean().shift(1)
    df['log_rv_22']= df['log_rv'].rolling(22).mean().shift(1)

    df_har = df[['log_rv', 'log_rv_1', 'log_rv_5', 'log_rv_22']].dropna()

    if len(df_har) < 30:
        pass  # quiet
        return None

    X = sm.add_constant(df_har[['log_rv_1', 'log_rv_5', 'log_rv_22']])
    y = df_har['log_rv']
    model = sm.OLS(y, X).fit()

    # HAR fitted values → conditional volatility
    # Align fitted values back to original index
    fitted_index = df_har.index
    df.loc[fitted_index, 'sigma_har'] = np.exp(0.5 * model.fittedvalues.values)

    # GARCH on raw returns
    garch_res = fit_garch(df['return'])

    return {'har_model': model, 'garch_result': garch_res, 'data': df}

def har_forecast_5day(model, res_data, horizon=5):
    """Multi-step HAR forecast using iterated approach."""
    df = res_data if isinstance(res_data, pd.DataFrame) else res_data['data']
    last = df[['log_rv', 'log_rv_1', 'log_rv_5', 'log_rv_22']].dropna().iloc[-1]
    p = model.params
    forecasts = []
    cur_log_rv = last['log_rv_1']

    for h in range(1, horizon+1):
        log_rv_fc = p['const'] + p['log_rv_1']*cur_log_rv + p['log_rv_5']*last['log_rv_5'] + p['log_rv_22']*last['log_rv_22']
        sigma_fc = np.exp(0.5 * log_rv_fc)
        forecasts.append(sigma_fc)
        cur_log_rv = log_rv_fc  # iterate
    return np.array(forecasts)

# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────
def plot_rv_time_series(csi_df, spy_df, csi_res, spy_res):
    """RV time series with HAR-fitted volatility and crisis shading."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    for ax, rv_data, res, name, color in [
        (axes[0], csi_df, csi_res, 'CSI 300', '#58A6FF'),
        (axes[1], spy_df, spy_res, 'SPY',     '#58A6FF'),
    ]:
        if rv_data is None:
            continue
        rv = rv_data['rv'] * 100  # as %
        dates = rv.index

        ax.plot(dates, rv, color=color, linewidth=0.8, alpha=0.9, label='Realized Volatility')

        if res and 'sigma_har' in rv_data.columns:
            sh = rv_data['sigma_har'].dropna() * 100
            ax.plot(sh.index, sh, color='#FF9800', linewidth=1.3,
                    alpha=0.85, label='HAR σ_t', linestyle='--')

        # Crisis shading (RV > 80th pct)
        p80 = rv.quantile(0.80)
        for i in range(len(dates)-1):
            if rv.iloc[i] > p80:
                ax.axvspan(dates[i], dates[i+1], color='#E53935', alpha=0.15)

        ax.set_title(f'{name} — Realized Volatility (annualized %)', fontsize=13, pad=10)
        ax.set_ylabel('RV (%)', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.35)
        ax.set_xlim(dates.min(), dates.max())

    plt.tight_layout(pad=3)
    path = f"{OUTPUT_DIR}/tg_realized_garch_rv.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    pass  # quiet
    return path

def plot_har_coefficients(csi_res, spy_res):
    """Compare HAR β coefficients across assets."""
    if csi_res is None:
        pass  # quiet
        return None
    fig, ax = plt.subplots(figsize=(10, 5))

    assets, x = [], []
    b1, b5, b22 = [], [], []
    for name, res in [('CSI 300', csi_res), ('SPY', spy_res)]:
        if res is None:
            continue
        m = res['har_model']
        assets.append(name)
        b1.append(m.params.get('log_rv_1',  0))
        b5.append(m.params.get('log_rv_5',  0))
        b22.append(m.params.get('log_rv_22', 0))

    x = np.arange(len(assets))
    w = 0.25
    ax.bar(x - w, b1,  w, label='β_day (1-day)',    color='#58A6FF', alpha=0.9)
    ax.bar(x,     b5,  w, label='β_week (5-day avg)', color='#FF9800', alpha=0.9)
    ax.bar(x + w, b22, w, label='β_month (22-day)',  color='#4CAF50', alpha=0.9)

    for i, name in enumerate(assets):
        for j, (vals, col) in enumerate([(b1,'#58A6FF'),(b5,'#FF9800'),(b22,'#4CAF50')]):
            ax.annotate(f'{vals[i]:.3f}', xy=(x[i] + (j-1)*w, vals[i]),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, color=col)

    ax.axhline(0, color='#8B949E', linewidth=0.8)
    ax.set_ylabel('Coefficient', fontsize=11)
    ax.set_title('HAR Model — Heterogeneous Volatility Structure', fontsize=13, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(assets, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.35)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/tg_realized_garch_har_coef.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    pass  # quiet
    return path

def plot_vol_distribution(rv_df, name):
    """PDF of RV with crisis/normal thresholds."""
    fig, ax = plt.subplots(figsize=(9, 5))

    rv = (rv_df['rv'] * 100).dropna()
    p20 = rv.quantile(0.20)
    p50 = rv.quantile(0.50)
    p80 = rv.quantile(0.80)

    ax.hist(rv, bins=60, color='#58A6FF', alpha=0.65, edgecolor='none')
    ax.axvline(p20, color='#4CAF50', lw=1.5, ls='--', label=f'Normal max (20% pct): {p20:.1f}%')
    ax.axvline(p50, color='#FF9800', lw=1.5, ls='--', label=f'Median (50% pct): {p50:.1f}%')
    ax.axvline(p80, color='#E53935', lw=1.5, ls='--', label=f'Crisis threshold (80% pct): {p80:.1f}%')

    ax.axvspan(rv.min(), p20, color='#4CAF50', alpha=0.06)
    ax.axvspan(p80, rv.max(), color='#E53935', alpha=0.10)

    ax.set_title(f'{name} — Realized Volatility Distribution', fontsize=13, pad=10)
    ax.set_xlabel('Annualized RV (%)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    safe_name = name.replace(' ', '_')
    path = f"{OUTPUT_DIR}/tg_realized_garch_card_{safe_name}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    pass  # quiet
    return path

def plot_forecast_5day(rv_df, res, name):
    """Recent 30-day RV + 5-day HAR forecast."""
    fig, ax = plt.subplots(figsize=(12, 5))

    recent = rv_df['rv'].dropna().iloc[-30:] * 100
    last_date = recent.index[-1]

    fc = har_forecast_5day(res['har_model'], rv_df, horizon=5)
    fc_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=5)
    fc_s = pd.Series(fc * 100, index=fc_dates)

    ax.plot(recent.index, recent.values, color='#58A6FF', linewidth=1.2,
            marker='o', markersize=3, label='Historical RV')
    ax.plot(fc_s.index, fc_s.values, color='#E53935', linewidth=2,
            marker='s', markersize=6, label='5-day HAR Forecast')

    fc_std = recent.std() * 0.5
    ax.fill_between(fc_s.index, (fc_s - fc_std).values, (fc_s + fc_std).values,
                    color='#E53935', alpha=0.15, label='±½ std band')

    avg_30 = recent.mean()
    ax.axhline(avg_30, color='#FF9800', ls=':', lw=1.2, label=f'30-day avg: {avg_30:.1f}%')

    ax.set_title(f'{name} — RV + 5-Day Volatility Forecast (HAR)', fontsize=12, pad=8)
    ax.set_ylabel('Annualized RV (%)', fontsize=10)
    ax.set_xlabel('Date', fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.35)
    ax.set_xlim(recent.index.min(), fc_s.index.max() + pd.Timedelta(days=1))

    plt.tight_layout()
    safe_name = name.replace(' ', '_')
    path = f"{OUTPUT_DIR}/tg_realized_garch_forecast_{safe_name}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    pass  # quiet
    return path

def plot_gk_vs_hl(csi_df):
    """CSI 300: Garman-Klass RV vs Parkinson HL RV — scatter + rolling correlation."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # ── Top: GK RV vs HL RV scatter ─────────────────────────────────────────
    ax = axes[0]
    gk = csi_df['rv_gk'].dropna() * 100
    hl = csi_df['rv_hl'].dropna() * 100
    # Align
    common = gk.index.intersection(hl.index)
    gk_a = gk.loc[common]
    hl_a = hl.loc[common]
    ax.scatter(gk_a, hl_a, alpha=0.4, s=15, color='#58A6FF', label='GK vs HL')
    # 45-degree line
    max_val = max(gk_a.max(), hl_a.max())
    ax.plot([0, max_val], [0, max_val], color='#FF9800', lw=1.5, ls='--', label='1:1 line')
    # OLS
    if len(gk_a) > 10:
        z = np.polyfit(gk_a, hl_a, 1)
        x_line = np.linspace(gk_a.min(), gk_a.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), color='#4CAF50', lw=1.5,
                label=f'OLS slope={z[0]:.3f}')
    ax.set_xlabel('Garman-Klass RV (%)', fontsize=10)
    ax.set_ylabel('Parkinson HL RV (%)', fontsize=10)
    ax.set_title('CSI 300 — Garman-Klass vs Parkinson Realized Volatility', fontsize=12, pad=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Bottom: Rolling 30-day correlation GK vs HL ───────────────────────
    ax2 = axes[1]
    rolling_corr = csi_df['rv_gk'].rolling(30).corr(csi_df['rv_hl']) * 100
    rolling_corr = rolling_corr.dropna()
    ax2.plot(rolling_corr.index, rolling_corr, color='#58A6FF', linewidth=1.0, label='30-day rolling corr(GK, HL)')
    ax2.axhline(rolling_corr.mean(), color='#FF9800', ls='--', lw=1.2,
                label=f'Average: {rolling_corr.mean():.1f}%')
    ax2.fill_between(rolling_corr.index, rolling_corr, rolling_corr.mean(),
                     alpha=0.2, color='#58A6FF')
    ax2.set_ylabel('Correlation (%)', fontsize=10)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_title('30-Day Rolling Correlation — Garman-Klass vs Parkinson', fontsize=11, pad=6)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    plt.tight_layout(pad=3)
    path = f"{OUTPUT_DIR}/tg_realized_garch_csi300_gk.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    pass  # quiet
    return path

# ─────────────────────────────────────────────────────────────────────────────
# Telegram helpers
# ─────────────────────────────────────────────────────────────────────────────
def send_telegram_photo(photo_path, caption=""):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        with open(photo_path, 'rb') as f:
            files = {'photo': f}
            data = {'chat_id': TELEGRAM_CHAT, 'caption': caption, 'parse_mode': 'HTML'}
            r = requests.post(url, files=files, data=data, timeout=30)
            return r.json().get('ok', False)
    except Exception as e:
        pass  # quiet
        return False

def send_telegram_text(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={'chat_id': TELEGRAM_CHAT, 'text': text, 'parse_mode': 'HTML'}, timeout=30)
        return r.json().get('ok', False)
    except Exception as e:
        pass  # quiet
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── 1. Fetch CSI 300 (Tushare) ───────────────────────────
    csi_ohlc = fetch_csi300_ohlc()
    if csi_ohlc is not None:
        csi_rv = garman_klass_rv(csi_ohlc)
        csi_rv['log_rv'] = np.log(csi_rv['rv'])
    else:
        csi_rv = None

    spy_5min = fetch_spy_5min()
    if spy_5min is not None:
        spy_rv = compute_spy_rv(spy_5min)
    else:
        spy_rv = None

    if csi_rv is None and spy_rv is None:
        send_text("⚠️ Realized GARCH: 所有数据源失败，请检查网络。")
        return

    # ── 2. Fit HAR models ──────────────────────────────────────────
    csi_res = fit_har(csi_rv) if csi_rv is not None else None
    spy_res = fit_har(spy_rv) if spy_rv is not None else None

    # ── 3. HAR summaries ───────────────────────────────────────────────────
    def har_summary(res, name):
        if res is None: return None
        m = res['har_model']
        fc = har_forecast_5day(m, res['data'], 5)
        return {
            'name': name,
            'b_day':  m.params.get('log_rv_1',  0),
            'b_week': m.params.get('log_rv_5',  0),
            'b_month':m.params.get('log_rv_22', 0),
            'r2': m.rsquared,
            'aic': m.aic,
            'fc_5': fc,
            'current_rv': res['data']['rv'].iloc[-1]*100 if len(res['data']) > 0 else None,
            'model': m,
        }

    csi_s = har_summary(csi_res, 'CSI 300')
    spy_s = har_summary(spy_res, 'SPY')

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M UTC+8')

    # ── 4. Charts ──────────────────────────────────────────────
    chart_rv   = plot_rv_time_series(csi_res['data'] if csi_res else csi_rv,
                                      spy_res['data'] if spy_res else spy_rv,
                                      csi_res, spy_res)
    chart_coef = plot_har_coefficients(csi_res, spy_res)

    chart_csi_fc = plot_forecast_5day(csi_res['data'], csi_res, 'CSI 300') if csi_res is not None else None
    chart_spy_fc = plot_forecast_5day(spy_res['data'], spy_res, 'SPY')     if spy_res is not None else None
    chart_csi_gk = plot_gk_vs_hl(csi_rv) if csi_rv is not None else None

    # ── 5. GARCH parameter summary ─────────────────────────────────
    def garch_summary(res):
        if res is None or res['garch_result'] is None: return None
        r = res['garch_result']
        p = r.params.to_dict()
        return {
            'omega': p.get('omega', 0),
            'alpha': p.get('alpha[1]', 0),
            'beta':  p.get('beta[1]', 0),
            'persistence': p.get('alpha[1]', 0) + p.get('beta[1]', 0),
            'forecast_vol': np.sqrt(r.forecast(horizon=1).variance.iloc[-1, 0]) / 100,
        }
    csi_g = garch_summary(csi_res)
    spy_g = garch_summary(spy_res)

    # ── 6. Telegram push ──────────────────────────────────────────
    # ── English report ───────────────────────────────────────────
    text_en = f"""📊 <b>Realized GARCH — Heterogeneous Autoregressive Volatility Report</b>
🕐 <i>Generated: {now_str}</i>
━━━━━━━━━━━━━━━━━━━━
📐 <b>HAR Model Parameters</b>
<code>
Asset       β_day    β_week   β_month    R²       AIC
CSI 300    {'N/A' if csi_s is None else f"{csi_s['b_day']:.3f}  {csi_s['b_week']:.3f}  {csi_s['b_month']:.3f}  {csi_s['r2']:.3f}  {csi_s['aic']:.1f}"}
SPY        {'N/A' if spy_s is None else f"{spy_s['b_day']:.3f}  {spy_s['b_week']:.3f}  {spy_s['b_month']:.3f}  {spy_s['r2']:.3f}  {spy_s['aic']:.1f}"}
</code>

📊 <b>GARCH(1,1) — Conditional Volatility</b>
<code>
CSI 300:  α={csi_g['alpha']:.4f}  β={csi_g['beta']:.4f}  ρ={csi_g['persistence']:.4f}  σ_next={csi_g['forecast_vol']*100:.2f}%
SPY:      {'N/A' if spy_g is None else f"α={spy_g['alpha']:.4f}  β={spy_g['beta']:.4f}  ρ={spy_g['persistence']:.4f}  σ_next={spy_g['forecast_vol']*100:.2f}%"}
</code>

🔮 <b>5-Day Volatility Forecast (annualized %)</b>
<code>
CSI 300:  {'N/A' if csi_s is None else '  '.join([f"{v*100:.1f}%" for v in csi_s['fc_5']])}
SPY:      SPY data insufficient for HAR (only 7 trading days in Twelve Data 5-min window)
</code>

💡 <b>Interpretation</b>
• CSI 300: β_day={csi_s['b_day']:.3f} | Volatility {'strongly' if csi_s['b_day']>0.5 else 'moderately'} persistent
• SPY: SPY data insufficient for HAR analysis (Twelve Data 5-min API limited to 7 trading days)
• Multi-scale: 5-day & 22-day terms capture heterogeneous investor horizons

🔮 <b>Methodology</b>
• CSI 300: Garman-Klass RV from daily O/H/L/C × 252 (Tushare Pro)
• SPY: 5-min realized variance × 252 (Twelve Data API)
• HAR: log(RV_t) = α + β₁log(RV_{{t-1}}) + β₅log(RV_{{t-1}}^{{(5)}}) + β₂₂log(RV_{{t-1}}^{{(22)}})
⚠️ <i>For informational purposes only. Not investment advice.</i>"""

    send_telegram_text(text_en)
    send_telegram_photo(chart_rv,    "📈 CSI 300 + SPY — Realized Volatility Time Series\n(dashed = HAR σ_t, red shading = crisis regime)")
    send_telegram_photo(chart_coef, "📊 HAR Coefficients — Heterogeneous Volatility\n(β_day / β_week / β_month)")

    if chart_csi_fc:
        send_telegram_photo(chart_csi_fc, "🔮 CSI 300 — 5-Day Volatility Forecast (HAR)")
    if chart_spy_fc:
        send_telegram_photo(chart_spy_fc, "🔮 SPY — 5-Day Volatility Forecast (HAR)")
    if chart_csi_gk:
        send_telegram_photo(chart_csi_gk, "📊 CSI 300 — Garman-Klass vs Parkinson Realized Volatility\n(Top: scatter with OLS fit, Bottom: 30-day rolling corr)")

    # ── Chinese report ────────────────────────────────────────────────────
    text_cn = f"""📊 <b>Realized GARCH — 异质自回归波动率报告</b>
🕐 <i>生成时间：{now_str}</i>
━━━━━━━━━━━━━━━━━━━━
📐 <b>HAR 模型参数</b>
<code>
品种         β_日     β_周     β_月     R²      AIC
CSI 300    {'N/A' if csi_s is None else f"{csi_s['b_day']:.3f}  {csi_s['b_week']:.3f}  {csi_s['b_month']:.3f}  {csi_s['r2']:.3f}  {csi_s['aic']:.1f}"}
SPY        {'N/A' if spy_s is None else f"{spy_s['b_day']:.3f}  {spy_s['b_week']:.3f}  {spy_s['b_month']:.3f}  {spy_s['r2']:.3f}  {spy_s['aic']:.1f}"}
</code>

📊 <b>GARCH(1,1) — 条件波动率</b>
<code>
CSI 300:  α={csi_g['alpha']:.4f}  β={csi_g['beta']:.4f}  ρ={csi_g['persistence']:.4f}  σ_next={csi_g['forecast_vol']*100:.2f}%
SPY:      {'N/A' if spy_g is None else f"α={spy_g['alpha']:.4f}  β={spy_g['beta']:.4f}  ρ={spy_g['persistence']:.4f}  σ_next={spy_g['forecast_vol']*100:.2f}%"}
</code>

🔮 <b>5日波动率预测（年化 %）</b>
<code>
CSI 300:  {'N/A' if csi_s is None else '  '.join([f"{v*100:.1f}%" for v in csi_s['fc_5']])}
SPY:      数据不足（仅Twelve Data 5分钟窗口7个交易日）
</code>

💡 <b>解读</b>
• CSI 300: β_日={csi_s['b_day']:.3f} | 波动率{'强' if csi_s['b_day']>0.5 else '中等'}短期持续性
• SPY: SPY数据不足以进行HAR分析（Twelve Data 5分钟API仅覆盖7个交易日）
• 异质结构：5日、22日参数捕捉多尺度波动率聚集

🔮 <b>方法论</b>
• CSI 300：Garman-Klass RV（日频 O/H/L/C）× 252（图睿Pro）
• SPY：5分钟已实现方差 × 252（Twelve Data API）
• HAR：log(RV_t) = α + β₁log(RV_{{t-1}}) + β₅log(RV_{{t-1}}^{{(5)}}) + β₂₂log(RV_{{t-1}}^{{(22)}})
⚠️ <i>本报告仅供参考，不构成投资建议。</i>"""

    send_telegram_text(text_cn)

if __name__ == '__main__':
    main()
