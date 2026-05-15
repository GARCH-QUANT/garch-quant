#!/usr/bin/env python3
"""
Realized GARCH — Telegram-native multi-panel figure
====================================================
Format: Telegram-friendly 16:9 style, readable on mobile
"""

import warnings, os, sys
warnings.filterwarnings('ignore')
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

# ── Figure style: dark Telegram theme ────────────────────────────────────────
OUTPUT_DIR = os.path.expanduser("~/.hermes/cron/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DPI = 150

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

# ── CSI 300 data via Tushare ──────────────────────────────────────────────────
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")

def load_csi300():
    import tushare as ts
    tushare_token = os.environ.get("TUSHARE_TOKEN", "")
    ts.set_token(tushare_token)
    pro = ts.pro_api(tushare_token)
    df = pro.index_daily(
        ts_code='000300.SH',
        start_date='20240101',
        end_date=datetime.now().strftime('%Y%m%d')
    )
    df['date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('date').set_index('date')
    df = df[['open', 'high', 'low', 'close']].astype(float)
    df['log_co'] = np.log(df['close'] / df['open'])
    df['rv_gk'] = (0.5*(np.log(df['high']/df['low'])**2) - (2*np.log(2)-1)*(df['log_co']**2)) * 252
    df['rv_hl'] = (np.log(df['high']/df['low'])**2) / (4*np.log(2)) * 252
    df['rv'] = df['rv_gk']
    df['return'] = np.log(df['close'] / df['close'].shift(1))
    return df[['rv', 'rv_gk', 'rv_hl', 'return', 'close']].dropna()

# ── HAR model ──────────────────────────────────────────────────────────────────
def fit_har(rv_df):
    import statsmodels.api as sm
    df = rv_df.copy()
    df['log_rv']    = np.log(df['rv'])
    df['log_rv_1']  = df['log_rv'].shift(1)
    df['log_rv_5']  = df['log_rv'].rolling(5).mean().shift(1)
    df['log_rv_22'] = df['log_rv'].rolling(22).mean().shift(1)
    df_har = df[['log_rv', 'log_rv_1', 'log_rv_5', 'log_rv_22']].dropna()
    X = sm.add_constant(df_har[['log_rv_1', 'log_rv_5', 'log_rv_22']])
    model = sm.OLS(df_har['log_rv'], X).fit()
    fitted_index = df_har.index
    df.loc[fitted_index, 'sigma_har'] = np.exp(0.5 * model.fittedvalues.values)
    return model, df

# ─────────────────────────────────────────────────────────────────────────────
def main():
    csi_df = load_csi300()
    model, csi_enr = fit_har(csi_df)

    rv    = csi_df['rv'] * 100
    sigma = csi_enr['sigma_har'] * 100
    gk    = csi_df['rv_gk'] * 100
    hl    = csi_df['rv_hl'] * 100

    # ── Chart 1: RV time series + HAR σ̂_t ─────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)

    for ax, series, color, label in [
        (axes[0], rv,    '#58A6FF', 'Garman-Klass RV'),
        (axes[1], sigma, '#FF9800', 'HAR Conditional σ̂_t'),
    ]:
        ax.fill_between(series.index, 0, series.values, color=color, alpha=0.15, linewidth=0)
        ax.plot(series.index, series.values, color=color, linewidth=1.0, alpha=0.9)
        p80 = series.quantile(0.80)
        for i in range(len(series)-1):
            if series.iloc[i] > p80:
                ax.axvspan(series.index[i], series.index[i+1], color='#E53935', alpha=0.15, linewidth=0)
        ax.set_ylabel('Annualized Volatility (%)', fontsize=10)
        ax.legend(loc='upper right', fontsize=9, frameon=False)
        ax.grid(True, alpha=0.35)
        ymax = series.quantile(0.97) * 1.15
        ax.set_ylim(0, ymax)

    axes[0].set_title('CSI 300 — Garman-Klass Realized Volatility', fontsize=13, pad=8)
    axes[1].set_title('CSI 300 — HARF Conditional Volatility σ̂_t', fontsize=13, pad=8)
    plt.tight_layout(pad=3)
    path1 = f"{OUTPUT_DIR}/rg_tg_rv_ts.png"
    plt.savefig(path1, dpi=DPI, bbox_inches='tight', facecolor='#0D1117')
    plt.close()

    # ── Chart 2: HAR coefficients ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    coeffs = {
        'β₁ (day)':    model.params.get('log_rv_1',  0),
        'β₅ (week)':   model.params.get('log_rv_5',  0),
        'β₂₂ (month)': model.params.get('log_rv_22', 0),
    }
    bar_colors = ['#58A6FF', '#FF9800', '#4CAF50']
    x = np.arange(len(coeffs))
    bars = list(coeffs.values())
    ax.bar(x, bars, color=bar_colors, width=0.5, linewidth=0, alpha=0.9)
    for i, (k, v) in enumerate(coeffs.items()):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=11, color='#E6EDF3')
    ax.axhline(0, color='#8B949E', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(list(coeffs.keys()), fontsize=11)
    ax.set_ylabel('Coefficient', fontsize=11)
    ax.set_title('HAR Model — Heterogeneous Volatility Coefficients (CSI 300)', fontsize=13, pad=10)
    ax.grid(True, axis='y', alpha=0.35)
    plt.tight_layout()
    path2 = f"{OUTPUT_DIR}/rg_tg_har_coef.png"
    plt.savefig(path2, dpi=DPI, bbox_inches='tight', facecolor='#0D1117')
    plt.close()

    # ── Chart 3: GK vs HL scatter + rolling corr ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    common = gk.index.intersection(hl.index)
    gk_a, hl_a = gk.loc[common], hl.loc[common]

    axes[0].scatter(gk_a, hl_a, s=12, color='#58A6FF', alpha=0.5, linewidths=0)
    max_val = max(gk_a.max(), hl_a.max())
    axes[0].plot([0, max_val], [0, max_val], color='#FF9800', lw=1.5, ls='--', label='1:1 line')
    z = np.polyfit(gk_a, hl_a, 1)
    x_l = np.linspace(gk_a.min(), gk_a.max(), 100)
    axes[0].plot(x_l, np.polyval(z, x_l), color='#4CAF50', lw=1.5, label=f'OLS slope={z[0]:.3f}')
    axes[0].set_xlabel('Garman-Klass RV (%)', fontsize=10)
    axes[0].set_ylabel('Parkinson HL RV (%)', fontsize=10)
    axes[0].set_title('GK vs Parkinson RV (CSI 300)', fontsize=12, pad=8)
    axes[0].legend(fontsize=9, frameon=False)
    axes[0].grid(True, alpha=0.35)

    rolling = csi_df['rv_gk'].rolling(30).corr(csi_df['rv_hl']) * 100
    rolling = rolling.dropna()
    axes[1].plot(rolling.index, rolling, color='#58A6FF', linewidth=1.0)
    axes[1].fill_between(rolling.index, rolling, rolling.mean(), alpha=0.2, color='#58A6FF')
    axes[1].axhline(rolling.mean(), color='#FF9800', ls='--', lw=1.2, label=f'Mean: {rolling.mean():.1f}%')
    axes[1].set_ylabel('Correlation (%)', fontsize=10)
    axes[1].set_xlabel('Date', fontsize=10)
    axes[1].set_title('30-Day Rolling Corr(GK, HL)', fontsize=12, pad=8)
    axes[1].legend(fontsize=9, frameon=False)
    axes[1].grid(True, alpha=0.35)
    axes[1].set_ylim(0, 105)

    plt.tight_layout(pad=3)
    path3 = f"{OUTPUT_DIR}/rg_tg_gk_hl.png"
    plt.savefig(path3, dpi=DPI, bbox_inches='tight', facecolor='#0D1117')
    plt.close()

    return [path1, path2, path3]

if __name__ == '__main__':
    main()
