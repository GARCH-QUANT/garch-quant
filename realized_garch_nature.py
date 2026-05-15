#!/usr/bin/env python3
"""
Realized GARCH — Nature-style Single-panel Figure
==================================================
Format: Nature articles require:
  • Single column: 89 mm wide (3.5 in)
  • Double column: 183 mm wide (7.2 in)
  • Text: 7 pt Helvetica/Arial
  • Line weight: 0.5–1 pt
  • No decorative elements, clean grid
  • PDF/EPS preferred for publication; PNG at 600 dpi

Content: 3 panels (a, b, c)
  (a) CSI 300 Realized Volatility + HAR-fitted σ_t  [time series]
  (b) HAR model coefficients (β_day, β_week, β_month)
  (c) Garman-Klass vs Parkinson: scatter + rolling correlation
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
from matplotlib.font_manager import FontProperties
from datetime import datetime

# ── Nature figure dimensions ──────────────────────────────────────────────────
# Nature single column: 89 mm = 3.504 in; height usually 0.75–0.9 × width
NW = 3.504   # inches (single column)
NH = 8.5     # inches (tall 3-panel)

DPI = 600
OUTPUT_DIR = "/home/agentuser/.hermes/papers/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load CSI 300 data (re-use cached Tushare data) ───────────────────────────
import os
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

    # Garman-Klass RV
    df['log_co'] = np.log(df['close'] / df['open'])
    df['rv_gk'] = (0.5 * (np.log(df['high'] / df['low'])**2)
                   - (2*np.log(2) - 1) * (df['log_co']**2)) * 252
    # Parkinson HL RV
    df['rv_hl'] = (np.log(df['high'] / df['low'])**2) / (4*np.log(2)) * 252
    df['rv'] = df['rv_gk']
    df['return'] = np.log(df['close'] / df['close'].shift(1))
    return df[['rv', 'rv_gk', 'rv_hl', 'return', 'close']].dropna()

# ── HAR model ────────────────────────────────────────────────────────────────
def fit_har(rv_df):
    import statsmodels.api as sm
    df = rv_df.copy()
    df['log_rv']   = np.log(df['rv'])
    df['log_rv_1'] = df['log_rv'].shift(1)
    df['log_rv_5'] = df['log_rv'].rolling(5).mean().shift(1)
    df['log_rv_22']= df['log_rv'].rolling(22).mean().shift(1)
    df_har = df[['log_rv', 'log_rv_1', 'log_rv_5', 'log_rv_22']].dropna()
    X = sm.add_constant(df_har[['log_rv_1', 'log_rv_5', 'log_rv_22']])
    model = sm.OLS(df_har['log_rv'], X).fit()
    fitted_index = df_har.index
    df.loc[fitted_index, 'sigma_har'] = np.exp(0.5 * model.fittedvalues.values)
    return model, df

# ── Nature font setup ────────────────────────────────────────────────────────
def nature_font():
    # Liberation Sans is metric-compatible with Arial — Nature-compliant
    plt.rcParams['font.family'] = 'Liberation Sans'
    plt.rcParams['font.sans-serif'] = ['Liberation Sans']
    plt.rcParams.update({
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'lines.linewidth': 0.8,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 2,
        'ytick.major.size': 2,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_nature_figure():
    nature_font()

    fig = plt.figure(figsize=(NW, NH))

    # Layout: panel (a) taller, (b)(c) share bottom half
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.6, 1],
                           hspace=0.45)

    # ── Panel (a): RV time series ─────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0])

    csi_df = load_csi300()
    model, csi_enriched = fit_har(csi_df)

    rv = csi_df['rv'] * 100
    sigma = csi_enriched['sigma_har'] * 100

    ax_a.fill_between(rv.index, 0, rv.values,
                       color='#D0D0D0', alpha=0.35, linewidth=0)
    ax_a.plot(rv.index, rv.values,
              color='#404040', linewidth=0.6, alpha=0.9, label='RV (GK, annualised)')
    ax_a.plot(sigma.index, sigma.values,
              color='#000000', linewidth=0.9, linestyle='--',
              alpha=0.9, label='HAR σ̂_t')

    # Crisis shading (RV > 80th pct)
    p80 = rv.quantile(0.80)
    for i in range(len(rv)-1):
        if rv.iloc[i] > p80:
            ax_a.axvspan(rv.index[i], rv.index[i+1],
                         color='#E53935', alpha=0.12, linewidth=0)

    ax_a.set_ylabel('Annualised volatility (%)', fontsize=7)
    ax_a.set_xlabel('Date', fontsize=7)
    ax_a.set_xlim(rv.index.min(), rv.index.max())
    ymax = rv.quantile(0.98) * 1.1
    ax_a.set_ylim(0, ymax)
    ax_a.legend(loc='upper right', frameon=False, fontsize=6,
                handlelength=1.5, handletextpad=0.4)

    # Panel label
    ax_a.text(0.01, 0.95, 'a', transform=ax_a.transAxes,
              fontweight='bold', fontsize=8, va='top')

    # ── Panels (b)(c): side by side within bottom half ───────────────────────
    gs_bottom = gridspec.GridSpec(1, 2,
                                  wspace=0.35,
                                  left=0.13, right=0.95,
                                  top=0.96, bottom=0.08)
    ax_b = fig.add_subplot(gs_bottom[0])
    ax_c = fig.add_subplot(gs_bottom[1])

    # ── Panel (b): HAR coefficients ─────────────────────────────────────────
    m = model
    coeffs = {
        'β₁ (day)':   m.params.get('log_rv_1',  0),
        'β₅ (week)':  m.params.get('log_rv_5',  0),
        'β₂₂ (month)':m.params.get('log_rv_22', 0),
    }
    bars = list(coeffs.values())
    x_pos = np.arange(len(bars))
    bar_colors = ['#404040', '#808080', '#C0C0C0']
    ax_b.bar(x_pos, bars, color=bar_colors, width=0.6, linewidth=0.5)

    for i, (k, v) in enumerate(coeffs.items()):
        ax_b.text(i, v + 0.01, f'{v:.3f}',
                  ha='center', va='bottom', fontsize=6)

    ax_b.axhline(0, color='black', linewidth=0.5)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(list(coeffs.keys()), fontsize=6)
    ax_b.set_ylabel('Coefficient', fontsize=7)
    ax_b.set_title('HAR model coefficients', fontsize=7, pad=3)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.tick_params(axis='both', which='major', labelsize=6)
    ax_b.text(0.01, 0.95, 'b', transform=ax_b.transAxes,
              fontweight='bold', fontsize=8, va='top')

    # ── Panel (c): GK vs HL scatter ──────────────────────────────────────────
    gk = csi_df['rv_gk'].dropna() * 100
    hl = csi_df['rv_hl'].dropna() * 100
    common = gk.index.intersection(hl.index)
    gk_a, hl_a = gk.loc[common], hl.loc[common]

    ax_c.scatter(gk_a, hl_a, s=3, color='#404040', alpha=0.4, linewidths=0)

    max_val = max(gk_a.max(), hl_a.max())
    ax_c.plot([0, max_val], [0, max_val],
              color='black', linewidth=0.6, linestyle='--', alpha=0.7, label='1:1')

    if len(gk_a) > 10:
        z = np.polyfit(gk_a, hl_a, 1)
        x_l = np.linspace(gk_a.min(), gk_a.max(), 100)
        ax_c.plot(x_l, np.polyval(z, x_l), color='black', linewidth=0.7,
                  label=f'OLS slope={z[0]:.2f}')

    ax_c.set_xlabel('Garman–Klass RV (%)', fontsize=7)
    ax_c.set_ylabel('Parkinson HL RV (%)', fontsize=7)
    ax_c.set_title('GK vs Parkinson RV', fontsize=7, pad=3)
    ax_c.legend(loc='upper left', frameon=False, fontsize=6,
                handlelength=1.2, handletextpad=0.3)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.tick_params(axis='both', which='major', labelsize=6)
    ax_c.text(0.01, 0.95, 'c', transform=ax_c.transAxes,
              fontweight='bold', fontsize=8, va='top')

    # ── Save ──────────────────────────────────────────────────────────────────
    base = f"{OUTPUT_DIR}/realized_garch_nature"
    for ext, dpi in [('.png', DPI), ('.pdf', DPI)]:
        path = f"{base}{ext}"
        plt.savefig(path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        pass  # quiet

    plt.close()

if __name__ == '__main__':
    plot_nature_figure()
