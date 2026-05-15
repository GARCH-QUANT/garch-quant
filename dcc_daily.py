"""
DCC-GARCH Daily Morning Report
- Pushes volatility briefing at 8:55 every trading day
- Triggers extra report if any asset moves >2%
"""
import json, os, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from arch import arch_model
from datetime import datetime

# ── Load saved DCC params ────────────────────────────────
with open('/tmp/dcc_params.json') as f:
    PARAMS = json.load(f)

ASSETS = ['CSI300', 'EURUSD', 'BTCUSD', 'USDCNH']
DCC_A = PARAMS['a_opt']
DCC_B = PARAMS['b_opt']

CSV_PATH = '/tmp/dcc_data.csv'

# ── Load & prep price series ─────────────────────────────
def load_prices():
    df = pd.read_csv(CSV_PATH, index_col='date', parse_dates=True)
    price_cols = ['CSI300', 'EURUSD', 'BTCUSD', 'USDCNH']
    prices = df[price_cols].dropna().sort_index()
    return prices

# ── Fit GARCH on full history → get conditional vol series ──
def fit_garch_full(prices):
    rets = (prices.pct_change().dropna() * 100).replace([np.inf, -np.inf], np.nan).dropna()
    rets = rets[abs(rets) < 30]
    vol_series = {}
    for asset in ASSETS:
        if asset in rets.columns:
            r = rets[asset].dropna().values
            if len(r) < 30:
                continue
            model = arch_model(r, vol='Garch', p=1, q=1, dist='t', rescale=False)
            res = model.fit(disp='off')
            cond_vol = res.conditional_volatility  # in pct units
            dates_idx = rets.index[len(rets)-len(r):]
            vol_series[asset] = pd.Series(cond_vol, index=dates_idx)
    return rets, vol_series

# ── Build report ─────────────────────────────────────────
def build_report(prices, rets, vol_series, is_trigger=False):
    omega = PARAMS['garch_omega']
    alpha = PARAMS['garch_alpha']
    beta = PARAMS['garch_beta']

    last_ret = rets.iloc[-1]

    # 1-step-ahead vol using saved params
    today_vol = {}
    for asset in ASSETS:
        if asset in vol_series and len(vol_series[asset]) > 0:
            today_vol[asset] = vol_series[asset].iloc[-1] * np.sqrt(252)

    # Std residuals for DCC
    std_resid = {}
    for asset in ASSETS:
        if asset in rets.columns and asset in vol_series and len(vol_series[asset]) > 0:
            vol_arr = vol_series[asset].values
            r_arr = rets[asset].dropna().values
            # align lengths
            min_len = min(len(vol_arr), len(r_arr))
            std_resid[asset] = r_arr[-min_len] / (vol_arr[-min_len] + 1e-10)

    # Build DCC Q_bar from full-sample eps matrix
    available = [a for a in ASSETS if a in std_resid]
    if len(available) < 2:
        return None

    # Long-run Q_bar = uncorrelated covariance matrix of std resids
    eps_arrays = []
    for a in available:
        r = rets[a].dropna().values
        if a in vol_series:
            v = vol_series[a].values
            min_l = min(len(r), len(v))
            sr = r[-min_l:] / (v[-min_l:] + 1e-10)
            eps_arrays.append(sr)
    min_L = min(len(e) for e in eps_arrays)
    eps_matrix = np.column_stack([e[-min_L:] for e in eps_arrays])
    Q_bar = np.cov(eps_matrix.T)

    # 1-step DCC update
    pairs = []
    for i in range(len(available)):
        for j in range(i+1, len(available)):
            rho = dcc_step(std_resid[available[i]], std_resid[available[j]],
                           Q_bar[i, i], Q_bar[j, j], Q_bar[i, j], DCC_A, DCC_B)
            pairs.append((f"{available[i]}/{available[j]}", rho))

    date_str = datetime.today().strftime('%Y-%m-%d')
    prefix = "🚨 *事件触发 · 市场异动专报*" if is_trigger else "🌅 *GARCH Quant · 早盘波动率简报*"

    lines = [prefix, f"_{date_str} 08:55 UTC+8_", "─────────────────────",
             "**一、隐含波动率（年化）**"]
    for asset in ASSETS:
        if asset in today_vol:
            v = today_vol[asset]
            emoji = "🔴" if v > 30 else "🟡" if v > 18 else "🟢"
            lines.append(f"{emoji} {asset}: **{v:.1f}%**")

    lines += ["", "**二、DCC 相关性快照**"]
    for name, rho in pairs:
        sig = "🔴" if abs(rho) > 0.4 else "🟡" if abs(rho) > 0.15 else "🟢"
        lines.append(f"{sig} {name}: **{rho:.3f}**")

    mover = last_ret.abs().max()
    mover_asset = last_ret.abs().idxmax()
    mover_val = last_ret[mover_asset]
    if mover > 2:
        lines += ["", "**三、异动预警**",
                 f"⚠️ {mover_asset} 单日变动 **{mover_val:+.2f}%**，已触发对冲建议检查"]

    lines += ["", "*GARCH(1,1) + DCC(1,1) · 每交易日 08:55 自动更新*"]
    return "\n".join(lines)

def dcc_step(eps1, eps2, var1, var2, cov12, a, b):
    """Single-step DCC update for a pair."""
    Q_bar_ij = cov12 / (np.sqrt(var1) * np.sqrt(var2) + 1e-10)
    e = np.array([eps1, eps2]).reshape(2, 1)
    Q11 = (1 - a - b) * var1 + a * eps1**2 + b * var1
    Q22 = (1 - a - b) * var2 + a * eps2**2 + b * var2
    Q12 = (1 - a - b) * cov12 + a * eps1 * eps2 + b * cov12
    denom = np.sqrt(Q11 * Q22 + 1e-10)
    return float(Q12 / denom)

#!/usr/bin/env python3
"""
GARCH Quant · 早盘波动率图表
Nature-style publication figure
"""
import warnings, os
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = '/home/agentuser/.hermes/papers/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Nature figure constants ─────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

# ── Nature palette ─────────────────────────────────────────
PALETTE = {
    "blue_main":    "#0F4D92",   # CSI300
    "orange":       "#E07B39",   # EURUSD
    "dark_blue":    "#1a3a5c",   # BTCUSD
    "green":        "#27ae60",   # USDCNH
}
COLORS = [PALETTE["blue_main"], PALETTE["orange"], PALETTE["dark_blue"], PALETTE["green"]]
ASSET_NAMES = ["CSI 300", "EUR/USD", "BTC/USD", "USD/CNH"]


def apply_publication_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })


def build_chart(vol_series, output_path):
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(7, 3.5))

    for k, asset in enumerate(['CSI300', 'EURUSD', 'BTCUSD', 'USDCNH']):
        if asset in vol_series and len(vol_series[asset]) > 0:
            vs = vol_series[asset] * np.sqrt(252) * 100
            dates = vs.index
            ax.plot(dates, vs.values,
                    color=COLORS[k], lw=1.8,
                    label=ASSET_NAMES[k], alpha=0.9)

    # Axis labels
    ax.set_ylabel("Annualized Volatility (%)", fontsize=8)
    ax.set_xlabel("Date", fontsize=8)

    # Legend outside
    ax.legend(fontsize=7, loc="upper right", frameon=False,
              handlelength=1.2, handletextpad=0.5)

    # Tick size
    ax.tick_params(axis='both', labelsize=7)

    # Title
    ax.set_title("GARCH(1,1) Conditional Volatility — 90-Day Window",
                 fontsize=9, fontweight='bold', pad=6)

    # No grid for Nature style — sparse y-ticks only
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")

    fig.tight_layout(pad=1.5)

    # Save SVG primary + PNG secondary
    base = output_path.rsplit('.', 1)[0]
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    pass  # quiet


if __name__ == '__main__':
    # Import and run data pipeline
    import sys
    sys.path.insert(0, '/home/agentuser/.hermes/scripts')
    from dcc_daily import load_prices, fit_garch_full

    prices = load_prices()
    rets, vol_series = fit_garch_full(prices)
    chart_path = '/home/agentuser/.hermes/papers/figures/dcc_daily_chart.png'
    build_chart(vol_series, chart_path)

# ── Main ─────────────────────────────────────────────────
if __name__ == '__main__':
    prices = load_prices()
    rets, vol_series = fit_garch_full(prices)
    report = build_report(prices, rets, vol_series, is_trigger=False)

    if report is None:
        pass  # quiet
    else:
        with open('/tmp/dcc_daily_report.txt', 'w') as f:
            f.write(report)
        os.makedirs('/home/agentuser/.hermes/papers/figures', exist_ok=True)
        chart_path = '/home/agentuser/.hermes/papers/figures/dcc_daily_chart.png'
        build_chart(vol_series, chart_path)
        pass  # quiet
