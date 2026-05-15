#!/usr/bin/env python3
"""DCC-GARCH Daily Report Runner — generates report and chart"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, json, os, sys
from datetime import datetime, timedelta
from arch import arch_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Step 1: Fetch and save data ─────────────────────────────────────────────
from dcc_garch import get_tushare_data, get_twelve_data

ASSETS = [
    ('CSI 300', '000300.SH'),
    ('EURUSD', 'EUR/USD'),
    ('USDCNH', 'USD/CNH'),
    ('BTCUSD', 'BTC/USD'),
]

end = datetime.today().strftime('%Y-%m-%d')
start = (datetime.today() - timedelta(days=900)).strftime('%Y-%m-%d')

raw_data = {}
for idx, (ticker, code) in enumerate(ASSETS):
    series = None
    if ticker == 'CSI 300':
        series = get_tushare_data('000300.SH', start, end)
    if series is None or len(series) < 100:
        series = get_twelve_data(code, start, end)
    if series is not None and len(series) > 100:
        raw_data[ticker] = series.dropna()
    import time; time.sleep(5)

if len(raw_data) < 2:
    with open('/tmp/dcc_daily_report.txt', 'w') as f:
        f.write('INSUFFICIENT DATA')
    sys.exit(1)

df = pd.DataFrame(raw_data).sort_index().dropna()
name_map = {'CSI 300': 'CSI300', 'EURUSD': 'EURUSD', 'BTCUSD': 'BTCUSD', 'USDCNH': 'USDCNH'}
df.columns = [name_map.get(c, c) for c in df.columns]
df.to_csv('/tmp/dcc_data.csv')
print(f'Data: {df.shape}  {df.index[0].date()} → {df.index[-1].date()}')

# ── Step 2: DCC-GARCH ─────────────────────────────────────────────────────────
ASSET_NAMES = ['CSI300', 'EURUSD', 'BTCUSD', 'USDCNH']

rets = df.pct_change().dropna()
rets = rets[abs(rets) < 0.25]
T = len(rets)
N = len(ASSET_NAMES)

garch_vol = {}
std_resid = {}

for name in ASSET_NAMES:
    if name not in rets.columns:
        continue
    r = rets[name].values * 100
    if len(r) < 30:
        continue
    model = arch_model(r, vol='Garch', p=1, q=1, dist='t', rescale=False)
    res = model.fit(disp='off', show_warning=False)
    vol = res.conditional_volatility / 100
    sr = r / 100 / vol
    min_len = min(len(vol), T)
    garch_vol[name] = vol[-min_len:]
    std_resid[name] = sr[-min_len:]
    print(f'GARCH OK: {name}')

min_len = min(len(v) for v in garch_vol.values())
eps_matrix = np.column_stack([std_resid[n][-min_len:] for n in ASSET_NAMES if n in garch_vol])
aligned_dates = rets.index[-min_len:]
aligned_vols = {n: garch_vol[n][-min_len:] for n in ASSET_NAMES if n in garch_vol}
print(f'Eps: {eps_matrix.shape}')

# DCC MLE
def estimate_dcc(eps):
    T, N = eps.shape
    def neg_ll(params):
        a, b = params
        if a <= 0 or b <= 0 or a + b >= 1:
            return 1e10
        Q_bar = np.cov(eps.T)
        Q_t = Q_bar.copy()
        ll = 0.0
        for t in range(1, T):
            e_t = eps[t].reshape(-1, 1)
            e_prev = eps[t-1].reshape(-1, 1)
            Q_t = (1 - a - b) * Q_bar + a * (e_prev @ e_prev.T) + b * Q_t
            d = np.diag(1.0 / np.sqrt(np.diag(Q_t) + 1e-6))
            R_t = d @ Q_t @ d
            R_t = R_t + np.eye(N) * 1e-6
            try:
                sign, logdet = np.linalg.slogdet(R_t)
                if sign <= 0:
                    ll -= 1e5
                else:
                    ll -= 0.5 * logdet
                    inv = np.linalg.inv(R_t)
                    ll -= 0.5 * (e_t.T @ inv @ e_t)[0, 0]
            except:
                ll -= 1e5
        return -ll

    from scipy.optimize import minimize
    res = minimize(neg_ll, [0.05, 0.90], bounds=[(0.001, 0.3), (0.5, 0.999)])
    a, b = res.x
    Q_bar = np.cov(eps.T)
    Q_t = Q_bar.copy()
    Rt_series = []
    for t in range(1, T):
        e_prev = eps[t-1].reshape(-1, 1)
        Q_t = (1 - a - b) * Q_bar + a * (e_prev @ e_prev.T) + b * Q_t
        d = np.diag(1.0 / np.sqrt(np.diag(Q_t) + 1e-6))
        R_t = d @ Q_t @ d
        R_t = R_t + np.eye(N) * 1e-6
        Rt_series.append(R_t.copy())
    return a, b, Q_bar, Rt_series

a_opt, b_opt, Q_bar, Rt_series = estimate_dcc(eps_matrix)
print(f'DCC: a={a_opt:.4f}  b={b_opt:.4f}  a+b={a_opt+b_opt:.4f}')

# Save params
params = {'a_opt': float(a_opt), 'b_opt': float(b_opt),
           'garch_omega': 0.0, 'garch_alpha': 0.0, 'garch_beta': 0.0}
with open('/tmp/dcc_params.json', 'w') as f:
    json.dump(params, f, indent=2)

# Pairs
last_Rt = Rt_series[-1]
pairs = []
for i in range(N):
    for j in range(i+1, N):
        pairs.append((f'{ASSET_NAMES[i]}/{ASSET_NAMES[j]}', last_Rt[i, j]))

# Annualized vols
today_vol = {}
for name in ASSET_NAMES:
    if name in aligned_vols and len(aligned_vols[name]) > 0:
        today_vol[name] = aligned_vols[name][-1] * np.sqrt(252) * 100

# ── Step 3: Build report ─────────────────────────────────────────────────────
date_str = datetime.today().strftime('%Y-%m-%d')

def emoji_vol(v):
    return 'RED' if v > 30 else 'YLW' if v > 18 else 'GRN'

def emoji_rho(rho):
    return 'RED' if abs(rho) > 0.4 else 'YLW' if abs(rho) > 0.15 else 'GRN'

lines = [
    "GARCH Quant - Morning Volatility Report",
    f"{date_str} 08:55 UTC+8",
    "=" * 40,
    "1. Annualized Volatility (%)",
]
for asset in ASSET_NAMES:
    if asset in today_vol:
        v = today_vol[asset]
        e = emoji_vol(v)
        lines.append(f"[{e}] {asset}: {v:.1f}%")

lines += ['', "2. DCC Correlation Snapshot"]
for name, rho in pairs:
    e = emoji_rho(rho)
    lines.append(f"[{e}] {name}: {rho:.3f}")

last_ret = rets.iloc[-1]
mover = last_ret.abs().max()
mover_asset = last_ret.abs().idxmax()
mover_val = last_ret[mover_asset]
if mover > 2:
    lines += ['', "3. Alert - Large Move Detected",
              f"WARNING: {mover_asset} moved {mover_val:+.2f}% - hedge check triggered"]

lines += ['', "GARCH(1,1) + DCC(1,1) | Auto-update 08:55 daily"]
report = '\n'.join(lines)

with open('/tmp/dcc_daily_report.txt', 'w') as f:
    f.write(report)
print('REPORT SAVED')
print()
print(report)

# ── Step 4: Build chart ──────────────────────────────────────────────────────
OUTPUT_DIR = '/home/agentuser/.hermes/papers/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 8,
    'axes.spines.right': False, 'axes.spines.top': False,
    'axes.linewidth': 0.8, 'legend.frameon': False,
})

ASSET_DISPLAY = ['CSI 300', 'EUR/USD', 'BTC/USD', 'USD/CNH']
COLS = ['#0F4D92', '#E07B39', '#1a3a5c', '#27ae60']

fig, ax = plt.subplots(figsize=(7, 3.5))
for k, name in enumerate(ASSET_NAMES):
    if name in aligned_vols and len(aligned_vols[name]) > 0:
        vs = aligned_vols[name] * np.sqrt(252) * 100
        ax.plot(vs.index, vs.values, color=COLS[k], lw=1.8, label=ASSET_DISPLAY[k], alpha=0.9)

ax.set_ylabel('Annualized Volatility (%)', fontsize=8)
ax.set_xlabel('Date', fontsize=8)
ax.legend(fontsize=7, loc='upper right', frameon=False, handlelength=1.2, handletextpad=0.5)
ax.tick_params(axis='both', labelsize=7)
ax.set_title('GARCH(1,1) Conditional Volatility - 90-Day Window', fontsize=9, fontweight='bold', pad=6)
ax.set_facecolor('#fafafa')
fig.patch.set_facecolor('white')
fig.tight_layout(pad=1.5)

chart_path = f'{OUTPUT_DIR}/dcc_daily_chart.png'
fig.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Chart saved: {chart_path}')
