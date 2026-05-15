#!/usr/bin/env python3.12
"""
DCC-GARCH 多资产动态条件相关 + 最优对冲比例
==============================================
Stage 1: Univariate GARCH(1,1) for each asset
Stage 2: DCC(1,1) on standardized residuals
Output: Dynamic correlations, hedge ratios, correlation heatmap
"""

import warnings, os, io, time, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta

# ── Config ─────────────────────────────────────────────────
OUTPUT_DIR = "/home/agentuser/.hermes/papers/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLR_BG   = "#0D1117"
CLR_TEXT = "#E6EDF3"
CLR_MUTED= "#8B949E"
CLR_GRID = "#21262D"

plt.rcParams.update({
    "axes.facecolor"   : CLR_BG,
    "figure.facecolor" : CLR_BG,
    "axes.edgecolor"   : "#30363D",
    "axes.labelcolor"  : CLR_TEXT,
    "text.color"       : CLR_TEXT,
    "xtick.color"      : CLR_MUTED,
    "ytick.color"      : CLR_MUTED,
    "grid.color"       : CLR_GRID,
    "grid.alpha"       : 0.5,
    "font.family"      : "sans-serif",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

import os
TWELVE_DATA_KEY = os.environ.get("TWELVE_DATA_KEY", "")
TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID         = os.environ.get("TELEGRAM_CHANNEL_ID", "-1003786012521")
TUSHARE_TOKEN   = os.environ.get("TUSHARE_TOKEN", "")

# ── Color palette for assets ────────────────────────────────
CLR_ASSETS = {
    "CSI 300":   "#58A6FF",
    "EURUSD":    "#F0C040",
    "USDCNH":    "#3FB950",
    "BTCUSD":    "#F78166",
}

# ── Data Fetch ────────────────────────────────────────────
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
            return df.dropna()
        else:
            pass  # quiet
    except Exception as e:
        pass  # quiet
    return None

def get_twelve_data(symbol, start, end, interval="1day"):
    import requests
    for attempt in range(3):
        try:
            url = "https://api.twelvedata.com/time_series"
            params = {"symbol": symbol, "interval": interval, "outputsize": 500,
                      "format": "JSON", "apikey": TWELVE_DATA_KEY}
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
            if "values" not in data:
                import time; time.sleep(5)
                continue
            vals = data["values"]
            records = [(v["datetime"], float(v["close"])) for v in reversed(vals)]
            idx = pd.to_datetime([r[0] for r in records])
            return pd.Series([r[1] for r in records], index=idx, name=symbol)
        except:
            import time; time.sleep(5)
    return None

def fetch_data(display_name, api_code):
    """display_name: 'CSI 300', 'EURUSD', etc.
       api_code: '000300.SH' for Tushare, 'EUR/USD' for Twelve Data"""
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=900)).strftime("%Y-%m-%d")

    if display_name == "CSI 300":
        s = get_tushare_data(api_code, start, end)
        if s is not None and len(s) > 100:
            return s.dropna()
        return None

    s = get_twelve_data(api_code, start, end)
    if s is not None and len(s) > 100:
        return s.dropna()
    return None

# ── DCC-GARCH Implementation ──────────────────────────────
def garch_standardize(resids, vol):
    """Standardize residuals by conditional vol"""
    return resids / np.sqrt(vol)

def dcc_likelihood(a, b, q_bar, qt):
    """
    DCC log-likelihood.
    qt = (1-a-b)*q_bar + a*eps_{t-1}*eps_{t-1}' + b*q_{t-1}
    """
    C = 1e-6  # numerical stability
    q_t = (1 - a - b) * q_bar + a * qt + b * qt
    # Normalize to correlation
    d = np.diag(1.0 / np.sqrt(np.diag(q_t) + C))
    R_t = d @ q_t @ d
    R_t = R_t + np.eye(len(R_t)) * C  # ensure positive definite
    return q_t, R_t

def estimate_dcc(eps):
    """
    Estimate DCC(1,1) parameters a, b via two-stage MLE.
    eps: T x N matrix of standardized residuals from univariate GARCH
    Returns: a, b, Q_bar, Q_T (time-varying correlation matrices)
    """
    T, N = eps.shape
    pass  # quiet

    # Sample splits for numerical stability
    def dcc_neg_ll(params):
        a, b = params
        if a <= 0 or b <= 0 or a + b >= 1:
            return 1e10

        Q_bar = np.cov(eps.T)  # unconditional covariance of eps
        Q_t = Q_bar.copy()

        ll = 0.0
        for t in range(1, T):
            e_t = eps[t].reshape(-1, 1)
            e_prev = eps[t-1].reshape(-1, 1)
            Q_t = (1 - a - b) * Q_bar + a * (e_prev @ e_prev.T) + b * Q_t
            # Normalize to correlation
            d = np.diag(1.0 / np.sqrt(np.diag(Q_t) + 1e-6))
            R_t = d @ Q_t @ d
            # Add small ridge for stability
            R_t = R_t + np.eye(N) * 1e-6
            try:
                L = np.linalg.cholesky(R_t)
                maha = float(e_t.T @ np.linalg.inv(R_t + np.eye(N)*1e-6) @ e_t)
                ll -= 0.5 * (np.log(np.linalg.det(R_t) + 1e-10) + maha)
            except:
                ll -= 0.5 * N  # penalty

        return -ll  # negative because we minimize

    # Grid search + refine
    from scipy.optimize import minimize, differential_evolution

    bounds = [(0.001, 0.3), (0.5, 0.98)]
    result = differential_evolution(dcc_neg_ll, bounds, seed=42,
                                    maxiter=200, polish=True, workers=1)
    a_opt, b_opt = result.x
    pass  # quiet

    # Recompute full Qt series with optimal params
    Q_bar = np.cov(eps.T)
    Q_t = Q_bar.copy()
    Qt_series = [Q_t.copy()]
    Rt_series = []

    for t in range(1, T):
        e_prev = eps[t-1].reshape(-1, 1)
        Q_t = (1 - a_opt - b_opt) * Q_bar + a_opt * (e_prev @ e_prev.T) + b_opt * Q_t
        d = np.diag(1.0 / np.sqrt(np.diag(Q_t) + 1e-6))
        R_t = d @ Q_t @ d
        R_t = R_t + np.eye(N) * 1e-6
        Qt_series.append(Q_t.copy())
        Rt_series.append(R_t.copy())

    return a_opt, b_opt, Q_bar, Qt_series, Rt_series

def compute_hedge_ratio(r_hedge, r_asset, h_asset, h_hedge, rho):
    """
    Optimal hedge ratio: h = cov(r_asset, r_hedge) / var(r_hedge)
    Using DCC correlation: h = rho * sqrt(h_asset) / sqrt(h_hedge)
    """
    return rho * np.sqrt(h_asset) / np.sqrt(h_hedge)

# ── Plotting ──────────────────────────────────────────────
def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=CLR_BG, edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_correlation_heatmap(Rt_series, dates, asset_names, title_prefix=""):
    """Plot a grid of pairwise dynamic correlations"""
    N = len(asset_names)
    n_pairs = N * (N - 1) // 2
    fig, axes = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 4),
                              sharex=True)
    if n_pairs == 1:
        axes = [axes]

    idx = 0
    pair_colors = ["#58A6FF", "#F78166", "#3FB950", "#BC8CFF", "#F0C040", "#FF6B9D"]
    for i in range(N):
        for j in range(i+1, N):
            ax = axes[idx]
            corr = np.array([Rt_series[t][i, j] for t in range(len(Rt_series))])
            ax.fill_between(dates, corr, alpha=0.3, color=pair_colors[idx])
            ax.plot(dates, corr, color=pair_colors[idx], lw=1.0)
            ax.set_title(f"{asset_names[i]} / {asset_names[j]}", fontsize=10)
            ax.set_ylim(-0.3, 1.05)
            ax.axhline(0, color=CLR_MUTED, lw=0.5, ls="--", alpha=0.5)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor(CLR_BG)
            ax.tick_params(labelsize=8)
            idx += 1

    fig.suptitle(f"{title_prefix}Dynamic Conditional Correlations — DCC-GARCH",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig_to_buf(fig)

def plot_hedge_ratios(rhos, h_asset, h_hedge, dates, pair_names, title_prefix=""):
    """Plot optimal hedge ratios over time"""
    n_pairs = len(pair_names)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(12, 3 * n_pairs),
                              sharex=True)
    if n_pairs == 1:
        axes = [axes]

    colors = ["#58A6FF", "#F78166", "#3FB950", "#BC8CFF", "#F0C040"]

    for k, (ax, name) in enumerate(zip(axes, pair_names)):
        # Hedge ratio: h_ij = rho_ij * sqrt(h_j) / sqrt(h_i)
        # Position in asset j needed to hedge 1 unit of asset i
        hr = rhos[:, k]
        ax.fill_between(dates, hr, alpha=0.3, color=colors[k])
        ax.plot(dates, hr, color=colors[k], lw=1.0)
        ax.set_ylabel("Hedge Ratio", fontsize=9)
        ax.set_title(f"Hedge 1 unit of {name.split(' / ')[0]} with {name.split(' / ')[1]}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor(CLR_BG)
        ax.tick_params(labelsize=8)
        # Add mean line
        ax.axhline(np.nanmean(hr), color="white", lw=1, ls="--", alpha=0.6)
        ax.annotate(f"Avg={np.nanmean(hr):.3f}", xy=(0.97, 0.85),
                    xycoords="axes fraction", fontsize=8,
                    ha="right", color=CLR_TEXT,
                    bbox=dict(boxstyle='round', facecolor=CLR_BG, alpha=0.7))

    axes[-1].set_xlabel("Date", fontsize=10)
    fig.suptitle(f"{title_prefix}Optimal Hedge Ratios — DCC-GARCH",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig_to_buf(fig)

def plot_vol_comparison(vols_dict, dates, asset_names, title_prefix=""):
    """Compare GARCH volatilities of different assets"""
    N = len(asset_names)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#58A6FF", "#F0C040", "#F78166", "#3FB950", "#BC8CFF"]
    for k, name in enumerate(asset_names):
        vol_ann = np.sqrt(vols_dict[name]) * np.sqrt(252) * 100
        ax.plot(dates, vol_ann, color=colors[k], lw=1.2, label=name, alpha=0.9)

    ax.set_ylabel("Annualized Volatility (%)", fontsize=11)
    ax.set_title(f"{title_prefix}GARCH(1,1) Conditional Volatilities",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(CLR_BG)
    plt.tight_layout()
    return fig_to_buf(fig)

def plot_correlation_summary_table(rhos, dates, pair_names, a, b):
    """Create a summary table figure of average correlations"""
    n = len(pair_names)
    means = np.nanmean(rhos, axis=0)
    stds  = np.nanstd(rhos, axis=0)
    mins  = np.nanmin(rhos, axis=0)
    maxs  = np.nanmax(rhos, axis=0)

    # Recent = last 60 days
    recent = rhos[-60:]
    recent_means = np.nanmean(recent, axis=0)

    fig, ax = plt.subplots(figsize=(12, 3 + n * 0.6))
    ax.axis('off')
    ax.set_facecolor(CLR_BG)

    headers = ["Pair", "Avg ρ", "Std", "Min", "Max", "Last 60d Avg", "Signal"]
    col_w = [0.28, 0.10, 0.10, 0.10, 0.10, 0.14, 0.18]

    def draw_cell(ax, x, y, w, h, text, facecolor, textcolor, fontsize=9, bold=False):
        rect = plt.Rectangle((x, y), w, h, transform=ax.transAxes,
                              facecolor=facecolor, edgecolor="#30363D", lw=0.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, transform=ax.transAxes,
                ha='center', va='center', fontsize=fontsize,
                color=textcolor, fontweight='bold' if bold else 'normal')

    total_w = sum(col_w)
    for col, (hdr, cw) in enumerate(zip(headers, col_w)):
        x = sum(col_w[:col]) / total_w
        draw_cell(ax, x, 0.72, cw/total_w, 0.2, hdr, "#1E3A5F", "white", 9, True)

    for row in range(n):
        y = 0.72 - (row + 1) * 0.2
        fc = "#1A1A2E" if row % 2 == 0 else "#16213E"
        # Pair name
        x = 0
        draw_cell(ax, x, y, col_w[0]/total_w, 0.2, pair_names[row], fc, CLR_TEXT, 8)
        # Stats
        vals = [f"{means[row]:.3f}", f"{stds[row]:.3f}", f"{mins[row]:.3f}",
                f"{maxs[row]:.3f}", f"{recent_means[row]:.3f}"]
        for col, val in enumerate(vals, 1):
            x = sum(col_w[:col]) / total_w
            draw_cell(ax, x, y, col_w[col]/total_w, 0.2, val, fc, CLR_TEXT, 8)
        # Signal
        signal = "🔴 High ρ" if recent_means[row] > 0.6 else \
                 "🟡 Med ρ" if recent_means[row] > 0.2 else "🟢 Low ρ"
        draw_cell(ax, sum(col_w[:-1])/total_w, y, col_w[-1]/total_w, 0.2,
                  signal, fc, CLR_TEXT, 8)

    # DCC params annotation
    fig.text(0.99, 0.01,
             f"DCC params: a={a:.4f}, b={b:.4f} | a+b={a+b:.4f}",
             ha='right', va='bottom', fontsize=8, color=CLR_MUTED)

    fig.suptitle("DCC-GARCH Correlation Summary", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig_to_buf(fig)

# ── Main ───────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("DCC-GARCH — Multi-Asset Dynamic Conditional Correlation")
    print("=" * 60)

    ASSETS = [
        ("CSI 300", "000300.SH"),
        ("EURUSD",  "EUR/USD"),
        ("USDCNH",  "USD/CNH"),
        ("BTCUSD",  "BTC/USD"),
    ]

    # ── Step 1: Load or fetch data ─────────────────────────
    csv_path = "/tmp/dcc_data.csv"
    import os
    if os.path.exists(csv_path):
        print(f"\n[1/8] Loading cached data from {csv_path}")
        df_csv = pd.read_csv(csv_path, index_col='date', parse_dates=True)
        raw_data = {}
        for col in ['CSI300', 'EURUSD', 'BTCUSD', 'USDCNH']:
            if col in df_csv.columns:
                raw_data[col] = df_csv[col].dropna()
        if len(raw_data) >= 2:
            print(f"    Loaded {len(raw_data)} assets from CSV")
        else:
            raw_data = {}
    else:
        raw_data = {}

    # Fallback: fetch if CSV not available
    if len(raw_data) < 2:
        print("\n[1/8] CSV missing or sparse — fetching data live...")
        for idx, (display_name, api_code) in enumerate(ASSETS):
            print(f"    Fetching {display_name}...")
            series = fetch_data(display_name, api_code)
            if series is not None and len(series) > 50:
                raw_data[display_name] = series
                print(f"      ✓ {display_name}: {len(series)} rows")
            else:
                print(f"      ✗ {display_name}: failed")
            if idx < len(ASSETS) - 1:
                import time; time.sleep(6)

    if len(raw_data) < 2:
        print("[ERROR] Less than 2 assets available. Aborting.")
        return None

    # ── Step 2: Align ──────────────────────────────────────
    print(f"\n[2/8] Aligning {len(raw_data)} assets...")
    df = pd.DataFrame(raw_data).sort_index().dropna()
    rets = df.pct_change().dropna()
    rets = rets[abs(rets) < 0.25]
    asset_names = list(rets.columns)
    N = len(asset_names)
    T = len(rets)
    print(f"    Aligned: {T} shared trading days, {N} assets")

    # ── Step 3: Fit univariate GARCH(1,1) ──────────────────
    from arch import arch_model

    garch_vol = {}   # name -> T x 1 conditional vol series
    std_resid = {}   # name -> T x 1 standardized residuals

    for name in asset_names:
        r = rets[name].values * 100  # scale to percentage for numerical stability
        model = arch_model(r, vol='Garch', p=1, q=1, dist='t', rescale=False)
        res = model.fit(disp='off', show_warning=False)
        vol = res.conditional_volatility / 100  # back to decimal
        resid_standardized = (r / 100) / vol

        # Align lengths
        min_len = min(len(vol), T)
        garch_vol[name] = vol[-min_len:]
        std_resid[name] = resid_standardized[-min_len:]

        omega = res.params.get('omega', 0)
        alpha = res.params.get('alpha[1]', 0)
        beta  = res.params.get('beta[1]', 0)
        nu    = res.params.get('nu', 10)

    # ── Step 4: Build standardized residual matrix ──────────
    # Align all to same length (use min)
    min_len = min(len(v) for v in garch_vol.values())
    eps_matrix = np.column_stack([std_resid[name][-min_len:] for name in asset_names])
    aligned_dates = rets.index[-min_len:]
    aligned_vols = {name: garch_vol[name][-min_len:] for name in asset_names}
    pass  # quiet

    # ── Step 5: Estimate DCC ────────────────────────────────
    pass  # quiet
    a_opt, b_opt, Q_bar, Qt_series, Rt_series = estimate_dcc(eps_matrix)

    # Extract pairwise correlations over time
    pair_indices = []
    pair_names = []
    for i in range(N):
        for j in range(i+1, N):
            pair_indices.append((i, j))
            pair_names.append(f"{asset_names[i]} / {asset_names[j]}")

    rhos = np.array([[Rt_series[t][i, j] for t in range(len(Rt_series))]
                     for (i, j) in pair_indices]).T  # T' x n_pairs (T' = T-1, starts at t=1)

    # ── Step 6: Compute hedge ratios ─────────────────────────
    # h_hedge = rho * sqrt(h_asset) / sqrt(h_hedge)
    # We hedge the first asset (CSI 300) with each other
    hedge_pairs = []
    hedge_rhos = []
    # rhos starts at t=1, so hedge_h_asset must also start at t=1
    hedge_h_asset = aligned_vols[asset_names[0]][1:]  # length T-1
    for k, (i, j) in enumerate(pair_indices):
        if i == 0:  # pairs where first asset is involved
            h_h = aligned_vols[asset_names[j]][1:]
            rho = rhos[:, k]
            hr = rho * np.sqrt(hedge_h_asset) / np.sqrt(h_h + 1e-10)
            hedge_pairs.append(pair_names[k])
            hedge_rhos.append(hr)

    hedge_rhos = np.array(hedge_rhos).T  # T' x n_hedge_pairs
    hedge_dates = aligned_dates[1:]

    # ── Step 7: Generate charts ─────────────────────────────
    pass  # quiet
    sys.stdout.flush()
    _saved_out = sys.stdout
    _saved_err = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    # Correlation heatmap
    corr_buf = plot_correlation_heatmap(Rt_series, aligned_dates[1:], asset_names)
    corr_path = f"{OUTPUT_DIR}/dcc_correlations.png"
    with open(corr_path, 'wb') as f:
        f.write(corr_buf.getvalue())
    pass  # quiet

    # Volatility comparison
    vol_buf = plot_vol_comparison(aligned_vols, aligned_dates, asset_names)
    vol_path = f"{OUTPUT_DIR}/dcc_vol_comparison.png"
    with open(vol_path, 'wb') as f:
        f.write(vol_buf.getvalue())
    pass  # quiet

    # Hedge ratios
    if len(hedge_pairs) > 0:
        hedge_buf = plot_hedge_ratios(hedge_rhos, hedge_h_asset,
                                       aligned_vols, hedge_dates, hedge_pairs)
        hedge_path = f"{OUTPUT_DIR}/dcc_hedge_ratios.png"
        with open(hedge_path, 'wb') as f:
            f.write(hedge_buf.getvalue())
        pass  # quiet

    # Summary table
    table_buf = plot_correlation_summary_table(rhos, aligned_dates[1:], pair_names, a_opt, b_opt)
    table_path = f"{OUTPUT_DIR}/dcc_summary_table.png"
    with open(table_path, 'wb') as f:
        f.write(table_buf.getvalue())
    pass  # quiet
    plt.close('all')
    sys.stdout = _saved_out
    sys.stderr = _saved_err

    # ── Step 8: Print summary ───────────────────────────────
    pass  # quiet
    pass  # quiet
    pass  # quiet

    pass  # quiet
    pass  # quiet
    if a_opt + b_opt < 0.95:
        pass  # quiet
    else:
        pass  # quiet

    pass  # quiet
    means = np.nanmean(rhos, axis=0)
    for name, mean in zip(pair_names, means):
        trend = "📈" if np.corrcoef(np.arange(len(rhos)), rhos[:, list(pair_names).index(name)])[0,1] > 0 else "📉"
        pass  # quiet

    pass  # quiet
    for pair, hr in zip(hedge_pairs, hedge_rhos.T):
        pass  # quiet
        pass  # quiet
        pass  # quiet

    # ── Step 8: Send Telegram report ──────────────────────────
    try:
        import requests

        # Build text report
        date_str = datetime.now().strftime('%Y-%m-%d %H:%M UTC+8')
        means = np.nanmean(rhos, axis=0)
        recent = rhos[-60:] if len(rhos) >= 60 else rhos

        # Pair correlation summary
        pair_lines = []
        for k, (name, mean) in enumerate(zip(pair_names, means)):
            recent_mean = np.nanmean(recent[:, k]) if len(recent) > 0 else mean
            sig = "🔴" if recent_mean > 0.5 else "🟡" if recent_mean > 0.15 else "🟢"
            pair_lines.append(f"{sig} {name}: <b>{recent_mean:+.3f}</b> (avg {mean:.3f})")

        # Asset volatility
        vol_lines = []
        for name in asset_names:
            if name in aligned_vols:
                v = np.sqrt(aligned_vols[name][-30:].mean()) * np.sqrt(252) * 100
                emoji = "🔴" if v > 25 else "🟡" if v > 16 else "🟢"
                vol_lines.append(f"{emoji} {name}: <b>{v:.1f}%</b>")

        report = (
            f"📊 <b>DCC-GARCH 多资产动态相关性报告</b>\n"
            f"🕐 {date_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📐 <b>资产波动率（年化）</b>\n" + "\n".join(vol_lines) + "\n\n"
            f"🔗 <b>DCC 条件相关性</b> (a={a_opt:.3f}, b={b_opt:.3f})\n" + "\n".join(pair_lines) + "\n\n"
            f"📈 <b>CSI 300 最优对冲比例</b>\n"
        )
        for hp in hedge_pairs:
            idx = hedge_pairs.index(hp)
            hr_vals = hedge_rhos[:, idx]
            report += f"  {hp}: avg=<b>{np.nanmean(hr_vals):.4f}</b>\n"

        report += (
            f"\n⚠️ <i>仅供参考，不构成投资建议</i>\n"
            f"#DCC #GARCH #波动率 #对冲"
        )

        # Send text
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={
            'chat_id': TELEGRAM_CHAT,
            'text': report,
            'parse_mode': 'HTML'
        }, timeout=20)

        # Send charts
        chart_files = [
            (f"{OUTPUT_DIR}/dcc_correlations.png", "🔥 DCC 相关性热图"),
            (f"{OUTPUT_DIR}/dcc_vol_comparison.png", "📊 多资产条件波动率对比"),
            (f"{OUTPUT_DIR}/dcc_hedge_ratios.png", "📈 CSI 300 对冲比例时序"),
            (f"{OUTPUT_DIR}/dcc_summary_table.png", "📋 DCC 相关性统计摘要"),
        ]
        for path, caption in chart_files:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    files = {'photo': f}
                    data = {'chat_id': TELEGRAM_CHAT, 'caption': caption, 'parse_mode': 'HTML'}
                    requests.post(
                        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                        files=files, data=data, timeout=30
                    )
        print("[DCC] Telegram report sent successfully")
    except Exception as e:
        print(f"[DCC] Telegram push error: {e}")

    return {
        "a": a_opt, "b": b_opt,
        "pair_names": pair_names,
        "pair_rhos": rhos,
        "hedge_pairs": hedge_pairs,
        "hedge_rhos": hedge_rhos,
        "asset_names": asset_names,
        "Rt_series": Rt_series,
        "aligned_dates": aligned_dates,
        "aligned_vols": aligned_vols,
    }

if __name__ == "__main__":
    import os, warnings; warnings.filterwarnings('ignore')
    results = main()
