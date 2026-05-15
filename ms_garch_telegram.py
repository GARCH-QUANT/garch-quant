#!/usr/bin/env python3.12
"""
MS-GARCH Regime Analysis — Telegram Report
Beautiful HTML report with charts for Telegram channel push
"""

import warnings, os, sys, io, time
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ── Config ─────────────────────────────────────────────────
import os
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")
TWELVE_DATA_KEY   = os.environ.get("TWELVE_DATA_KEY", "")

CLR_NORMAL   = "#4CAF50"
CLR_CRISIS   = "#E53935"
CLR_NEUTRAL  = "#2196F3"
CLR_ACCENT   = "#FF9800"
CLR_BG       = "#0D1117"
CLR_CARD_BG  = "#161B22"
CLR_GRID     = "#21262D"
CLR_TEXT     = "#E6EDF3"
CLR_MUTED    = "#8B949E"
CLR_BORDER   = "#30363D"

plt.rcParams.update({
    "axes.facecolor"   : CLR_BG,
    "figure.facecolor" : CLR_BG,
    "axes.edgecolor"   : CLR_GRID,
    "axes.labelcolor"  : CLR_TEXT,
    "text.color"       : CLR_TEXT,
    "xtick.color"      : CLR_MUTED,
    "ytick.color"      : CLR_MUTED,
    "grid.color"       : CLR_GRID,
    "grid.alpha"       : 0.5,
    "font.family"      : "sans-serif",
})

TICKERS = {
    "CSI 300":  {"tushare": "000300.SH", "av": "000300", "td": "HSI:IND"},
    "HSI":      {"tushare": None,        "av": "HSI",    "td": "HSI:IND"},
    "S&P 500":  {"tushare": None,        "av": "SPY",    "td": "SPX:IND"},
    "AAPL":    {"tushare": None,        "av": "AAPL",   "td": "AAPL"},
    "TSLA":    {"tushare": None,        "av": "TSLA",   "td": "TSLA"},
}

CRISIS_EVENTS = [
    ("2020-01-01", "COVID-19",              0.30),
    ("2020-03-01", "US Circuit Breaker",     0.35),
    ("2022-03-01", "Russia-Ukraine War",     0.25),
    ("2022-06-01", "Fed Tightening",         0.20),
    ("2023-03-01", "SVB Crisis",             0.25),
    ("2024-01-01", "Middle East Tension",    0.20),
    ("2025-04-01", "Tariff War 2.0",        0.35),
]

# ── Data Fetch ─────────────────────────────────────────────
def get_tushare_data(ts_code, start, end):
    try:
        import tushare as ts
        tushare_token = os.environ.get("TUSHARE_TOKEN", "")
        ts.set_token(tushare_token)
        pro = ts.pro_api(tushare_token)
        if ts_code.endswith(".SH") or ts_code.endswith(".SZ"):
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
        if not records:
            return None
        idx = pd.to_datetime([r[0] for r in records])
        return pd.Series([r[1] for r in records], index=idx, name="close")
    except:
        return None

def fetch_all_data():
    import time
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
    data  = {}

    for name, sources in TICKERS.items():
        pass  # quiet
        tried = []

        if sources.get("tushare"):
            s = get_tushare_data(sources["tushare"], start, end)
            if s is not None and len(s) > 100:
                data[name] = s.dropna()
                pass  # quiet
                continue
            tried.append("tushare")

        td_sym = sources.get("td", "")
        if td_sym:
            time.sleep(1)
            s = get_twelve_data(td_sym, start, end)
            if s is not None and len(s) > 100:
                data[name] = s.dropna()
                pass  # quiet
                continue
            tried.append(f"TwelveData({td_sym})")

        if name not in data:
            pass  # quiet

    return data

# ── MS-GARCH Core ───────────────────────────────────────────
def fit_regime_garch(returns, name="Asset"):
    r = returns.values.astype(float)
    T = len(r)
    dates = returns.index

    window = 20
    realized_vol = np.zeros(T)
    for t in range(window, T):
        realized_vol[t] = np.std(r[t-window:t], ddof=1) * np.sqrt(252)
    for t in range(window):
        realized_vol[t] = realized_vol[window] if window < T else realized_vol[0]

    realized_vol_series = pd.Series(realized_vol, index=dates)

    low_q  = np.percentile(realized_vol, 40)
    high_q = np.percentile(realized_vol, 80)
    normal_mask = realized_vol < low_q
    crisis_mask = realized_vol > high_q

    n_normal = normal_mask.sum()
    n_crisis = crisis_mask.sum()

    from arch import arch_model

    def fit_garch_subset(subset_r):
        if len(subset_r) < 30:
            return None
        try:
            am = arch_model(subset_r * 100, vol='Garch', p=1, q=1,
                            mean='Constant', dist='normal')
            res = am.fit(disp='off', show_warning=False)
            omega = res.params.get('omega', 0) / 10000
            alpha = res.params.get('alpha[1]', 0)
            beta  = res.params.get('beta[1]', 0)
            llv = omega / (1 - alpha - beta + 1e-10)
            return np.sqrt(llv) * np.sqrt(252)
        except:
            return None

    vol_normal = fit_garch_subset(r[normal_mask])
    vol_crisis = fit_garch_subset(r[crisis_mask])

    if vol_normal is None:
        vol_normal = 0.20
    if vol_crisis is None:
        vol_crisis = vol_normal * 1.5

    vol_ratio = vol_crisis / vol_normal if vol_normal > 0 else 1.0

    prob = (realized_vol - low_q) / (high_q - low_q + 1e-10)
    prob = np.clip(prob, 0, 1)
    cond_vol_arr = prob * vol_crisis + (1 - prob) * vol_normal
    cond_vol = pd.Series(cond_vol_arr, index=dates)
    regime_prob = pd.Series(1 - prob, index=dates)

    crisis_periods = []
    in_crisis = False
    crisis_start = None
    for date, p in regime_prob.items():
        if p > 0.5 and not in_crisis:
            in_crisis, crisis_start = True, date
        elif p <= 0.3 and in_crisis:
            in_crisis = False
            crisis_periods.append((crisis_start, date))
    if in_crisis:
        crisis_periods.append((crisis_start, dates[-1]))

    return regime_prob, cond_vol, crisis_periods, vol_normal, vol_crisis, vol_ratio

# ── Chart Generators ───────────────────────────────────────
def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=CLR_BG, edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_regime_card(name, rp, cv, periods):
    """Single asset regime probability card"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.2]})

    ax = axes[0]
    ax.plot(cv.index, cv.values * 100, color=CLR_NEUTRAL, lw=1.5,
            label="Conditional Vol (Ann.)")
    ax.fill_between(cv.index, 0, cv.values * 100,
                    where=(rp.values > 0.5), color=CLR_CRISIS, alpha=0.25,
                    label="Crisis Zone")
    for start, end in periods[-3:]:
        ax.axvspan(start, end, color=CLR_CRISIS, alpha=0.12)
    ax.set_ylabel("Ann. Volatility (%)", fontsize=11, color=CLR_TEXT)
    ax.set_title(f"{name} — Regime-Switching Volatility", fontsize=14,
                 fontweight='bold', color=CLR_TEXT)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    ax.set_facecolor(CLR_BG)

    ax = axes[1]
    ax.fill_between(rp.index, 0, rp.values, color=CLR_CRISIS, alpha=0.7,
                    label="Crisis Probability")
    ax.plot(rp.index, rp.values, color=CLR_CRISIS, lw=1.2)
    ax.axhline(0.5, color="white", lw=1.2, linestyle="--", alpha=0.8,
               label="Threshold (0.5)")
    ax.set_ylabel("Crisis Prob.", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(CLR_BG)

    plt.tight_layout(pad=2)
    return fig_to_buf(fig)

def plot_multi_vol_comparison(cond_vols_dict, rp_dict):
    """Multi-asset volatility comparison"""
    names = list(cond_vols_dict.keys())
    n = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, n))

    for i, name in enumerate(names):
        ax = axes[i]
        cv = cond_vols_dict[name]
        rp = rp_dict[name]
        ax.plot(cv.index, cv.values * 100, color=colors[i], lw=1.4,
                label=f"{name} Vol")
        ax.fill_between(rp.index, 0, rp.values * 100,
                        color=CLR_CRISIS, alpha=0.2, label="Crisis Prob")
        ax.set_ylabel("Ann. Vol (%)", fontsize=10)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)
        ax.set_facecolor(CLR_BG)

    plt.tight_layout(pad=2)
    return fig_to_buf(fig)

def plot_bars(summary_df):
    """Bar chart: Normal vs Crisis volatility per asset"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(summary_df))
    w = 0.35

    ax = axes[0]
    bars1 = ax.bar(x - w/2, summary_df["Normal Vol"], w,
                   label="Normal", color=CLR_NORMAL, alpha=0.85)
    bars2 = ax.bar(x + w/2, summary_df["Crisis Vol"], w,
                   label="Crisis", color=CLR_CRISIS, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["Asset"], fontsize=11)
    ax.set_ylabel("Ann. Volatility (%)", fontsize=11)
    ax.set_title("Normal vs Crisis Volatility", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_facecolor(CLR_BG)
    for b in bars1 + bars2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.5, f"{h:.1f}%",
                ha='center', va='bottom', fontsize=9, color=CLR_TEXT)

    ax = axes[1]
    colors_b = [CLR_CRISIS if r > 2 else CLR_ACCENT for r in summary_df["Multiplier"]]
    bars = ax.bar(summary_df["Asset"], summary_df["Multiplier"], color=colors_b, alpha=0.85)
    ax.axhline(1, color="white", lw=1, linestyle="--", alpha=0.5)
    ax.set_ylabel("Vol Multiplier (Crisis/Normal)", fontsize=11)
    ax.set_title("Crisis Vol Multiplier", fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_facecolor(CLR_BG)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.03, f"{h:.2f}x",
                ha='center', va='bottom', fontsize=10, color=CLR_TEXT)

    plt.tight_layout()
    return fig_to_buf(fig)

def plot_heatmap(rp_dict, cond_vols_dict):
    """Correlation heatmap of regime probabilities"""
    names = list(rp_dict.keys())
    n = len(names)
    if n < 2:
        return None

    dates = None
    for rp in rp_dict.values():
        dates = rp.index
        break

    min_len = min(len(rp_dict[n].values) for n in names)
    mat = np.array([rp_dict[n].values[:min_len] for n in names])
    corr = np.corrcoef(mat)

    fig, ax = plt.subplots(figsize=(max(6, n*2.5), n*2))
    im = ax.imshow(corr, cmap="RdYlGn_r", vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=11)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_title("Regime Probability Correlation", fontsize=13, fontweight='bold')

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i,j]:.2f}",
                    ha='center', va='center', fontsize=11,
                    color="white" if abs(corr[i,j]) > 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Correlation")
    plt.tight_layout()
    return fig_to_buf(fig)

# ── Telegram HTML Builder ──────────────────────────────────
def build_telegram_html(summary_df, rp_dict, cond_vols_dict, crisis_dict, params_dict):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M")

    lines = [
        f"📊 <b>MS-GARCH Regime-Switching Volatility Report</b>",
        f"🕐 <i>Generated: {date_str} UTC+8</i>",
        "━━━━━━━━━━━━━━━━━━━━",
        ""
    ]

    # ── Key Metrics Card ───────────────────────────────
    lines.append("📐 <b>Key Metrics</b>")
    lines.append("<code>")
    header = f"{'Asset':<12} {'Normal':>8} {'Crisis':>8} {'Mult':>6}  {'Crisis%':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for _, row in summary_df.iterrows():
        asset  = row["Asset"]
        norm   = f"{row['Normal Vol']:.1f}%"
        crisis = f"{row['Crisis Vol']:.1f}%"
        mult   = f"{row['Multiplier']:.2f}x"
        crp    = f"{row['Crisis Prob']:.1f}%"
        lines.append(f"{asset:<12} {norm:>8} {crisis:>8} {mult:>6}  {crp:>8}")

    lines.append("</code>")
    lines.append("")

    # ── Interpretation ─────────────────────────────────
    lines.append("💡 <b>Interpretation</b>")
    for _, row in summary_df.iterrows():
        asset  = row["Asset"]
        mult   = row["Multiplier"]
        crp    = row["Crisis Prob"]
        nv     = row["Normal Vol"]
        cvv    = row["Crisis Vol"]

        if mult > 1.8:
            level = "🔴 HIGH LEVERAGE"
            desc  = f"Crisis vol is {mult:.2f}x normal — elevated tail risk"
        elif mult > 1.3:
            level = "🟡 MODERATE"
            desc  = f"Volatility amplifies {mult:.2f}x in down markets"
        else:
            level = "🟢 LOW"
            desc  = "Regime stability, low crisis premium"

        if crp > 70:
            status = "⚠️ Currently in CRISIS regime"
        elif crp > 50:
            status = "⚡ Transition zone"
        else:
            status = "✅ Normal regime"

        lines.append(f"<b>{asset}</b>: {level}")
        lines.append(f"  {desc}")
        lines.append(f"  {status} ({crp:.1f}% crisis prob)")
        lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("")
    lines.append("📅 <b>Recent Crisis Periods</b>")

    for name, periods in crisis_dict.items():
        if periods:
            last3 = periods[-3:]
            lines.append(f"<b>{name}</b>:")
            for s, e in last3:
                days = (e - s).days + 1
                lines.append(f"  {s.strftime('%Y-%m-%d')} → {e.strftime('%Y-%m-%d')} ({days}d)")

    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("🔮 <b>Methodology</b>")
    lines.append("• Rolling 20-day realized volatility")
    lines.append("• GARCH(1,1) estimated separately for Normal/Crisis regimes")
    lines.append("• Crisis: top 20th percentile | Normal: bottom 40th percentile")
    lines.append("• Conditional vol = probability-weighted blend")
    lines.append("")
    lines.append("⚠️ <i>For informational purposes only. Not investment advice.</i>")

    return "<br>".join(lines)

# ── Main ───────────────────────────────────────────────────
def main():
    import json as _json

    data = fetch_all_data()
    if not data:
        return

    returns_dict = {}
    for name, series in data.items():
        rets = series.pct_change().dropna()
        rets = rets[abs(rets) < 0.25]
        returns_dict[name] = rets

    rp_dict, cv_dict, crisis_dict = {}, {}, {}
    params_dict = {}
    records = []

    for name, rets in returns_dict.items():
        rp, cv, periods, vn, vc, vr = fit_regime_garch(rets, name)
        rp_dict[name]  = rp
        cv_dict[name]  = cv
        crisis_dict[name] = periods
        crp = (rp > 0.5).mean() * 100
        params_dict[name] = {"normal": vn, "crisis": vc, "ratio": vr, "crisis_prob": crp}
        records.append({
            "Asset": name,
            "Normal Vol": round(vn * 100, 1),
            "Crisis Vol": round(vc * 100, 1),
            "Multiplier": round(vr, 2),
            "Crisis Prob": round(crp, 1),
        })

    summary_df = pd.DataFrame(records)

    # ── Generate Charts ─────────────────────────────────
    chart_bufs = {}

    buf = plot_multi_vol_comparison(cv_dict, rp_dict)
    chart_bufs["multi_vol"] = buf

    buf = plot_bars(summary_df)
    chart_bufs["bars"] = buf

    for name in returns_dict:
        buf = plot_regime_card(name, rp_dict[name], cv_dict[name], crisis_dict[name])
        chart_bufs[f"card_{name}"] = buf

    buf = plot_heatmap(rp_dict, cv_dict)
    if buf:
        chart_bufs["heatmap"] = buf

    # ── Save Charts & JSON ──────────────────────────────
    out_dir = "/home/agentuser/.hermes/papers/figures"
    os.makedirs(out_dir, exist_ok=True)
    chart_paths = {}

    for key, buf in chart_bufs.items():
        path = f"{out_dir}/tg_ms_garch_{key}.png"
        with open(path, 'wb') as f:
            f.write(buf.getvalue())
        chart_paths[key] = path

    # ── Build HTML Report ─────────────────────────────────
    html_report = build_telegram_html(
        summary_df, rp_dict, cv_dict, crisis_dict, params_dict
    )

    # Write JSON with computed metrics for the agent to read
    result = {
        "summary": summary_df.to_dict(orient="records"),
        "chart_paths": {k: v for k, v in chart_paths.items()},
        "params": params_dict,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M UTC+8"),
    }
    json_path = f"{out_dir}/tg_ms_garch_result.json"
    with open(json_path, 'w') as f:
        _json.dump(result, f, indent=2, ensure_ascii=False)

    # Print HTML to stdout for cron capture
    print(html_report)

    return result

if __name__ == "__main__":
    main()
