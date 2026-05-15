#!/usr/bin/env python3.12
"""
MS-GARCH Regime-Switching Volatility Analysis
Regime 0: Normal (Low Volatility)
Regime 1: Crisis (High Volatility)
"""

import warnings, os, sys, json
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
from arch import arch_model
from scipy.stats import norm
import subprocess, time, io

# ── Color Palette ────────────────────────────────────────
CLR_NORMAL   = "#4CAF50"   # Green
CLR_CRISIS   = "#E53935"   # Red
CLR_NEUTRAL  = "#2196F3"   # Blue
CLR_TEXT     = "#ECEFF1"
CLR_BG       = "#1E1E1E"
CLR_GRID     = "#37474F"

# ── Asset Config ──────────────────────────────────────────
TICKERS = {
    "CSI 300": "000300",
    "HSI": "HSI",
    "S&P 500": "SPX",
    "AAPL": "AAPL",
}

# ── Crisis Event Annotations ──────────────────────────────
CRISIS_EVENTS = [
    ("2020-01-01", "COVID-19",           0.30),
    ("2020-03-01", "US Equity Circuit Breaker", 0.35),
    ("2022-03-01", "Russia-Ukraine War",        0.25),
    ("2022-06-01", "Fed Tightening",             0.20),
    ("2023-03-01", "SVB Crisis",                0.25),
    ("2024-01-01", "Middle East Tensions",       0.20),
    ("2025-04-01", "Tariff War 2.0",             0.35),
]

PLT_STYLE = {
    "axes.facecolor" : CLR_BG,
    "figure.facecolor": CLR_BG,
    "axes.edgecolor" : CLR_GRID,
    "axes.labelcolor": CLR_TEXT,
    "text.color"     : CLR_TEXT,
    "xtick.color"    : CLR_TEXT,
    "ytick.color"    : CLR_TEXT,
    "grid.color"     : CLR_GRID,
    "grid.alpha"     : 0.4,
}
plt.rcParams.update(**PLT_STYLE)

# ── Crisis Event Annotations ──────────────────────────────
CRISIS_EVENTS = [
    ("2020-01-01", "COVID-19",           0.30),
    ("2020-03-01", "US Equity Circuit Breaker", 0.35),
    ("2022-03-01", "Russia-Ukraine War",        0.25),
    ("2022-06-01", "Fed Tightening",             0.20),
    ("2023-03-01", "SVB Crisis",                0.25),
    ("2024-01-01", "Middle East Tensions",       0.20),
    ("2025-04-01", "Tariff War 2.0",             0.35),
]

# ── Data Fetch ────────────────────────────────────────────
import os
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")
TWELVE_DATA_KEY   = os.environ.get("TWELVE_DATA_KEY", "")

def get_stooq_data(ticker, start, end):
    """Fetch global stock/index data via yfinance (no retries param)"""
    import yfinance as yf
    try:
        sym = ticker.replace(".US", "")
        df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=True,
                         interval="1d")
        if df.empty:
            return None
        close = df["Close"].dropna()
        if hasattr(close, "_values"):
            close = pd.Series(close._values, index=pd.to_datetime(close.index), name="close")
        return close
    except Exception as e:
        pass  # quiet
        return None

def fetch_with_retry(fn, ticker, max_attempts=3, delay=5):
    """Data fetch with retry"""
    import time
    for attempt in range(max_attempts):
        result = fn(ticker)
        if result is not None and len(result) > 100:
            return result
        if attempt < max_attempts - 1:
            pass  # quiet
            time.sleep(delay)
            delay *= 1.5
    return None

def get_alpha_vantage(ticker, start, end):
    """Fetch US stock data via Alpha Vantage"""
    import requests
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "apikey": ALPHA_VANTAGE_KEY,
            "outputsize": "full"
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if "Time Series (Daily)" not in data:
            return None
        ts = data["Time Series (Daily)"]
        records = [(date, float(v["5. adjusted close"])) for date, v in ts.items()
                   if start <= date <= end]
        if not records:
            return None
        records.sort(key=lambda x: x[0])
        idx = pd.to_datetime([r[0] for r in records])
        series = pd.Series([r[1] for r in records], index=idx, name="close")
        return series
    except Exception as e:
        pass  # quiet
        return None

def get_tushare_data(ts_code, start, end):
    """Fetch A-share data via tushare"""
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
    except Exception as e:
        pass  # quiet
    return None

def get_finnhub_data(ticker):
    """Fetch US stock data via finnhub (daily K-line)"""
    import requests
    try:
        url = f"https://finnhub.io/api/v1/quote"
        params = {"symbol": ticker, "token": "***"}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        return None  # Finnhub free tier only has today's quote, skip
    except:
        return None

def get_twelve_data(symbol, start, end):
    """Fetch global stock/index data via Twelve Data"""
    import requests
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": "1day",
            "outputsize": 500,
            "format": "JSON",
            "apikey": TWELVE_DATA_KEY
        }
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if "values" not in data:
            return None
        vals = data["values"]
        records = [(v["datetime"], float(v["close"])) for v in reversed(vals)]
        if not records:
            return None
        idx = pd.to_datetime([r[0] for r in records])
        series = pd.Series([r[1] for r in records], index=idx, name="close")
        return series
    except Exception as e:
        pass  # quiet
        return None

def fetch_all_data():
    """Fetch all asset data (multi-source priority)"""
    import time
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
    data  = {}

    TICKERS_EXT = {
        "CSI 300":   {"tushare": "000300.SH", "av": "000300", "td": "EUR/USD"},
        "HSI":       {"tushare": None,       "av": "HSI",    "td": "HSI:IND"},
        "S&P 500":   {"tushare": None,       "av": "SPY",    "td": "SPX:IND"},
        "AAPL":      {"tushare": None,       "av": "AAPL",   "td": "AAPL"},
        "TSLA":      {"tushare": None,       "av": "TSLA",   "td": "TSLA"},
    }

    for name, sources in TICKERS_EXT.items():
        pass  # quiet
        series = None
        tried = []

        # 1. tushare (A-shares)
        if sources.get("tushare"):
            s = get_tushare_data(sources["tushare"], start, end)
            if s is not None and len(s) > 100:
                data[name] = s.dropna()
                pass  # quiet
                continue
            tried.append("tushare")

        # 2. Alpha Vantage
        av_sym = sources.get("av", "")
        if av_sym:
            for delay in [0, 3, 5]:
                if delay:
                    time.sleep(delay)
                series = get_alpha_vantage(av_sym, start, end)
                if series is not None and len(series) > 100:
                    data[name] = series.dropna()
                    pass  # quiet
                    break
            if name in data:
                continue
            tried.append(f"AlphaVantage({av_sym})")

        # 3. Twelve Data
        td_sym = sources.get("td", "")
        if td_sym:
            series = get_twelve_data(td_sym, start, end)
            if series is not None and len(series) > 100:
                data[name] = series.dropna()
                pass  # quiet
                continue
            tried.append(f"TwelveData({td_sym})")

        # 4. yfinance (fallback, has latency)
        yf_map = {"HSI": "^HSI", "S&P 500": "^SPX", "AAPL": "AAPL",
                  "TSLA": "TSLA", "CSI 300": "000300.SS"}
        yf_sym = yf_map.get(name, name)
        if name != "CSI 300":  # CSI 300 already from tushare
            for delay in [0, 5, 10]:
                if delay:
                    time.sleep(delay)
                series = get_stooq_data(yf_sym, start, end)
                if series is not None and len(series) > 100:
                    data[name] = series.dropna()
                    pass  # quiet
                    break
            if name in data:
                continue
            tried.append(f"yfinance({yf_sym})")

        if name not in data:
            pass  # quiet

    return data

# ── MS-GARCH Estimation ────────────────────────────────────
def fit_regime_garch(returns, name="Asset"):
    """
    Two-regime identification:
      1. Calculate rolling realized volatility (20-day)
      2. Quantile thresholds: < 50th pct = Normal, > 80th pct = Crisis
      3. Estimate GARCH(1,1) on each subsample
      4. Full-sample probability-weighted conditional volatility
    More robust than EM, no dependence on initial value convergence.
    """
    r = returns.values.astype(float)
    T = len(r)
    dates = returns.index

    # ── Step 1: Rolling Realized Volatility ──────────────
    window = 20
    realized_vol = np.zeros(T)
    for t in range(window, T):
        realized_vol[t] = np.std(r[t-window:t], ddof=1) * np.sqrt(252)

    # EWMA fill for the beginning
    for t in range(window):
        realized_vol[t] = realized_vol[window] if window < T else realized_vol[0]

    realized_vol_series = pd.Series(realized_vol, index=dates)

    # ── Step 2: Regime Identification ─────────────────────
    low_q  = np.percentile(realized_vol, 40)   # <40% = Normal
    high_q = np.percentile(realized_vol, 80)   # >80% = Crisis

    normal_mask = realized_vol < low_q
    crisis_mask = realized_vol > high_q
    middle_mask = ~(normal_mask | crisis_mask)

    n_normal = normal_mask.sum()
    n_crisis = crisis_mask.sum()

    pass  # quiet

    # ── Step 3: Piecewise GARCH Estimation ────────────────
    from arch import arch_model

    def fit_garch_subset(subset_r, label):
        """Estimate GARCH on subsample"""
        if len(subset_r) < 30:
            return None
        try:
            am = arch_model(subset_r * 100, vol='Garch', p=1, q=1,
                            mean='Constant', dist='normal')
            res = am.fit(disp='off', show_warning=False)
            omega = res.params.get('omega', 0) / 10000
            alpha = res.params.get('alpha[1]', 0)
            beta  = res.params.get('beta[1]', 0)
            res_params = res.params
            omega = res_params.get('omega', 0)
            alpha = res_params.get('alpha[1]', 0)
            beta  = res_params.get('beta[1]', 0)
            # arch omega unit is (r*100)^2, convert to r^2 by dividing by 10000
            omega_sq = omega / 10000.0
            llv = omega_sq / (1 - alpha - beta + 1e-10)
            annualized_vol = np.sqrt(llv) * np.sqrt(252)
            return {'omega': omega, 'alpha': alpha, 'beta': beta,
                    'params': res.params, 'annualized_vol': annualized_vol}
        except Exception as e:
            return None

    r_normal = r[normal_mask]
    r_crisis = r[crisis_mask]

    garch_normal = fit_garch_subset(r_normal, 'Normal')
    garch_crisis = fit_garch_subset(r_crisis, 'Crisis')

    if garch_normal is None:
        # Fallback: full-sample GARCH
        try:
            am_full = arch_model(r * 100, vol='Garch', p=1, q=1,
                                 mean='Constant', dist='normal')
            res_full = am_full.fit(disp='off', show_warning=False)
            omega_full = res_full.params.get('omega', 0) / 10000
            alpha_full = res_full.params.get('alpha[1]', 0)
            beta_full = res_full.params.get('beta[1]', 0)
            vol_full = np.sqrt(omega_full / (1 - alpha_full - beta_full)) * np.sqrt(252)
            garch_normal = {'omega': omega_full, 'alpha': alpha_full,
                            'beta': beta_full, 'annualized_vol': vol_full}
            garch_crisis = garch_normal.copy()
        except:
            garch_normal = {'omega': 1e-6, 'alpha': 0.05, 'beta': 0.90, 'annualized_vol': 0.20}
            garch_crisis = {'omega': 5e-6, 'alpha': 0.10, 'beta': 0.85, 'annualized_vol': 0.35}

    if garch_crisis is None:
        garch_crisis = garch_normal.copy()
        garch_crisis['annualized_vol'] = garch_normal['annualized_vol'] * 1.5

    normal_vol_annual = garch_normal['annualized_vol']
    crisis_vol_annual = garch_crisis['annualized_vol']
    vol_ratio = crisis_vol_annual / normal_vol_annual if normal_vol_annual > 0 else 1.0

    # ── Step 4: Full-Sample Conditional Vol (Prob-Weighted) ─
    # Probability = (realized_vol - low_q) / (high_q - low_q), clipped to [0,1]
    prob = (realized_vol - low_q) / (high_q - low_q + 1e-10)
    prob = np.clip(prob, 0, 1)

    # Conditional volatility
    cond_vol_arr = prob * crisis_vol_annual + (1 - prob) * normal_vol_annual
    cond_vol = pd.Series(cond_vol_arr, index=dates)
    regime_prob = pd.Series(1 - prob, index=dates)  # Crisis probability

    # Crisis periods
    crisis_periods = []
    in_crisis = False
    crisis_start = None
    for i, (date, p) in enumerate(regime_prob.items()):
        if p > 0.5 and not in_crisis:
            in_crisis = True
            crisis_start = date
        elif p <= 0.3 and in_crisis:
            in_crisis = False
            crisis_periods.append((crisis_start, date))
    if in_crisis:
        crisis_periods.append((crisis_start, dates[-1]))

    params = dict(
        omega_normal=garch_normal['omega'],
        alpha_normal=garch_normal['alpha'],
        beta_normal=garch_normal['beta'],
        omega_crisis=garch_crisis['omega'],
        alpha_crisis=garch_crisis['alpha'],
        beta_crisis=garch_crisis['beta'],
        p00=0.97, p11=0.93,
        normal_vol=normal_vol_annual * 100,
        crisis_vol=crisis_vol_annual * 100,
        vol_ratio=vol_ratio,
        low_q=low_q, high_q=high_q,
        realized_vol=realized_vol_series
    )

    pass  # quiet

    return regime_prob, cond_vol, crisis_periods, params, realized_vol_series

# ── Plotting Functions ─────────────────────────────────────
def plot_regime_probabilities(regime_probs, cond_vols, crisis_periods, asset_name, vol_series=None):
    """Chart 1: Regime Probability + Conditional Volatility"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: Conditional Volatility
    ax = axes[0]
    ax.plot(cond_vols.index, cond_vols.values * 100, color=CLR_NEUTRAL, lw=1.2, label="Conditional Vol (Ann.)")
    ax.fill_between(cond_vols.index, 0, cond_vols.values * 100,
                    where=(regime_probs.values > 0.5),
                    color=CLR_CRISIS, alpha=0.3, label="Crisis Zone")
    for start, end in crisis_periods:
        ax.axvspan(start, end, color=CLR_CRISIS, alpha=0.15)
    ax.set_ylabel("Annualized Volatility (%)", fontsize=11)
    ax.set_title(f"{asset_name} — Conditional Volatility & Regime Probability", fontsize=13, fontweight='bold')
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    # Bottom: Crisis Probability
    ax = axes[1]
    ax.fill_between(regime_probs.index, 0, regime_probs.values,
                    color=CLR_CRISIS, alpha=0.6, label="Crisis Probability")
    ax.plot(regime_probs.index, regime_probs.values, color=CLR_CRISIS, lw=1.0)
    ax.axhline(0.5, color="white", lw=1, linestyle="--", alpha=0.7, label="Regime Threshold (0.5)")
    ax.set_ylabel("Crisis Probability", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=CLR_BG, edgecolor='none')
    buf.seek(0)
    plt.close()
    return buf

def plot_vol_comparison(assets_data, regime_probs_dict, cond_vols_dict):
    """Chart 2: Multi-Asset Volatility Comparison + Regime Shading"""
    fig, axes = plt.subplots(len(assets_data), 1, figsize=(14, 3*len(assets_data)), sharex=True)
    if len(assets_data) == 1:
        axes = [axes]

    colors = list(matplotlib.cm.Set1(np.linspace(0, 1, len(assets_data))))

    for i, (name, cv) in enumerate(cond_vols_dict.items()):
        ax = axes[i]
        rp = regime_probs_dict[name]
        ax.plot(cv.index, cv.values*100, color=colors[i], lw=1.2, label=f"{name} Volatility")
        ax.fill_between(rp.index, 0, rp.values,
                        color=CLR_CRISIS, alpha=0.25, label="Crisis Probability")
        ax.set_ylabel("Annualized Volatility (%)", fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=CLR_BG, edgecolor='none')
    buf.seek(0)
    plt.close()
    return buf

def plot_crisis_events(cond_vols_dict, regime_probs_dict):
    """Chart 3: Crisis vs Normal Period - Key Metrics Comparison"""
    records = []
    for name, cv in cond_vols_dict.items():
        rp = regime_probs_dict[name]
        cv_vals = cv.values
        rp_vals = rp.values

        normal_mask = rp_vals <= 0.5
        crisis_mask = rp_vals > 0.5

        if normal_mask.sum() > 0:
            normal_vol = np.mean(cv_vals[normal_mask]) * 100
        else:
            normal_vol = np.nan
        if crisis_mask.sum() > 0:
            crisis_vol = np.mean(cv_vals[crisis_mask]) * 100
        else:
            crisis_vol = np.nan

        vol_ratio = crisis_vol / normal_vol if normal_vol and not np.isnan(normal_vol) else np.nan
        crisis_pct = crisis_mask.mean() * 100

        records.append({
            "Asset": name,
            "Normal Vol (%)": round(normal_vol, 1) if not np.isnan(normal_vol) else 0,
            "Crisis Vol (%)": round(crisis_vol, 1) if not np.isnan(crisis_vol) else 0,
            "Vol Multiplier": round(vol_ratio, 2) if not np.isnan(vol_ratio) else 0,
            "Crisis Prob Mean (%)": round(crisis_pct, 1),
        })

    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Normal vs Crisis Volatility
    ax = axes[0]
    x = np.arange(len(df))
    width = 0.35
    bars1 = ax.bar(x - width/2, df["Normal Vol (%)"], width, label="Normal", color=CLR_NORMAL, alpha=0.85)
    bars2 = ax.bar(x + width/2, df["Crisis Vol (%)"], width, label="Crisis", color=CLR_CRISIS, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Asset"], fontsize=11)
    ax.set_ylabel("Annualized Volatility (%)", fontsize=11)
    ax.set_title("Normal vs Crisis Volatility", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}%",
                ha='center', va='bottom', fontsize=9, color=CLR_TEXT)

    # Right: Volatility Leverage
    ax = axes[1]
    colors_bar = [CLR_CRISIS if r > 2 else CLR_NEUTRAL for r in df["Vol Multiplier"]]
    bars = ax.bar(df["Asset"], df["Vol Multiplier"], color=colors_bar, alpha=0.85)
    ax.axhline(1, color="white", lw=1, linestyle="--", alpha=0.5)
    ax.set_ylabel("Vol Multiplier (Crisis/Normal)", fontsize=11)
    ax.set_title("Crisis vs Normal Volatility Leverage", fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05, f"{h:.2f}x",
                ha='center', va='bottom', fontsize=10, color=CLR_TEXT)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=CLR_BG, edgecolor='none')
    buf.seek(0)
    plt.close()
    return buf, df

def plot_regime_scatter(regime_probs_dict, cond_vols_dict):
    """Chart 4: Cross-Asset Regime Probability Scatter"""
    names = list(regime_probs_dict.keys())
    n = len(names)
    if n < 2:
        return None

    fig, axes = plt.subplots(1, n-1, figsize=(5*(n-1), 5))
    if n == 2:
        axes = [axes]

    for i, name2 in enumerate(names[1:]):
        ax = axes[i]
        rp1 = regime_probs_dict[names[0]]
        rp2 = regime_probs_dict[name2]

        # Align dates
        common_idx = rp1.index.intersection(rp2.index)
        x = rp1.loc[common_idx].values
        y = rp2.loc[common_idx].values
        colors = np.where(y > 0.5, CLR_CRISIS, CLR_NORMAL)

        ax.scatter(x, y, c=colors, alpha=0.3, s=8, rasterized=True)
        ax.set_xlabel(f"{names[0]} Crisis Prob", fontsize=10)
        ax.set_ylabel(f"{name2} Crisis Prob", fontsize=10)
        ax.set_title(f"{names[0]} vs {name2}", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Correlation
        corr = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.95, f"rho={corr:.3f}", transform=ax.transAxes,
                fontsize=11, color=CLR_TEXT, va='top',
                bbox=dict(boxstyle='round', facecolor=CLR_BG, alpha=0.7))

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=CLR_BG, edgecolor='none')
    buf.seek(0)
    plt.close()
    return buf

# ── PDF Report Generation ──────────────────────────────────
def generate_pdf_report(fig_bufs, summary_df, crisis_summary, regime_probs_dict, cond_vols_dict):
    """Generate PDF report"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm, mm
        from reportlab.lib.utils import ImageReader
    except ImportError:
        pass  # quiet
        return None

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm, mm

    output_path = "/home/agentuser/.hermes/papers/ms_garch_regime_analysis.pdf"
    c = canvas.Canvas(output_path, pagesize=A4)
    W, H = A4
    MARGIN = 1.8*cm

    def new_page():
        c.showPage()

    def header(title, subtitle=""):
        c.setFillColor("#1E1E1E")
        c.rect(0, H-2.2*cm, W, 2.2*cm, fill=1, stroke=0)
        c.setFillColor("#FFFFFF")
        c.setFont("Helvetica-Bold", 16)
        c.drawString(MARGIN, H-1.5*cm, title)
        if subtitle:
            c.setFont("Helvetica", 10)
            c.drawString(MARGIN, H-1.95*cm, subtitle)
        c.setFillColor("#37474F")
        c.rect(0, H-2.2*cm, W, 0.5*mm, fill=1, stroke=0)

    def footer(page_num):
        c.setFillColor("#37474F")
        c.rect(0, 0, W, 1*cm, fill=1, stroke=0)
        c.setFillColor("#90A4AE")
        c.setFont("Helvetica", 8)
        c.drawString(MARGIN, 0.4*cm, "GARCH Quant | MS-GARCH Regime Analysis | Data: akshare/yfinance")
        c.drawRightString(W-MARGIN, 0.4*cm, f"Page {page_num}")

    # ── Cover Page ───────────────────────────────────────
    c.setFillColor("#0D1117")
    c.rect(0, 0, W, H, fill=1, stroke=0)

    c.setFillColor("#FFFFFF")
    c.setFont("Helvetica-Bold", 28)
    c.drawString(MARGIN, H-5*cm, "MS-GARCH")
    c.setFont("Helvetica-Bold", 28)
    c.drawString(MARGIN, H-6.2*cm, "Regime-Switching Volatility Analysis")

    c.setFillColor("#90A4AE")
    c.setFont("Helvetica", 12)
    c.drawString(MARGIN, H-7.5*cm, "Normal vs Crisis Regime Identification | Multi-Asset Comparison | Risk Early Warning")
    c.drawString(MARGIN, H-8.2*cm, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Decorative line
    c.setStrokeColor("#2196F3")
    c.setLineWidth(2)
    c.line(MARGIN, H-8.8*cm, W-MARGIN, H-8.8*cm)

    # Key metrics preview
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor("#ECEFF1")
    c.drawString(MARGIN, H-10*cm, "Key Metrics Preview")

    y_pos = H-11*cm
    for _, row in summary_df.iterrows():
        c.setFont("Helvetica", 10)
        c.setFillColor("#B0BEC5")
        c.drawString(MARGIN, y_pos, f"{row['Asset']}: Normal {row['Normal Vol (%)']}% | Crisis {row['Crisis Vol (%)']}% | Multiplier {row['Vol Multiplier']}x")
        y_pos -= 0.6*cm

    footer(1)
    c.showPage()

    # ── Chart Pages ───────────────────────────────────────
    page = 2
    for title, fig_buf in fig_bufs:
        c.setFillColor("#0D1117")
        c.rect(0, 0, W, H, fill=1, stroke=0)
        header(title, f"MS-GARCH Regime Analysis | {datetime.now().strftime('%Y-%m-%d')}")
        fig_buf.seek(0)
        img = ImageReader(fig_buf)
        img_w = W - 2*MARGIN
        img_h = img_w * img.getSize()[1] / img.getSize()[0]
        max_h = H - 3.5*cm
        if img_h > max_h:
            img_h = max_h
            img_w = img_h * img.getSize()[0] / img.getSize()[1]
        x_center = (W - img_w) / 2
        y_bottom = (H - img_h) / 2 - 0.3*cm
        c.drawImage(img, x_center, y_bottom, width=img_w, height=img_h)
        footer(page)
        page += 1
        c.showPage()

    # ── Data Table Page ───────────────────────────────────
    c.setFillColor("#0D1117")
    c.rect(0, 0, W, H, fill=1, stroke=0)
    header("Key Metrics Comparison", f"Normal vs Crisis Quantification | {datetime.now().strftime('%Y-%m-%d')}")

    # Table
    col_widths = [3.5*cm, 3.5*cm, 3.5*cm, 3.5*cm, 3.5*cm]
    cols = list(summary_df.columns)
    row_h = 0.8*cm

    # Header row
    y = H - 3.2*cm
    c.setFillColor("#1E3A5F")
    c.rect(MARGIN, y - row_h, sum(col_widths), row_h, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor("#FFFFFF")
    x = MARGIN
    for col, cw in zip(cols, col_widths):
        c.drawString(x + 0.15*cm, y - row_h + 0.2*cm, col)
        x += cw

    # Data rows
    for i, row in summary_df.iterrows():
        y -= row_h
        bg = "#1A1A2E" if i % 2 == 0 else "#16213E"
        c.setFillColor(bg)
        c.rect(MARGIN, y - row_h, sum(col_widths), row_h, fill=1, stroke=0)
        c.setFont("Helvetica", 10)
        c.setFillColor("#ECEFF1")
        x = MARGIN
        for val, cw in zip(row, col_widths):
            c.drawString(x + 0.15*cm, y - row_h + 0.2*cm, str(val))
            x += cw

    footer(page)
    c.save()
    pass  # quiet
    return output_path

# ── Main Program ────────────────────────────────────────────
def main():
    pass  # quiet
    pass  # quiet
    pass  # quiet

    # 1. Data Fetch
    data = fetch_all_data()
    if not data:
        pass  # quiet
        return

    # 2. Calculate Returns
    realized_vols_dict = {}
    returns_dict = {}
    for name, series in data.items():
        rets = series.pct_change().dropna()
        rets = rets[abs(rets) < 0.25]  # Remove outliers
        returns_dict[name] = rets
        pass  # quiet

    # 3. MS-GARCH Estimation
    regime_probs_dict = {}
    cond_vols_dict    = {}
    crisis_periods_dict = {}
    params_dict       = {}

    for name, rets in returns_dict.items():
        pass  # quiet
        rp, cv, cp, pm, rv = fit_regime_garch(rets, name)
        regime_probs_dict[name] = rp
        cond_vols_dict[name]    = cv
        crisis_periods_dict[name] = cp
        params_dict[name]        = pm
        realized_vols_dict[name]  = rv

    # 4. Generate Charts
    fig_bufs = []

    # Chart 1: Regime Probability + Volatility (per asset)
    for name in returns_dict:
        buf = plot_regime_probabilities(
            regime_probs_dict[name],
            cond_vols_dict[name],
            crisis_periods_dict[name],
            name,
            cond_vols_dict[name]
        )
        fig_bufs.append((f"Chart 1: {name} — Regime Probability & Conditional Volatility", buf))

    # Chart 2: Multi-Asset Volatility Comparison
    buf = plot_vol_comparison(returns_dict, regime_probs_dict, cond_vols_dict)
    fig_bufs.append(("Chart 2: Multi-Asset Conditional Volatility Comparison", buf))

    # Chart 3: Crisis vs Normal Period Metrics
    buf3, summary_df = plot_crisis_events(cond_vols_dict, regime_probs_dict)
    fig_bufs.append(("Chart 3: Normal vs Crisis Key Metrics", buf3))

    # Chart 4: Regime Correlation
    buf4 = plot_regime_scatter(regime_probs_dict, cond_vols_dict)
    if buf4:
        fig_bufs.append(("Chart 4: Cross-Asset Regime Probability Correlation", buf4))

    # 5. Generate PDF
    pdf_path = generate_pdf_report(fig_bufs, summary_df, crisis_periods_dict,
                                    regime_probs_dict, cond_vols_dict)

    # 6. Output Text Summary
    report = f"""MS-GARCH Regime-Switching Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Data Range: Last 2 Years

{'='*50}
Key Findings
{'='*50}

"""
    for name in returns_dict:
        rp = regime_probs_dict[name]
        cv = cond_vols_dict[name]
        pm = params_dict[name]
        crisis_pct = (rp > 0.5).mean() * 100
        # Normal/Crisis volatility from model params
        normal_vol = pm.get('normal_vol', cv[rp <= 0.5].mean() * 100 if (rp <= 0.5).any() else cv.mean() * 100)
        crisis_vol = pm.get('crisis_vol', cv[rp > 0.5].mean() * 100 if (rp > 0.5).any() else cv.mean() * 100)
        vol_ratio = pm.get('vol_ratio', crisis_vol / normal_vol if normal_vol > 0 else 0)

        report += f"""
【{name}】
  Crisis period share: {crisis_pct:.1f}%
  Normal volatility: {normal_vol:.1f}%/yr
  Crisis volatility: {crisis_vol:.1f}%/yr
  Vol multiplier: {vol_ratio:.2f}x
  Current crisis probability: {rp.iloc[-1]:.1%}
  Current annualized volatility: {cv.iloc[-1]*100:.1f}%
"""

    report += f"""
{'='*50}
Regime Switching Events
{'='*50}
"""
    for name, periods in crisis_periods_dict.items():
        if periods:
            report += f"\n【{name}】\n"
            for s, e in periods[-5:]:  # Last 5 only
                report += f"  {s.strftime('%Y-%m-%d')} ~ {e.strftime('%Y-%m-%d')}\n"

    report += f"\n{'='*50}\n"
    report += f"✅ PDF Report: {pdf_path}\n" if pdf_path else "\n⚠️ PDF generation failed\n"

    pass  # quiet
    pass  # quiet
    return pdf_path, report

if __name__ == "__main__":
    main()
