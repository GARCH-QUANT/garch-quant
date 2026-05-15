#!/usr/bin/env python3.12
"""
GARCH 波动率日报
数据源优先级：yfinance(缓存) > Alpha Vantage > stooq > akshare
品种：SPY, QQQ, AAPL, TSLA
"""

import warnings, os, time, json
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import norm
from datetime import datetime, timedelta
import urllib.request
import urllib.error

TODAY = datetime.now().strftime("%Y-%m-%d")
TICKERS = ["SPY", "QQQ", "AAPL", "TSLA"]
CACHE_DIR = os.path.expanduser("~/.hermes/garch_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")
CACHE_TTL_HOURS = 4

# ── 缓存层 ─────────────────────────────────────────────────
def cache_path(ticker: str) -> str:
    return f"{CACHE_DIR}/{ticker}.json"

def read_cache(ticker: str) -> pd.DataFrame | None:
    """读缓存，TTL 内直接返回"""
    fpath = cache_path(ticker)
    if not os.path.exists(fpath):
        return None
    age = time.time() - os.path.getmtime(fpath)
    if age > CACHE_TTL_HOURS * 3600:
        return None
    with open(fpath) as f:
        d = json.load(f)
    df = pd.DataFrame(d["data"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()

def write_cache(ticker: str, df: pd.DataFrame):
    fpath = cache_path(ticker)
    records = df.reset_index().to_dict("records")
    for r in records:
        if "date" in r and hasattr(r["date"], "isoformat"):
            r["date"] = r["date"].isoformat()
    with open(fpath, "w") as f:
        json.dump({"data": records}, f)

# ── 数据源 1: yfinance（带重试） ───────────────────────────
def fetch_yfinance(ticker: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        df = t.history(period="3mo", auto_adjust=True)
        if df.empty or len(df) < 30:
            return None
        return df[["Close"]].rename(columns={"Close": "close"})
    except Exception as e:
        pass  # quiet
        return None

# ── 数据源 2: Alpha Vantage ─────────────────────────────────
def fetch_alpha_vantage(ticker: str) -> pd.DataFrame | None:
    if not ALPHA_VANTAGE_KEY:
        return None
    url = (f"https://www.alphavantage.co/query"
           f"?function=TIME_SERIES_DAILY&symbol={ticker}"
           f"&apikey={ALPHA_VANTAGE_KEY}&outputsize=compact")
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            d = json.loads(r.read())
        ts = d.get("Time Series (Daily)", {})
        if not ts:
            return None
        records = [{"date": k, "close": float(v["4. close"])} for k, v in ts.items()]
        df = pd.DataFrame(records).sort_values("date")
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")[["close"]]
    except Exception as e:
        pass  # quiet
        return None

# ── 数据源 3: stooq（无需 key） ──────────────────────────────
def fetch_stooq(ticker: str) -> pd.DataFrame | None:
    # stooq 用 .US 后缀
    sym = ticker if "." in ticker else f"{ticker}.US"
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=120)).strftime("%Y%m%d")
    url = f"https://stooq.com/q/d/l/?s={sym.lower()}&d1={start}&d2={end}&i=d"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            txt = r.read().decode()
        lines = txt.strip().split("\n")
        if len(lines) < 5:
            return None
        cols = lines[0].split(",")
        data = []
        for row in lines[1:]:
            parts = row.split(",")
            if len(parts) >= 5:
                try:
                    data.append({
                        "date": parts[0].strip(),
                        "close": float(parts[4].strip())
                    })
                except:
                    pass
        if not data:
            return None
        df = pd.DataFrame(data).sort_values("date")
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")[["close"]]
    except Exception as e:
        pass  # quiet
        return None

# ── 数据源 4: akshare（东方财富备用） ───────────────────────
def fetch_akshare(ticker: str) -> pd.DataFrame | None:
    try:
        import akshare as ak
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=120)).strftime("%Y%m%d")
        df = ak.stock_us_hist(symbol=ticker, period="daily",
                               start_date=start, end_date=end, adjust="qfq")
        df = df[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.dropna().sort_values("date").set_index("date")[["close"]]
    except Exception as e:
        pass  # quiet
        return None

# ── 主取数逻辑（优先级 + 缓存） ─────────────────────────────
def get_stock_data(ticker: str) -> pd.DataFrame:
    # 1. 先读缓存
    cached = read_cache(ticker)
    if cached is not None and len(cached) >= 30:
        pass  # quiet
        return cached

    # 2. 尝试各数据源
    fetchers = [
        ("yfinance",   fetch_yfinance),
        ("AlphaVantage", lambda t: fetch_alpha_vantage(t)),
        ("stooq",      fetch_stooq),
        ("akshare",    fetch_akshare),
    ]
    for name, fetcher in fetchers:
        pass  # quiet
        df = fetcher(ticker)
        if df is not None and len(df) >= 30:
            pass  # quiet
            write_cache(ticker, df)
            return df
        pass  # quiet
        time.sleep(1)

    # 3. 全部失败：用缓存（哪怕过期）
    if cached is not None:
        pass  # quiet
        return cached
    return pd.DataFrame()

# ── GARCH(1,1) ──────────────────────────────────────────────
def fit_garch(returns: pd.Series) -> dict:
    try:
        r = returns.dropna() * 100
        if len(r) < 30:
            return {}
        model = arch_model(r, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        res = model.fit(disp="off", options={"maxiter": 500})
        params = res.params.to_dict()
        forecast = res.forecast(horizon=1)
        cond_var = forecast.variance.iloc[-1, 0]
        vol_next = np.sqrt(cond_var) / 100
        persistence = params.get("alpha[1]", 0) + params.get("beta[1]", 0)
        return {
            "mu": params.get("mu", 0),
            "omega": params.get("omega", 0),
            "alpha": params.get("alpha[1]", 0),
            "beta": params.get("beta[1]", 0),
            "vol_next": vol_next,
            "persistence": persistence,
        }
    except Exception as e:
        pass  # quiet
        return {}

def calc_var(returns: pd.Series, vol: float, confidence: float = 0.95) -> float:
    mu = returns.mean()
    z = norm.ppf(1 - confidence)
    return -(mu + z * vol)

# ── 报告生成 ────────────────────────────────────────────────
def generate_report() -> str:
    lines = [
        f"📊 GARCH 波动率日报 {TODAY}",
        "",
        "【波动率预测 & 风险指标】",
        "",
    ]

    all_results = []
    for ticker in TICKERS:
        df = get_stock_data(ticker)
        if df.empty:
            lines.append(f"• {ticker}: 数据获取失败 ❌")
            continue

        ret = df["close"].pct_change().dropna()
        garch = fit_garch(ret)

        if not garch:
            lines.append(f"• {ticker}: GARCH 拟合失败 ❌")
            continue

        vol_annual = garch["vol_next"] * np.sqrt(252)
        var_95 = calc_var(ret, garch["vol_next"], 0.95)

        lines.append(
            f"• {ticker}: "
            f"预测波动率 {vol_annual*100:.1f}%/年 | "
            f"1d VaR(95%) {var_95*100:.2f}%"
        )
        all_results.append((ticker, vol_annual, var_95))

    if len(all_results) >= 2:
        lines.append("")
        lines.append("【波动率排序】（年化）")
        for i, (t, v, _) in enumerate(sorted(all_results, key=lambda x: x[1], reverse=True), 1):
            lines.append(f"  {i}. {t}: {v*100:.1f}%")

    return "\n".join(lines)

if __name__ == "__main__":
    import sys, os
    # 静默所有调试输出，只留报告内容到 stdout
    sys.stderr = open(os.devnull, "w")
    report = generate_report()
    sys.stderr = sys.__stderr__
    pass  # quiet
