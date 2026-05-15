#!/usr/bin/env python3.12
"""
MS-GARCH Crisis Alert — 实时监控 + 推送预警
每小时检查一次，crisis probability > 80% 或波动率偏离 > 2σ 即时推送
"""

import warnings, os, json, time
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Config ─────────────────────────────────────────────────
import os
TOKEN             = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID           = os.environ.get("TELEGRAM_CHANNEL_ID", "-1003786012521")
COOLDOWN_FILE     = "/tmp/ms_garch_alert_cooldown.json"
ALERT_THRESHOLD   = 0.80      # crisis prob > 80% 触发预警
VOL_DEVIATION_THRESHOLD = 2.0  # 偏离 > 2σ 触发预警
COOLDOWN_MINUTES  = 5        # 同品种5分钟内不重复报警

# ── Data Sources (reused from ms_garch_telegram.py) ──────────
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")
TWELVE_DATA_KEY   = os.environ.get("TWELVE_DATA_KEY", "")

TICKERS = {
    "CSI 300":  {"tushare": "000300.SH", "av": "000300", "td": "HSI:IND"},
    "AAPL":    {"tushare": None,        "av": "AAPL",   "td": "AAPL"},
    "TSLA":    {"tushare": None,        "av": "TSLA",   "td": "TSLA"},
}

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
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
    data  = {}
    for name, sources in TICKERS.items():
        tried = []
        if sources.get("tushare"):
            s = get_tushare_data(sources["tushare"], start, end)
            if s is not None and len(s) > 100:
                data[name] = s.dropna()
                continue
            tried.append("tushare")
        td_sym = sources.get("td", "")
        if td_sym:
            time.sleep(1)
            s = get_twelve_data(td_sym, start, end)
            if s is not None and len(s) > 100:
                data[name] = s.dropna()
                continue
            tried.append(f"TwelveData({td_sym})")
    return data

# ── MS-GARCH Core ────────────────────────────────────────────
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

    prob = (realized_vol - low_q) / (high_q - low_q + 1e-10)
    prob = np.clip(prob, 0, 1)
    cond_vol_arr = prob * vol_crisis + (1 - prob) * vol_normal

    # Current values (latest)
    current_prob = float(prob[-1])
    current_vol  = float(cond_vol_arr[-1])
    current_rvol = float(realized_vol[-1])

    # Vol deviation: current realized vol vs 30-day mean
    vol_30d_mean = np.mean(realized_vol[-30:]) if len(realized_vol) >= 30 else current_rvol
    vol_std_30d  = np.std(realized_vol[-30:]) if len(realized_vol) >= 30 else current_rvol * 0.1
    vol_zscore   = (current_rvol - vol_30d_mean) / (vol_std_30d + 1e-10)

    return {
        "name": name,
        "crisis_prob": current_prob,
        "cond_vol": current_vol,
        "realized_vol": current_rvol,
        "vol_normal": vol_normal,
        "vol_crisis": vol_crisis,
        "vol_zscore": vol_zscore,
        "vol_ratio": vol_crisis / vol_normal if vol_normal > 0 else 1.0,
    }

# ── Cooldown Management ───────────────────────────────────────
def load_cooldown():
    if os.path.exists(COOLDOWN_FILE):
        try:
            with open(COOLDOWN_FILE) as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cooldown(cooldown):
    with open(COOLDOWN_FILE, 'w') as f:
        json.dump(cooldown, f, ensure_ascii=False)

def is_cooling_down(name, cooldown):
    if name not in cooldown:
        return False
    last = datetime.fromisoformat(cooldown[name]["last_alert"])
    return (datetime.now() - last).total_seconds() < COOLDOWN_MINUTES * 60

def should_alert(asset_result, cooldown):
    name = asset_result["name"]
    cp   = asset_result["crisis_prob"]
    zscore = abs(asset_result["vol_zscore"])

    # Crisis prob > threshold
    if cp > ALERT_THRESHOLD and not is_cooling_down(name, cooldown):
        return True, f"危机概率 {cp*100:.1f}% 突破 {ALERT_THRESHOLD*100:.0f}% 阈值"

    # Vol deviation > 2σ
    if zscore > VOL_DEVIATION_THRESHOLD and not is_cooling_down(name + "_vol", cooldown):
        direction = "↑" if asset_result["vol_zscore"] > 0 else "↓"
        return True, f"波动率{direction} {zscore:.1f}σ，偏离30日均值超过阈值"

    return False, ""

# ── Alert Builder ────────────────────────────────────────────
def _interpret_asset(r):
    """Generate one-paragraph interpretation for a single asset."""
    cp = r["crisis_prob"]
    cv = r["cond_vol"]
    z  = r["vol_zscore"]
    vr = r.get("vol_ratio", 1.0)
    vn = r.get("vol_normal", 0.0)
    vc = r.get("vol_crisis", 0.0)

    parts = []
    # Crisis regime
    if cp > 0.8:
        parts.append(f"危机概率 {cp*100:.0f}%，已进入高波动状态。")
    elif cp > 0.5:
        parts.append(f"危机概率 {cp*100:.0f}%，处于中间过渡阶段。")
    elif cp > 0.2:
        parts.append(f"危机概率 {cp*100:.0f}%，低风险状态。")
    else:
        parts.append(f"危机概率 {cp*100:.0f}%，市场平稳。")

    # Vol deviation
    if abs(z) > 2:
        direction = "急剧上升" if z > 0 else "显著偏低"
        parts.append(f"波动率{direction}（{z:+.2f}σ），偏离30日均值超过阈值。")
    elif abs(z) > 1:
        direction = "偏高" if z > 0 else "偏低"
        parts.append(f"波动率{direction}（{z:+.2f}σ），在正常区间内。")
    else:
        parts.append(f"波动率基本正常（{z:+.2f}σ）。")

    # Cond vol context
    if cv > 0.40:
        parts.append(f"年化条件波动率 {cv*100:.1f}% 属历史高位。")
    elif cv > 0.25:
        parts.append(f"年化条件波动率 {cv*100:.1f}% 处于中等水平。")
    else:
        parts.append(f"年化条件波动率 {cv*100:.1f}% 属历史低位。")

    return "".join(parts)


def build_alert_html(alerts, all_results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC+8")

    lines = [
        f"🚨 <b>MS-GARCH 危机预警</b>",
        f"🕐 <i>{now}</i>",
        "━━━━━━━━━━━━━━━━━━━━",
        ""
    ]

    # Always interpret ALL assets, not just triggered ones
    for r in all_results:
        emoji = "🔴" if r["crisis_prob"] > 0.8 else ("🟠" if abs(r["vol_zscore"]) > 2 else "🟢")
        lines.append(f"{emoji} <b>{r['name']}</b>")
        lines.append(f"   {_interpret_asset(r)}")
        lines.append(f"   危机概率: {r['crisis_prob']*100:.1f}% | 条件波动率: {r['cond_vol']*100:.1f}%/年 | 波动率偏离: {r['vol_zscore']:+.2f}σ")
        lines.append("")

    # Summary table
    lines.append("📐 <b>当前状态快照</b>")
    lines.append("<code>")
    lines.append(f"{'Asset':<10} {'Crisis%':>8} {'CondVol':>8} {'Vol Z':>8}")
    lines.append("-" * 36)
    for r in all_results:
        cp   = r["crisis_prob"] * 100
        cv   = r["cond_vol"] * 100
        z    = r["vol_zscore"]
        flag = " ⚠️" if cp > 80 or abs(z) > 2 else ""
        lines.append(f"{r['name']:<10} {cp:>7.1f}% {cv:>7.1f}%  {z:>+7.2f}{flag}")
    lines.append("</code>")
    lines.append("")
    lines.append("⚠️ <i>仅供研究参考，非投资建议。</i>")

    return "\n".join(lines)

# ── Telegram Push ────────────────────────────────────────────
def send_telegram(text):
    import requests
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    r = requests.post(url, data={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=30)
    return r.json().get("ok", False)

# ── Main ────────────────────────────────────────────────────
def main():
    import sys
    # Suppress routine logs; only print when alerting
    class SilentLogger:
        def write(self, msg): pass
        def flush(): pass
    sys.stdout = SilentLogger()
    sys.stderr = SilentLogger()

    # Load cooldown
    cooldown = load_cooldown()

    # Fetch data
    data = fetch_all_data()
    if not data:
        return

    # Compute metrics
    results = []
    for name, series in data.items():
        rets = series.pct_change().dropna()
        rets = rets[abs(rets) < 0.25]
        if len(rets) < 30:
            continue
        result = fit_regime_garch(rets, name)
        results.append(result)

    if not results:
        return

    # Check alerts
    triggered = []
    for r in results:
        alert, reason = should_alert(r, cooldown)
        if alert:
            triggered.append({"result": r, "reason": reason})

    now_str = datetime.now().isoformat()

    if triggered:
        html = build_alert_html(triggered, results)
        ok = send_telegram(html)
        if ok:
            for alert in triggered:
                name = alert["result"]["name"]
                cooldown[name] = {"last_alert": now_str}
                if "vol" in alert["reason"]:
                    cooldown[name + "_vol"] = {"last_alert": now_str}
            save_cooldown(cooldown)

if __name__ == "__main__":
    main()
