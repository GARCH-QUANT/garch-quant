#!/usr/bin/env python3
"""
GARCH-VIX Volatility Signal Strategy
=====================================
CSI 300 GARCH(1,1) Forecast + VIX Cross-Asset Confirmation
→ Trading signals: LONG / SHORT / NEUTRAL on vol expansion/contraction

Signals:
  - Vol level:   Low(<15%) | Medium(15-25%) | High(>25%)
  - Vol change:  Expanding | Stable | Contracting
  - VIX regime:  Calm(<20) | Elevated(20-30) | Stress(>30)
  - Signal:      🟢 BUY vol (expect rise) | 🔴 SELL vol (expect fall) | ⚪ NEUTRAL

Data: Tushare Pro (CSI 300) + Yahoo Finance (VIX proxy)
Charts: dark theme → push to Telegram
"""

import warnings, os, sys, requests
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ── Config ──────────────────────────────────────────────────────────────────
import os
TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT   = os.environ.get("TELEGRAM_CHANNEL_ID", "-1003786012521")
TUSHARE_TOKEN   = os.environ.get("TUSHARE_TOKEN", "")
OUTPUT_DIR      = os.path.expanduser("~/.hermes/cron/output")
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

# ── Data Fetch ──────────────────────────────────────────────────────────────
def fetch_csi300():
    import tushare as ts
    tushare_token = os.environ.get("TUSHARE_TOKEN", "")
    ts.set_token(tushare_token)
    pro = ts.pro_api(tushare_token)
    end = datetime.today().strftime('%Y%m%d')
    start = (datetime.today() - timedelta(days=600)).strftime('%Y%m%d')
    df = pro.index_daily(ts_code='000300.SH', start_date=start, end_date=end)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    return df

def fetch_vix():
    """VIX proxy via yfinance — try ^VIX first, then VIXY, then use fallback"""
    try:
        import yfinance as yf
        end = datetime.today().strftime('%Y-%m-%d')
        start = (datetime.today() - timedelta(days=600)).strftime('%Y-%m-%d')
        vix = yf.download('^VIX', start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(vix, tuple):
            vix = vix[0]
        if hasattr(vix.columns, 'droplevel'):
            vix.columns = vix.columns.droplevel(1) if hasattr(vix.columns, 'levels') else vix.columns
        vix.index = pd.to_datetime(vix.index).tz_localize(None) if vix.index.tz else vix.index
        vix.columns = [c.lower() for c in vix.columns]
        vix = vix[['close']].rename(columns={'close': 'vix'})
        if vix['vix'].dropna().empty or vix['vix'].dropna().min() < 5:
            raise ValueError("VIX data too sparse")
        vix['vix'] = vix['vix'].ffill().bfill()
        print(f"    VIX: {vix['vix'].iloc[-1]:.1f} (^VIX)")
        return vix
    except Exception as e:
        print(f"    [VIX ^VIX error: {e}]")
        try:
            import yfinance as yf
            vix2 = yf.download('VIXY', start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(vix2, tuple):
                vix2 = vix2[0]
            if hasattr(vix2.columns, 'droplevel'):
                vix2.columns = vix2.columns.droplevel(1) if hasattr(vix2.columns, 'levels') else vix2.columns
            vix2.index = pd.to_datetime(vix2.index).tz_localize(None) if vix2.index.tz else vix2.index
            vix2.columns = [c.lower() for c in vix2.columns]
            vix2 = vix2[['close']].rename(columns={'close': 'vix'})
            vix2['vix'] = vix2['vix'].ffill().bfill()
            print(f"    VIX: {vix2['vix'].iloc[-1]:.1f} (VIXY ETF)")
            return vix2
        except Exception as e2:
            print(f"    [VIXY also failed: {e2}]")
            # Return None; caller will use a fallback series aligned to CSI dates
            return None

def build_vix_fallback(csi_index):
    """Build a flat VIX=20 fallback series aligned to CSI index."""
    return pd.Series(20.0, index=csi_index)

# ── GARCH(1,1) Fit & Forecast ──────────────────────────────────────────────
def fit_garch(ret, horizon=5):
    """Fit GARCH(1,1) and produce 1-step and multi-step forecasts."""
    from arch import arch_model

    r = ret.dropna().values.astype(float)
    if len(r) < 100:
        raise ValueError(f"Too few observations: {len(r)}")
    model = arch_model(r, vol='Garch', p=1, q=1, dist='normal')
    res = model.fit(disp='off', show_warning=False, options={'maxiter': 500})

    # Extract params
    omega  = res.params.get('omega', 0)
    alpha  = res.params.get('alpha[1]', 0)
    beta   = res.params.get('beta[1]', 0)
    nu     = res.params.get('nu', 10)

    # Conditional vol series (back to decimal)
    cond_vol = res.conditional_volatility / 100

    # 1-step ahead forecast (annualized)
    h_1 = omega + (alpha + beta) * (cond_vol[-1] * 100)**2
    vol_1step = np.sqrt(h_1) * np.sqrt(252) * 100

    # Multi-step: converges to unconditional variance ω/(1-α-β)
    h_forecast = np.zeros(horizon)
    h_forecast[0] = h_1
    for t in range(1, horizon):
        h_forecast[t] = omega + (alpha + beta) * h_forecast[t - 1]

    vol_forecast = np.sqrt(h_forecast) * np.sqrt(252) * 100

    return {
        'omega': omega, 'alpha': alpha, 'beta': beta, 'nu': nu,
        'cond_vol': cond_vol,          # T x 1 (decimal)
        'vol_1step': vol_1step,        # float (annualized %)
        'vol_forecast': vol_forecast, # horizon x 1 (annualized %)
        'params': res.params,
        'aic': res.aic,
    }

# ── Volatility Signal Engine ───────────────────────────────────────────────
def compute_signal(garch_res, vix_val, cond_vol_series):
    """
    Combine GARCH forecast + VIX level + recent vol trend
    → LONG / SHORT / NEUTRAL with confidence
    """
    vol_now   = cond_vol_series.iloc[-5:].mean() * np.sqrt(252) * 100  # recent 5d avg ann vol
    vol_1m    = cond_vol_series.iloc[-22:].mean() * np.sqrt(252) * 100  # 1-month avg
    vol_3m    = cond_vol_series.iloc[-66:].mean() * np.sqrt(252) * 100  # 3-month avg
    vol_1step = garch_res['vol_1step']
    vol_fc    = garch_res['vol_forecast']

    # Vol regime thresholds
    vol_level = "HIGH" if vol_now > 25 else "LOW" if vol_now < 15 else "MEDIUM"

    # Trend: is vol expanding or contracting?
    vol_trend = vol_now - vol_1m
    trend_label = "EXPANDING" if vol_trend > 1.0 else "CONTRACTING" if vol_trend < -1.0 else "STABLE"

    # VIX regime
    vix_level = "STRESS" if vix_val > 30 else "CALM" if vix_val < 18 else "ELEVATED"

    # Directional signal from GARCH forecast vs current level
    signal_strength = 0  # -2 to +2
    vol_change = vol_1step - vol_now
    if vol_1step > vol_now + 2:
        signal_strength += 1   # GARCH says vol will rise
    elif vol_1step < vol_now - 2:
        signal_strength -= 1   # GARCH says vol will fall

    # VIX confirmation: high VIX supports LONG vol (stress), low VIX supports SHORT vol
    if vix_val > 28 and signal_strength >= 0:
        signal_strength += 1   # VIX confirms vol expansion
    elif vix_val < 18 and signal_strength <= 0:
        signal_strength -= 1   # VIX confirms vol contraction

    # 5-day trend confirmation
    if vol_fc[-1] > vol_now + 3:
        signal_strength += 1
    elif vol_fc[-1] < vol_now - 3:
        signal_strength -= 1

    # Direction label for rationale
    vol_direction = "expanding" if vol_change > 0 else "contracting"

    # Practical interpretation
    if signal_strength >= 2:
        conf_label = "HIGH"
        sig_label = "🔴 SELL VOL"
        interp = (
            f"当前波动率处于{vol_level}区间（{vol_now:.1f}%），"
            f"GARCH 预测 1-step 升至 {vol_1step:.1f}%。"
            f"VIX={vix_val:.1f}（{vix_level}）提供跨市场确认。\n"
            f"策略含义：波动率从低位扩张，机构交易活跃度上升，"
            f"趋势持续中不宜盲目做空，应等待波动率见顶信号（VIX>30 或 1-day 预测>25%）。"
        )
    elif signal_strength <= -2:
        conf_label = "HIGH"
        sig_label = "🟢 BUY VOL"
        interp = (
            f"波动率从高位回落或处于{vol_level}区间（{vol_now:.1f}%），"
            f"GARCH 预测 {vol_1step:.1f}%，VIX={vix_val:.1f}（{vix_level}）。\n"
            f"策略含义：市场平静期权便宜，波动率回升时做多 gamma 可获得超额收益。"
        )
    elif signal_strength > 0:
        conf_label = "MEDIUM"
        sig_label = "🟡 BUY VOL (cautious)"
        interp = f"信号偏弱，波动率趋势不明确，建议观望。"
    elif signal_strength < 0:
        conf_label = "MEDIUM"
        sig_label = "🟡 SELL VOL (cautious)"
        interp = f"信号偏弱，波动率趋势不明确，建议观望。"
    else:
        conf_label = "LOW"
        sig_label = "⚪ NEUTRAL"
        interp = f"波动率处于中性区间（{vol_now:.1f}%），无明确方向，等待突破。"

    signal = sig_label
    confidence = conf_label
    rationale = f"Vol {vol_direction} ({vol_now:.1f}%→{vol_1step:.1f}%), VIX={vix_val:.1f} confirm" if signal_strength >= 2 else (
        f"Vol {vol_direction} ({vol_now:.1f}%→{vol_1step:.1f}%), VIX={vix_val:.1f} stress" if signal_strength <= -2 else
        f"Moderate signal, vol={vol_now:.1f}%→{vol_1step:.1f}%" if abs(signal_strength) == 1 else
        f"Vol stable near {vol_now:.1f}%, VIX={vix_val:.1f}"
    )

    return {
        'signal': signal,
        'confidence': confidence,
        'vol_now': vol_now,
        'vol_1step': vol_1step,
        'vol_5day': vol_fc,
        'vol_trend': vol_trend,
        'vol_level': vol_level,
        'vix_val': vix_val,
        'vix_level': vix_level,
        'trend_label': trend_label,
        'rationale': rationale,
        'interpretation': interp,
        'alpha': garch_res['alpha'],
        'beta': garch_res['beta'],
    }

# ── Charts ──────────────────────────────────────────────────────────────────
def chart_vol_signal(cond_vol, dates, vol_forecast, signal_info, output_path, horizon=5):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Top: vol time series with signal zone
    ax = axes[0]
    ann_vol = cond_vol * np.sqrt(252) * 100

    ax.plot(dates, ann_vol, color='#58A6FF', lw=1.2, label='GARCH(1,1) σ (ann.)')

    # Forecast zone
    last_date = dates[-1]
    fc_dates = pd.date_range(last_date, periods=horizon+1, freq='B')[1:]
    ax.plot(fc_dates, vol_forecast, 's--', color='#FFA726', lw=2,
            label=f'5-day forecast ({signal_info["vol_1step"]:.1f}%)', alpha=0.9)

    # Signal zone coloring
    vol_now = signal_info['vol_now']
    ax.axhline(vol_now, color='#58A6FF', lw=0.5, ls=':', alpha=0.5)
    ax.fill_between(dates, 0, ann_vol, alpha=0.15, color='#58A6FF')

    # Regime bands
    ax.axhline(25, color='#E53935', lw=0.8, ls='--', alpha=0.5, label='High (25%)')
    ax.axhline(15, color='#4CAF50', lw=0.8, ls='--', alpha=0.5, label='Low (15%)')
    ax.axhline(vol_now, color='white', lw=0.8, ls='-', alpha=0.3)

    ax.set_facecolor('#0D1117')
    ax.set_ylabel('Annualized Volatility (%)', color='#E6EDF3')
    ax.set_title(f"CSI 300 — GARCH Volatility Signal: {signal_info['signal']}",
                 color='#E6EDF3', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom: signal panel
    ax2 = axes[1]
    signal = signal_info['signal']
    conf = signal_info['confidence']

    colors_map = {'🔴 SELL VOL': '#E53935', '🟢 BUY VOL': '#4CAF50',
                  '🟡 BUY VOL (cautious)': '#FFA726', '🟡 SELL VOL (cautious)': '#FFA726',
                  '⚪ NEUTRAL': '#8B949E'}
    bg_color = colors_map.get(signal, '#8B949E')

    ax2.set_facecolor(bg_color + '22')
    ax2.text(0.5, 0.65, signal, fontsize=22, fontweight='bold',
             ha='center', va='center', color='white',
             transform=ax2.transAxes)
    ax2.text(0.5, 0.35, f"Confidence: {conf} | {signal_info['rationale']}",
             fontsize=11, ha='center', va='center', color='#E6EDF3',
             transform=ax2.transAxes)

    ax2.text(0.02, 0.15, f"VIX: {signal_info['vix_val']:.1f} ({signal_info['vix_level']})\n"
                          f"Vol: {signal_info['vol_now']:.1f}% → {signal_info['vol_1step']:.1f}% ({signal_info['trend_label']})\n"
                          f"α={signal_info['alpha']:.3f} β={signal_info['beta']:.3f}",
             fontsize=9, ha='left', va='center', color='#8B949E',
             transform=ax2.transAxes)

    ax2.axis('off')
    ax2.set_title("Signal Panel", color='#E6EDF3', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    return output_path

def chart_vix_comparison(dates, vix_series, csi_vol, output_path):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax2 = ax.twinx()

    vix_s = vix_series.reindex(dates).ffill().fillna(20)
    csi_v = csi_vol * np.sqrt(252) * 100

    ax.plot(dates, vix_s, color='#F78166', lw=1.2, label='VIX', alpha=0.8)
    ax2.plot(dates, csi_v, color='#58A6FF', lw=1.2, label='CSI 300 σ (GARCH)', alpha=0.8)

    ax.set_ylabel('VIX', color='#F78166', fontsize=10)
    ax2.set_ylabel('Annualized Volatility (%)', color='#58A6FF', fontsize=10)
    ax.set_title('VIX vs CSI 300 GARCH Volatility', color='#E6EDF3', fontsize=12)
    ax.legend(loc='upper left'); ax2.legend(loc='upper right')
    ax.set_facecolor('#0D1117')
    ax.tick_params(axis='y', labelcolor='#F78166')
    ax2.tick_params(axis='y', labelcolor='#58A6FF')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    return output_path

# ── Telegram Helpers ────────────────────────────────────────────────────────
def send_telegram_photo(photo_path, caption=""):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        with open(photo_path, 'rb') as f:
            files = {'photo': f}
            data = {'chat_id': TELEGRAM_CHAT, 'caption': caption, 'parse_mode': 'HTML'}
            r = requests.post(url, files=files, data=data, timeout=30)
            return r.json().get('ok', False)
    except Exception as e:
        print(f"[send_photo error] {e}")
        return False

def send_telegram_text(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={'chat_id': TELEGRAM_CHAT, 'text': text, 'parse_mode': 'HTML'}, timeout=30)
        return r.json().get('ok', False)
    except Exception as e:
        print(f"[send_text error] {e}")
        return False

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("GARCH-VIX Volatility Signal Strategy — CSI 300")
    print("=" * 60)

    # 1. Fetch data
    print("\n[1/5] Fetching CSI 300...")
    csi = fetch_csi300()
    print(f"    Rows: {len(csi)}, {csi['trade_date'].min().date()} → {csi['trade_date'].max().date()}")

    print("\n[2/5] Fetching VIX...")
    vix = fetch_vix()
    vix_val = vix['vix'].iloc[-1] if vix is not None and len(vix) > 0 else 20.0
    print(f"    VIX: {vix_val:.1f}")

    # 2. Merge
    print("\n[3/5] Merging & computing...")
    df = csi.set_index('trade_date')

    # Align VIX to CSI 300 dates (fallback if VIX data unavailable)
    if vix is not None and len(vix) > 0:
        vix_clean = vix['vix'].reindex(df.index).ffill().bfill()
    else:
        vix_clean = build_vix_fallback(df.index)

    df['vix'] = vix_clean
    df = df.dropna(subset=['returns'])
    ret = df['returns']
    print(f"    Merged dataset: {len(df)} rows, {df.index.min().date()} → {df.index.max().date()}")
    print(f"    Returns: {len(ret)} obs, range [{ret.min():.4f}, {ret.max():.4f}]")
    if len(ret) < 100:
        raise ValueError(f"Too few observations after merge: {len(ret)}")

    # 3. Fit GARCH
    print("\n[4/5] Fitting GARCH(1,1)...")
    res = fit_garch(ret, horizon=5)
    print(f"    ω={res['omega']:.2e}  α={res['alpha']:.4f}  β={res['beta']:.4f}  ν={res['nu']:.1f}")
    print(f"    AIC={res['aic']:.1f}")

    # 4. Compute signal
    print("\n[5/5] Computing signal...")
    signal = compute_signal(res, vix_val, df['returns'])
    print(f"    Signal: {signal['signal']} ({signal['confidence']})")
    print(f"    Vol: {signal['vol_now']:.1f}% → {signal['vol_1step']:.1f}%")
    print(f"    VIX: {signal['vix_val']:.1f} ({signal['vix_level']}), Trend: {signal['trend_label']}")

    # 5. Charts
    print("\n[6/6] Generating charts...")
    dates = df.index.values
    chart_path = f"{OUTPUT_DIR}/garch_vix_signal.png"
    chart_vol_signal(df['returns'], dates, signal['vol_5day'], signal, chart_path)

    vix_path = f"{OUTPUT_DIR}/garch_vix_comparison.png"
    chart_vix_comparison(dates, df['vix'], df['returns'], vix_path)

    # 6. Telegram
    print("\n[7/7] Sending Telegram report...")
    report = (
        f"📊 <b>GARCH-VIX 波动率交易信号</b>\n"
        f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M UTC+8')}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{signal['signal']}  |  置信度: <b>{signal['confidence']}</b>\n\n"
        f"📐 <b>波动率状态</b>\n"
        f"  当前: <b>{signal['vol_now']:.1f}%</b> ({signal['vol_level']})\n"
        f"  预测: <b>{signal['vol_1step']:.1f}%</b> (GARCH 1-step)\n"
        f"  趋势: {signal['trend_label']} ({signal['vol_trend']:+.1f}% vs 1M avg)\n\n"
        f"🌐 <b>VIX 确认</b>\n"
        f"  当前: <b>{signal['vix_val']:.1f}</b> ({signal['vix_level']})\n\n"
        f"💡 <b>理由</b>\n{signal['rationale']}\n\n"
        f"💡 <b>解读</b>\n{signal['interpretation']}\n\n"
        f"🔮 <b>5日波动率路径</b>\n" +
        "".join([f"  Day {i+1}: <b>{v:.1f}%</b>\n" for i, v in enumerate(signal['vol_5day'])]) +
        f"\n⚠️ <i>仅供参考，不构成投资建议</i>\n"
        f"#GARCH #VIX #波动率信号"
    )
    send_telegram_text(report)
    send_telegram_photo(chart_path, "📊 GARCH-VIX 波动率信号图")
    send_telegram_photo(vix_path, "🌐 VIX vs CSI 300 波动率对比")
    print("[DONE]")
    return signal

if __name__ == '__main__':
    main()