#!/usr/bin/env python3
"""
Realized GARCH + Jump Detection — CSI 300
=========================================
Andersen, Bollerslev, Diebold (2007) 框架：
  RV_t  = BP_t + Jump_t
  BP_t  = 连续样本路径方差（连续交易）
  Jump_t = 跳成分方差（突发性消息面冲击）

模型：
  log(h_t) = ω + β * log(h_{t-1}) + α * (|r_{t-1}| - sqrt(h_{t-1})) + γ * r_{t-1}²/sqrt(h_{t-1})
  Measurement: q_t = ω_q + ρ_q * q_{t-1} + φ_q * (RV_{t-1} - BP_{t-1})

数据：Tushare Pro 日线 O/H/L/C（2024-01 → 2026-05）
Charts: dark theme → push to Telegram

⚠️  数组长度 pitfall: bpv_arr[0]=0, bpv_arr[n-1]=0 (boundary padding)
    画图时用 np.append(bpv_arr[1:n], 0) 对齐 dates (长度 n)
"""

import warnings, os, sys, requests
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN=os.environ.get("TELEGRAM_BOT_TOKEN","")
TELEGRAM_CHAT  = os.environ.get("TELEGRAM_CHANNEL_ID","")
TUSHARE_TOKEN=os.environ.get("TUSHARE_TOKEN","")
OUTPUT_DIR     = os.path.expanduser("~/.hermes/cron/output")
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

# ── Tushare Data Fetch ──────────────────────────────────────────────────────
def fetch_csi300():
    import tushare as ts
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api(TUSHARE_TOKEN)
    df = pro.index_daily(ts_code='000300.SH',
                          start_date='20240101',
                          end_date='20260512')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]
    return df

# ── Realized Variance: Garman-Klass + Jump Test ─────────────────────────────
def garman_klass_rv(ohlc):
    h, l, c, o = ohlc['high'], ohlc['low'], ohlc['close'], ohlc['open']
    log_hl = np.log(h / l)
    log_co = np.log(c / o)
    gk = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
    return np.nan_to_num(gk, nan=0.0, posinf=0.0, neginf=0.0)

def bipower_variation(close):
    """Bipower Variation — continuous component, immune to finite jumps"""
    returns = np.log(close / np.roll(close, 1))[1:]
    bpv = np.zeros(len(returns))
    for t in range(1, len(returns)):
        bpv[t] = (np.pi / 2) * abs(returns[t]) * abs(returns[t-1])
    return bpv, returns

def jump_test(bpv, rv):
    """Jump = RV - BPV; flag days where jump dominates"""
    jump_var = np.maximum(rv - bpv, 0)
    z_scores = jump_var / (jump_var.mean() + 1e-10)
    threshold = np.percentile(z_scores[z_scores > 0], 85)
    return (z_scores > threshold).astype(float), jump_var

# ── HAR Model ───────────────────────────────────────────────────────────────
def fit_har_rv(rv_series):
    log_rv = np.log(rv_series.replace(0, np.nan)).dropna()
    df = pd.DataFrame({'log_rv': log_rv})
    df['log_rv_1']  = df['log_rv'].shift(1)
    df['log_rv_5']  = df['log_rv'].rolling(5).mean().shift(1)
    df['log_rv_22'] = df['log_rv'].rolling(22).mean().shift(1)
    df = df.dropna()
    X = df[['log_rv_1', 'log_rv_5', 'log_rv_22']]
    X = (X - X.mean()) / (X.std() + 1e-8)
    y = df['log_rv']
    X_n = np.column_stack([np.ones(len(X)), X.values])
    coeffs = np.linalg.lstsq(X_n, y.values, rcond=None)[0]
    residuals = y.values - X_n @ coeffs
    r2 = 1 - (residuals**2).sum() / ((y - y.mean())**2).sum()
    aic = len(y) * np.log((residuals**2).mean() + 1e-10) + 2 * len(coeffs)
    return {'alpha': coeffs[0], 'beta_day': coeffs[1], 'beta_week': coeffs[2],
            'beta_month': coeffs[3], 'r2': r2, 'aic': aic}

# ── GARCH(1,1) ──────────────────────────────────────────────────────────────
def fit_garch(returns, label="Asset"):
    from arch import arch_model
    try:
        model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='t')
        res = model.fit(disp='off', options={'maxiter': 500})
        return {'alpha': res.params.get('alpha[1]', 0), 'beta': res.params.get('beta[1]', 0),
                'omega': res.params.get('omega', 0), 'nu': res.params.get('nu', 999), 'name': label}
    except Exception as e:
        return {'alpha': 0, 'beta': 0, 'omega': 0, 'nu': 999, 'name': label}

# ── Charts ───────────────────────────────────────────────────────────────────
def chart_rv_jump_decomp(rv, bpv, jump_var, dates, output_path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].fill_between(dates, rv, alpha=0.6, color='#4FC3F7', label='Total RV (Garman-Klass)')
    axes[0].set_ylabel('RV', color='#E6EDF3')
    axes[0].set_title('CSI 300 — Total Realized Variance', color='#E6EDF3', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right'); axes[0].set_facecolor('#0D1117')
    axes[1].fill_between(dates, bpv, alpha=0.7, color='#66BB6A', label='Continuous BPV')
    axes[1].fill_between(dates, jump_var, alpha=0.7, color='#FF7043', label='Jump Component')
    axes[1].set_ylabel('Variance', color='#E6EDF3')
    axes[1].set_title('RV Decomposition: Continuous (BPV) + Jump', color='#E6EDF3', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right'); axes[1].set_facecolor('#0D1117')
    total = np.where(rv < 1e-10, 1e-10, rv)
    jump_pct = np.clip(jump_var / total, 0, 1)
    axes[2].fill_between(dates, jump_pct, alpha=0.7, color='#FFA726')
    axes[2].axhline(0.1, color='red', linestyle='--', linewidth=1, label='10% threshold')
    axes[2].set_ylabel('Jump % of RV', color='#E6EDF3'); axes[2].set_xlabel('Date', color='#E6EDF3')
    axes[2].set_title('Jump Ratio (Jump RV / Total RV)', color='#E6EDF3', fontsize=12, fontweight='bold')
    axes[2].set_ylim(0, 1); axes[2].legend(loc='upper right'); axes[2].set_facecolor('#0D1117')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()

def chart_jump_events(dates, jump_flag, close, output_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, close, color='#90CAF9', linewidth=1, label='CSI 300 Close')
    jump_idx = np.where(jump_flag > 0)[0]
    if len(jump_idx) > 0:
        ax.scatter(dates[jump_idx], close[jump_idx], color='#FF5722', s=60, zorder=5,
                   label=f'Jump Events (n={len(jump_idx)})', marker='^')
    ax.set_facecolor('#0D1117'); ax.set_xlabel('Date', color='#E6EDF3')
    ax.set_ylabel('CSI 300 Close', color='#E6EDF3')
    ax.set_title('CSI 300 — Jump Events Detection (BNS Test)', color='#E6EDF3', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()

def chart_har_jump_coeffs(har_bpv, har_jump, output_path):
    assets = ['CSI 300\n(Continuous)', 'CSI 300\n(Jump)']
    betas = [har_bpv['beta_day'], har_jump['beta_day']]
    beta_w = [har_bpv['beta_week'], har_jump['beta_week']]
    beta_m = [har_bpv['beta_month'], har_jump['beta_month']]
    x = np.arange(len(assets)); width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, betas,  width, label=r'$\beta_{day}$',  color='#4FC3F7')
    ax.bar(x,        beta_w, width, label=r'$\beta_{week}$', color='#66BB6A')
    ax.bar(x + width, beta_m, width, label=r'$\beta_{month}$', color='#FFA726')
    ax.set_xticks(x); ax.set_xticklabels(assets, color='#E6EDF3', fontsize=11)
    ax.set_ylabel('HAR Coefficient', color='#E6EDF3')
    ax.set_title('HAR Coefficients: Continuous vs Jump Component', color='#E6EDF3', fontsize=13, fontweight='bold')
    ax.set_facecolor('#0D1117'); ax.legend(); ax.axhline(0, color='white', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()

# ── Telegram Helpers ─────────────────────────────────────────────────────────
def send_telegram_photo(photo_path, caption=""):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        with open(photo_path, 'rb') as f:
            files = {'photo': f}
            data = {'chat_id': TELEGRAM_CHAT, 'caption': caption, 'parse_mode': 'HTML'}
            r = requests.post(url, files=files, data=data, timeout=30)
            return r.json().get('ok', False)
    except Exception as e:
        print(f"[send_photo error] {e}"); return False

def send_telegram_text(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={'chat_id': TELEGRAM_CHAT, 'text': text, 'parse_mode': 'HTML'}, timeout=30)
        return r.json().get('ok', False)
    except Exception as e:
        print(f"[send_text error] {e}"); return False

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Realized GARCH + Jump — CSI 300")
    print("=" * 60)

    print("\n[1/6] Fetching CSI 300 daily data...")
    df = fetch_csi300()
    print(f"    Rows: {len(df)}, Range: {df['trade_date'].min()} → {df['trade_date'].max()}")

    print("\n[2/6] Computing Garman-Klass RV + Bipower Variation...")
    gk = garman_klass_rv(df[['high','low','close','open']])
    close = df['close'].values
    dates = df['trade_date'].values

    bpv_arr, returns = bipower_variation(close)
    n = len(gk) - 1  # 566

    jump_flag, jump_arr = jump_test(bpv_arr, gk[1:])
    print(f"    Total trading days: {len(returns)}")
    print(f"    Jump events detected: {int(jump_flag.sum())} ({jump_flag.mean()*100:.1f}%)")

    print("\n[3/6] Fitting HAR models...")
    rv_full = pd.Series((returns ** 2), name='rv')
    bpv_s   = pd.Series(bpv_arr[1:], name='bpv')  # drop first NaN index
    har_bpv = fit_har_rv(bpv_s[bpv_s > 0] if bpv_s.sum() > 0 else bpv_s)
    har_jump = fit_har_rv(pd.Series(jump_arr[1:], name='jump')[jump_arr[1:] > 0] if jump_arr[1:].sum() > 0 else bpv_s)
    print(f"    HAR(BPV)  β_day={har_bpv['beta_day']:.3f}  β_week={har_bpv['beta_week']:.3f}  β_month={har_bpv['beta_month']:.3f}")
    print(f"    HAR(Jump) β_day={har_jump['beta_day']:.3f}  β_week={har_jump['beta_week']:.3f}  β_month={har_jump['beta_month']:.3f}")

    print("\n[4/6] Fitting GARCH(1,1) on continuous vs total...")
    g_full = fit_garch(returns, "Total")
    g_cont = fit_garch(pd.Series(np.sqrt(np.maximum(bpv_arr, 0)), index=range(len(returns))).replace(0, np.nan).dropna(), "Continuous")
    print(f"    GARCH(Total):     α={g_full.get('alpha',0):.4f}  β={g_full.get('beta',0):.4f}")
    print(f"    GARCH(Continuous):α={g_cont.get('alpha',0):.4f}  β={g_cont.get('beta',0):.4f}")

    print("\n[5/6] Generating charts...")
    # ⚠️ 关键对齐：bpv_arr[1:n] 长度为 n-1，末尾补 0
    bpv_trimmed  = np.append(bpv_arr[1:n], 0)   # 565→566
    jump_trimmed = np.append(jump_arr[1:n], 0)  # 565→566
    rv_trimmed   = gk[1:n+1]                    # 566
    dates_trim   = dates[1:n+1]                 # 566

    chart_rv   = f"{OUTPUT_DIR}/rgarch_jump_decomp.png"
    chart_evts = f"{OUTPUT_DIR}/rgarch_jump_events.png"
    chart_har  = f"{OUTPUT_DIR}/rgarch_har_jump.png"

    chart_rv_jump_decomp(rv_trimmed, bpv_trimmed, jump_trimmed, dates_trim, chart_rv)
    chart_jump_events(dates_trim, jump_flag[1:n+1], close[1:n+1], chart_evts)
    chart_har_jump_coeffs(har_bpv, har_jump, chart_har)
    print(f"    Saved: rgarch_jump_decomp.png, rgarch_jump_events.png, rgarch_har_jump.png")

    print("\n[6/6] Sending Telegram report...")
    n_jumps = int(jump_flag.sum())
    avg_jump_pct = jump_arr.sum() / (rv_full.sum() + 1e-10) * 100

    text_en = f"""📊 <b>Realized GARCH + Jump — CSI 300 Report</b>
🕐 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} CST
━━━━━━━━━━━━━━━━━━━━

📐 <b>HAR Models: Continuous vs Jump</b>
<code>
Component     β_day   β_week  β_month
BPV (cont)   {har_bpv['beta_day']:>7.3f}  {har_bpv['beta_week']:>7.3f}  {har_bpv['beta_month']:>8.3f}
Jump          {har_jump['beta_day']:>7.3f}  {har_jump['beta_week']:>7.3f}  {har_jump['beta_month']:>8.3f}
</code>

📊 <b>GARCH(1,1) Parameters</b>
<code>
Series        α      β      Persistence
Total        {g_full['alpha']:>6.4f}  {g_full['beta']:>6.4f}  {g_full['alpha']+g_full['beta']:>10.4f}
Continuous   {g_cont['alpha']:>6.4f}  {g_cont['beta']:>6.4f}  {g_cont['alpha']+g_cont['beta']:>10.4f}
</code>

💥 <b>Jump Statistics</b>
• Detected jump days: <b>{n_jumps}</b> / {len(returns)} ({n_jumps/len(returns)*100:.1f}%)
• Avg jump contribution: <b>{avg_jump_pct:.1f}%</b> of total RV

💡 <b>Interpretation</b>
• BPV β_week={har_bpv['beta_week']:.3f} {'→ weekly volatility dominates' if har_bpv['beta_week'] > har_bpv['beta_day'] else '→ daily volatility dominates'}
• Jump component {'mean-reverting (β_day<0)' if har_jump['beta_day'] < 0 else 'persistent (β_day>0)'}

🔮 <b>Methodology</b>
• RV: Garman-Klass | Continuous: Bipower Variation | Jump: RV - BPV
• HAR: log(RV_t) = α + β₁log(RV_d₋₁) + β₅log(RV_d₋₁⁽⁵⁾) + β₂₂log(RV_d₋₁⁽²²⁾)
• GARCH(1,1): σ²_t = ω + αε²_{t-1} + βσ²_{t-1}
• Data: CSI 300 (000300.SH) via Tushare Pro · {len(returns)} trading days
⚠️ For informational purposes only. Not investment advice.
"""

    text_cn = f"""📊 <b>Realized GARCH + Jump — CSI 300 分析报告</b>
🕘 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')} CST
━━━━━━━━━━━━━━━━━━━━

📐 <b>HAR 模型：连续路径 vs 跳成分</b>
<code>
成分        β_日    β_周    β_月
BPV(连续)  {har_bpv['beta_day']:>7.3f}  {har_bpv['beta_week']:>7.3f}  {har_bpv['beta_month']:>8.3f}
Jump(跳)   {har_jump['beta_day']:>7.3f}  {har_jump['beta_week']:>7.3f}  {har_jump['beta_month']:>8.3f}
</code>

📊 <b>GARCH(1,1) 参数</b>
<code>
序列         α       β      持续性
Total       {g_full['alpha']:>6.4f}  {g_full['beta']:>6.4f}  {g_full['alpha']+g_full['beta']:>10.4f}
Continuous  {g_cont['alpha']:>6.4f}  {g_cont['beta']:>6.4f}  {g_cont['alpha']+g_cont['beta']:>10.4f}
</code>

💥 <b>跳成分统计</b>
• 检测到跳事件天数：<b>{n_jumps}</b> / {len(returns)} ({n_jumps/len(returns)*100:.1f}%)
• 跳成分平均贡献：<b>{avg_jump_pct:.1f}%</b>（占 RV 总方差）

💡 <b>解读</b>
• BPV β_周={har_bpv['beta_week']:.3f} {'→ 周度波动率主导' if har_bpv['beta_week'] > har_bpv['beta_day'] else '→ 日内波动率主导'}
• 跳成分 {'呈均值回归（负 β_日）' if har_jump['beta_day'] < 0 else '呈持续性（正 β_日）'}

🔮 <b>方法论</b>
• RV：Garman-Klass | 连续路径：Bipower Variation | 跳：RV - BPV
• HAR：log(RV_t) = α + β₁log(RV_{t-1}) + β₅log(RV_{t-1}^{(5)}) + β₂₂log(RV_{t-1}^{(22)})
• 数据：CSI 300 (000300.SH) via Tushare Pro · {len(returns)} 交易日
⚠️ 本报告仅供参考，不构成投资建议。
"""

    send_telegram_text(text_en)
    send_telegram_photo(chart_rv,   "📈 RV Decomposition — Continuous BPV vs Jump")
    send_telegram_photo(chart_evts,  "💥 CSI 300 — Price with Jump Events (orange ▲ = jump)")
    send_telegram_photo(chart_har,   "📊 HAR Coefficients: Continuous (BPV) vs Jump")
    send_telegram_text(text_cn)
    print("\n✅ Done — all outputs sent to Telegram.")

if __name__ == '__main__':
    main()
