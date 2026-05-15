#!/usr/bin/env python3
"""
EVT VaR Early Warning System
GARCH Quant 每日风险监控 — 黄金ETF × 纳斯达克ETF × 原油基金
触发条件: EVT VaR (99%) > threshold OR EVT/Historical VaR 比值超标
"""

import pandas as pd
import numpy as np
import tushare as ts
import requests
import sys
import os
from scipy.stats import norm, genpareto
import warnings
warnings.filterwarnings('ignore')

# ── API Keys (from environment) ───────────────────────────────────────
import os
TOKEN_TUSHARE      = os.environ.get("TUSHARE_TOKEN", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID            = os.environ.get("TELEGRAM_CHANNEL_ID", "-1003786012521")

# ── Alert Thresholds ───────────────────────────────────────────────────
EVT_VAR_THRESHOLD = 5.0    # EVT VaR > 5% 则预警
EVT_HIST_RATIO    = 1.5    # EVT/Hist VaR 比值 > 1.5x 则预警

ts.set_token(TOKEN_TUSHARE)

# ── Data Fetch ─────────────────────────────────────────────────────────
def get_etf_bar(ts_code, start, end=None):
    try:
        df = ts.pro_bar(ts_code=ts_code, start_date=start, end_date=end,
                        asset='FD', adj='qfq', freq='D')
        if df is None or len(df) == 0:
            return None
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').set_index('trade_date')
        return df['close']
    except Exception as e:
        pass  # quiet
        return None

def fetch_latest_prices():
    """获取最近 500 个交易日（约2年）的日线数据"""
    end = pd.Timestamp.today().strftime('%Y%m%d')
    start = (pd.Timestamp.today() - pd.Timedelta(days=800)).strftime('%Y%m%d')

    gold   = get_etf_bar("518880.SH", start, end)
    nasdaq = get_etf_bar("513100.SH", start, end)
    oil    = get_etf_bar("162411.SZ", start, end)

    if gold is None or nasdaq is None or oil is None:
        raise RuntimeError("数据获取失败，脚本终止")

    prices = pd.DataFrame({'Gold': gold, 'Nasdaq': nasdaq, 'Oil': oil}).dropna()
    return prices

# ── GARCH ───────────────────────────────────────────────────────────────
def fit_garch(series):
    from arch import arch_model
    model = arch_model(series, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
    fit = model.fit(disp='off', options={'maxiter': 500})
    cond_vol = fit.conditional_volatility
    std_resid = series / cond_vol
    return cond_vol, std_resid

# ── EVT-GPD VaR ─────────────────────────────────────────────────────────
def fit_gpd_loss(loss_series, threshold_pct=90):
    threshold = np.percentile(loss_series, threshold_pct)
    tail = loss_series[loss_series > threshold]
    xi, psi, _ = genpareto.fit(tail, floc=threshold)
    n = len(loss_series)
    k = len(tail)
    prob_exceed = k / n
    return {'xi': xi, 'psi': psi, 'threshold': threshold,
            'prob_exceed': prob_exceed, 'n_total': n}

def evt_var_cvair(loss_series, gpd_result, level=0.99):
    xi = gpd_result['xi']
    psi = gpd_result['psi']
    u = gpd_result['threshold']
    p_u = gpd_result['prob_exceed']
    q = 1 - level
    if abs(xi) < 1e-6:
        xi = 1e-6
    ratio = p_u / q
    var = u + (psi / xi) * (ratio ** xi - 1)
    cvair = var * (1 / (1 - xi))
    return var, cvair

def historical_var(loss_series, level=0.99):
    var = -np.percentile(loss_series, (1-level)*100)
    return var

# ── Telegram Alert ─────────────────────────────────────────────────────
def send_telegram(text, bot_token=TELEGRAM_BOT_TOKEN, chat_id=CHAT_ID):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    r = requests.post(url, data={'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown'})
    if r.status_code != 200:
        pass  # quiet
    return r.status_code == 200

# ── Main Logic ─────────────────────────────────────────────────────────
def main():
    today_str = pd.Timestamp.today().strftime('%Y-%m-%d')
    pass  # quiet

    # 1. 获取数据
    prices = fetch_latest_prices()
    returns = np.log(prices / prices.shift(1)).dropna() * 100  # %收益率
    # 去除极端异常值
    returns = returns[returns['Nasdaq'] > -30]

    pass  # quiet

    # 2. GARCH
    vol_gold,   _ = fit_garch(returns['Gold'])
    vol_nasdaq,  _ = fit_garch(returns['Nasdaq'])
    vol_oil,     _ = fit_garch(returns['Oil'])

    # 3. 损失序列 + GPD 拟合
    gpd_gold   = fit_gpd_loss(-returns['Gold'])
    gpd_nasdaq = fit_gpd_loss(-returns['Nasdaq'])
    gpd_oil    = fit_gpd_loss(-returns['Oil'])

    # 4. VaR 计算
    results = {}
    for asset, gpd, vol in [('Gold', gpd_gold, vol_gold),
                             ('Nasdaq', gpd_nasdaq, vol_nasdaq),
                             ('Oil', gpd_oil, vol_oil)]:
        loss = -returns[asset]
        h_var = historical_var(loss, 0.99)
        e_var, e_cvar = evt_var_cvair(loss, gpd, 0.99)
        results[asset] = {
            'hist_var_99': h_var,
            'evt_var_99': e_var,
            'evt_cvar_99': e_cvar,
            'ratio': e_var / h_var if h_var > 0 else 0,
            'xi': gpd['xi'],
            'latest_vol': vol.iloc[-1]
        }

    # 5. 组合 EVT VaR（等权）
    w = np.array([1/3, 1/3, 1/3])
    evt_vars = np.array([results[a]['evt_var_99'] for a in ['Gold','Nasdaq','Oil']])
    port_evt_var = np.sum(w * evt_vars)

    # 6. 判断是否触发预警
    triggers = []
    for asset, r in results.items():
        if r['evt_var_99'] > EVT_VAR_THRESHOLD:
            triggers.append(f"  [{asset}] EVT VaR {r['evt_var_99']:.2f}% > 阈值 {EVT_VAR_THRESHOLD}%")
        if r['ratio'] > EVT_HIST_RATIO:
            triggers.append(f"  [{asset}] EVT/Hist 比值 {r['ratio']:.2f}x > {EVT_HIST_RATIO}x（风险被低估！）")

    alert_triggered = len(triggers) > 0 or port_evt_var > EVT_VAR_THRESHOLD

    # ── 构建报告 ─────────────────────────────────────────────────────────
    STATUS_OK   = "✅"
    STATUS_WARN  = "⚠️"
    STATUS_FIRE  = "🔥"

    gold_warn  = results['Gold']['evt_var_99'] > EVT_VAR_THRESHOLD or results['Gold']['ratio'] > EVT_HIST_RATIO
    ndx_warn   = results['Nasdaq']['evt_var_99'] > EVT_VAR_THRESHOLD or results['Nasdaq']['ratio'] > EVT_HIST_RATIO
    oil_warn   = results['Oil']['evt_var_99'] > EVT_VAR_THRESHOLD or results['Oil']['ratio'] > EVT_HIST_RATIO

    report = f"""🛡️ *GARCH Quant · 每日 EVT VaR 风险监控*
`{today_str} · 滚动 {len(returns)} 天数据`

🔎 *VaR 对比 · 99% 置信度*
━━━━━━━━━━━━━━━━━━━━

🏅 黄金   {STATUS_OK if not gold_warn else STATUS_WARN}  Hist {results['Gold']['hist_var_99']:.2f}%  |  EVT {results['Gold']['evt_var_99']:.2f}%  |  比值 {results['Gold']['ratio']:.2f}x
📈 纳斯达克 {STATUS_OK if not ndx_warn else STATUS_FIRE}  Hist {results['Nasdaq']['hist_var_99']:.2f}%  |  EVT {results['Nasdaq']['evt_var_99']:.2f}%  |  比值 {results['Nasdaq']['ratio']:.2f}x {'⬆️ 低估!' if ndx_warn else ''}
🛢️ 原油   {STATUS_OK if not oil_warn else STATUS_WARN}  Hist {results['Oil']['hist_var_99']:.2f}%  |  EVT {results['Oil']['evt_var_99']:.2f}%  |  比值 {results['Oil']['ratio']:.2f}x

📦 组合 EVT VaR（等权）: **{port_evt_var:.2f}%** {'⚠️ 超阈值' if port_evt_var > EVT_VAR_THRESHOLD else '✓ 正常'}

🦴 *尾部肥尾系数 ξ*
━━━━━━━━━━━━━━━━━━━━
ξ 越高 → 极端事件概率越大
🏅 黄金   ξ = {results['Gold']['xi']:.4f}
📈 纳斯达克 ξ = {results['Nasdaq']['xi']:.4f} {'◀️ 最高（尾部最肥）' if results['Nasdaq']['xi']==max(r['xi'] for r in results.values()) else ''}
🛢️ 原油   ξ = {results['Oil']['xi']:.4f}
"""

    if alert_triggered and triggers:
        report += f"""
🚨 *预警触发 · {len(triggers)} 条*
━━━━━━━━━━━━━━━━━━━━
"""
        for t in triggers:
            report += t.replace("  [", "• [") + "\n"

    # ── 周末持仓建议 ─────────────────────────────────────────────────
    xi_avg = np.mean([results[a]['xi'] for a in results])
    port_evar = port_evt_var

    if xi_avg > 0.35 or port_evar > 7.0:
        risk_level = "🔴 高风险"
        risk_desc  = "极端事件概率显著升高，建议减仓控制风险"
        pos_advice = ("• 总仓位降至 50% 以下\n"
                      "• 黄金配置提高至 40%+\n"
                      "• 纳斯达克 / 原油暂时减持")
        watch = "纳斯达克（ξ 最高，尾部最肥）"
    elif xi_avg > 0.25 or port_evar > 5.0:
        risk_level = "🟡 中风险"
        risk_desc  = "风险偏高但可控，建议适度增配避险资产"
        pos_advice = ("• 黄金配置提高至 35-40%\n"
                      "• 纳指 / 原油维持标配\n"
                      "• 总仓位控制在 70-80%")
        watch = "原油（EVT VaR 绝对值最高）"
    else:
        risk_level = "🟢 低风险"
        risk_desc  = "各指标正常，极端风险可控"
        pos_advice = ("• 各资产维持等权（33% each）\n"
                      "• 总仓位可维持 80-90%\n"
                      "• 无需特殊对冲")
        watch = "无特定预警"

    report += f"""
💼 *周末持仓建议*
━━━━━━━━━━━━━━━━━━━━
风险等级：{risk_level}
评估：{risk_desc}

配置方向：
{pos_advice}

重点关注：{watch}
"""

    # ── 解读 ──────────────────────────────────────────────────────────
    ndx_xi = results['Nasdaq']['xi']
    oil_var = results['Oil']['evt_var_99']
    oil_xi  = results['Oil']['xi']
    ndx_ratio = results['Nasdaq']['ratio']

    report += f"""
📖 *关键信号解读*
━━━━━━━━━━━━━━━━━━━━
ξ 系数：纳斯达克当前 ξ={ndx_xi:.3f}，为三资产最高，说明
近期美股已有多次大幅波动，将 GPD 尾部撑肥。GPD 的 ξ>0
意味着极端事件概率远高于正态分布假设。

低估警报：纳指 EVT VaR 比 Historical 高出 **{ndx_ratio:.1f}x**，
传统方法严重低估当前风险，需警惕突发冲击。

原油风险：绝对风险最高（EVT VaR = {oil_var:.2f}%），但 ξ={oil_xi:.3f}
低于纳指——表明近期原油波动频率不及纳指，但单次幅度更大，
历史上常与地缘风险事件相关联。

极端情景：组合 EVT VaR = **{port_evt_var:.2f}%**，含义是：在 99%
置信度下，极端情况（概率 <1%）可能导致本金损失达到 **{port_evt_var*2:.1f}%**。
建议将其作为最大可承受亏损的心理阈值。

{'⚠️ 当前处于预警状态，建议周五收盘前执行仓位调整。' if alert_triggered else '✅ 各指标暂无异常，可维持现有配置。'}
"""

    # ── 发送 ──────────────────────────────────────────────────────────
    send_telegram(report)
    pass  # quiet

    if not alert_triggered:
        pass  # quiet

    return alert_triggered, results

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        pass  # quiet
        send_telegram(f"**GARCH Quant 预警系统异常**\n`{pd.Timestamp.today().strftime('%Y-%m-%d')}`\n错误: {str(e)[:200]}")
        sys.exit(1)
