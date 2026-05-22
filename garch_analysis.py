#!/usr/bin/env python3
"""
GARCH Family Models for Bitcoin — Complete Analysis Suite
Models: GARCH(1,1) · GJR-GARCH(1,1) · EGARCH(1,1)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
import datetime, os, sys

# ── 配置 ─────────────────────────────────────────────────────────────────
TICKER     = "BTC-USD"
START_DATE = "2019-01-01"
END_DATE   = datetime.date.today().strftime("%Y-%m-%d")
FORECAST_H = 30
CI_LEVEL   = 0.95
PLOT_DPI   = 150
OUT_DIR    = os.path.dirname(os.path.abspath(__file__)) or "."
CACHE_FILE = os.path.join(OUT_DIR, "btc_cache.pkl")

np.random.seed(42)
ZH_FONT = "WenQuanYi Zen Hei"
plt.rcParams.update({
    "figure.facecolor": "#0d1117",  "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",  "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",  "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",  "grid.color":       "#21262d",
    "grid.linewidth":   0.8,
    "axes.titlesize":   12,          "axes.labelsize":   11,
    "figure.titlesize": 14,
    "font.family":      ZH_FONT,
    "axes.unicode_minus": False,
})

# ─────────────────────────────────────────────────────────────────────────
# 1. 数据获取（yfinance → 缓存 → 模拟数据兜底）
# ─────────────────────────────────────────────────────────────────────────
print("=" * 62)
print("  GARCH QUANT — Bitcoin Volatility Analysis")
print("=" * 62)
print(f"\n[1/9] 拉取行情数据  {TICKER}  {START_DATE} → {END_DATE}")
print("-" * 62)

data_source = "yfinance"
try:
    raw = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False, timeout=15)
    if raw is None or len(raw) < 100:
        raise ValueError(f"Bad data: {len(raw) if raw is not None else 0} rows")
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.to_pickle(CACHE_FILE)
    print(f"  ✓ yfinance 拉取成功 {len(raw):,} 条")
except Exception as e:
    print(f"  ! yfinance 失败: {e}")
    if os.path.exists(CACHE_FILE):
        print("  → 使用本地缓存...")
        raw = pd.read_pickle(CACHE_FILE)
        data_source = "cache"
    else:
        print("  → 生成模拟 BTC 数据（5年日线）...")
        np.random.seed(42)
        n = 2000
        dates = pd.bdate_range("2019-01-01", periods=n)
        mu_d, vol_d = 0.0008, 0.04          # 日均收益/波动率
        log_rets = np.random.normal(mu_d, vol_d, n)
        sigma = np.zeros(n); sigma[0] = vol_d
        omega, a, b = 1e-6, 0.08, 0.88
        for t in range(1, n):
            sigma[t] = np.sqrt(omega + a * log_rets[t-1]**2 + b * sigma[t-1]**2)
            log_rets[t] = np.random.normal(mu_d, sigma[t])
        prices = 4000 * np.exp(np.cumsum(log_rets))
        raw = pd.DataFrame({
            "Open": prices * 0.995, "High": prices * 1.015,
            "Low":  prices * 0.985, "Close": prices,
            "Volume": np.random.randint(1e9, 5e9, n)
        }, index=dates)
        raw.to_pickle(CACHE_FILE)
        data_source = "synthetic"
        print(f"  ✓ 模拟数据生成完毕 {len(raw):,} 条")

price   = raw["Close"].dropna()
returns = np.log(price / price.shift(1)).dropna() * 100

print(f"  行情点数 : {len(price):,} 条")
print(f"  日期范围 : {price.index[0].date()} → {price.index[-1].date()}")
print(f"  最新价   : ${price.iloc[-1]:,.0f}")
print(f"  数据来源 : {data_source}")

# ─────────────────────────────────────────────────────────────────────────
# 2. 描述性统计
# ─────────────────────────────────────────────────────────────────────────
print(f"\n[2/9] 对数收益率统计")
print("-" * 62)
sr = returns.mean() / returns.std() * np.sqrt(252)
sk = stats.skew(returns)
kt = stats.kurtosis(returns)
jb_stat, jb_pval = stats.jarque_bera(returns)
sig_jb = "***" if jb_pval < 0.01 else "**" if jb_pval < 0.05 else "*" if jb_pval < 0.1 else "N.S."

print(f"  {'均值 (μ)':<20s}: {returns.mean():>10.4f}%")
print(f"  {'标准差 (σ)':<20s}: {returns.std():>10.4f}%")
print(f"  {'偏度 (Skew)':<20s}: {sk:>10.4f}")
print(f"  {'峰度 (Kurt)':<20s}: {kt:>10.4f}  (正态=0)")
print(f"  {'JB 统计量':<20s}: {jb_stat:>10.2f}  p={jb_pval:.4f}  {sig_jb}")
print(f"  {'年化 Sharpe':<20s}: {sr:>10.2f}")

# ─────────────────────────────────────────────────────────────────────────
# 3. ARCH 效应检验
# ─────────────────────────────────────────────────────────────────────────
print(f"\n[3/9] ARCH 效应检验 (Ljung-Box on r²)")
print("-" * 62)
lb = acorr_ljungbox(returns**2, lags=[10, 20, 30], return_df=True)
print(f"  {'滞后阶数':<10s}  {'LB 统计量':>12s}  {'p-value':>10s}  {'结论':>10s}")
print("  " + "-" * 48)
for lag in [10, 20, 30]:
    s, p = lb.loc[lag, "lb_stat"], lb.loc[lag, "lb_pvalue"]
    sig  = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "N.S."
    print(f"  lag={lag:<6d}  {s:>12.4f}  {p:>10.4f}  {sig:>10s}")

arch_ok = lb.loc[10, "lb_pvalue"] < 0.05
print(f"\n  → ARCH 效应 {'【显著】建议使用 GARCH 模型' if arch_ok else '【不显著】GARCH 可能非最优'}")

# ─────────────────────────────────────────────────────────────────────────
# 4. 模型拟合
# ─────────────────────────────────────────────────────────────────────────
print(f"\n[4/9] 拟合 GARCH 族模型 (p=1, q=1)")
print("-" * 62)

r = returns.values

def fit_model(name, vol_type, **kw):
    print(f"\n  ▶ {name}")
    try:
        am = arch_model(r, vol=vol_type, p=1, q=1, dist="normal", **kw)
        res = am.fit(disp="off", show_warning=False)
        print(f"    AIC = {res.aic:,.2f}    BIC = {res.bic:,.2f}")
        return res
    except Exception as e:
        print(f"    ✗ 拟合失败: {e}")
        return None

res_g  = fit_model("标准 GARCH(1,1)",     "GARCH")
res_jr = fit_model("GJR-GARCH(1,1)",      "Garch", o=1)
res_eg = fit_model("EGARCH(1,1)",         "EGARCH")

# ─────────────────────────────────────────────────────────────────────────
# 5. 参数报表
# ─────────────────────────────────────────────────────────────────────────
print(f"\n[5/9] 模型参数对比报表")
print("=" * 62)
header = f"  {'参数':<22s}  {'GARCH':>12s}  {'GJR-GARCH':>12s}  {'EGARCH':>12s}"
print(header)
print("  " + "-" * 64)

params_map = {
    "omega":   "ω (方差常数)",
    "alpha[1]":"α (ARCH 项)",
    "beta[1]": "β (GARCH 项)",
    "gamma[1]":"γ (不对称)",
    "delta[1]":"δ (EGARCH)",
    "mu":      "μ (均值)",
}

def pv(res, p):
    return f"{res.params[p]:12.6f}" if res is not None and p in res.params else "      N/A   "

for p, label in params_map.items():
    g = pv(res_g,  p); jr = pv(res_jr, p); eg = pv(res_eg, p)
    if g == "      N/A   " and jr == "      N/A   " and eg == "      N/A   ":
        continue
    print(f"  {label:<22s}  {g}  {jr}  {eg}")

# 波动率持久性 α+β (+ γ/2 for GJR)
print("  " + "-" * 64)
for name, res, extra in [
    ("GARCH",     res_g,  ""),
    ("GJR-GARCH", res_jr, "+(γ/2)"),
    ("EGARCH",    res_eg,  ""),
]:
    if res is None: continue
    a1 = res.params.get("alpha[1]", 0)
    b1 = res.params.get("beta[1]",  0)
    g1 = res.params.get("gamma[1]", 0)
    if name == "GJR-GARCH": p = a1 + g1/2 + b1
    else:                   p = a1 + b1
    g_str  = f"{p:.6f}" if name == "GARCH"     else "          "
    jr_str = f"{p:.6f}" if name == "GJR-GARCH" else "          "
    eg_str = f"{p:.6f}" if name == "EGARCH"    else "          "
    print(f"  {'波动率持久性 α+β'+extra:<22s}  {g_str:>12}  {jr_str:>12}  {eg_str:>12}")

# ─────────────────────────────────────────────────────────────────────────
# 6. AIC/BIC 模型选择
# ─────────────────────────────────────────────────────────────────────────
print(f"\n[6/9] 模型选择")
print("-" * 62)
results = [
    ("GARCH",     res_g,  res_g.aic,  res_g.bic),
    ("GJR-GARCH", res_jr, res_jr.aic if res_jr else np.inf, res_jr.bic if res_jr else np.inf),
    ("EGARCH",    res_eg, res_eg.aic if res_eg else np.inf, res_eg.bic if res_eg else np.inf),
]
aic_best = min(results, key=lambda x: x[2])[0]
bic_best = min(results, key=lambda x: x[3])[0]

print(f"  {'模型':<12s}  {'AIC':>14s}  {'BIC':>14s}  {'最优':>8s}")
print("  " + "-" * 56)
for name, _, aic, bic in results:
    mark = "★AIC" if name == aic_best else ("★BIC" if name == bic_best else "")
    print(f"  {name:<12s}  {aic:>14,.2f}  {bic:>14,.2f}  {mark:>8s}")

print(f"\n  → AIC 最优: {aic_best}    → BIC 最优: {bic_best}")

# 使用 AIC 最优模型
best_res = next(r_ for n, r_, a, b in results if n == aic_best)
best_name = aic_best

# ─────────────────────────────────────────────────────────────────────────
# 7. 波动率预测
# ─────────────────────────────────────────────────────────────────────────
print(f"\n[7/9] 波动率预测 (最优: {best_name}, 前向 {FORECAST_H} 天)")
print("-" * 62)

fore = best_res.forecast(horizon=FORECAST_H, reindex=False)
h_var   = fore.variance.values[-1]
h_std   = np.sqrt(h_var)          # 日度 σ (%)
ann_vol = h_std * np.sqrt(252)    # 年化 σ (%)

z = stats.norm.ppf(1 - CI_LEVEL)

print(f"  {'预测日':<10s}  {'σ_日 (%)':>12s}  {'σ_年 (%)':>12s}  {'95% VaR':>14s}")
print("  " + "-" * 56)
for i in range(0, FORECAST_H, 5):
    var_d = -z * h_std[i]  / 100 * price.iloc[-1]
    print(f"  Day {i+1:<5d}  {h_std[i]:>12.4f}%  {ann_vol[i]:>12.2f}%  ${var_d:>12,.0f}")

# 最后一行
var_d30 = -z * h_std[-1] / 100 * price.iloc[-1]
var_d1  = -z * h_std[0]  / 100 * price.iloc[-1]
print(f"\n  → 30日年化波动率预测: {ann_vol[-1]:.2f}%")
print(f"  → 95% VaR (1日):    ${var_d1:,.0f}")
print(f"  → 95% VaR (30日):   ${var_d30:,.0f}")

# ─────────────────────────────────────────────────────────────────────────
# 8. 模型诊断
# ─────────────────────────────────────────────────────────────────────────
print(f"\n[8/9] 最优模型诊断 ({best_name})")
print("-" * 62)
stdres = best_res.std_resid
lb2 = acorr_ljungbox(stdres**2, lags=[10, 20], return_df=True)
for lag in [10, 20]:
    s, p = lb2.loc[lag, "lb_stat"], lb2.loc[lag, "lb_pvalue"]
    sig  = "OK (N.S.)" if p > 0.05 else "仍显著 **"
    print(f"  LB 检验 (残差平方, lag={lag}): stat={s:.4f}  p={p:.4f}  → {sig}")
print(f"  条件均值 μ: {best_res.params.get('mu', 0):.4f}%")
print(f"  残差均值:  {np.mean(stdres):.6f}   残差标准差: {np.std(stdres):.4f}")

# ─────────────────────────────────────────────────────────────────────────
# 9. 可视化
# ─────────────────────────────────────────────────────────────────────────
print(f"\n[9/9] 生成可视化图表")
print("-" * 62)

roll_std = returns.rolling(30).std() * np.sqrt(252)

fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                          gridspec_kw={"height_ratios": [3, 2, 2]})
fig.suptitle(
    f"GARCH Volatility Analysis — {TICKER}\n"
    f"{price.index[0].strftime('%Y-%m-%d')} → {price.index[-1].strftime('%Y-%m-%d')}"
    f"  |  最优模型: {best_name}  |  95% VaR(30d): ${var_d30:,.0f}",
    fontsize=13, fontweight="bold", color="#c9d1d9", y=0.98)

# Panel 1: 价格
ax = axes[0]
ax.plot(price.index, price.values, color="#58a6ff", linewidth=1.0, label="BTC Price")
ax.fill_between(price.index, price.values, alpha=0.07, color="#58a6ff")
ax.set_ylabel("Price (USD)")
ax.set_title("Price Series", fontsize=11)
ax.legend(loc="upper left"); ax.grid(True, alpha=0.35)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.set_xlim(price.index[0], price.index[-1])

# Panel 2: 收益率
ax = axes[1]
clr = np.where(returns.values < 0, "#f85149", "#3fb950")
ax.bar(returns.index, returns.values, color=clr, alpha=0.65, width=1)
ax.axhline(0, color="#8b949e", lw=0.7)
ax.set_ylabel("Log Return (%)"); ax.set_title("Daily Log Returns", fontsize=11)
ax.grid(True, alpha=0.35); ax.set_xlim(price.index[0], price.index[-1])

# Panel 3: 波动率
ax = axes[2]
last_dt  = price.index[-1]
pred_dt  = pd.bdate_range(last_dt, periods=FORECAST_H+1)[1:]
ax.plot(roll_std.index, roll_std.values, color="#d2a8ff", linewidth=0.9,
        label="Rolling 30d σ (annualized)", alpha=0.8)
ax.plot(pred_dt, ann_vol, color="#f0883e", linewidth=2.2,
        label=f"Forecast σ ({best_name})", linestyle="--")
ax.fill_between(pred_dt, ann_vol*0.85, ann_vol*1.15,
                alpha=0.13, color="#f0883e", label="±15% band")
ax.axhline(ann_vol[-1], color="#f0883e", lw=1, linestyle=":",
           label=f"Final: {ann_vol[-1]:.1f}%")
ax.set_ylabel("Annualized Vol (%)"); ax.set_xlabel("Date")
ax.set_title(f"Historical & Forecast Volatility  |  95% VaR(30d) = ${var_d30:,.0f}", fontsize=11)
ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.35)
ax.set_xlim(price.index[0], pred_dt[-1])

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(OUT_DIR, f"garch_{TICKER.replace('-','_')}_{datetime.date.today().strftime('%Y%m%d')}.png")
plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  ✓ 图表已保存: {out}")

print("\n" + "=" * 62)
print("  ✅ 分析完成")
print("=" * 62)
