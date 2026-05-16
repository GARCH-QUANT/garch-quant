---
name: garch-quant
description: GARCH Quant Skills — 专业量化分析工具箱，涵盖波动率建模(GARCH/SABR)、期权定价、因子投资、技术分析、回测引擎、组合绩效分析
version: 1.0.0
---

# GARCH Quant Skills

专业量化分析技能包，波动率建模为核心方向。

---

## Python 环境

**已安装库**（`~/.hermes/hermes-agent/venv`）：
```
arch            # GARCH 族波动率模型
pysabr          # SABR 波动率曲面
finoptions      # 期权定价 (R fOptions Python 实现)
quantstats      # 组合绩效分析
backtrader      # 策略回测引擎
finta           # 技术指标 100+
ta              # Pandas 技术分析 addon
statsmodels     # 计量经济 (ADF/ARIMA/OLS)
scipy           # 科学计算
akshare         # A/H股实时数据
yfinance        # 美股数据
pandas, numpy   # 核心数值
```

---

## 1. 波动率建模（核心方向）

### GARCH 模型
```python
import pandas as pd
import numpy as np
from arch import arch_model

# 获取数据
import akshare as ak
df = ak.stock_zh_a_hist(symbol='600519', period='daily',
                         start_date='20230101', end_date='20260422', adjust='qfq')
df['returns'] = np.log(df['收盘'] / df['收盘'].shift(1)).dropna() * 100

# GARCH(1,1)
model = arch_model(df['returns'].dropna(), vol='Garch', p=1, q=1, dist='t')
result = model.fit(disp='off')
print(result.summary())

# 滚动预测
forecasts = result.forecast(horizon=5)
print(forecasts.variance.iloc[-1])
```

### Realized Volatility（已实现波动率）
```python
# 日线数据计算 RV（对数收益率平方和）
df['rv'] = (df['returns'] ** 2).rolling(20).sum()
df['rv_annualized'] = df['rv'] * np.sqrt(252)

# 如果有分钟数据：
# rv = (5-min returns ** 2).sum() * (252 * 78)  # 78 = 6.5h * 60 / 5
```

### Realized GARCH + Jump（已实现波动率分解）
Andersen-Bollerslev-Diebold (2007) 框架：RV = BPV + Jump
```python
import numpy as np

# Garman-Klass RV（日线 O/H/L/C）
def garman_klass_rv(h, l, c, o):
    gk = 0.5 * np.log(h/l)**2 - (2*np.log(2)-1) * np.log(c/o)**2
    return gk.fillna(0).replace([np.inf, -np.inf], 0)

# Bipower Variation（连续路径方差，对跳不敏感）
# BPV_t = (π/2) * |r_t| * |r_{t-1}|  （注意：长度比 returns 少 1）
returns = np.log(close / close.shift(1)).dropna().values  # len=n
bpv = np.zeros(n)
for t in range(1, n):
    bpv[t] = (np.pi / 2) * abs(returns[t]) * abs(returns[t-1])
# bpv[0] = 0（padding），有效数据从 index 1 开始

# Jump detection: z-score thresholding on (RV - BPV)
jump_var = np.maximum(returns**2 - bpv, 0)
z = jump_var / (jump_var.mean() + 1e-10)
jump_flag = (z > np.percentile(z[z>0], 85)).astype(float)

# HAR on continuous vs jump separately
# log(RV_t) = α + β₁log(RV_{t-1}) + β₅log(RV_{t-1}^{(5)}) + β₂₂log(RV_{t-1}^{(22)})
```

**关键陷阱：**
- `bpv_arr` / `jump_arr` 长度 = `len(returns)` = `n`，但 `bpv[0]=0` 为 padding，有效数据从 `bpv[1:n]` 开始
- 作图时若用 `dates[1:n+1]`（长度 n）配 `bpv[1:n]`（长度 n-1）会报 `ValueError: x has size N, but y1 has size N-1`
- 修复：用 `np.append(bpv_arr[1:n], 0)` 把最后一个 slot pad 成 0，使长度对齐为 n
- f-string 中 `{t-1}` 会被 Python 循环变量 `t` 插值 → 下标公式用 `{{t-1}}` 双括号转义

**解读框架：**
- β_day >> |β_week| + β_month → 日内冲击主导，均值回归在周度 → 牛市缓涨格局
- Jump β_day > 0 → 跳事件有持续性，一旦出现倾向于聚集
- GARCH(Continuous) α≈0.92, β≈0 → 连续路径方差接近随机游走
- 波动率持续性主要来自跳成分通道，非连续交易
```python
from pysabr import Hagan2002
import numpy as np

# SABR 参数
f = 0.03      # forward rate
K = 0.029     # strike
T = 2.0       # maturity
alpha = 0.03  # SABR alpha (ATM vol)
rho = -0.3    # correlation
nu = 0.5      # vol of vol

sabr_vol = Hagan2002().lognormal_vol(f, K, T, alpha, rho, nu)
print(f"SABR implied vol: {sabr_vol:.4f}")
```

### HAR 族模型（异质投资者模型）
```python
import pandas as pd
import statsmodels.api as sm

# HAR: RV_d = a + b*RV_d-1 + c*RV_d-5 + d*RV_d-22 + e
df['log_rv'] = np.log(df['rv'].replace(0, np.nan))
df['log_rv_1'] = df['log_rv'].shift(1)
df['log_rv_5'] = df['log_rv'].rolling(5).mean().shift(1)
df['log_rv_22'] = df['log_rv'].rolling(22).mean().shift(1)

df_clean = df[['log_rv', 'log_rv_1', 'log_rv_5', 'log_rv_22']].dropna()
X = sm.add_constant(df_clean[['log_rv_1', 'log_rv_5', 'log_rv_22']])
model = sm.OLS(df_clean['log_rv'], X).fit()
print(model.summary())
```

### GARCH 多步预测陷阱（⚠️ 重要）
```python
# 错误写法：多步预测时错误地用 r[-1]（同一期收益）重复代入
for t in range(1, horizon):
    h_forecast[t] = omega + alpha * (r[-1] * 100)**2 + beta * h_forecast[t-1]
# 结果：Day 2+ 重复加同一个 r[-1]^2，导致数值爆炸（Day 2 → 700%+）

# 正确写法：收敛到无条件方差 ω/(1-α-β)
for t in range(1, horizon):
    h_forecast[t] = omega + (alpha + beta) * h_forecast[t - 1]
# 结果：多步预测平滑收敛，不再爆炸
```

**信号解读文字陷阱：**
rationale 中 contracting/expanding 不能硬编码，必须根据 `vol_change = vol_1step - vol_now` 动态判断：
```python
vol_direction = "expanding" if (vol_1step - vol_now) > 0 else "contracting"
```

### Realized GARCH + Jump 分解（ Andersen-Bollerslev-Diebold 框架）

将已实现方差分解为连续路径成分（BPV）和跳成分：
```python
import numpy as np
import pandas as pd

def garman_klass_rv(ohlc):
    """Garman-Klass realized variance from O/H/L/C"""
    h, l, c, o = ohlc['high'], ohlc['low'], ohlc['close'], ohlc['open']
    gk = 0.5 * np.log(h/l)**2 - (2*np.log(2) - 1) * np.log(c/o)**2
    return np.nan_to_num(gk, nan=0.0, posinf=0.0, neginf=0.0)

def bipower_variation(close):
    """Bipower Variation — continuous component, immune to finite jumps"""
    returns = np.log(close / np.roll(close, 1))[1:]  # drop index 0
    bpv = np.zeros(len(returns))
    for t in range(1, len(returns)):
        bpv[t] = (np.pi / 2) * abs(returns[t]) * abs(returns[t-1])
    return bpv, returns  # bpv[0]=0, bpv[n-1]=0 (boundary padding)

def jump_test(bpv, rv):
    """Jump = RV - BPV; flag days where jump dominates"""
    jump_var = np.maximum(rv - bpv, 0)
    z_scores = jump_var / (jump_var.mean() + 1e-10)
    threshold = np.percentile(z_scores[z_scores > 0], 85)
    return (z_scores > threshold).astype(float), jump_var
```

**⚠️ Pitfall — BPV/RV 数组长度对齐**：
```
close:    indices 0..n   (length n+1)
returns:  indices 0..n-1 (length n, returns[0]=NaN)
bpv_arr:  indices 0..n-1 (length n, bpv_arr[0]=0, bpv_arr[n-1]=0 padding)
dates:    indices 0..n
rv_gk:    indices 0..n
```
画图时必须 trim：`bpv_trimmed = np.append(bpv_arr[1:n], 0)`（565→566），否则 matplotlib fill_between 报 `ValueError: 'x' has size N, but 'y1' has size N-1`。见 `realized_garch_jump.py` 第 351–365 行。

**数据源（Tushare Pro） — 所有凭证通过环境变量注入：**
```python
import os
import tushare as ts
tushare_token = os.environ.get("TUSHARE_TOKEN", "")   # 不再硬编码
ts.set_token(tushare_token)
pro = ts.pro_api(tushare_token)
df = pro.index_daily(ts_code='000300.SH', start_date='20240101', end_date='20260512')
```
其他数据源环境变量：`ALPHA_VANTAGE_KEY`、`TWELVE_DATA_KEY`、`TELEGRAM_BOT_TOKEN`、`TELEGRAM_CHANNEL_ID`。

**脚本**：`~/.hermes/scripts/realized_garch_jump.py` — Realized GARCH + Jump CSI 300 全流程（HAR + GARCH + BPV 分解 + Telegram 推送）

---

## 2. 期权定价 / 隐含波动率

### Black-Scholes + Greeks
```python
from finoptions import BlackScholes, GarmanKohlhagen, MertonJump

# Black-Scholes
bs = BlackScholes()
print(bs.call(S=100, K=100, t=0.5, r=0.05, sig=0.2))
print(bs.greeks(S=100, K=100, t=0.5, r=0.05, sig=0.2))
```

### 隐含波动率求解
```python
from scipy.stats import norm
from scipy.optimize import brentq

def implied_vol(price, S, K, T, r, q=0, option_type='call'):
    def objective(sigma):
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        if option_type == 'call':
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2) - price
        else:
            return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1) - price
    return brentq(objective, 1e-6, 5.0)
```

### 波动率曲面
```python
# 用 pysabr 构建立方体
# 参考: pysabr/examples 目录
```

---

## 3. 股票数据获取

```python
# A股
import akshare as ak
df_a = ak.stock_zh_a_hist(symbol='600519', period='daily',
                            start_date='20240101', end_date='20260422', adjust='qfq')

# 港股
df_hk = ak.stock_hk_hist(symbol='00700', period='daily',
                          start_date='20240101', end_date='20260422', adjust='qfq')

# 美股
import yfinance as yf
df_us = yf.download('NVDA', start='2024-01-01', end='2026-04-22')
```

---

## 4. 技术指标

```python
from finta import TA
from ta import add_bollinger_bands, add_rsi_indicator

# finta: 100+ 指标
sma20 = TA.SMA(df, 20)
sma60 = TA.SMA(df, 60)
rsi = TA.RSI(df, 14)
macd = TA.MACD(df)
boll = TA.BBANDS(df, 20)

# ta: pandas-addon 风格
import ta
df = add_rsi_indicator(df, close='close', n=14)
df = add_bollinger_bands(df, close='close', n=20, n_std=2)
```

---

## 5. 策略回测

```python
import backtrader as bt

class SMAStrategy(bt.Strategy):
    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=20)

    def next(self):
        if not self.position:
            if self.data.close > self.sma:
                self.buy()
        else:
            if self.data.close < self.sma:
                self.sell()

cerebro = bt.Cerebro()
cerebro.addstrategy(SMAStrategy)
data = bt.feeds.PandasData(dataname=df)
cerebro.adddata(data)
cerebro.run()
print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')
```

---

## 6. 组合绩效分析

```python
import quantstats as qs
qs.extend_pandas()

# returns = pd.Series of daily returns
qs.stats.sharpe(returns)
qs.stats.max_drawdown(returns)
qs.stats.win_rate(returns)

# 可视化
qs.plots.snapshot(returns, title='Performance')
```

---

## 7. 因子分析

```python
import statsmodels.api as sm

# 多因子模型
X = sm.add_constant(factor_df[['mkt', 'smb', 'hml', 'rmw', 'cma']])
model = sm.OLS(returns, X).fit()
print(model.summary())
```

---

## Additional Research Modules

The following specialized modules extend the core garch-quant workflow and are
incorporated into the research pipeline via garch-report-generator.

---

### garch-evt-tail-risk (EVT 尾部风险 + 压力测试)
GPD 尾部建模、尾部相关系数 λ、VaR/CVaR 三方法对比（Historical / Parametric / EVT-GPD）、
六情景压力测试。数据源：Tushare Pro（A股 ETF）。

**⚠️ Cron 脚本凭证规范（2026-05-15 起已全面脱敏）：**
所有脚本凭证改用 `os.environ.get()`，不再硬编码。
```python
# ✅ 正确写法
TELEGRAM_BOT_TOKEN=os.environ.get("TELEGRAM_BOT_TOKEN", "")
TUSHARE_TOKEN=os.environ.get("TUSHARE_TOKEN", "")

# ❌ 错误写法（已全部替换）
os.environ['TELEGRAM_BOT_TOKEN'] = '***'
```
完整脱敏记录见 `finance/garch-quant-open-source` skill（凭证清单、脱敏检查命令、开源结构建议）。

**核心分析步骤：**
1. 数据获取：黄金 ETF (518880.SH)、纳斯达克 ETF (513100.SH)、原油基金 (162411.SZ)
2. GARCH(1,1) 条件波动率
3. EVT-GPD 尾部建模（threshold=90th pct）
4. 尾部相关系数 λ（上尾/下尾）
5. VaR/CVaR 三方法对比
6. 六情景压力测试

**脚本**: `~/.hermes/scripts/evt_var_alert.py` — 每日预警（周一至周五 15:30）
**Cron**: `0 9 * * 1-5`（每周一 09:00 报告推送 Telegram）

---

### garch-hedge-ratio-analysis (波动传导 + 动态最优对冲)
基于 BEKK-GARCH 波动溢出、DCC-GARCH 动态相关、20天滚动最优对冲比例。
包含正常期 vs 危机期对比 + 三条交易规则。

**核心分析步骤：**
1. BEKK-GARCH 波动溢出检验
2. DCC-GARCH 动态条件相关
3. 滚动 OLS + GARCH 条件相关调整最优对冲比例
4. 正常期 vs 危机期（相关 > 75th pct）对比
5. 三条交易规则 + Telegram 推送

**适用场景**：黄金+纳指、原油+美元、沪深300+中证500、个股+行业ETF

---

### garch-report-generator (报告生成 + 定时推送)
MS-GARCH 区制转换波动率报告生成器 + 多个辅助分析脚本。

**主脚本**：`ms_garch_telegram.py`（周一 09:00 北京时间）
**输出图表**：条件波动率对比、区制切换柱状图、CSI300/AAPL/TSLA 区制卡片、热力图

**辅助脚本（均为 garch-quant 框架的延伸）：**
- `realized_garch_jump.py` — Andersen-Bollerslev-Diebold 框架：BPV 连续路径 + Jump 成分分解
  Cron: `6 8 * * *`（每日 08:06）
- `midas_garch.py` — MIDAS-GARCH：宏观因子（VIX/USD-CNY/SHIBOR/10Y国债）→ 日度波动率
  Cron: `30 18 * * *`（每日 18:30）
- `dcc_garch.py` — DCC-GARCH：多资产动态条件相关 + 最优对冲比例
  Cron: `55 8 * * 1-5`（周一至周五 08:55）

**已知调试经验（garch-report-generator references/）：**
- `references/dcc-garch-debugging.md` — DCC-GARCH 调试会话
  关键修复：Twelve Data `fetch_data(display_name, api_code)` 参数顺序、
  符号格式（`EUR/USD` 而非 `EURUSD`）、裸 `pass` 语句导致脚本静默退出
- `references/garch-script-debugging.md` — GARCH 脚本调试经验

## Notes
**参考文档：**
- `references/garch-script-debugging.md` — GARCH 脚本通用调试（含 cron stdout/Telegram 陷阱、Tushare token 注入、多步预测爆炸、BPV/RV 长度对齐）
- `references/telegram-html-render-pitfall.md` — Telegram HTML 模式 `<br>`、`{{}}`、HTML字符转义陷阱
- `references/ms-garch-alert-debugging.md` — ms_garch_alert.py 调试会话（SilentLogger + exit 120 + `<br>` 修复）
- `references/dcc-garch-debugging.md` — DCC-GARCH 调试会话（含参数顺序、符号格式陷阱）
- `scripts/realized_garch_jump.py` — Realized GARCH + Jump 完整脚本（可修改复现）
- `scripts/realized_garch_jump.py` — Realized GARCH + Jump 完整脚本（可修改复现）
