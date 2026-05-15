# GARCH Quant — 专业量化波动率建模工具箱

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **GARCH Quant** 是一套完整的波动率建模工具箱，涵盖 Bayesian GARCH、DCC-GARCH、MIDAS-GARCH、马尔可夫区制 GARCH（MS-GARCH）、Realized GARCH（含 Jump 分解）和 EVT VaR 风险预警等模块。全部脚本已脱敏，API Key 通过环境变量注入，可直接 fork 使用。

---

## 📦 已包含模型

| 模型 | 脚本 | 简介 |
|------|------|------|
| **Bayesian GARCH(1,1)** | `bayesian_garch.py` | MCMC 贝叶斯参数估计，带后验可视化 |
| **DCC-GARCH** | `dcc_garch.py` | 动态条件相关模型 + 最优对冲比例 |
| **MIDAS-GARCH** | `midas_garch.py` | 混频 GARCH，整合 VIX / 汇率 / SHIBOR 宏观因子 |
| **MS-GARCH** | `ms_garch_analysis.py` | 马尔可夫区制切换，含危机概率估计 |
| **MS-GARCH 预警** | `ms_garch_alert.py` | 实时危机监控（波动率偏离 > 2σ 触发） |
| **Realized GARCH** | `realized_garch_telegram.py` | Garman-Klass RV + HAR 模型 |
| **Realized GARCH + Jump** | `realized_garch_jump.py` | 已实现波动率分解：连续成分 + Jump 成分 |
| **GARCH-VIX 信号** | `garch_vix_signal.py` | CSI 300 波动率预测 + VIX 跨资产确认 |
| **GARCH 日报** | `garch_daily.py` | GARCH(1,1) 日频波动率预测 + 1d VaR |
| **EVT VaR 预警** | `evt_var_alert.py` | 极值理论 VaR，监控 ETF/纳斯达克/原油 |
| **DCC 日报** | `dcc_daily.py` | 多资产动态相关性日报 |
| **市场情绪简报** | `market_briefing.py` / `sentiment_news_briefing.py` | Twelve Data 实时行情 + FinBERT 情绪分析 |

---

## 🔧 环境配置

### 1. 克隆仓库
```bash
git clone https://github.com/GARCH-QUANT/garch-quant.git
cd garch-quant
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置环境变量
```bash
# A股数据（Tushare Pro）
export TUSHARE_TOKEN="your_tushare_pro_token"

# 全球行情（Twelve Data）
export TWELVE_DATA_KEY="your_twelve_data_key"

# 美股数据（Alpha Vantage）
export ALPHA_VANTAGE_KEY="your_alpha_vantage_key"

# Telegram 推送（可选）
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHANNEL_ID="your_channel_id"
```

### 4. 运行示例
```bash
# Bayesian GARCH 日波动率预测
python bayesian_garch.py

# MS-GARCH 危机预警
python ms_garch_alert.py

# Realized GARCH + Jump 检测
python realized_garch_jump.py
```

---

## 📁 目录结构

```
garch-quant/
├── README.md
├── requirements.txt
├── .gitignore
├── bayesian_garch.py        # Bayesian GARCH(1,1) + MCMC
├── dcc_garch.py             # DCC-GARCH 动态相关性
├── midas_garch.py           # MIDAS-GARCH 宏观波动率
├── ms_garch_analysis.py     # MS-GARCH 全套分析
├── ms_garch_alert.py        # MS-GARCH 危机实时监控
├── ms_garch_telegram.py     # MS-GARCH Telegram 报告
├── realized_garch_telegram.py  # Realized GARCH + HAR
├── realized_garch_jump.py   # Realized GARCH + Jump 分解
├── realized_garch_nature.py # Nature 风格单图输出
├── realized_garch_tg.py     # Telegram 多面板图表
├── garch_vix_signal.py      # GARCH-VIX 交易信号
├── garch_daily.py           # GARCH 日波动率日报
├── dcc_daily.py             # DCC 日报
├── evt_var_alert.py         # EVT VaR 风险预警
├── market_briefing.py       # 全球宏观日报
├── sentiment_news_briefing.py  # FinBERT 情绪分析
└── ai_news_briefing.py      # AI 科技新闻简报
```

---

## 📊 模型方法论

### Bayesian GARCH(1,1)
- 先验：Normal-Gamma / Jeffreys
- MCMC：Gibbs Sampling（5000 迭代，burn-in 1000）
- 输出：后验均值估计 +  credible interval

### DCC-GARCH (Engle, 2002)
- Stage 1：单资产 GARCH(1,1) 条件方差
- Stage 2：DCC(1,1) 动态相关矩阵
- 输出：时变相关热力图 + 最优对冲比例

### MIDAS-GARCH (Engle-Ghysels-Sohn, 2013)
- 低频宏观因子（VIX、USD/CNY、SHIBOR 3M、10Y 国债收益率）
- Beta-weight 混频回归
- 日频 GARCH 波动率预测

### MS-GARCH Regime-Switching
- 40th / 80th 分位数划分 Normal / Crisis 区间
- 分段 GARCH(1,1) 估计
- 危机概率 = (RV - low_q) / (high_q - low_q)

### Realized GARCH + Jump (Andersen-Bollerslev-Diebold, 2007)
- Garman-Klass 已实现波动率：RV = Σ r²
- BP 连续成分 + Jump 成分分解
- HAR 模型：log(RV_t) ~ β₁ log(RV_{t-1}) + β₅ log(RV_{t-1}^{(5)}) + β₂₂ log(RV_{t-1}^{(22)})

### EVT VaR (McNeil-Frey, 2000)
- Peaks-over-Threshold (POT) + Generalized Pareto Distribution
- VaR(99%) 估计，阈值自动选择
- EVT/Historical VaR 比值监控

---

## ⚠️ 免责声明

本工具箱仅供研究与学习用途，不构成任何投资建议。实盘使用请自行承担风险。

---

## 📄 License

MIT License — © GARCH-QUANT
