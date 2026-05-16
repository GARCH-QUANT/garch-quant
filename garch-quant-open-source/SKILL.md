---
name: garch-quant-open-source
description: GARCH Quant 项目 GitHub 开源可行性评估 & 脱敏规范
triggers:
  - "哪些项目可以开源"
  - "上传 GitHub"
  - "脱敏"
  - "open source garch"
---

# GARCH Quant — 开源评估报告

## 一、脚本资产总览

| 脚本 | 行数 | 核心功能 | 可脱敏程度 |
|------|------|----------|------------|
| `bayesian_garch.py` | 708 | 贝叶斯 GARCH(1,1) 参数估计 | ✅ 可开源 |
| `dcc_garch.py` | 624 | DCC-GARCH 动态相关性建模 | ✅ 可开源 |
| `midas_garch.py` | 591 | MIDAS-GARCH 宏观波动率 | ✅ 可开源 |
| `ms_garch_analysis.py` | 847 | 马尔可夫区制 GARCH 全套 | ✅ 可开源 |
| `realized_garch_telegram.py` | 616 | Realized GARCH + Telegram 推送 | ⚠️ 需清理推送模块 |
| `realized_garch_jump.py` | 483 | Realized GARCH + Jump 分解 | ✅ 可开源 |
| `garch_vix_signal.py` | 442 | GARCH-VIX 信号交易 | ⚠️ 需清理 Telegram 模块 |
| `ms_garch_telegram.py` | 522 | MS-GARCH + Telegram 推送 | ⚠️ 需清理推送模块 |
| `ms_garch_alert.py` | 332 | 危机预警（Telegram） | ⚠️ 需清理 Telegram 模块 |
| `dcc_report_pdf.py` | 453 | DCC-GARCH PDF 报告生成 | ⚠️ 需清理 Telegram 模块 |
| `realized_garch_nature.py` | 226 | Nature 风格学术报告 | ✅ 可开源 |
| `garch_daily.py` | 245 | GARCH 日波动率日报 | ⚠️ 清理 API key 后可开源 |
| `dcc_daily.py` | 254 | DCC 日报 | ⚠️ 清理 API key 后可开源 |
| `evt_var_alert.py` | 275 | EVT VaR 风险预警 | ✅ 可开源 |
| `sentiment_news_briefing.py` | 222 | FinBERT 情绪分析 | ✅ 可开源 |
| `ai_news_briefing.py` | 192 | AI 科技新闻简报 | ✅ 可开源 |
| `market_briefing.py` | 271 | 市场情绪简报 | ✅ 可开源 |
| `medallion_article.py` | 11 | 大奖章基金文章推送 | ⚠️ 仅文章推送，无量化逻辑 |

---

## 二、必须脱敏的内容（硬编码凭证）

以下四类凭证分布在几乎所有脚本中，开源前**必须替换为环境变量或占位符**：

### 1. Tushare Pro Token（A股数据）
```python
# 当前（脱敏版，已是部分脱敏）
token = "3953d8e4941aad5cf1a2b5212856d208b4b5c0a5259ad0ef46df04a7"

# 开源标准写法
import os
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")
```
**出现在**: `bayesian_garch`, `dcc_garch`, `ms_garch_*`, `midas_garch`, `realized_garch_*`, `evt_var_alert`, `garch_vix_signal`

### 2. Twelve Data API Key
```python
# 当前
TWELVE_DATA_KEY = "f00f38fce5e1494db68ffcdd2683eb66"

# 开源标准写法
TWELVE_DATA_KEY = os.environ.get("TWELVE_DATA_KEY", "")
```
**出现在**: `bayesian_garch`, `dcc_garch`, `ms_garch_*`, `midas_garch`, `realized_garch_telegram`, `market_briefing`

### 3. Alpha Vantage API Key
```python
# 当前
ALPHA_VANTAGE_KEY = "G45JJUVUQ1ZO7JM0"

# 开源标准写法
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")
```
**出现在**: `ms_garch_analysis`, `ms_garch_telegram`, `garch_daily`

### 4. Telegram Bot Token
```python
# 当前（部分脱敏）
TOKEN = "830302...CeVU"

# 开源标准写法
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHANNEL_ID", "-1003786012521")
```
**出现在**: `dcc_garch`, `ms_garch_alert`, `garch_vix_signal`, `midas_garch`, `realized_garch_jump`, `realized_garch_telegram`

---

## 三、推荐开源结构

```
garch-quant/
├── README.md
├── requirements.txt
├── bayesian_garch/
│   ├── __init__.py
│   ├── model.py          # 贝叶斯 GARCH 核心（去 API key）
│   └── data.py            # 数据获取（API key 用 env var）
├── dcc_garch/
│   ├── __init__.py
│   ├── model.py
│   └── data.py
├── midas_garch/
│   ├── __init__.py
│   ├── model.py
│   └── data.py
├── ms_garch/
│   ├── __init__.py
│   ├── model.py           # 马尔可夫区制 GARCH
│   └── data.py
├── realized_garch/
│   ├── __init__.py
│   ├── har_model.py       # HAR 模型
│   ├── jump_detect.py     # Jump 识别
│   └── data.py
├── evt_var/
│   ├── __init__.py
│   └── var_alert.py       # EVT VaR（无外部依赖）
├── sentiment/
│   ├── __init__.py
│   └── finbert_news.py    # FinBERT 情绪分析
└── examples/
    ├── bayesian_example.py
    ├── dcc_example.py
    └── ms_garch_example.py
```

---

## 四、脱敏检查清单（每次提交前必做）

```bash
# 1. 确保没有真实 token 泄漏（除 env var 引用外）
grep -rn "3953d8e4941aad5cf\|8303023127\|G45JJUVUQ1ZO7JM0\|f00f38fce5e1494" *.py

# 2. 确保所有 API key 都通过 os.environ.get() 获取
grep -rn "ALPHA_VANTAGE_KEY = \"[A-Z0-9]\{10\}\"" *.py   # 应无输出

# 3. 确保 requirements.txt 不含 credentials
cat requirements.txt | grep -v "^#" | grep -v "^$"
```

---

**状态：已完成脱敏（2026-05-15）**

所有凭证已替换为 `os.environ.get()`，语法检查全部通过。

## 开源优先级建议

**第一梯队（方法论纯度高、无平台耦合）：**
1. `bayesian_garch.py` — MCMC + GARCH，学术价值强
2. `realized_garch_jump.py` — HAR-RV + Jump 分解，业界稀缺
3. `evt_var_alert.py` — EVT 极值理论，无外部 API 依赖

**第二梯队（需拆分清理推送逻辑）：**
4. `ms_garch_analysis.py` — 全套 MS-GARCH，定量分析完整
5. `dcc_garch.py` — DCC 动态相关性，模型部分可独立

**第三梯队（含 Telegram/微信推送，清理后开源）：**
6. `ms_garch_alert.py`
7. `garch_vix_signal.py`
8. `realized_garch_telegram.py`
