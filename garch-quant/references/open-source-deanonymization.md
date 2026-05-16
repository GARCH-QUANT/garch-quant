# GARCH Quant 脱敏开源工作记录（2026-05-15）

## 凭证清单

| 凭证 | 真实值 | 环境变量 |
|------|--------|----------|
| Tushare Pro Token | `***` | `TUSHARE_TOKEN` |
| Twelve Data API Key | `***` | `TWELVE_DATA_KEY` |
| Alpha Vantage API Key | `***` | `ALPHA_VANTAGE_KEY` |
| Telegram Bot Token | `***` | `TELEGRAM_BOT_TOKEN` |
| Telegram Channel ID | `-100XXXXXXXX` | `TELEGRAM_CHANNEL_ID` |

## 已脱敏脚本（14个，全部通过语法检查）

```
bayesian_garch.py          ✅
dcc_garch.py               ✅
midas_garch.py             ✅
ms_garch_alert.py          ✅
ms_garch_analysis.py       ✅（修复了 pass+f-string 残留语法碎片）
ms_garch_telegram.py       ✅
realized_garch_jump.py     ✅
realized_garch_nature.py   ✅
realized_garch_telegram.py ✅
realized_garch_tg.py       ✅（严重损坏→完整重写）
garch_daily.py             ✅
garch_vix_signal.py        ✅（修复了缩进损坏的 fetch_csi300）
market_briefing.py         ✅
evt_var_alert.py           ✅
```

## 替换规范

```python
# Config 区 — 顶格写 import，变量全部用 os.environ.get()
import os
TOKEN             = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID           = os.environ.get("TELEGRAM_CHANNEL_ID", "-1003786012521")
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")
TWELVE_DATA_KEY   = os.environ.get("TWELVE_DATA_KEY", "")
TUSHARE_TOKEN     = os.environ.get("TUSHARE_TOKEN", "")

# 函数内 tushare 调用 — 局部读取
def get_tushare_data(ts_code, start, end):
    import tushare as ts
    tushare_token = os.environ.get("TUSHARE_TOKEN", "")
    ts.set_token(tushare_token)
    pro = ts.pro_api(tushare_token)
    ...
```

## 验证命令

```bash
# 1. 凭证残留检查（应无输出）
grep -rn "3953d8e4941aad5cf1a2b5212856d208b4b5c0a5259ad0ef46df04a7\|8303023127:AAHfttpDnJdwBWLHErbVXbCeqkY20lpCeVU\|G45JJUVUQ1ZO7JM0\|f00f38fce5e1494db68ffcdd2683eb66" *.py

# 2. 语法检查（应无输出）
python3 -m py_compile bayesian_garch.py dcc_garch.py midas_garch.py ms_garch_alert.py ...
```

## 典型 patch 失败原因

| 错误 | 原因 | 解法 |
|------|------|------|
| `Found 2 matches` | 同一字符串在文件内出现多次（常为 `pass  # quiet`） | 加更多上下文唯一化 |
| `Could not find a match` | 并行 patch 时另一个 subagent 已修改了文件 | 先 `read_file` 再 patch |
| `IndentationError` | 替换时破坏了缩进结构（如 `fetch_csi300` 函数体缩进级别错误） | 完整重写该 section |
| 语法解析后 lint 报错 | patch 替换了函数头但留下了函数体残片 | lint 失败立刻 read_file 定位，完整重写 |

## 开源优先级

**第一梯队（方法论纯度高、无平台耦合）：**
`bayesian_garch.py` · `realized_garch_jump.py` · `evt_var_alert.py`

**第二梯队（需拆分 Telegram 模块）：**
`ms_garch_analysis.py` · `dcc_garch.py`

**第三梯队（含推送逻辑，清理后开源）：**
`ms_garch_alert.py` · `garch_vix_signal.py` · `realized_garch_telegram.py`
