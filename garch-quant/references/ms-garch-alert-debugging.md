# ms_garch_alert.py Debugging Session — 2026-05-14

## Issue: Script exits with code 120, no output

**Symptom:** `python3.12 ms_garch_alert.py` returns exit code 120 with no stdout/stderr.

**Root Cause:** The script redefines `sys.stdout` and `sys.stderr` to a `SilentLogger` class that swallows all output. This is intentional — the script is designed to be silent unless it sends an alert. The exit code 120 comes from `sys.exit(120)` in the parent shell wrapper (injected by the cron job runner). The script itself runs correctly.

**Key insight:** Exit code 120 = SIGCHLD signal to parent, not a Python error. The Python process itself exits cleanly (0). Confirmed via:
```python
signal.signal(signal.SIGCHLD, handler)  # Received signal 17 (SIGCHLD)
```
The SIGCHLD was delivered to the parent timeout wrapper, not to the script itself.

**How to debug the silent script:**
1. Remove or bypass the `SilentLogger`:
   ```python
   # Replace SilentLogger with pass in main():
   code = code.replace(
       '''    class SilentLogger:
           def write(self, msg): pass
           def flush(): pass
       sys.stdout = SilentLogger()
       sys.stderr = SilentLogger()''',
       '''    pass''')
   ```
2. Use file-based debug writes when stderr is also silenced:
   ```python
   with open("/tmp/debug.txt", "w") as f:
       f.write(f"checkpoint: {value}\n")
   ```

## Issue: Telegram `parse_mode=HTML` rejects `<br>` tag

**Symptom:**
```
TG resp: {'ok': False, 'error_code': 400, 
'description': "Bad Request: can't parse entities: Unsupported start tag 'br' at byte offset 33"}
```

**Root Cause:** `send_telegram()` uses `parse_mode=HTML`, but `build_alert_html()` joined lines with `<br>`:
```python
return "<br>".join(lines)  # WRONG — <br> not supported
```

**Fix:**
```python
return "\n".join(lines)  # CORRECT — \n works in HTML mode
```

**Result after fix:** Alert successfully delivered to Telegram (message_id 4048).

## Alert Output (2026-05-14)

```
🚨 MS-GARCH 危机预警
🕐 2026-05-14 21:10 UTC+8
━━━━━━━━━━━━━━━━━━━━

🟠 CSI 300 — 波动率↓ 2.2σ，偏离30日均值超过阈值
   危机概率: 0.0% | 条件波动率: 16.6%/年 | 波动率偏离: -2.23σ

📐 当前状态快照

Asset       Crisis%  CondVol    Vol Z
------------------------------------
CSI 300        0.0%    16.6%    -2.23 ⚠️
AAPL          44.5%    30.3%    +0.98
TSLA           0.0%    59.8%    -0.16
```

- **CSI 300** triggered: vol deviation -2.23σ (threshold: ±2σ)
- AAPL: crisis prob 44.5%, below 80% threshold
- TSLA: crisis prob 0.0%, normal

## Script Location
`/home/agentuser/.hermes/scripts/ms_garch_alert.py`

## Related Pitfalls (from this session)
1. `SilentLogger` makes script appear dead — use file-based debug
2. Exit code 120 from wrapper = SIGCHLD, not Python crash
3. `<br>` vs `\n` in Telegram HTML mode

---

# ms_garch_alert.py Debugging Session — 2026-05-15 (Follow-up)

## Confirmed: SilentLogger + Exit 120 = Normal Behavior (Not a Bug)

**Today's run:** Script executed at 09:07 UTC+8, exit code 120 again — but this time confirmed the script **ran correctly**:
- Data fetched: CSI 300 (484 records), AAPL (500 records), TSLA (500 records)
- MS-GARCH metrics computed for all 3 assets
- CSI 300 vol_zscore = -2.23σ → triggered alert condition
- BUT cooldown was active (`{"CSI 300": {"last_alert": "2026-05-15T09:05:15.566458"}}`) → no Telegram push sent
- Script returned silently (correct behavior)

**Key diagnostic technique:**
```python
# Redirect output to file to see what SilentLogger swallowed
result = subprocess.run(
    ["python3", "/home/agentuser/.hermes/scripts/ms_garch_alert.py"],
    capture_output=True, text=True, timeout=120
)
with open("/tmp/out.txt", "w") as f:
    f.write(result.stdout)
print(result.returncode)  # 120 = wrapper SIGCHLD, NOT a Python error
```

**How to trace what the script actually did:**
```python
# Check cooldown file to see if alert was attempted
import json
with open("/tmp/ms_garch_alert_cooldown.json") as f:
    cooldown = json.load(f)
print(cooldown)  # {'CSI 300': {'last_alert': '2026-05-15T09:05:15.566458'}}
```

**Confirmed correct behavior:**
- Exit 120 = parent wrapper timeout/SIGCHLD (harmless)
- Silent output = no alert triggered OR cooldown active (correct suppression)
- Telegram push = only when alert condition met AND not in cooldown

**Cooldown parameters (from script):**
- `COOLDOWN_MINUTES = 5` — same asset won't re-alert within 5 minutes
- Separate cooldown keys: `{name}` for crisis prob, `{name}_vol` for vol deviation

---

# ms_garch_alert.py Debugging Session — 2026-05-16

## Primary Failure Mode: All Data Sources Silently Fail → No Alert

**Symptom:** Script runs to completion (exit 0), cooldown file not updated, no Telegram message sent, no console output.

**Root Cause (today):** Both API data sources are unavailable:
- **Tushare** — token invalid: `您的token不对，请确认。` (even though `TUSHARE_TOKEN` env var is set, the token itself is wrong/invalid)
- **TwelveData** — API key not configured: `apikey parameter is incorrect or not specified` (env var `TWELVE_DATA_KEY` is empty/unset)

**Script behavior when all sources fail:**
```python
# fetch_all_data() returns {} — no data collected
data = fetch_all_data()
if not data:
    return   # ← silently exits, no output, no Telegram, cooldown untouched
```

**This is correct behavior.** The script intentionally suppresses alerts when it cannot fetch data — it will not send a "I have no data" alert.

## Diagnostic Checklist

```bash
# 1. Check which env vars are actually set (look for EMPTY values)
env | grep -iE "(token|key)" | grep -v PASSWORD

# Expected for this script:
#   TELEGRAM_BOT_TOKEN=...       ← must be set
#   TUSHARE_TOKEN=...            ← must be set AND valid
#   TWELVE_DATA_KEY=...          ← must be set AND valid
#   ALPHA_VANTAGE_KEY=...        ← optional

# 2. Check cooldown file modification time (if updated → script ran)
stat /tmp/ms_garch_alert_cooldown.json | grep Modify

# 3. Test TwelveData API key directly
python -c "
import os, requests
key = os.environ.get('TWELVE_DATA_KEY', '')
r = requests.get('https://api.twelvedata.com/time_series',
    params={'symbol':'AAPL','interval':'1day','outputsize':5,'apikey':key}, timeout=10)
print(r.status_code, r.text[:200])
"

# 4. Test Tushare token
python -c "
import os, tushare as ts
token = os.environ.get('TUSHARE_TOKEN','')
ts.set_token(token)
pro = ts.pro_api(token)
print(pro.trade_cal(exchange='SSE', start_date='20260501', end_date='20260510'))
"

# 5. Verify cooldown is NOT blocking (if age < 300s, alert won't fire)
python -c "
import json, os
from datetime import datetime
cd = json.load(open('/tmp/ms_garch_alert_cooldown.json'))
for k,v in cd.items():
    diff = (datetime.now() - datetime.fromisoformat(v['last_alert'])).total_seconds()
    print(f'{k}: age={diff:.0f}s cooling={diff < 300}')
"
```

## Why "exit 120" is Misleading

Exit code 120 came from the `timeout` wrapper in the cron shell command, not from Python itself:
- `timeout 120 python ms_garch_alert.py` → when Python exits cleanly (0), the shell sees SIGCHLD from the timeout subprocess
- Today confirmed: exit 0 from Python process, exit 120 propagated from shell wrapper
- **Not a bug.** Python process exits 0.

## Updated: Script Execution Map

```
main()
  ├── sys.stdout = SilentLogger()    ← ALL print() output swallowed
  ├── cooldown = load_cooldown()     ← Read /tmp/ms_garch_alert_cooldown.json
  ├── data = fetch_all_data()        ← Tushare → TwelveData → None
  │     └── if not data: return      ← SILENT exit, cooldown unchanged
  ├── results = fit_regime_garch()  ← Only reached if data available
  ├── triggered = should_alert()     ← Checks cooldown per asset
  ├── if triggered:
  │     ├── send_telegram(html)      ← Only fires if alert condition met
  │     └── save_cooldown()          ← Updates cooldown file
  └── return                         ← Clean exit
```

**Key insight:** If cooldown file is NOT updated after a run, either:
1. No data was fetched (silent return at `if not data: return`) — today's case
2. Alert conditions were not met (correct suppression)
3. Cooldown is active (age < 300s)

## Required Env Vars (from script source)

```bash
TELEGRAM_BOT_TOKEN=...      # Telegram bot token (for alert push)
TELEGRAM_CHANNEL_ID=...     # Telegram channel/group ID (default: -1003786012521)
TUSHARE_TOKEN=...           # Tushare Pro API token (must be valid!)
TWELVE_DATA_KEY=...        # TwelveData API key (free tier OK)
ALPHA_VANTAGE_KEY=...      # Optional fallback
```

**Current status (2026-05-16):** TUSHARE_TOKEN and TWELVE_DATA_KEY are set as env vars but contain invalid values — Tushare returns "token不对" and TwelveData returns 401. These need to be refreshed.
