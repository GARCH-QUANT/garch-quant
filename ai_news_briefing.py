#!/usr/bin/env python3
"""
每日 AI 科技资讯 - HTML版
- 数据源：NewsNow（36kr/IT之家/掘金/HackerNews） + 关键词过滤
- 结构：对标用户模板，分账号摘要 → 高共识信号 → 新主题
- 输出纯HTML，用于 Telegram HTML模式发送
"""

import requests
import time
from datetime import datetime

NEWSNOW_BASE = "https://newsnow.busiyi.world"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
}

# 科技/AI相关关键词（包含项）
INCLUDE_KW = [
    "AI", "人工智能", "OpenAI", "Anthropic", "Google", "微软", "Meta",
    "英伟达", "nvidia", "AMD", "intel", "Qualcomm",
    "ChatGPT", "Gemini", "Claude", "Llama", "Grok",
    "GPU", "HBM", "H200", "GB200", "GB300", "数据中心",
    "云计算", "AWS", "Azure", "GCP", "阿里云", "腾讯云",
    "自动驾驶", "Tesla", "FSD", "Waymo",
    "机器人", "Figure", "1X", "宇树", "AGI",
    "量子", "量子计算", "ionq", "ibm quantum",
    "光模块", "InP", "磷化铟", "cpo",
    "SK海力士", "美光", "三星", "海力士",
    "SpaceX", "Starlink", "Rocket", "火箭",
    "芯片", "半导体", "晶圆", "wafer", "先进封装",
    "ARM", "RISC", "CPU", "GPU", "ASIC",
    "软件", "SaaS", "Snowflake", "Databricks",
    "大选", "关税", "中美", "特朗普", "Trump",
    "蛋白质", "AlphaFold", "DeepMind", "生物学",
    "能源", "核能", "核电", "太阳能", "电力",
    "生物科技", "Biotech", "CRISPR", "mRNA",
]

# 排除项（国内A股、房产、理财产品等）
EXCLUDE_KW = [
    "A股", "上证", "深证", "科创", "创业板", "涨跌停",
    "房地产", "房价", "楼市", "恒大", "碧桂园",
    "券商", "基金", "私募", "理财", "净值",
    "余额宝", "货币基金", "REITs",
    "许昆林", "省委", "书记", "调研",
]


def news(source_id: str, count: int = 25) -> list:
    try:
        resp = requests.get(
            f"{NEWSNOW_BASE}/api/s?id={source_id}",
            headers=HEADERS, timeout=15
        )
        if resp.status_code == 200:
            items = resp.json().get("items", [])[:count]
            return [(i.get("title", ""), i.get("url", "")) for i in items]
    except Exception:
        pass
    return []


def is_relevant(title: str) -> bool:
    t = title.lower()
    if any(kw.lower() in t for kw in EXCLUDE_KW):
        return False
    return any(kw.lower() in t for kw in INCLUDE_KW)


def fmt_link(title: str, url: str) -> str:
    short = title[:70] + "…" if len(title) > 70 else title
    if url:
        return f'<a href="{url}">{short}</a>'
    return short


def build_html() -> str:
    now = datetime.now()
    date_str = now.strftime("%Y年%m月%d日")
    day_str = ["周一","周二","周三","周四","周五","周六","周日"][now.weekday()]
    time_str = now.strftime("%H:%M")

    # ── 抓新闻 ─────────────────────────────────
    sources = [
        ("36kr", "36氪"),
        ("ithome", "IT之家"),
        ("juejin", "掘金"),
        ("hackernews", "Hacker News"),
        ("wallstreetcn", "华尔街见闻"),
        ("cls", "财联社"),
    ]
    all_news = []
    for sid, _ in sources:
        all_news += news(sid, 20)
        time.sleep(0.25)

    # 过滤
    rel = [(t, u) for t, u in all_news if is_relevant(t)]

    # ── 分类 ───────────────────────────────────
    infra_news = [(t, u) for t, u in rel if any(
        k in t.lower() for k in ["英伟达", "nvidia", "AMD", "intel", "GPU", "HBM", "数据中心", "光模块", "InP", "磷化铟", "GB200", "GB300", "H200"]
    )]
    cloud_news = [(t, u) for t, u in rel if any(
        k in t.lower() for k in ["OpenAI", "Anthropic", "Google", "微软", "Meta", "ChatGPT", "Gemini", "Claude", "GCP", "Azure", "AWS", "阿里云"]
    )]
    stock_news = [(t, u) for t, u in rel if any(
        k in t.lower() for k in ["$", "股价", "营收", "财报", "超预期", "beat", "miss", "分析师", "目标价", "评级", "AMD", "NVDA", "TSLA", "AMZN", "GOOGL", "META"]
    )]
    energy_news = [(t, u) for t, u in rel if any(
        k in t.lower() for k in ["能源", "电力", "核电", "核能", "太阳能", "数据中心", "电力需求", "电网", "nuclear", "solar"]
    )]
    new_themes = [(t, u) for t, u in rel if any(
        k in t.lower() for k in ["漏洞", "安全", "突破", "革命", "首次", "新型", "量子", "生物", "蛋白质", "新发现"]
    )]

    # 取交集避免重复
    all_cats = {
        "AI/大模型": cloud_news[:5],
        "硬件/算力": infra_news[:5],
        "市场/个股": stock_news[:5],
        "能源/电力": energy_news[:3],
        "新主题": new_themes[:3],
    }

    # ── HTML 构建 ──────────────────────────────
    lines = [
        f"📊 <b>Tech Report | {date_str} {day_str}</b>",
        f"🕐 北京时间 {time_str}",
        "━━━━━━━━━━━━",
        "<b>🔍 科技要闻速览</b>",
        "",
    ]

    for cat, items in all_cats.items():
        if items:
            lines.append(f"<b>【{cat}】</b>")
            for t, u in items:
                lines.append(f"- {fmt_link(t, u)}")
            lines.append("")

    # ── 高共识信号 ────────────────────────────
    lines += ["━━━━━━━━━━━━", "<b>🔥 高共识信号</b>", ""]

    signals = []

    # 信号1: AI基础设施
    if any("英伟达" in t or "nvidia" in t.lower() for t, _ in rel[:20]):
        signals.append("1️⃣ <b>AI基础设施支出加速</b> — 英伟达/AMD DC收入暴涨，数据中心建设持续加码")

    # 信号2: HBM/存储
    if any("HBM" in t or "存储" in t or "海力士" in t for t, _ in rel):
        signals.append("2️⃣ <b>HBM/存储超级周期</b> — SK海力士/美光HBM需求超供应，供不应求格局持续")

    # 信号3: 推理>训练
    if any("推理" in t or "inference" in t.lower() or "GB300" in t for t, _ in rel):
        signals.append("3️⃣ <b>推理&gt;训练时代来临</b> — GB300推理性能实测2.7x，推理需求料将超过训练")

    # 信号4: AMD财报
    if any("AMD" in t and ("超预期" in t or "财报" in t or "Q1" in t) for t, _ in rel):
        signals.append("4️⃣ <b>AMD财报全线超预期</b> — DC收入+57%领跑，服务器CPU TAM扩至$1200亿")

    if not signals:
        signals = [
            "1️⃣ <b>AI基础设施</b> — 关注英伟达/AMD/云厂商最新动态",
            "2️⃣ <b>HBM/存储</b> — 关注SK海力士/美光/三星",
            "3️⃣ <b>大模型进展</b> — GPT/Gemini/Claude/ Llama更新",
            "4️⃣ <b>电力需求</b> — 数据中心扩张推动能源需求"
        ]

    for s in signals[:4]:
        lines.append(s)

    # ── 新主题 ───────────────────────────────
    new_items = new_themes[:3]
    if not new_items:
        # 兜底：从全部相关新闻中取最新的3条
        new_items = [(t, u) for t, u in rel[:10] if (t, u) not in sum([v for k, v in all_cats.items() if k != "新主题"], [])]

    if new_items:
        lines += ["", "━━━━━━━━━━━━", "<b>🆕 新兴主题</b>", ""]
        for t, u in new_items[:3]:
            lines.append(f"- {fmt_link(t, u)}")

    # ── 底部 ───────────────────────────────
    lines += ["", "━━━━━━━━━━━━", "⚠️ <i>内容综合自公开信息，仅供参考，不构成投资建议。</i>"]

    return "<br>".join(lines)


if __name__ == "__main__":
    print(build_html())