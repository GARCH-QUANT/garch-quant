#!/usr/bin/env python3
"""
全球宏观日报 - HTML版
- 市场数据：Twelve Data
- 新闻：财联社 + 精选关键词过滤（排除国内A股/房产等无关内容）
- 输出纯HTML，用于 Telegram HTML模式发送
"""

import requests
import time
from datetime import datetime

import os
TWELVE_DATA_KEY = os.environ.get("TWELVE_DATA_KEY", "")
NEWSNOW_BASE = "https://newsnow.busiyi.world"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
}

# 负面关键词：用来过滤国内财经噪音
EXCLUDE_KW = [
    "A股", "上证", "深证", "创业板", "科创板", "沪深", "涨停", "跌停",
    "房地产", "房价", "楼市", "开发商", "恒大", "碧桂园", "融创",
    "券商", "基金", "私募", "理财", "净值",
    "许昆林", "省委", "书记", "调研", "沈阳市", "数字经济",
    "天弘余额宝", "余额宝", "公募",
]


def td_price(symbols: list) -> dict:
    try:
        resp = requests.get(
            "https://api.twelvedata.com/price",
            params={"symbol": ",".join(symbols), "apikey": TWELVE_DATA_KEY},
            timeout=15
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def td_quote(symbol: str) -> dict:
    try:
        resp = requests.get(
            "https://api.twelvedata.com/quote",
            params={"symbol": symbol, "apikey": TWELVE_DATA_KEY},
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def news(source_id: str, count: int = 15) -> list:
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
    title_lower = title.lower()
    exclude_any = any(kw.lower() in title_lower for kw in EXCLUDE_KW)
    if exclude_any:
        return False
    # 必须包含以下任一关键词才算是宏观相关内容
    relevant_kw = [
        "美国", "中国", "全球", "美联储", "特朗普", "拜登", "鲍威尔",
        "利率", "国债", "债券", "收益率", "通胀", "CPI", "PPI",
        "非农", "就业", "GDP", "经济", "衰退", "央行", "加息", "降息",
        "原油", "石油", "OPEC", "黄金", "美元", "欧元", "英镑", "日元",
        "伊朗", "俄乌", "中东", "制裁", "能源", "LNG", "天然气",
        "英伟达", "AMD", "苹果", "Meta", "特斯拉", "AI",
        "标普", "纳斯达克", "道琼斯", "期货", "期权",
        "英国", "欧洲", "德国", "法国", "日本", "澳洲", "加拿大",
        "G7", "G20", "IMF", "世界银行", "OECD",
        "贸易战", "关税", "制裁", "核", "大选",
    ]
    return any(kw in title_lower for kw in relevant_kw)


def fmt_chg(chg) -> str:
    if chg is None:
        return "—"
    try:
        return f"{float(chg):+.2f}%"
    except Exception:
        return "—"


def safe_float(val):
    try:
        return float(val)
    except Exception:
        return None


def build_html() -> str:
    now = datetime.now()
    date_str = now.strftime("%Y年%m月%d日")
    day_str = ["周一","周二","周三","周四","周五","周六","周日"][now.weekday()]
    time_str = now.strftime("%H:%M")

    # ── 市场数据（Twelve Data）────────────────────
    prices = td_price(["SPY", "QQQ", "DIA", "GLD", "USO", "TLT"])
    time.sleep(6)   # 等 rate limit

    spy_quote = td_quote("SPY")
    qqq_quote = td_quote("QQQ")
    time.sleep(1)

    def get_p(sym):
        d = prices.get(sym) if isinstance(prices, dict) else None
        return d.get("price") if isinstance(d, dict) else None

    spy_p = get_p("SPY")
    qqq_p = get_p("QQQ")
    dia_p = get_p("DIA")
    gld_p = get_p("GLD")
    uso_p = get_p("USO")
    tlt_p = get_p("TLT")

    def get_pc(sym, quote_obj):
        if isinstance(quote_obj, dict):
            return quote_obj.get("percent_change")
        return None

    spy_pc = get_pc("SPY", spy_quote)
    qqq_pc = get_pc("QQQ", qqq_quote)

    # ── 新闻（财联社 + 过滤）───────────────────────
    raw = news("cls", 30) + news("wallstreetcn", 20)
    time.sleep(0.3)

    # 过滤 + 分类
    relevant = [(t, u) for t, u in raw if is_relevant(t)]

    rate_news = [(t, u) for t, u in relevant if any(
        k in t.lower() for k in ["利率", "国债", "债券", "收益率", "美联储", "加息", "降息", "鲍威尔", "央行", "Powell", "yield", "bond", "rate"]
    )][:4]
    infl_news = [(t, u) for t, u in relevant if any(
        k in t.lower() for k in ["通胀", "CPI", "PPI", "物价", "原油", "石油", "能源", "汽油", "大宗", "inflation", "oil", "price"]
    )][:4]
    labor_news = [(t, u) for t, u in relevant if any(
        k in t.lower() for k in ["非农", "就业", "失业", "GDP", "经济", "衰退", "消费", "零售", "job", "employment", "gdp", "economy"]
    )][:3]
    geo_news = [(t, u) for t, u in relevant if any(
        k in t.lower() for k in ["伊朗", "俄乌", "中东", "制裁", "能源", "LNG", "OPEC", "石油", "核", "特朗普", "G7", "关税", "贸易"]
    )][:4]
    dollar_news = [(t, u) for t, u in relevant if any(
        k in t.lower() for k in ["美元", "黄金", "欧元", "英镑", "日元", "汇率", "外汇", "央行", "储备", "dollar", "gold", "fx"]
    )][:3]
    tech_news = [(t, u) for t, u in relevant if any(
        k in t.lower() for k in ["英伟达", "nvidia", "AMD", "苹果", "apple", "Meta", "特斯拉", "TSLA", "AI", "chatgpt", "openai"]
    )][:3]

    # ── HTML 构建 ───────────────────────────────
    def link_item(title, url):
        short = title[:65] + "…" if len(title) > 65 else title
        return f'- <a href="{url}">{short}</a>'

    def section(title, items):
        out = [title]
        if items:
            out += [link_item(t, u) for t, u in items]
        else:
            out.append("- 暂无最新数据")
        return out

    lines = [
        f"🌍 <b>Macro Report | {date_str} {day_str}</b>",
        f"🕐 北京时间 {time_str}",
        "━━━━━━━━━━━━",
    ]

    # 利率市场
    lines += section("📈 <b>利率市场</b>", rate_news)

    # 通胀压力
    if not infl_news and gld_p:
        g = safe_float(gld_p)
        if g:
            infl_news = [(f"黄金(GLD) <b>${g:.2f}</b>（{fmt_chg(get_pc('GLD', td_quote('GLD')))}）", "")]
    lines += section("🔥 <b>通胀压力</b>", infl_news)

    # 就业/经济
    lines += section("💼 <b>就业/经济</b>", labor_news)

    # 地缘与能源
    lines += section("🌐 <b>地缘与能源</b>", geo_news)

    # 美元与黄金
    if not dollar_news and gld_p:
        g = safe_float(gld_p)
        gc = get_pc("GLD", td_quote("GLD"))
        if g:
            dollar_news = [(f"黄金(GLD) <b>${g:.2f}</b>（{fmt_chg(gc)}）", "")]
    lines += section("💵 <b>美元与黄金</b>", dollar_news)

    # 市场大盘
    mkt = []
    if spy_p:
        s = safe_float(spy_p)
        if s:
            mkt.append((f"标普500(SPY) <b>${s:.2f}</b>（{fmt_chg(spy_pc)}）", ""))
    if qqq_p:
        q = safe_float(qqq_p)
        if q:
            mkt.append((f"纳斯达克100(QQQ) <b>${q:.2f}</b>（{fmt_chg(qqq_pc)}）", ""))
    if dia_p:
        d = safe_float(dia_p)
        if d:
            dc = get_pc("DIA", td_quote("DIA"))
            mkt.append((f"道琼斯(DIA) <b>${d:.2f}</b>（{fmt_chg(dc)}）", ""))
    if uso_p:
        o = safe_float(uso_p)
        if o:
            oc = get_pc("USO", td_quote("USO"))
            mkt.append((f"WTI原油ETF(USO) <b>${o:.2f}</b>（{fmt_chg(oc)}）", ""))
    if gld_p:
        g = safe_float(gld_p)
        if g:
            gc = get_pc("GLD", td_quote("GLD"))
            mkt.append((f"黄金(GLD) <b>${g:.2f}</b>（{fmt_chg(gc)}）", ""))

    lines += section("📊 <b>市场大盘</b>", mkt if mkt else [])

    # 科技动态
    if tech_news:
        lines += ["", "🔥 <b>科技动态</b>"]
        for t, u in tech_news:
            short = t[:65] + "…" if len(t) > 65 else t
            lines.append(f'- <a href="{u}">{short}</a>')

    # 观点提示
    lines += ["", "━━━━━━━━━━━━", "🔴 <b>SPY 观点提示</b>", ""]

    signals = []
    if spy_p and spy_pc is not None:
        s = safe_float(spy_p)
        if s:
            signals.append(f"SPY {fmt_chg(spy_pc)} → ${s:.2f}")
    if qqq_p and qqq_pc is not None:
        q = safe_float(qqq_p)
        if q:
            signals.append(f"QQQ {fmt_chg(qqq_pc)} → ${q:.2f}")
    if gld_p:
        g = safe_float(gld_p)
        if g:
            gc = get_pc("GLD", td_quote("GLD"))
            signals.append(f"黄金 {fmt_chg(gc)} → ${g:.2f}")

    if signals:
        lines.append("当前信号：" + " / ".join(signals))
    lines.append("")
    lines.append("⚠️ 仅供参考，不构成投资建议。实时数据请以交易所/彭博终端为准。")

    return "<br>".join(lines)


if __name__ == "__main__":
    print(build_html())