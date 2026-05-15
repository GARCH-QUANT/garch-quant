#!/usr/bin/env python3
"""
每日 AI 科技资讯 - 情绪增强版
在原有新闻聚合基础上增加 FinBERT/LLM 情绪打分
- FinBERT 模式：本地高速推理，零 API 成本
- LLM 模式：通过 stdin 传递新闻列表，由 Agent 调用 LLM 分析后回填
"""

import requests
import time
import json
import os
import sys
from datetime import datetime

NEWSNOW_BASE = "https://newsnow.busiyi.world"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
}

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
    "大选", "关税", "中美", "特朗普", "Trump",
    "蛋白质", "AlphaFold", "DeepMind", "生物学",
    "能源", "核能", "核电", "太阳能", "电力",
    "生物科技", "Biotech", "CRISPR", "mRNA",
]

EXCLUDE_KW = [
    "A股", "上证", "深证", "科创", "创业板", "涨跌停",
    "房地产", "房价", "楼市", "恒大", "碧桂园",
    "券商", "基金", "私募", "理财", "净值",
    "余额宝", "货币基金", "REITs",
    "许昆林", "省委", "书记", "调研",
]


def news(source_id: str, count: int = 25) -> list:
    try:
        resp = requests.get(f"{NEWSNOW_BASE}/api/s?id={source_id}", headers=HEADERS, timeout=15)
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


def sentiment_score_keyword(title: str) -> tuple:
    """基于关键词的简易情绪打分（兜底方案）"""
    pos_kw = ["超预期", "暴涨", "突破", "首", "增长", "合作", "增持", "利好", "突破", "革命", "新", "开放", "史上"]
    neg_kw = ["暴跌", "亏损", "危机", "裁员", "衰退", "违约", "崩盘", "下调", "风险", "制裁", "告", "起诉", "倒"]
    pos_count = sum(1 for k in pos_kw if k in title)
    neg_count = sum(1 for k in neg_kw if k in title)
    net = pos_count - neg_count
    if net >= 2:
        return 0.6 + net * 0.1, "positive"
    elif net <= -2:
        return -0.6 - abs(net) * 0.1, "negative"
    elif net == 1:
        return 0.25, "positive"
    elif net == -1:
        return -0.25, "negative"
    return 0.0, "neutral"


def analyze_sentiment_finbert(texts: list) -> list:
    """使用 FinBERT 本地分析（需安装 transformers + torch）"""
    try:
        import torch
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        from transformers.utils import logging as tf_logging
        tf_logging.set_verbosity_error()

        bert_model = os.getenv("BERT_SENTIMENT_MODEL", "uer/roberta-base-finetuned-chinanews-chinese")
        pipe = pipeline("sentiment-analysis", model=bert_model, tokenizer=bert_model, device=-1)
        results = pipe(texts, truncation=True, max_length=512)

        scores = []
        for r in results:
            label = r['label'].lower()
            score = r['score']
            if 'negative' in label:
                score = -score
            elif 'neutral' in label:
                score = 0.0
            final_label = 'positive' if score > 0.1 else ('negative' if score < -0.1 else 'neutral')
            scores.append({"score": round(score, 3), "label": final_label})
        return scores
    except ImportError as e:
        print(f"⚠️ FinBERT 依赖未安装: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"⚠️ FinBERT 分析失败: {e}", file=sys.stderr)
        return None


def build_html(sentiment_mode: str = "keyword") -> str:
    """
    sentiment_mode: 'finbert' (本地模型) 或 'keyword' (关键词兜底)
    """
    now = datetime.now()
    date_str = now.strftime("%Y年%m月%d日")
    day_str = ["周一","周二","周三","周四","周五","周六","周日"][now.weekday()]
    time_str = now.strftime("%H:%M")

    # ── 抓新闻 ─────────────────────────────────
    sources = [
        ("36kr", "36氪"), ("ithome", "IT之家"), ("juejin", "掘金"),
        ("hackernews", "Hacker News"), ("wallstreetcn", "华尔街见闻"), ("cls", "财联社"),
    ]
    all_news = []
    for sid, _ in sources:
        all_news += news(sid, 20)
        time.sleep(0.25)

    rel = [(t, u) for t, u in all_news if is_relevant(t)]

    # ── 情绪分析 ───────────────────────────────
    if sentiment_mode == "finbert":
        texts = [t for t, _ in rel]
        sentiments = analyze_sentiment_finbert(texts)
        if sentiments is None:
            sentiments = [{"score": 0.0, "label": "neutral"}] * len(rel)
    else:
        sentiments = [sentiment_score_keyword(t) for t, _ in rel]
        sentiments = [{"score": s, "label": l} for s, l in sentiments]

    # 给新闻附上情绪
    news_with_sentiment = list(zip(rel, sentiments))
    news_with_sentiment.sort(key=lambda x: abs(x[1]["score"]), reverse=True)

    # ── 分类（保留Top 5） ───────────────────────
    categories = {
        "AI/大模型": [(k, v) for k, v in news_with_sentiment if any(w in k[0].lower() for w in ["openai", "anthropic", "google", "微软", "meta", "chatgpt", "gemini", "claude", "gcp", "azure", "aws", "阿里云"])],
        "硬件/算力": [(k, v) for k, v in news_with_sentiment if any(w in k[0] for w in ["英伟达", "nvidia", "AMD", "GPU", "HBM", "GB200", "GB300", "光模块", "InP"])],
        "市场/个股": [(k, v) for k, v in news_with_sentiment if any(w in k[0] for w in ["$", "股价", "营收", "财报", "超预期", "beat", "miss", "分析师"])],
        "能源/电力": [(k, v) for k, v in news_with_sentiment if any(w in k[0] for w in ["能源", "电力", "核电", "核能", "太阳能", "数据中心"])],
        "新主题": [(k, v) for k, v in news_with_sentiment if any(w in k[0] for w in ["漏洞", "安全", "突破", "首次", "新型", "量子", "生物", "蛋白质"])],
    }

    # ── HTML 构建 ──────────────────────────────
    emoji_map = {"positive": "🟢", "negative": "🔴", "neutral": "⚪️"}

    lines = [
        f"📊 <b>Tech Report | {date_str} {day_str}</b>",
        f"🕐 北京时间 {time_str} | 情绪模式: {sentiment_mode}",
        "━━━━━━━━━━━━",
        "<b>🔍 科技要闻 + 情绪打分</b>",
        "",
    ]

    for cat, items in categories.items():
        if items:
            lines.append(f"<b>【{cat}】</b>")
            for (t, u), sent in items[:4]:
                emo = emoji_map.get(sent["label"], "⚪️")
                score_str = f"{sent['score']:+.2f}"
                short = t[:60] + "…" if len(t) > 60 else t
                link = fmt_link(t, u) if u else short
                lines.append(f"{emo} {score_str} | {link}")
            lines.append("")

    # ── 高共识信号 ────────────────────────────
    top_negative = [(t, u, s) for (t, u), s in news_with_sentiment if s["label"] == "negative"][:2]
    top_positive = [(t, u, s) for (t, u), s in news_with_sentiment if s["label"] == "positive"][:2]

    lines += ["━━━━━━━━━━━━", "<b>🔥 情绪信号</b>", ""]
    if top_positive:
        lines.append("<b>🟢 利好信号：</b>")
        for t, u, s in top_positive:
            lines.append(f"  • {fmt_link(t, u)} ({s['score']:+.2f})")
        lines.append("")
    if top_negative:
        lines.append("<b>🔴 利空信号：</b>")
        for t, u, s in top_negative:
            lines.append(f"  • {fmt_link(t, u)} ({s['score']:+.2f})")
        lines.append("")

    # ── 情绪分布 ────────────────────────────
    pos_n = sum(1 for s in sentiments if s["label"] == "positive")
    neg_n = sum(1 for s in sentiments if s["label"] == "negative")
    neu_n = len(sentiments) - pos_n - neg_n
    avg_score = sum(s["score"] for s in sentiments) / len(sentiments) if sentiments else 0

    lines += ["━━━━━━━━━━━━", "<b>📈 情绪分布</b>", ""]
    lines.append(f"🟢 积极: {pos_n}  ⚪️ 中性: {neu_n}  🔴 消极: {neg_n}")
    lines.append(f"平均情绪: {avg_score:+.3f}  {'偏正面 😄' if avg_score > 0.1 else ('偏负面 😟' if avg_score < -0.1 else '中性 😐')}")
    lines += ["", "━━━━━━━━━━━━", "⚠️ <i>内容综合自公开信息，仅供参考，不构成投资建议。</i>"]

    return "<br>".join(lines)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "keyword"
    print(build_html(sentiment_mode=mode))
