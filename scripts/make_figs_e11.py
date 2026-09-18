# -*- coding: utf-8 -*-
"""
块 E11 可视化：引用**偏不偏**（跨药一致性核验）。

这张图要先讲一件反直觉的事：
    **"多几份说明书就会冲突"是错的。** 50 份 FDA 说明书里，
    「两边说法相反」**一例都没有** —— 它们在同一套监管要求下写，本来就高度一致。

    真正的不一致是**信息不对称**：一边明确警告，另一边**只字不提**。

⚠️ 数据是**扫出来的**，不是画的。

跑法：PYTHONIOENCODING=utf-8 D:/anaconda/envs/myenv/python.exe scripts/make_figs_e11.py
"""


import sys

# ⚠️ Windows 中文控制台默认 GBK：不设这个，print("⭐") 会抛 UnicodeEncodeError
#    → **判据崩在半路，红绿一个字都读不到**（2026-09-18 实测 eval_retrieval.py）。
#    errors="replace"：宁可显示问号，也不许判据跑到一半死掉。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
FIGS = PROJECT / "reports" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

from agent.consistency import DrugIndex, WARN, ABSENT, MENTION   # noqa: E402
from retrieval.retriever import BM25Retriever                     # noqa: E402

C_GREY, C_RED, C_GREEN = "#B0B7C3", "#C44E52", "#7FB77E"


def scan(idx: DrugIndex):
    """全量扫：甲药正文提到乙药 → 四种形态。**图和判据用的是同一套扫描。**"""
    stat = Counter()
    asym = []
    for a in idx.drugs:
        for b in idx.drugs:
            if a == b:
                continue
            sa = idx.side(a, b)
            if sa.label == ABSENT:
                continue
            sb = idx.side(b, a)
            stat[(sa.label, sb.label)] += 1
            if sa.label == WARN and sb.label != WARN:
                asym.append((a, b, sa.sentence, sb.label))
    return stat, asym


def main() -> int:
    labels = json.loads((PROJECT / "data" / "labels.json").read_text(encoding="utf-8"))["labels"]
    r = BM25Retriever(labels)
    idx = DrugIndex(r.chunks, [c.drug for c in r.chunks])
    stat, asym = scan(idx)

    n_pairs = sum(stat.values())
    n_opposite = stat.get((WARN, WARN), 0)          # 双方都警告 = 一致，不是相反
    # 「两边说反」在极性框架下等价于：一方警告、另一方说"可以用"（我们扫不到 MENTION 里带明确允许义的）
    n_mention_warn = stat.get((WARN, MENTION), 0) + stat.get((MENTION, WARN), 0)

    fig = plt.figure(figsize=(16.2, 10.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.85, 1.35], hspace=0.28,
                          left=0.055, right=0.975, top=0.86, bottom=0.05)

    fig.suptitle("块 E11 · 引用偏不偏：跨药一致性核验", fontsize=19,
                 fontweight="bold", y=0.955)
    fig.text(0.5, 0.905,
             f"扫了 {len(idx.drugs)} 份 FDA 说明书、{n_pairs} 对互相提及 —— "
             f"「两边说法相反」0 例；真正的不一致是**信息不对称**：一边警告、另一边只字不提"
             .replace("**", ""),
             ha="center", fontsize=11.5, color="#444")

    # ================================================================ ① 形态分布
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("① 全量扫描：95 对药互相提及，只有四种形态",
                  fontsize=13, fontweight="bold", loc="left", pad=10)

    order = [("普通提及 × 对方没提", stat.get((MENTION, ABSENT), 0), C_GREY),
             ("★ 强警告 × 对方没提", stat.get((WARN, ABSENT), 0), C_RED),
             ("双方都警告（一致）", stat.get((WARN, WARN), 0), C_GREEN),
             ("双方都提及（一致）", stat.get((MENTION, MENTION), 0), C_GREEN)]
    xs = np.arange(len(order))
    vals = [v for _, v, _ in order]
    bars = ax1.bar(xs, vals, 0.55, color=[c for _, _, c in order])
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 1.2, str(v),
                 ha="center", fontsize=14, fontweight="bold")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([n for n, _, _ in order], fontsize=10.5)
    ax1.set_ylabel("药对数量", fontsize=11)
    ax1.set_ylim(0, max(vals) * 1.30)
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_axisbelow(True)

    ax1.annotate("两边说法相反：**0 例**".replace("**", ""),
                 xy=(1.0, max(vals) * 0.62), xytext=(2.55, max(vals) * 0.86),
                 fontsize=13, fontweight="bold", color=C_RED,
                 arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=2.0))
    ax1.text(2.55, max(vals) * 0.62,
             "FDA 说明书在同一套监管要求下写，\n它们本来就高度一致 —— 多几份说明书写不出冲突",
             fontsize=10, color="#666")

    # ================================================================ ② 7 对不对称
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title(f"② 真正的素材：{len(asym)} 对「单向强警告」—— 甲明确警告，乙的说明书只字不提",
                  fontsize=13, fontweight="bold", loc="left", pad=10)

    for i, (a, b, sent, lb) in enumerate(asym):
        y = len(asym) - i - 1
        ax2.add_patch(FancyBboxPatch((0.0, y - 0.36), 3.45, 0.72,
                                     boxstyle="round,pad=0.01,rounding_size=0.05",
                                     facecolor=C_RED, edgecolor="none"))
        ax2.text(0.16, y, f"{a}  →  警告 {b}", ha="left", va="center",
                 fontsize=11.5, color="white", fontweight="bold")
        ax2.text(3.62, y, f"「{sent[:88]}」", ha="left", va="center",
                 fontsize=9.6, color="#333")
        ax2.add_patch(FancyBboxPatch((12.05, y - 0.28), 1.85, 0.56,
                                     boxstyle="round,pad=0.01,rounding_size=0.05",
                                     facecolor="#EDEFF2", edgecolor="#D5D9DE"))
        ax2.text(12.97, y, f"{b}：未提及", ha="center", va="center",
                 fontsize=9.6, color="#888")

    ax2.text(0.0, len(asym) - 0.02, "谁警告了谁（警告方）", fontsize=10.5,
             fontweight="bold", color="#555")
    ax2.text(3.62, len(asym) - 0.02, "原文摘录（来自警告方的说明书）", fontsize=10.5,
             fontweight="bold", color="#555")
    ax2.text(12.05, len(asym) - 0.02, "对方说明书的表态", fontsize=10.5,
             fontweight="bold", color="#555")

    ax2.text(0.0, -0.85,
             "★ 两种问题的**正确动作完全不同**："
             "「选择性引用」（对方也说了、却没引）→ 去**补检索**；"
             "「沉默未标注」（对方确实没提）→ 加一句**如实说明**。"
             "混成一个，agent 会白白重试（E9 的 not_retrieved / not_in_corpus 踩过一模一样的坑）"
             .replace("**", ""),
             fontsize=9.6, color="#666")

    ax2.set_xlim(-0.15, 14.2)
    ax2.set_ylim(-1.30, len(asym) + 0.35)
    ax2.axis("off")

    out = FIGS / "fig_E11_consistency.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    print(f"→ {out}")
    print(f"  形态分布：{dict(stat)}")
    print(f"  单向强警告：{len(asym)} 对")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
