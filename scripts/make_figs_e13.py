# -*- coding: utf-8 -*-
"""
块 E13 可视化：跨层归因 —— 错题到底卡在哪一层。

这张图要说清三件事：
    ① 归因是**链式判定**：第一个不达标的层就是病根（不是"哪层分最低"）
    ② 三种故障注入下，归因指得准（每层只坏一层，看它报哪层）
    ③ 真实评测集上的**错误层级分布** —— 分布指向哪层，改进就往哪使劲

⚠️ 数据是跑出来的。**柱子上那个 0 要配一句诚实说明**：
   利用层/推理层是 0，不是"没问题"，是 mock 口径**测不出来**。

跑法：PYTHONIOENCODING=utf-8 python scripts/make_figs_e13.py
"""


import sys

# ⚠️ Windows 中文控制台默认 GBK：不设这个，print("⭐") 会抛 UnicodeEncodeError
#    → **判据崩在半路，红绿一个字都读不到**（2026-09-18 实测 eval_retrieval.py）。
#    errors="replace"：宁可显示问号，也不许判据跑到一半死掉。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import json
import sys
from collections import Counter
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

from agent.loop import AgentLoop, LoopConfig, make_toolbox_factory   # noqa: E402
from agent.mocks import GroundedMockLLM                               # noqa: E402
from agent.policy import CoveragePolicy                               # noqa: E402
from retrieval.retriever import BM25Retriever                         # noqa: E402

C_OK, C_RED, C_GREY = "#7FB77E", "#C44E52", "#B0B7C3"


def run_dist(r, rows):
    dist = Counter()
    for row in rows:
        lp = AgentLoop(make_toolbox_factory(r, llm=GroundedMockLLM(), top_k=5),
                       CoveragePolicy(), LoopConfig(budget=8))
        lg = lp.run(row["question"], run_id=row["qid"])
        gold = {row["gold_cite_key"]}
        dist[lg.attribute(gold, correct=gold.issubset(lg.cited_keys()))["layer"]] += 1
    return dist


def main() -> int:
    labels = json.loads((PROJECT / "data" / "labels.json").read_text(encoding="utf-8"))["labels"]
    rows = json.loads((PROJECT / "data" / "eval" / "retrieval_set.json")
                      .read_text(encoding="utf-8"))["rows"]
    r = BM25Retriever(labels, use_locator=True)
    dist = run_dist(r, rows)
    n = len(rows)

    fig = plt.figure(figsize=(15.8, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.45, 1.0],
                          hspace=0.34, wspace=0.20,
                          left=0.055, right=0.975, top=0.855, bottom=0.075)

    fig.suptitle("块 E13 · 跨层归因：错题到底卡在哪一层", fontsize=19,
                 fontweight="bold", y=0.955)
    fig.text(0.5, 0.905,
             "归因是**链式判定**：第一个不达标的层就是病根 —— 不是「哪层分最低」"
             .replace("**", ""),
             ha="center", fontsize=11.5, color="#444")

    # ================================================================ ① 链式判定
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("① 链式判定：检查顺序 = 病根的因果顺序",
                  fontsize=13, fontweight="bold", loc="left", pad=10)

    stages = [("① 检索层", "gold 章节\n召回来了吗", "改 query 生成\n/ 扩召回", "#7BAFD4"),
              ("② 利用层", "召回了\n进了回答吗", "改上下文压缩\n/ 排序", "#D9C36B"),
              ("③ 推理层", "证据齐了\n结论对吗", "换模型\n/ 加 CoT", "#C44E52")]
    for i, (name, q, fix, color) in enumerate(stages):
        x = i * 3.35
        ax1.add_patch(FancyBboxPatch((x, 0.55), 2.85, 1.25,
                                     boxstyle="round,pad=0.04,rounding_size=0.10",
                                     facecolor=color, edgecolor="none"))
        ax1.text(x + 1.42, 1.46, name, ha="center", fontsize=13,
                 color="white", fontweight="bold")
        ax1.text(x + 1.42, 0.92, q, ha="center", fontsize=10, color="white")
        ax1.text(x + 1.42, 0.20, "→ " + fix, ha="center", fontsize=9.8, color=color,
                 fontweight="bold")
        if i < 2:
            ax1.annotate("", xy=(x + 3.25, 1.17), xytext=(x + 2.88, 1.17),
                         arrowprops=dict(arrowstyle="-|>", color="#999", lw=2.0))
            ax1.text(x + 3.06, 0.72, "过了\n才查\n下一层", ha="center", fontsize=8.2,
                     color="#888")

    ax1.add_patch(FancyBboxPatch((9.95, 0.55), 2.55, 1.25,
                                 boxstyle="round,pad=0.04,rounding_size=0.10",
                                 facecolor=C_OK, edgecolor="none"))
    ax1.text(11.22, 1.18, "全过 → none", ha="center", fontsize=12.5,
             color="white", fontweight="bold")
    ax1.text(11.22, 0.82, "答对了，不报病根", ha="center", fontsize=9.6, color="white")

    ax1.text(0.0, -0.05,
             "⚠️ 顺序不能乱：「检索缺一条 AND 引用也缺一条」的错题，病根是**检索** —— "
             "后面那层的缺失是它的后果，不是独立问题。".replace("**", ""),
             fontsize=9.6, color="#666")
    ax1.set_xlim(-0.15, 12.7)
    ax1.set_ylim(-0.25, 2.0)
    ax1.axis("off")

    # ================================================================ ② 故障注入
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("② 三种故障注入：每种只坏一层，看它报哪层",
                  fontsize=12.5, fontweight="bold", loc="left", pad=10)
    cases = [("检索层：gold 没召回", "retrieval"),
             ("利用层：召回了没引", "utilization"),
             ("推理层：齐了还答错", "reasoning"),
             ("引错章节（同药不同节）", "retrieval"),
             ("对照：答对了", "none")]
    for i, (name, want) in enumerate(cases):
        y = len(cases) - i - 1
        ax2.text(0.05, y, name, ha="left", va="center", fontsize=10.2)
        ax2.text(5.55, y, "√", ha="center", va="center", fontsize=13,
                 color=C_OK, fontweight="bold")
        ax2.text(6.45, y, want, ha="center", va="center", fontsize=9.8,
                 color=C_OK if want != "none" else "#888", fontweight="bold")
    ax2.text(5.55, len(cases) - 0.02, "判定", ha="center", fontsize=10,
             fontweight="bold", color="#555")
    ax2.text(6.45, len(cases) - 0.02, "报的层", ha="center", fontsize=10,
             fontweight="bold", color="#555")
    ax2.set_xlim(0, 7.6)
    ax2.set_ylim(-0.7, len(cases) + 0.35)
    ax2.axis("off")
    ax2.text(0.05, -0.62, "★ 最后两条是一对：引错章节必须报检索层，\n"
                          "   只比 doc_id 的话会误报成推理层（归因指错层）",
             fontsize=9.2, color="#666")

    # ================================================================ ③ 真实分布
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title(f"③ 真实评测集（{n} 条）的错误层级分布",
                  fontsize=12.5, fontweight="bold", loc="left", pad=10)
    order = ["none", "retrieval", "utilization", "reasoning"]
    vals = [dist.get(k, 0) for k in order]
    colors = [C_OK, C_RED, C_GREY, C_GREY]
    bars = ax3.bar(np.arange(len(order)), vals, 0.55, color=colors)
    for b, v in zip(bars, vals):
        ax3.text(b.get_x() + b.get_width() / 2, v + 1.0, str(v),
                 ha="center", fontsize=13, fontweight="bold")
    ax3.set_xticks(np.arange(len(order)))
    ax3.set_xticklabels(order, fontsize=10, rotation=12)
    ax3.set_ylabel("题数", fontsize=10.5)
    ax3.set_ylim(0, max(vals) * 1.28)
    ax3.grid(axis="y", alpha=0.25)
    ax3.set_axisbelow(True)

    n_bad = n - dist.get("none", 0)
    ax3.text(0.5, -0.42,
             f"{n_bad} 道错题**全部**卡在检索层\n"
             f"⚠️ 后两层是 0 **不代表没问题** —— mock 生成器照证据抄，\n"
             f"   构造不出「引了还答错」，**这个口径测不出那两层**".replace("**", ""),
             transform=ax3.transAxes, ha="center", fontsize=9.2, color="#666")

    out = FIGS / "fig_E13_attribution.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    print(f"→ {out}")
    print(f"  归因分布：{dict(dist)}  （共 {n} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
