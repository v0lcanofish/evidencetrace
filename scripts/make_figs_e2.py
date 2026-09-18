# -*- coding: utf-8 -*-
"""块 E2 可视化：编排循环 + 找到的两个 bug。

跑法：D:/anaconda/python.exe scripts/make_figs_e2.py
"""


import sys

# ⚠️ Windows 中文控制台默认 GBK：不设这个，print("⭐") 会抛 UnicodeEncodeError
#    → **判据崩在半路，红绿一个字都读不到**（2026-09-18 实测 eval_retrieval.py）。
#    errors="replace"：宁可显示问号，也不许判据跑到一半死掉。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIGS = PROJECT / "reports" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


def main() -> int:
    fig = plt.figure(figsize=(16, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.35, 1.15], wspace=0.26)

    # ---------------- ① 编排循环
    ax0 = fig.add_subplot(gs[0]); ax0.axis("off")
    ax0.set_xlim(0, 10); ax0.set_ylim(0, 10)
    ax0.set_title("① 编排循环（每步都写账本）", fontsize=13, fontweight="bold", pad=10)

    steps = [
        ("plan", "拆子问题", "#1565c0"),
        ("retrieve", "检索 → Doc", "#00897b"),
        ("retrieve", "再检索（去重后）", "#00897b"),
        ("synthesize", "写报告（带引用）", "#6a1b9a"),
    ]
    y = 8.6
    for i, (act, desc, c) in enumerate(steps, 1):
        ax0.add_patch(mp.FancyBboxPatch((0.5, y - 0.42), 9.0, 0.84,
                                        boxstyle="round,pad=0.08", fc=c, ec=c))
        ax0.text(0.85, y, f"step {i}", va="center", fontsize=9.5,
                 color="white", fontweight="bold")
        ax0.text(2.1, y, f"{act}", va="center", fontsize=9.8, color="white")
        ax0.text(4.6, y, desc, va="center", fontsize=9.3, color="#e0e0e0")
        y -= 1.35
        if i < len(steps):
            ax0.annotate("", xy=(5.0, y + 0.95), xytext=(5.0, y + 0.5),
                         arrowprops=dict(arrowstyle="-|>", color="#90a4ae", lw=1.8))

    ax0.add_patch(mp.FancyBboxPatch((0.5, 1.6), 9.0, 1.35,
                                    boxstyle="round,pad=0.1",
                                    fc="#ffebee", ec="#c62828", lw=2))
    ax0.text(5.0, 2.55, "★ 闭包断言", ha="center", fontsize=11,
             color="#c62828", fontweight="bold")
    ax0.text(5.0, 2.0, "每条引用都要能追到某一步的检索\n追不到 → 抛异常，该 run 标记 malformed",
             ha="center", fontsize=9.5, color="#37474f")

    # ---------------- ② 找到的两个 bug
    ax1 = fig.add_subplot(gs[1]); ax1.axis("off")
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
    ax1.set_title("② 接编排时暴露的两个 bug", fontsize=13, fontweight="bold", pad=10)

    # bug 1
    ax1.add_patch(mp.FancyBboxPatch((0.2, 5.3), 9.6, 3.6,
                                    boxstyle="round,pad=0.12",
                                    fc="#fff3e0", ec="#ef6c00", lw=2))
    ax1.text(0.55, 8.5, "bug 1：去重按 doc_id 做", fontsize=11,
             fontweight="bold", color="#e65100")
    ax1.text(0.55, 7.95, "一份说明书有多节，共用同一个 setid：", fontsize=9.6,
             color="#37474f")
    for k, (sec, lo) in enumerate([("Drug Interactions", "34073-7"),
                                   ("Contraindications", "34070-3")]):
        ax1.add_patch(mp.FancyBboxPatch((0.8, 7.15 - k * 0.62), 8.4, 0.5,
                                        boxstyle="round,pad=0.05",
                                        fc="#5a709591"[:7] if False else "#00897b",
                                        ec="#00695c"))
        ax1.text(1.0, 7.40 - k * 0.62, f"5a709591-... #{lo}  {sec}",
                 va="center", fontsize=9.2, color="white")
    ax1.text(0.55, 5.75, "→ 第二节被当重复滤掉，证据只剩半份",
             fontsize=9.8, color="#e65100", fontweight="bold")

    # bug 2
    ax1.add_patch(mp.FancyBboxPatch((0.2, 1.4), 9.6, 3.5,
                                    boxstyle="round,pad=0.12",
                                    fc="#ffebee", ec="#c62828", lw=2))
    ax1.text(0.55, 4.55, "bug 2：闭包不验章节（更严重）", fontsize=11,
             fontweight="bold", color="#c62828")
    ax1.text(0.55, 4.0, "key.split(\"#\")[0]  ← 把 LOINC 扔了", fontsize=9.6,
             color="#37474f",  )
    ax1.text(0.55, 3.45, "只比「哪份药」，不比「哪一节」", fontsize=9.6,
             color="#37474f")
    ax1.text(0.55, 2.75, "! 引入 LOINC 的全部意义就是验「引的是不是那一节」",
             fontsize=9.8, color="#c62828", fontweight="bold")
    ax1.text(0.55, 2.15, "→ 修完后：药对了但章节不对 = 追不到出处",
             fontsize=9.8, color="#2e7d32", fontweight="bold")
    ax1.text(0.55, 1.65, "（顺带：报告里把哮喘禁忌引到了 metformin 头上，"
                         "闭包居然还通过）",
             fontsize=9.0, color="#78909c")

    # ---------------- ③ 账本长什么样
    ax2 = fig.add_subplot(gs[2]); ax2.axis("off")
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
    ax2.set_title("③ 跑出来的账本（mock）", fontsize=13, fontweight="bold", pad=10)

    log = (
        "step 1  plan        拆出 2 个子问题\n"
        "step 2  retrieve    拿到 2 篇\n"
        "         进报告=True   丢弃=None\n"
        "step 3  retrieve    拿到 2 篇\n"
        "         进报告=False  丢弃=context_budget\n"
        "step 4  synthesize  用了 3 篇证据\n"
        "\n"
        "claims:\n"
        "  c1 → 5a709591#34073-7  (第 2 步)\n"
        "  c2 → 5a709591#34070-3  (第 2 步)\n"
        "\n"
        "闭包 √   同 seed 两次一致 √\n"
        "断网可跑 √   坏引用被拦 √"
    )
    ax2.add_patch(mp.FancyBboxPatch((0.15, 1.5), 9.7, 7.4,
                                    boxstyle="round,pad=0.15",
                                    fc="#263238", ec="#263238"))
    ax2.text(0.45, 8.55, log, fontsize=8.4, color="#b0bec5", va="top")
    ax2.text(5.0, 0.85, "★ 两条 claim 引的是同一份药的两个不同章节\n"
                        "—— 这正是 bug 1 修好之后才对的",
             ha="center", fontsize=9.3, color="#1b5e20", fontweight="bold")

    fig.suptitle("图 E2-1｜编排引擎：循环跑通 + 接入时暴露的两个 bug",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.015,
             "一句话：先搭假模型跑通逻辑再花钱接 API —— 这两个 bug 都是接编排时才暴露的，"
             "光看账本代码发现不了。",
             ha="center", fontsize=11.5, color="#1b5e20", fontweight="bold")
    out = FIGS / "fig_E2_agent.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  √ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
