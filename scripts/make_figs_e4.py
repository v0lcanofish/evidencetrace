# -*- coding: utf-8 -*-
"""块 E4 可视化：语料 + 15 道自建题。

跑法：D:/anaconda/python.exe scripts/make_figs_e4.py
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
import matplotlib.patches as mp

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIGS = PROJECT / "reports" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

DIFF_CN = {"single_hop": "单跳（1 节）", "multi_hop": "多跳（2 节）",
           "cross_drug": "跨药（2 节）"}
COLS = {"single_hop": "#1565c0", "multi_hop": "#ef6c00", "cross_drug": "#6a1b9a"}


def main() -> int:
    labels = json.loads((PROJECT / "data" / "labels.json").read_text(encoding="utf-8"))["labels"]
    qs = json.loads((PROJECT / "data" / "questions_v1.json").read_text(encoding="utf-8"))["questions"]

    fig = plt.figure(figsize=(16, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.35], wspace=0.24)

    # ---------------- ① 语料
    ax0 = fig.add_subplot(gs[0]); ax0.axis("off")
    ax0.set_xlim(0, 10); ax0.set_ylim(0, 10)
    ax0.set_title("① 真语料（DailyMed 抓的）", fontsize=13, fontweight="bold", pad=10)

    ax0.text(0.4, 9.3, f"{len(labels)} 份处方药说明书", fontsize=12,
             fontweight="bold", color="#0d47a1", va="center")
    y = 8.5
    for L in labels:
        ax0.add_patch(mp.FancyBboxPatch((0.4, y - 0.34), 5.6, 0.68,
                                        boxstyle="round,pad=0.06",
                                        fc="#e3f2fd", ec="#1565c0", lw=1.4))
        ax0.text(0.7, y, L["drug"], fontsize=10.5, color="#0d47a1",
                 fontweight="bold", va="center")
        ax0.text(6.2, y, f'{len(L["sections"])} 节', fontsize=9.8,
                 color="#546e7a", va="center")
        y -= 0.92
    n_sec = sum(len(L["sections"]) for L in labels)
    ax0.text(5.0, 2.35, f"共 {n_sec} 个章节\n每节带 LOINC 编码",
             ha="center", fontsize=11, color="#1b5e20", fontweight="bold")
    ax0.text(5.0, 1.35, "★ 章节是 HL7 SPL 标准切的\n不是按字数瞎切",
             ha="center", fontsize=9.6, color="#78909c")

    # ---------------- ② 题目分布
    ax1 = fig.add_subplot(gs[1])
    dc = Counter(q["difficulty"] for q in qs)
    order = ["single_hop", "multi_hop", "cross_drug"]
    vals = [dc.get(k, 0) for k in order]
    bars = ax1.bar([DIFF_CN[k] for k in order], vals,
                   color=[COLS[k] for k in order], width=0.58)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.12, str(v),
                 ha="center", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, max(vals) + 1.6)
    ax1.set_ylabel("题数", fontsize=12)
    ax1.set_title(f"② 15 道题的难度分布", fontsize=13, fontweight="bold", pad=10)
    ax1.tick_params(axis="x", labelsize=10)
    ax1.text(0.5, 0.06,
             "★ 10/15 需要综合 2 节以上\n单跳题只是对照 baseline",
             transform=ax1.transAxes, ha="center", fontsize=10,
             color="#b71c1c", fontweight="bold")

    # ---------------- ③ 一道题的结构
    ax2 = fig.add_subplot(gs[2]); ax2.axis("off")
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
    ax2.set_title("③ 一道题长什么样（q11）", fontsize=13, fontweight="bold", pad=10)

    q = next(x for x in qs if x["qid"] == "q11")
    ax2.add_patch(mp.FancyBboxPatch((0.2, 7.4), 9.6, 2.0,
                                    boxstyle="round,pad=0.12",
                                    fc="#e3f2fd", ec="#1565c0", lw=2))
    ax2.text(0.5, 8.95, "问题", fontsize=10.5, fontweight="bold", color="#0d47a1")
    ax2.text(0.5, 8.15, q["question"].replace(". ", ".\n"),
             fontsize=9.8, color="#263238", va="center")

    ax2.text(0.4, 7.0, "gold 证据集（回答它【必须】引到的章节）",
             fontsize=10.5, fontweight="bold", color="#e65100")
    y = 6.3
    for g in q["gold"]:
        ax2.add_patch(mp.FancyBboxPatch((0.4, y - 0.62), 9.2, 1.24,
                                        boxstyle="round,pad=0.1",
                                        fc="#fff3e0", ec="#ef6c00", lw=1.8))
        ax2.text(0.7, y + 0.20, f'{g["drug"]}  #{g["loinc"]}',
                 fontsize=10.5, fontweight="bold", color="#e65100", va="center")
        ax2.text(0.7, y - 0.28, g["why"], fontsize=9.3, color="#546e7a", va="center")
        y -= 1.55

    ax2.add_patch(mp.FancyBboxPatch((0.2, 1.15), 9.6, 2.0,
                                    boxstyle="round,pad=0.12",
                                    fc="#e8f5e9", ec="#2e7d32", lw=2))
    ax2.text(5.0, 2.75, "这 15 道题是【手写】的，不是程序化生成", ha="center",
             fontsize=10.8, color="#1b5e20", fontweight="bold")
    ax2.text(5.0, 2.15, "因为「该引哪几节」要读内容才能判，程序判不了\n"
                        "—— 这正是「垂类做深」的实质",
             ha="center", fontsize=9.5, color="#37474f")
    ax2.text(5.0, 1.5, "（对照：ToolHorizon 的 800 道是程序化造的，"
                       "因为那边 gold 是结构化动作）",
             ha="center", fontsize=8.8, color="#78909c")

    fig.suptitle("图 E4-1｜自建问题集：6 份真说明书 → 15 道带 gold 的题",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.015,
             "一句话：通用 benchmark 没法给你「回答这个问题必须引哪几节」的标注 —— "
             "这个标注粒度就是垂类做深的价值。",
             ha="center", fontsize=11.5, color="#1b5e20", fontweight="bold")
    out = FIGS / "fig_E4_questions.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  [OK] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
