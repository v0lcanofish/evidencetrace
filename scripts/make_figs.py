# -*- coding: utf-8 -*-
"""
EvidenceTrace 块 E1 / E3 的可视化。

  fig_E1_ledger.png    证据账本的结构与闭包不变量
  fig_E3_sources.png   数据源可达性对照（为什么换掉 NMPA）

跑法：python scripts/make_figs.py
"""


import sys

# ⚠️ Windows 中文控制台默认 GBK：不设这个，print("⭐") 会抛 UnicodeEncodeError
#    → **判据崩在半路，红绿一个字都读不到**（2026-09-18 实测 eval_retrieval.py）。
#    errors="replace"：宁可显示问号，也不许判据跑到一半死掉。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIGS = PROJECT / "reports" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

C_STEP = "#1565c0"
C_DOC = "#2e7d32"
C_CLAIM = "#ef6c00"
C_BAD = "#c62828"
C_OK = "#2e7d32"


def box(ax, x, y, w, h, text, fc, ec, fs=11, tc="white", bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.12",
                                fc=fc, ec=ec, lw=2.0, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3,
            fontweight="bold" if bold else "normal")


def arrow(ax, p1, p2, color, label="", style="-|>", ls="-", off=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=18,
                                 color=color, lw=2.2, linestyle=ls, zorder=1))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + off, label, ha="center", va="center", fontsize=10.5,
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))


# ================================================================ E1
def fig_ledger():
    fig, ax = plt.subplots(figsize=(13, 7.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    box(ax, 0.6, 5.7, 3.4, 1.15,
        "Step（一步动作）\nplan / retrieve / read / synthesize", C_STEP, C_STEP, 10.5)
    box(ax, 5.0, 5.7, 3.4, 1.15,
        "Doc（说明书章节）\ndoc_id = setid  ← 主键\n+ LOINC 章节码", C_DOC, C_DOC, 10.5)
    box(ax, 9.4, 5.7, 3.0, 1.15,
        "Claim（回答里的\n一条建议）", C_CLAIM, C_CLAIM, 11)

    arrow(ax, (4.0, 6.28), (5.0, 6.28), C_STEP, "retrieved", off=0.34)
    arrow(ax, (9.4, 6.28), (8.4, 6.28), C_CLAIM, "cite", off=0.34)

    # 闭包
    ax.add_patch(FancyArrowPatch((10.9, 5.7), (2.3, 5.7),
                                 connectionstyle="arc3,rad=0.42",
                                 arrowstyle="-|>", mutation_scale=20,
                                 color=C_BAD, lw=2.6, linestyle="--", zorder=1))
    ax.text(6.6, 3.95, "★ 闭包不变量：回答里每一条引用，\n都必须能在账本里追到出处",
            ha="center", va="center", fontsize=12.5, color=C_BAD, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.55", fc="#ffebee", ec=C_BAD, lw=2))

    # 底下两行说明
    ax.text(0.6, 3.0, "两个防呆设计（让归因能机械判定）：", fontsize=12,
            fontweight="bold", color="#37474f")
    ax.text(0.9, 2.45,
            "① 引用带章节码  setid#LOINC  →  不只验证文档存不存在，还验证【引的是不是那一节】",
            fontsize=11, color="#37474f")
    ax.text(0.9, 1.95,
            "② 「检索到了但没进回答」必须填原因是枚举值（预算不够/判不相关/重复/权威性低）",
            fontsize=11, color="#37474f")
    ax.text(0.9, 1.45,
            "    →  否则「利用层」归因无从判定",
            fontsize=11, color="#37474f")
    ax.text(0.9, 0.85,
            "追不到出处  →  无法判断是检索层、利用层还是推理层  →  整个归因机制作废",
            fontsize=11.5, color=C_BAD, fontweight="bold")

    ax.set_title("图 E1-1｜证据账本：三个实体 + 一条闭包不变量",
                 fontsize=15, fontweight="bold", pad=14)

    out = FIGS / "fig_E1_ledger.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  ✅ {out}")


# ================================================================ E3
def fig_sources():
    rows = [
        ("NMPA 主站",           "中文", "× HTTP 412", False, "WAF 反爬"),
        ("NMPA 数据共享平台",    "中文", "× HTTP 412", False, "WAF 反爬"),
        ("flk 法律法规库",       "中文", "! SPA/HTML", None, "需逆向 API"),
        ("天池中药说明书",       "中文", "! 需登录",  None, "仅 1997 份"),
        ("openFDA",             "英文", "√ 可用",     True,  "有 API"),
        ("DailyMed（采用）",     "英文", "√ 可用",     True,  "API + 分页 + setid + LOINC"),
    ]

    fig, ax = plt.subplots(figsize=(12.2, 6.0))
    y = range(len(rows))
    colors = [C_OK if ok else (C_BAD if ok is False else "#f9a825") for _, _, _, ok, _ in rows]
    names = [r[0] for r in rows]
    bars = ax.barh(list(y), [1] * len(rows), color=colors, height=0.62)
    for i, (n, lang, status, ok, note) in enumerate(rows):
        ax.text(-0.02, i, n, ha="right", va="center", fontsize=12,
                fontweight="bold" if ok else "normal")
        ax.text(0.5, i, f"{status}   ·   {note}", ha="center", va="center",
                fontsize=11.5, color="white", fontweight="bold")
        ax.text(1.03, i, lang, ha="left", va="center", fontsize=11, color="#546e7a")

    ax.set_xlim(-0.30, 1.12)
    ax.set_ylim(-0.7, len(rows) + 2.3)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # 分词：上三行中文（采集不到），下三行英文（采集得到）
    ax.annotate("", xy=(1.16, 2.5), xytext=(1.16, -0.5),
                xycoords="data", textcoords="data",
                arrowprops=dict(arrowstyle="-|>", color="#f9a825", lw=2.4))
    ax.annotate("", xy=(1.16, 5.5), xytext=(1.16, 2.5),
                xycoords="data", textcoords="data",
                arrowprops=dict(arrowstyle="-|>", color=C_OK, lw=2.4))

    worst = ("  换源之后反而更强：\n"
             "    · 唯一 ID = setid（UUID），可机械核验\n"
             "    · 章节带 LOINC 编码 → 「该引哪一节」机械可判\n"
             "    · gold 证据集可半自动构造   ")
    ax.text(0.41, 7.0, worst, ha="center", va="center", fontsize=11.5,
            color="#1b5e20",
            bbox=dict(boxstyle="round,pad=0.6", fc="#e8f5e9", ec=C_OK, lw=2))

    ax.set_title("图 E3-1｜数据源可达性实测 —— 为什么从 NMPA 换成 DailyMed",
                 fontsize=15, fontweight="bold", pad=16)

    out = FIGS / "fig_E3_sources.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  ✅ {out}")


if __name__ == "__main__":
    print("生成 EvidenceTrace 可视化 ...")
    fig_ledger()
    fig_sources()
    print("完成。输出目录：", FIGS)
