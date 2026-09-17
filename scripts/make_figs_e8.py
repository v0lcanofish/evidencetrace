# -*- coding: utf-8 -*-
"""
块 E8 可视化：把 0/1 判断换成算出来的覆盖度，好了多少。

跑法：D:/anaconda/python.exe scripts/make_figs_e8.py
（先跑 scripts/eval_coverage.py 生成 reports/coverage_eval.json）
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIGS = PROJECT / "reports" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
DATA = PROJECT / "reports" / "coverage_eval.json"

C_RULE, C_COV = "#B0B7C3", "#C44E52"


def main() -> int:
    d = json.loads(DATA.read_text(encoding="utf-8"))
    res = d["results"]
    names = list(res)
    rule, cov = res[names[0]], res[names[1]]

    fig = plt.figure(figsize=(15.5, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.22,
                          left=0.07, right=0.97, top=0.86, bottom=0.08)

    # ---------- ① 两策略对照
    ax1 = fig.add_subplot(gs[0, :])
    metrics = [("答对率\n（该答的答对）", "accuracy", True),
               ("拒答准确率\n（该拒的拒了）", "refuse_accuracy", True),
               ("硬答率\n（该拒的却答了）", "hard_answer_rate", False),
               ("误拒率\n（该答的却拒了）", "wrong_abstain_rate", False)]
    x = np.arange(len(metrics))
    w = 0.36
    v1 = [rule[k] for _, k, _ in metrics]
    v2 = [cov[k] for _, k, _ in metrics]
    b1 = ax1.bar(x - w / 2, v1, w, label="① 规则策略（0/1 判断）", color=C_RULE)
    b2 = ax1.bar(x + w / 2, v2, w, label="② 覆盖度策略（算出来的）", color=C_COV)
    for bars in (b1, b2):
        for b in bars:
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                     f"{b.get_height():.3f}", ha="center", fontsize=9.5, fontweight="bold")
    # ⚠️ 不画"从 A 到 B"的箭头：柱子高度已经把差距说清楚了，
    #    再加一条斜线只会变成视觉噪声（第一版画了，看着像根杂线）。
    ax1.set_xticks(x)
    ax1.set_xticklabels([m[0] for m in metrics], fontsize=10)
    ax1.set_ylim(0, 1.18)
    ax1.set_ylabel("比例", fontsize=10)
    ax1.set_title("① 同一批题（检索集 60 条 + 边界集 45 条）· 同一个循环 · 同一个生成器 —— **只换策略**",
                  fontsize=11.5, fontweight="bold", loc="left")
    ax1.legend(fontsize=10, frameon=False, loc="upper left")
    ax1.grid(axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # ---------- ② θ 标定热力图
    ax2 = fig.add_subplot(gs[1, 0])
    highs = sorted({g["theta_high"] for g in d["grid"]})
    lows = sorted({g["theta_low"] for g in d["grid"]})
    M = np.full((len(lows), len(highs)), np.nan)
    for g in d["grid"]:
        M[lows.index(g["theta_low"]), highs.index(g["theta_high"])] = g["score"]
    im = ax2.imshow(M, cmap="YlOrRd", aspect="auto", origin="lower")
    for i in range(len(lows)):
        for j in range(len(highs)):
            if not np.isnan(M[i, j]):
                ax2.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if M[i, j] > np.nanmax(M) * 0.75 else "#333")
    ax2.set_xticks(range(len(highs)))
    ax2.set_xticklabels([f"{h:g}" for h in highs], fontsize=9)
    ax2.set_yticks(range(len(lows)))
    ax2.set_yticklabels([f"{l:g}" for l in lows], fontsize=9)
    ax2.set_xlabel("θ_high（≥ 它才作答）", fontsize=10)
    ax2.set_ylabel("θ_low（< 它才拒答）", fontsize=10)
    ch = d["chosen"]
    ai, aj = lows.index(ch["theta_low"]), highs.index(ch["theta_high"])
    ax2.add_patch(plt.Rectangle((aj - 0.5, ai - 0.5), 1, 1, fill=False,
                                edgecolor="#111", lw=3))
    ax2.set_title(f"② θ 标定：扫 21 组，选 {ch['theta_high']:g} / {ch['theta_low']:g}"
                  f"（★ 最优在网格**内部**，不是边界）",
                  fontsize=10.5, fontweight="bold", loc="left")
    fig.colorbar(im, ax=ax2, fraction=0.045, label="综合分")

    # ---------- ③ 拒答率按边界类别拆开
    ax3 = fig.add_subplot(gs[1, 1])
    classes = sorted(cov["boundary_by_class"])
    y = np.arange(len(classes))
    r_rate = [rule["boundary_by_class"].get(c, {}).get("refused", 0)
              / max(1, rule["boundary_by_class"].get(c, {}).get("n", 1)) for c in classes]
    c_rate = [cov["boundary_by_class"][c]["refused"]
              / max(1, cov["boundary_by_class"][c]["n"]) for c in classes]
    ax3.barh(y - 0.19, r_rate, 0.36, color=C_RULE, label="① 规则")
    ax3.barh(y + 0.19, c_rate, 0.36, color=C_COV, label="② 覆盖度")
    for i, (a, b) in enumerate(zip(r_rate, c_rate)):
        ax3.text(a + 0.01, i - 0.19, f"{a:.2f}", va="center", fontsize=8.5)
        ax3.text(b + 0.01, i + 0.19, f"{b:.2f}", va="center", fontsize=8.5, fontweight="bold")
    lab = {"out_of_scope": "越界（不是药品问题）", "beyond_label": "超出说明书（要医疗建议）",
           "unknown_drug": "库里没这份药", "missing_section": "该章节没采"}
    ax3.set_yticks(y)
    ax3.set_yticklabels([lab.get(c, c) for c in classes], fontsize=9.5)
    ax3.set_xlim(0, 1.22)
    ax3.set_xlabel("拒答率（该拒的真的拒了）", fontsize=10)
    ax3.set_title("③ 拒答率按类别拆开 —— 看剩下的漏在哪一类",
                  fontsize=10.5, fontweight="bold", loc="left")
    ax3.legend(fontsize=9.5, frameon=False, loc="lower right")
    ax3.grid(axis="x", alpha=0.25)
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)

    fig.suptitle(
        "块 E8 · 把「够不够」从 0/1 换成算出来的覆盖度："
        f"硬答率 {rule['hard_answer_rate']:.3f} → {cov['hard_answer_rate']:.3f}"
        f"（−{rule['hard_answer_rate'] - cov['hard_answer_rate']:.3f}）· "
        f"答对率 {rule['accuracy']:.3f} → {cov['accuracy']:.3f}（持平）· "
        f"动作数 {rule['avg_actions']:.2f} → {cov['avg_actions']:.2f}（更省）",
        fontsize=12.5, fontweight="bold", y=0.965)

    out = FIGS / "fig_E8_coverage.png"
    fig.savefig(out, dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
