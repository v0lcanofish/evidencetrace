# -*- coding: utf-8 -*-
"""
块 E6 可视化：检索层补完 —— 修掉了什么、补上了什么、**以及尺子在哪里到顶**。

跑法：python scripts/make_figs_e6.py
（先跑 scripts/eval_retrieval.py 生成 reports/retrieval_eval.json）
"""


import sys

# ⚠️ Windows 中文控制台默认 GBK：不设这个，print("⭐") 会抛 UnicodeEncodeError
#    → **判据崩在半路，红绿一个字都读不到**（2026-09-18 实测 eval_retrieval.py）。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
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
DATA = PROJECT / "reports" / "retrieval_eval.json"

# 由浅到深：① 最弱 → ⑤ 完整；⑥ 是旁路对照（稠密单独）
C = ["#C9CED8", "#9AA4B5", "#6C7A93", "#C44E52", "#5B8C5A", "#4A78B5"]

# ⚠️⚠️ 标签**不写死**，按数字前缀从 JSON 反推（2026-09-18 修）：
#    上一版 SHORT 写死 5 个，而 eval_retrieval.py 后来加了第 ⑥ 层（稠密单独）
#    → `make_figs_e6.py` 当场 ValueError: shape mismatch (5,) vs (6,)。
#    **图脚本和判据脚本是两份会各自漂移的清单** —— 判据加了层，图不知道。
#    所以这里：不认识的前缀直接报错，宁可炸掉也不画一张少一层的图。
LAYER_LABELS = {
    "①": "① BM25\n单独",
    "②": "② +定位\n(旧:乘性boost)",
    "③": "③ 三路融合\n(无稠密)",
    # ⚠️ 图里不要放 emoji：Microsoft YaHei 没有 ⭐/⚠️ 的字形，
    #    实测渲染成豆腐块（savefig 只会 warning，不会报错）→ 图看起来"就是那样"。
    #    控制台打印照样用 ⭐，图里用纯文字。
    "④": "④ +稠密\n(完整)",
    "⑤": "⑤ +重排\n(带硬约束)",
    "⑥": "⑥ 稠密\n单独",
}


def _short(key: str) -> str:
    if key[:1] not in LAYER_LABELS:
        raise SystemExit(
            f"retrieval_eval.json 的 layers 里出现了没登记的前缀：{key!r}\n"
            f"  → 判据脚本加了新层，图脚本的 LAYER_LABELS 没跟着加。"
            f"补上标签再加图（别让图和判据对不上）。")
    return LAYER_LABELS[key[:1]]


def _bars(ax, names, vals, title, note=None, ylim=(0, 1.2)):
    x = np.arange(len(names))
    bars = ax.bar(x, vals, 0.62, color=C[:len(names)])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(*ylim)
    ax.set_ylabel("Recall@5", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.grid(axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if note:
        ax.text(0.5, 0.93, note, transform=ax.transAxes, ha="center",
                fontsize=9, color="#C44E52", fontweight="bold")


def main() -> int:
    d = json.loads(DATA.read_text(encoding="utf-8"))
    layers = d["layers"]
    keys = list(layers)
    SHORT = [_short(k) for k in keys]          # ← 由 JSON 反推，不再写死
    ret_v = [layers[k]["recall@5"] for k in keys]

    probe = d.get("probe", {})
    ptest = {k.split("|")[1]: v for k, v in probe.items() if k.startswith("探针 test")}
    ptest_v = [ptest.get(k, {}).get("recall@5", 0) for k in keys]

    # 三个关键层的下标也按**名字**查，不写死数字 —— 层数一变，写死的下标就会指错柱子
    i_dense = next((i for i, k in enumerate(keys) if "+稠密" in k), None)
    i_nodense = next((i for i, k in enumerate(keys) if "无稠密" in k), None)
    sat = d.get("saturated_on_retrieval_set", [])
    sat_txt = "".join(k[:1] for k in sat)      # 例如 [③④⑤] → "③④⑤"

    fig = plt.figure(figsize=(16.2, 10.0))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.20,
                          left=0.06, right=0.975, top=0.855, bottom=0.07)

    # ---------- ① 检索集：到顶了
    ax1 = fig.add_subplot(gs[0, 0])
    _bars(ax1, SHORT, ret_v,
          "① 检索集 60 —— 定位 60/60 全对，尺子到顶",
          note=f"注意：{sat_txt} 到 1.000 不是系统到顶，是这把尺子到顶了")

    # ---------- ② 探针集：真正能量出东西的地方
    ax2 = fig.add_subplot(gs[0, 1])
    _bars(ax2, SHORT, ptest_v,
          "② 探针 test 60（药名在、意图词不认识 → 结构先验失效）")
    contrib = d.get("probe_dense_contribution_test")
    if contrib is not None and i_dense is not None and i_nodense is not None:
        ax2.annotate("", xy=(i_dense, ptest_v[i_dense]),
                     xytext=(i_nodense, ptest_v[i_nodense]),
                     arrowprops=dict(arrowstyle="->", color="#C44E52", lw=2.4))
        ax2.text(i_dense + 0.42, (ptest_v[i_nodense] + ptest_v[i_dense]) / 2,
                 f"稠密的贡献\n{contrib:+.3f}", fontsize=10.5,
                 color="#C44E52", fontweight="bold", va="center")

    # ---------- ③ 互补性：谁强取决于查询
    ax3 = fig.add_subplot(gs[1, 0])
    comp = d.get("complementarity", {})
    names = list(comp)
    lab = [n.replace("检索集·", "检索集\n").replace("（", "\n（") for n in names]
    x = np.arange(len(names))
    w = 0.36
    b1 = ax3.bar(x - w / 2, [comp[n]["bm25"] for n in names], w,
                 label="BM25（词面）", color="#6C7A93")
    b2 = ax3.bar(x + w / 2, [comp[n]["dense"] for n in names], w,
                 label="稠密（语义）", color="#C44E52")
    for bars in (b1, b2):
        for b in bars:
            ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                     f"{b.get_height():.3f}", ha="center", fontsize=9.5, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{l}\n(n={comp[n]['n']})" for l, n in zip(lab, names)], fontsize=9)
    ax3.set_ylim(0, 1.2)
    ax3.set_ylabel("Recall@5", fontsize=10)
    ax3.set_title("③ 同两个检索器，谁强取决于查询里有没有判别性实词 → 融合权重必须算出来",
                  fontsize=11, fontweight="bold", loc="left")
    ax3.legend(fontsize=9.5, frameon=False, loc="upper center", ncol=2)
    ax3.grid(axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)

    # ---------- ④ 域外弃权：余弦门限
    ax4 = fig.add_subplot(gs[1, 1])
    cos = d.get("cosine_by_class", {})
    tau = d.get("dense_min_cosine", 0.60)
    order = ["out_of_scope", "unknown_drug", "beyond_label", "missing_section",
             "in_domain_probe"]
    zh = {"out_of_scope": "域外\n(不是药品问题)", "unknown_drug": "库里\n没这份药",
          "beyond_label": "超出说明书\n(要医疗建议)", "missing_section": "该章节\n没采",
          "in_domain_probe": "域内\n(探针 120)"}
    y = np.arange(len(order))
    for i, k in enumerate(order):
        v = cos.get(k)
        if not v:
            continue
        col = "#C44E52" if k == "out_of_scope" else "#6C7A93"
        ax4.plot([v["min"], v["max"]], [i, i], color=col, lw=9, alpha=0.32,
                 solid_capstyle="round")
        ax4.plot([v["mean"]], [i], "o", color=col, ms=9)
        ax4.text(v["max"] + 0.006, i, f"{v['mean']:.3f}", va="center",
                 fontsize=9, fontweight="bold", color=col)
    ax4.axvline(tau, color="#C44E52", ls="--", lw=2)
    ax4.text(tau - 0.004, len(order) - 0.35, f"门限 τ={tau:g}", rotation=90,
             ha="right", va="top", fontsize=9.5, color="#C44E52", fontweight="bold")
    ax4.set_yticks(y)
    ax4.set_yticklabels([zh[k] for k in order], fontsize=9)
    ax4.set_xlabel("稠密检索的 top-1 余弦", fontsize=10)
    ax4.set_xlim(0.40, 0.80)
    ax4.set_title("④ 稠密层的弃权门限 —— 稠密永远返回 top-k，没有它 agent 又看不到「什么都没查到」",
                  fontsize=11, fontweight="bold", loc="left")
    ax4.grid(axis="x", alpha=0.25)
    for s in ("top", "right"):
        ax4.spines[s].set_visible(False)

    fig.suptitle(
        "块 E6 · 检索层补完：修掉「强先验挂在弱信号上」那个洞，补上稠密与自适应融合"
        f"（探针 test 上稠密 {contrib:+.3f}）"
        if contrib is not None else "块 E6 · 检索层补完",
        fontsize=13, fontweight="bold", y=0.955)

    out = FIGS / "fig_E6_retrieval.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
