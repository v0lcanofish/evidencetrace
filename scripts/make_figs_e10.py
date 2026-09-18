# -*- coding: utf-8 -*-
"""
块 E10 可视化：多轮会话 —— 指代消解 + 跨轮复用。

这张图要回答两个问题：
    ① **指代消解**：第 2、3 轮的问题里没有药名，agent 知道在说哪个药吗？
    ② **跨轮复用**：第 1 轮查过的，第 2 轮还查吗？（省下的动作就是记忆的价值）

⚠️ 图是**跑出来的**，不是画的 —— 每根柱子来自一次真跑的 Session。

跑法：PYTHONIOENCODING=utf-8 D:/anaconda/envs/myenv/python.exe scripts/make_figs_e10.py
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
from agent.session import Session                                     # noqa: E402
from retrieval.retriever import BM25Retriever                         # noqa: E402

C_EXPLICIT, C_CONTEXT = "#7BAFD4", "#C44E52"
C_BAR, C_CTX = "#B0B7C3", "#C44E52"


def _mk_loop(r):
    return AgentLoop(make_toolbox_factory(r, llm=GroundedMockLLM(), top_k=5),
                     CoveragePolicy(), LoopConfig(budget=8))


def main() -> int:
    labels = json.loads((PROJECT / "data" / "labels.json").read_text(encoding="utf-8"))["labels"]
    r = BM25Retriever(labels, use_locator=True)
    known = sorted({c.drug for c in r.chunks if c.drug})
    A, B = known[0], known[1]

    # ---- 跑①：三轮真实追问
    lp = _mk_loop(r)
    s = Session(known_drugs=known)
    convo = [f"What does {A} interact with?",
             f"What about taking it with {B}?",
             "Is it contraindicated?"]
    for q in convo:
        s.ask(q, lp)
    turns = s.turns

    # ---- 跑②：复用对照（同一题，单轮 vs 带上下文）
    q1 = convo[0]
    lp_solo = _mk_loop(r)
    s_solo = Session(known_drugs=known)
    s_solo.ask(q1, lp_solo)
    solo_actions = s_solo.turns[0].n_actions
    solo_ev = len(s_solo.evidence)

    lp_ctx = _mk_loop(r)
    s_ctx = Session(known_drugs=known)
    s_ctx.ask(q1, lp_ctx)
    rec2 = s_ctx.ask(q1, lp_ctx)
    ctx_actions, ctx_new_ev = rec2.n_actions, rec2.n_new_evidence

    # ================================================================
    fig = plt.figure(figsize=(16.0, 9.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.55, 1.0],
                          hspace=0.36, wspace=0.24,
                          left=0.055, right=0.975, top=0.865, bottom=0.075)

    fig.suptitle("块 E10 · 多轮会话：指代消解 + 跨轮复用", fontsize=19,
                 fontweight="bold", y=0.955)
    fig.text(0.5, 0.912,
             "第 2、3 轮的问题里**一个药名都没有** —— agent 靠会话状态知道在说哪个药；"
             "第 1 轮查过的，第 2 轮不再重查".replace("**", ""),
             ha="center", fontsize=11.5, color="#444")

    # ================================================================ ① 三轮对话
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("① 三轮真实追问：每轮解析出哪个药、走了几个动作",
                  fontsize=13, fontweight="bold", loc="left", pad=10)

    for i, t in enumerate(turns):
        y = len(turns) - 1 - i
        res = t.resolution
        color = C_EXPLICIT if res.mode == "explicit" else C_CONTEXT

        ax1.add_patch(FancyBboxPatch((0.0, y - 0.40), 3.55, 0.80,
                                     boxstyle="round,pad=0.02,rounding_size=0.06",
                                     facecolor="#F4F6F8", edgecolor="#DDE1E6"))
        ax1.text(0.14, y + 0.13, f"轮{res.__dict__.get('turn', i+1)}  「{res.raw[:44]}」",
                 ha="left", va="center", fontsize=9.8, color="#333")
        ax1.text(0.14, y - 0.20,
                 f"解析模式 = {res.mode}"
                 + (f"（线索词: {res.cue}）" if res.cue else ""),
                 ha="left", va="center", fontsize=9.0, color=color, fontweight="bold")

        # 箭头
        ax1.annotate("", xy=(4.02, y), xytext=(3.60, y),
                     arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.6))

        # 解析出的药
        drugs = res.drugs or ["（不猜）"]
        for j, d in enumerate(drugs):
            ax1.add_patch(FancyBboxPatch((4.15 + j * 1.95, y - 0.27), 1.80, 0.54,
                                         boxstyle="round,pad=0.02,rounding_size=0.06",
                                         facecolor=color, edgecolor="none"))
            ax1.text(4.15 + j * 1.95 + 0.90, y, d[:17], ha="center", va="center",
                     fontsize=9.2, color="white", fontweight="bold")

        ax1.text(8.55, y + 0.13, f"动作 {t.n_actions} ｜ 搜索 {t.n_search}",
                 ha="left", va="center", fontsize=9.8, color="#333")
        ax1.text(8.55, y - 0.20,
                 f"证据累计 {t.n_evidence}（本轮新增 {t.n_new_evidence}）",
                 ha="left", va="center", fontsize=9.4, color="#666")

    ax1.set_xlim(-0.1, 12.4)
    ax1.set_ylim(-0.62, len(turns) - 0.30)
    ax1.axis("off")
    ax1.text(0.0, -0.58,
             "蓝 = 本轮点名了药（explicit）｜ 红 = 靠会话上下文解析（from_context）"
             "　·　轮2 解析出**两个**药：它问的是「A 和 B 一起」，"
             "只取新点名的那个会把 A 丢掉".replace("**", ""),
             fontsize=9.4, color="#666")

    # ================================================================ ② 复用
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("② 跨轮复用：同样的题，第 2 轮少走动作",
                  fontsize=12.5, fontweight="bold", loc="left", pad=10)
    x = np.arange(2)
    acts = [solo_actions, ctx_actions]
    bars = ax2.bar(x, acts, 0.5, color=[C_BAR, C_CTX])
    for b, v in zip(bars, acts):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.06, str(v),
                 ha="center", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["单轮\n（没有上下文）", "多轮第 2 轮\n（带着上一轮证据）"], fontsize=10)
    ax2.set_ylabel("动作数", fontsize=10.5)
    ax2.set_ylim(0, max(acts) * 1.35)
    ax2.grid(axis="y", alpha=0.25)
    ax2.set_axisbelow(True)
    ax2.text(0.5, max(acts) * 1.15, f"省下 {solo_actions - ctx_actions} 个动作",
             ha="center", fontsize=11, color=C_CTX, fontweight="bold")

    # ================================================================ ③ 证据累积
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title("③ 证据跨轮累积（不是每轮清零）", fontsize=12.5,
                  fontweight="bold", loc="left", pad=10)
    xs = np.arange(1, len(turns) + 1)
    cum = [t.n_evidence for t in turns]
    new = [t.n_new_evidence for t in turns]
    ax3.bar(xs, new, 0.45, label="本轮新增", color="#9CC69B")
    ax3.plot(xs, cum, "-o", color=C_CTX, lw=2.4, ms=8, label="累计证据")
    for a, b in zip(xs, cum):
        ax3.text(a, b + 0.30, str(b), ha="center", fontsize=10.5,
                 color=C_CTX, fontweight="bold")
    ax3.set_xticks(xs)
    ax3.set_xticklabels([f"轮{i}" for i in xs], fontsize=10)
    ax3.set_ylabel("证据条数", fontsize=10.5)
    ax3.set_ylim(0, max(cum) * 1.45)
    ax3.legend(fontsize=9.5, loc="upper left")
    ax3.grid(axis="y", alpha=0.25)
    ax3.set_axisbelow(True)
    ax3.text(0.5, -0.30,
             f"复用对照：单轮拿到 {solo_ev} 条 ｜ 多轮第 2 轮新增 {ctx_new_ev} 条",
             transform=ax3.transAxes, ha="center", fontsize=9.5, color="#666")

    out = FIGS / "fig_E10_session.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    print(f"→ {out}")
    print(f"  三轮解析：{[t.resolution.drugs for t in turns]}")
    print(f"  复用：单轮 {solo_actions} 动作（{solo_ev} 条证据）→ "
          f"多轮第2轮 {ctx_actions} 动作（新增 {ctx_new_ev} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
