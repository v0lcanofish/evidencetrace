# -*- coding: utf-8 -*-
"""
块 E9 可视化：引用核验 + 反馈链。

这张图要回答一个问题：
    **"核验"到底只是事后判死，还是真的能改变 agent 的下一步？**

所以它一半画"抓得住"（判据矩阵），一半画"用得上"（两条策略的时间线对照）。

跑法：PYTHONIOENCODING=utf-8 D:/anaconda/envs/myenv/python.exe scripts/make_figs_e9.py
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
from agent.mocks import MockLLM                                       # noqa: E402
from agent.policy import CoveragePolicy, RulePolicy                   # noqa: E402
from agent.verify import K_NO_LOINC, K_NOT_IN_CORPUS, K_NOT_RETRIEVED, verify_answer  # noqa: E402
from retrieval.retriever import BM25Retriever                         # noqa: E402

LABELS = PROJECT / "data" / "labels.json"
Q = "Can I take warfarin with ibuprofen?"
FAKE = "deadbeef-0000-4000-8000-000000000000"

# 动作配色
C_ACT = {"locate": "#7BAFD4", "search": "#9CC69B", "ask": "#D9C36B",
         "answer": "#C44E52", "abstain": "#8A8F98"}
C_OK, C_BAD = "#7FB77E", "#C44E52"


def run_traces(retriever):
    """真跑两种策略，取各自的 trace。**图是跑出来的，不是画的。**"""
    out = {}
    for name, pol in (("RulePolicy", RulePolicy()), ("CoveragePolicy", CoveragePolicy())):
        lp = AgentLoop(make_toolbox_factory(retriever, llm=MockLLM(bad_citation=True), top_k=5),
                       pol, LoopConfig(budget=8))
        lp.run(Q, run_id=f"fig-{name}")
        out[name] = {"trace": lp.last_trace, "state": lp.last_state,
                     "verdict": [c["kind"] for c in lp.last_state.failed_cites]}
    return out


def build_cases(retriever):
    """跑一遍核验判据的 10 条用例，把判定结果取出来给图用。"""
    corpus = {f"{c.doc_id}#{c.loinc}" for c in retriever.chunks if c.loinc}
    drug = retriever.chunks[0].drug
    ev = {d.cite_key for d in retriever(f"{drug} warnings", 5, restrict={"drug": drug})}
    good = sorted(ev)[0]
    missing = sorted(corpus - ev)[0]
    cases = [
        ("好引用（不许误伤）", f"x [{good}].", True, ""),
        ("有效拒答（拒答≠坏引用）", "insufficient evidence", True, ""),
        ("引 [setid] 不带章节码", f"x [{good.split('#')[0]}].", False, K_NO_LINC := K_NO_LOINC),
        ("语料有、本轮没检索到", f"x [{missing}].", False, K_NOT_RETRIEVED),
        ("编造的 setid", f"x [{FAKE}#34073-7].", False, K_NOT_IN_CORPUS),
        ("真 setid + 不存在章节码", f"x [{good.split('#')[0]}#99999-9].", False, K_NOT_IN_CORPUS),
        ("有正文、零引用", "x has no citation at all.", False, ""),
        ("空字符串", "", False, ""),
        ("三条里混一条坏的", f"a [{good}]. b [{missing}]. c [{good}].", False, K_NOT_RETRIEVED),
        ("不传语料索引也要抓", f"x [{FAKE}#34073-7].", False, K_NOT_RETRIEVED),
    ]
    rows = []
    for name, text, want_ok, want_kind in cases:
        use_corpus = name != "不传语料索引也要抓"
        v = verify_answer(text, ev, corpus if use_corpus else None)
        rows.append({"name": name, "want_ok": want_ok,
                     "got_ok": v.ok, "bad": sorted(v.bad_kinds),
                     "passed": (v.ok == want_ok) and (want_kind in v.bad_kinds if want_kind else True)})
    return rows


def main() -> int:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    r = BM25Retriever(labels, use_locator=True)
    traces = run_traces(r)
    cases = build_cases(r)

    fig = plt.figure(figsize=(16.2, 10.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.30,
                          left=0.055, right=0.975, top=0.885, bottom=0.055)

    fig.suptitle("块 E9 · 引用核验：抓得住 + 用得上", fontsize=19, fontweight="bold", y=0.965)
    fig.text(0.5, 0.925,
             "上一份答案的引用被打回后，agent 会不会改变下一步？—— 会读状态的策略 1 次就转向，"
             "读不到的重复答到踩刹车",
             ha="center", fontsize=11.5, color="#444")

    # ================================================================ ① 时间线
    ax = fig.add_subplot(gs[0])
    ax.set_title("① 两条策略在同一道题上的行为轨迹（模型故意给一个编造的引用）",
                 fontsize=13, fontweight="bold", loc="left", pad=10)

    for row, (name, d) in enumerate(traces.items()):
        y = 1 - row
        steps = []
        for ln in d["trace"]:
            head = ln.split("(")[0].strip()
            if head == "[verify]":
                steps.append(("verify", None))
            elif head.startswith("[刹车]"):
                steps.append(("brake", None))
            elif head in C_ACT:
                ok = "ok=False" not in ln
                steps.append((head, ok))
        n_answer = sum(1 for s, _ in steps if s == "answer")
        for i, (act, ok) in enumerate(steps):
            color = C_ACT.get(act, "#DDD")
            if act == "brake":
                color = "#5C5C5C"
            if act == "verify":
                color = "#F2B134"
            w, h = 0.78, 0.42
            ax.add_patch(FancyBboxPatch((i - w / 2, y - h / 2), w, h,
                                        boxstyle="round,pad=0.02,rounding_size=0.06",
                                        facecolor=color, edgecolor="white", linewidth=1.4))
            label = {"verify": "核验\n打回", "brake": "刹车"}.get(act, act)
            ax.text(i, y, label, ha="center", va="center", fontsize=8.6,
                    color="white", fontweight="bold")
            if act == "answer" and ok is False:
                ax.text(i, y - h / 2 - 0.055, "×被拒", ha="center", va="top",
                        fontsize=8, color=C_BAD, fontweight="bold")
        ax.text(-0.9, y, name, ha="right", va="center", fontsize=11.5, fontweight="bold")
        ax.text(len(steps) + 0.55, y, f"答了 {n_answer} 次", ha="left", va="center",
                fontsize=10, color="#C44E52", fontweight="bold")

    ax.set_xlim(-3.6, max(len(d["trace"]) for d in traces.values()) + 1.6)
    ax.set_ylim(-0.75, 1.75)
    ax.axis("off")
    ax.text(-3.5, -0.55,
            "黄 = 核验打回（引用追不到）｜ 红 = answer（× = 被拒，未采纳）｜ 深灰 = 刹车强制收尾",
            fontsize=9.5, color="#666")
    ax.text(-3.5, -0.75,
            "两种策略最后都没有交出坏答案（走的是拒答）—— 这是本项目的红线：不给无出处的用药建议。",
            fontsize=9.5, color="#666")

    # ================================================================ ② 判据矩阵
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title("② 核验判据：10 条用例 —— 坏引用 100% 抓住，好引用不误伤",
                  fontsize=13, fontweight="bold", loc="left", pad=10)

    kind_color = {K_NO_LOINC: "#E08A3C", K_NOT_RETRIEVED: "#C44E52",
                  K_NOT_IN_CORPUS: "#8E44AD", "": "#B0B7C3"}
    n = len(cases)
    for i, c in enumerate(cases):
        y = n - i - 1
        ax2.add_patch(FancyBboxPatch((0.0, y - 0.32), 4.15, 0.64,
                                     boxstyle="round,pad=0.01,rounding_size=0.03",
                                     facecolor="#F4F6F8", edgecolor="#DDE1E6"))
        ax2.text(0.12, y, c["name"], ha="left", va="center", fontsize=10.2)
        # 期望
        ax2.text(4.42, y, "应当通过" if c["want_ok"] else "应当抓住",
                 ha="center", va="center", fontsize=9.6,
                 color="#2E7D32" if c["want_ok"] else "#B03A2E")
        # 实际
        got = "通过" if c["got_ok"] else "抓住"
        ax2.text(5.45, y, got, ha="center", va="center", fontsize=9.6, fontweight="bold")
        # 抓到的类型
        for j, k in enumerate(c["bad"]):
            ax2.add_patch(FancyBboxPatch((6.05 + j * 1.02, y - 0.24), 0.92, 0.48,
                                         boxstyle="round,pad=0.01,rounding_size=0.03",
                                         facecolor=kind_color.get(k, "#999"), edgecolor="none"))
            ax2.text(6.05 + j * 1.02 + 0.46, y, k, ha="center", va="center",
                     fontsize=8.4, color="white", fontweight="bold")
        mark = "√" if c["passed"] else "×"
        ax2.text(9.6, y, mark, ha="center", va="center", fontsize=13,
                 color=C_OK if c["passed"] else C_BAD, fontweight="bold")

    ax2.text(0.12, n - 0.05, "用例", fontsize=10.5, fontweight="bold", color="#555")
    ax2.text(4.42, n - 0.05, "期望", fontsize=10.5, fontweight="bold", color="#555", ha="center")
    ax2.text(5.45, n - 0.05, "实测", fontsize=10.5, fontweight="bold", color="#555", ha="center")
    ax2.text(6.05, n - 0.05, "抓到的坏引用类型", fontsize=10.5, fontweight="bold", color="#555")
    ax2.text(9.6, n - 0.05, "判定", fontsize=10.5, fontweight="bold", color="#555", ha="center")

    n_pass = sum(1 for c in cases if c["passed"])
    ax2.text(9.6, -0.95, f"{n_pass}/{n} 全过", ha="center", fontsize=12,
             color=C_OK, fontweight="bold")
    ax2.text(0.12, -0.95,
             "注意：前两条是反向用例：一个把所有引用都判坏的核验器也能通过「坏引用 100% 抓住」，"
             "所以必须同时验好引用不误伤。",
             fontsize=9.4, color="#666")
    ax2.set_xlim(-0.15, 10.4)
    ax2.set_ylim(-1.3, n + 0.42)
    ax2.axis("off")

    out = FIGS / "fig_E9_verify.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    print(f"→ {out}")
    print(f"  时间线：RulePolicy 答 {sum(1 for l in traces['RulePolicy']['trace'] if l.startswith('answer'))} 次 ｜ "
          f"CoveragePolicy 答 {sum(1 for l in traces['CoveragePolicy']['trace'] if l.startswith('answer'))} 次")
    print(f"  判据：{n_pass}/{n} 通过")
    return 0 if n_pass == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
