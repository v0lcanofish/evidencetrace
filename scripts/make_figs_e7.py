# -*- coding: utf-8 -*-
"""
块 E7 可视化：agent 走出来的路。

跑法：D:/anaconda/python.exe scripts/make_figs_e7.py

━━━ 这张图要回答的问题 ━━━

    "agent 自己决定每一步做什么" —— 这句话到底成立不成立？

    上面板：同一批题（15 道），规则策略走出的路径。
            ⚠️ 结论是**难看**的：只有 2 种路径，15/15 都停在 answer。
               → 规则策略**本质还是一条带分支的流水线**，它不"看题"。
                 这**不是 E7 失败**，这是基线该有的样子，
                 也正是 E8 要治的东西。

    下面板：同一个循环，只换策略/预算/题目 → 行为立刻完全不同。
            → 证明**决定权真的在策略手里**，循环只是个骨架。
               没有这一面板，上面板的"路径单一"会被误读成"循环没起作用"。

    一句话：上面板说明**基线的病**，下面板说明**机制是真的**。
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
sys.path.insert(0, str(PROJECT))


from agent.loop import AgentLoop, LoopConfig, make_toolbox_factory       # noqa: E402
from agent.policy import RulePolicy                                      # noqa: E402
from agent.mocks import GroundedMockLLM                                  # noqa: E402
from agent.state import Action, A_SEARCH                                 # noqa: E402
from agent.tools import ScriptedUser                                     # noqa: E402
from retrieval import BM25Retriever                                      # noqa: E402

COLOR = {
    "locate": "#4C72B0",      # 蓝 —— 定位（买路钱）
    "retrieve": "#DD8452",    # 橙 —— 检索
    "search": "#DD8452",
    "ask": "#8172B3",         # 紫 —— 问用户
    "synthesize": "#55A868",  # 绿 —— 作答（收尾）
    "answer": "#55A868",
    "abstain": "#C44E52",     # 红 —— 拒答（收尾）
}
LABEL_CN = {
    "locate": "定位", "retrieve": "检索", "ask": "问用户",
    "synthesize": "作答", "abstain": "拒答",
}
OOD_Q = "What is the price of tea in China?"


class StuckPolicy:
    """故意写坏的策略：永远提同一个动作。用来验打转刹车是真的。"""
    name = "stuck"

    def __call__(self, state):
        return Action(A_SEARCH, {"query": "warfarin interaction"})


# ---------------------------------------------------------------- 跑数据

def run_one(retriever, question, policy=None, cfg=None, user=None):
    loop = AgentLoop(make_toolbox_factory(retriever, llm=GroundedMockLLM(), top_k=5),
                     policy or RulePolicy(), cfg or LoopConfig(budget=8))
    loop.run(question, run_id="fig", user=user)
    return [t.action for t in loop.last_state.tried], loop.last_stats


def collect(retriever, questions):
    rows = []
    for q in questions:
        acts, st = run_one(retriever, q["question"], user=ScriptedUser({"conditions": ["hypertension"]}))
        rows.append({"qid": q["qid"], "difficulty": q["difficulty"],
                     "question": q["question"], "actions": acts,
                     "terminated_by": st.terminated_by, "cost": st.cost_spent,
                     "n_evidence": st.n_evidence, "n_claims": st.n_claims,
                     "malformed": st.malformed})
    return rows


def collect_configs(retriever, q0):
    """同一批题/同一个循环，只换策略或预算 —— 行为立刻不同。"""
    outs = []

    acts, st = run_one(retriever, q0, RulePolicy(), LoopConfig(budget=8),
                       ScriptedUser({"conditions": ["hypertension"]}))
    outs.append(("规则策略 · 预算 8", acts, st.terminated_by, "正常收尾"))

    acts, st = run_one(retriever, q0, RulePolicy(), LoopConfig(budget=2))
    outs.append(("规则策略 · 预算 2", acts, st.terminated_by, "经济刹车：买不起下一个动作"))

    acts, st = run_one(retriever, q0, RulePolicy(), LoopConfig(budget=100, max_steps=2))
    outs.append(("规则策略 · 步数上限 2", acts, st.terminated_by, "硬上限：兜住策略失控"))

    acts, st = run_one(retriever, q0, StuckPolicy(), LoopConfig(budget=50, max_repeats=2))
    outs.append(("坏策略 · 永远同一个动作", acts, st.terminated_by, "打转刹车：不许原地踏步"))

    acts, st = run_one(retriever, OOD_Q, RulePolicy(), LoopConfig(budget=8))
    outs.append(("域外问题 · 无用户", acts, st.terminated_by, "该说不知道时就说不知道"))

    acts, st = run_one(retriever, OOD_Q, RulePolicy(), LoopConfig(budget=8),
                       ScriptedUser({"conditions": ["peptic ulcer"]}))
    outs.append(("域外问题 · 有用户", acts, st.terminated_by, "问了也补不上证据 → 仍拒答"))
    return outs


# ---------------------------------------------------------------- 画

def draw_strip(ax, labels, seqs, term_by, title, notes=None, row_h=1.0):
    """动作序列条带：每行一个 run，每格一个动作。

    notes: 每行右侧的"为什么"（跟终止原因一起挂，不另开说明框 —— 会打架）
    """
    n = len(labels)
    maxlen = max((len(s) for s in seqs), default=1)
    for i, (lab, seq, tb) in enumerate(zip(labels, seqs, term_by)):
        y = (n - i) * row_h
        for j, a in enumerate(seq):
            ax.add_patch(mp.Rectangle((j + 0.08, y - 0.30 * row_h), 0.84, 0.60 * row_h,
                                      color=COLOR.get(a, "#BBBBBB")))
            ax.text(j + 0.5, y, LABEL_CN.get(a, a[:2]), ha="center", va="center",
                    fontsize=6.4, color="white", fontweight="bold")
        tail = f"[{tb}]"
        if notes and notes[i]:
            tail += f"   {notes[i]}"
        ax.text(maxlen + 0.30, y, tail, ha="left", va="center",
                fontsize=7.6, color="#444444")
        ax.text(-0.30, y, lab, ha="right", va="center", fontsize=8.2)
    ax.set_xlim(-0.05, maxlen + 4.6)
    ax.set_ylim(0.15 * row_h, n * row_h + 0.85 * row_h)
    ax.set_yticks([])
    ax.set_xticks(range(maxlen))
    ax.set_xticklabels([str(i + 1) for i in range(maxlen)], fontsize=7)
    ax.set_xlabel("第几个动作", fontsize=8)
    ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left", pad=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def main() -> int:
    labels = json.loads((PROJECT / "data" / "labels.json").read_text(encoding="utf-8"))["labels"]
    questions = json.loads((PROJECT / "data" / "questions_v1.json").read_text(encoding="utf-8"))["questions"]
    r = BM25Retriever(labels, use_locator=True)

    rows = collect(r, questions)
    cfgs = collect_configs(r, questions[0]["question"])

    fig = plt.figure(figsize=(15.8, 10.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.02, 1.0], hspace=0.32,
                          left=0.155, right=0.985, top=0.905, bottom=0.115)

    # ---- 上面板：15 道题
    ax1 = fig.add_subplot(gs[0, 0])
    draw_strip(ax1,
               [f"{x['qid']}  {x['difficulty']}" for x in rows],
               [x["actions"] for x in rows],
               [x["terminated_by"] for x in rows],
               "① 同一批题（15 道）· 规则策略走出的路径   ——   "
               "只有 2 种路径、15/15 都停在 answer：它不看题，本质还是带分支的流水线")

    # ---- 下面板：换策略/预算 —— "为什么"直接挂在每行右边，不另开说明框（会打架）
    ax2 = fig.add_subplot(gs[1, 0])
    draw_strip(ax2, [c[0] for c in cfgs], [c[1] for c in cfgs], [c[2] for c in cfgs],
               "② 同一个循环，只换策略 / 预算 / 题目   ——   "
               "行为立刻不同：决定权真的在策略手里（循环只是骨架）",
               notes=[c[3] for c in cfgs])

    # ---- 图例（放在两面板之间的空白带，不压标题）
    handles = [mp.Patch(color=COLOR[k], label=f"{LABEL_CN[k]}（{k}）")
               for k in ("locate", "retrieve", "ask", "synthesize", "abstain")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, 0.012))

    term = Counter(x["terminated_by"] for x in rows)
    fig.suptitle(
        f"块 E7 · Agent Loop：动作由策略选，不是排好的队        "
        f"（15 题 · 平均 {sum(len(x['actions']) for x in rows)/len(rows):.1f} 个动作 · "
        f"平均 {sum(x['cost'] for x in rows)/len(rows):.1f} 个预算单位 · "
        f"终止分布 {dict(term)}）",
        fontsize=12.5, fontweight="bold", y=0.965)

    out = FIGS / "fig_E7_agent_loop.png"
    fig.savefig(out, dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- 存数据（台账要用）
    (PROJECT / "reports" / "agent_eval.json").write_text(json.dumps({
        "policy": "rule", "budget": 8, "n": len(rows),
        "n_distinct_paths": len({tuple(x["actions"]) for x in rows}),
        "terminated_by": dict(term),
        "avg_actions": sum(len(x["actions"]) for x in rows) / len(rows),
        "avg_cost": sum(x["cost"] for x in rows) / len(rows),
        "rows": rows,
        "configs": [{"name": c[0], "actions": c[1], "terminated_by": c[2], "why": c[3]}
                    for c in cfgs],
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"→ {out}")
    print(f"→ {PROJECT / 'reports' / 'agent_eval.json'}")
    print(f"\n15 题：路径种类 {len({tuple(x['actions']) for x in rows})} ｜ "
          f"终止分布 {dict(term)}｜平均动作 {sum(len(x['actions']) for x in rows)/len(rows):.1f}")
    print("配置对照：")
    for c in cfgs:
        print(f"   {c[0]:<24} 终止={c[2]:<11} {c[3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
