# -*- coding: utf-8 -*-
"""
块 E12 前置的判据：**决策日志里到底有没有东西可学**。

    PYTHONIOENCODING=utf-8 python scripts/eval_decision_signal.py
    PYTHONIOENCODING=utf-8 python scripts/eval_decision_signal.py --limit 12

━━━ 为什么这个脚本必须先跑 ━━━

E12 是「用 ①② 的决策日志 **SFT 小模型**」。开训之前必须先答一个问题：

    **这些决策点里，有多少是「选什么动作都一样」的？**

没有梯度的决策点，模仿它只是**把当时那套规则的巧合记下来**，
换一批题就失效 —— 训出来的东西在报告里看着像"策略③"，实际是噪声。
（ToolHorizon 已经栽过同形的坑：**40% 的题的 reward 与策略无关**。）

━━━ 判据三铁律怎么落的 ━━━

    可执行      —— 本文件，断言 + 四张表
    阈值不能拍  —— 「有梯度占比」是**量出来的**；对照基线是跑出来的
    区分「做到了」
    和「看起来做到了」—— ⭐ 两条：
        · 判据 ②：**majority baseline**。动作分布一偏斜，
          "模仿准确率 85%" 完全可能只是"全猜最常见的那个动作"。
          不报 baseline 的模仿准确率是骗人的。
        · 判据 ③：**用反事实量信号，不用模仿准确率量**。
          准确率量的是"像不像旧策略"，反事实量的是"这一步重不重要"。**后者才是能学的。**

━━━ ⚠️ 这个脚本不训练任何东西 ━━━
    它只回答"该不该训"。结论是"不该"的话，省下的是 GPU 时。
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT))

from agent.coverage import corpus_drugs_from                     # noqa: E402
from agent.decisions import (DecisionPoint, ForcedPolicy,         # noqa: E402
                             LoggingPolicy, available_actions)
from agent.loop import AgentLoop, LoopConfig, make_toolbox_factory  # noqa: E402
from agent.mocks import GroundedMockLLM                            # noqa: E402
from agent.policy import CoveragePolicy, RulePolicy                # noqa: E402
from agent.report import looks_like_abstention                     # noqa: E402
from retrieval import make_retriever                               # noqa: E402

LABELS = _PROJECT / "data" / "labels.json"
R_SET = _PROJECT / "data" / "eval" / "retrieval_set.json"
OUT = _PROJECT / "data" / "decisions.jsonl"
N = 60

FAILS: list = []
# 动作在训练集里的分布偏斜到这个程度就算"没有可学信号"
MAJORITY_ALARM = 0.70


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"   {'✓' if ok else '✗'} {name}" + (f"　{detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------- 跑一轮


def _loop(r, policy, qid, question):
    lp = AgentLoop(make_toolbox_factory(r, llm=GroundedMockLLM(), top_k=5),
                   policy, LoopConfig(budget=8))
    lg = lp.run(question, run_id=qid)
    return lp, lg


def baseline(r, make_policy, qid, question):
    """跑一条基线轨迹，返回 (决策点列表, 答对了没, 是不是拒答)。"""
    sink: list = []
    policy = LoggingPolicy(make_policy(), sink, qid=qid)
    lp, lg = _loop(r, policy, qid, question)
    gold = {r_gold[qid]}
    ok = gold.issubset(lg.cited_keys())
    abst = looks_like_abstention(lp.last_state.answer_text or "")
    for p in sink:
        p.outcome, p.abstained = ok, abst
    return sink, ok, abst


def counterfactual(r, make_policy, qid, question, i, action):
    """在第 i 个决策点强制换动作，返回结局（答对了没）。"""
    pol = ForcedPolicy(make_policy(), force_at=i, force_action=action,
                       question=question)
    _, lg = _loop(r, pol, qid, question)
    return {r_gold[qid]}.issubset(lg.cited_keys())


r_gold: dict = {}


# ---------------------------------------------------------------- 主流程


def main() -> int:
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    r = make_retriever(labels, use_locator=True)
    rows = json.loads(R_SET.read_text(encoding="utf-8"))["rows"][:N]
    if limit:
        rows = rows[:limit]
    for row in rows:
        r_gold[row["qid"]] = row["gold_cite_key"]

    makers = {
        "rule": lambda: RulePolicy(),
        "coverage": lambda: CoveragePolicy(),
    }

    all_points: list = []
    print(f"① 采集决策点（{len(rows)} 题 × 2 个策略，生成器=GroundedMockLLM）")
    print("   " + "-" * 80)
    for pname, mk in makers.items():
        n_dp, n_correct = 0, 0
        pts_this = []
        for row in rows:
            pts, ok, _ = baseline(r, mk, row["qid"], row["question"])
            pts_this.extend(pts)
            n_dp += len(pts)
            n_correct += int(ok)
        all_points.extend(pts_this)
        per_q = n_dp / max(1, len(rows))
        print(f"   {pname:<10} 决策点 {n_dp:>4}（{per_q:.1f}/题）  答对 {n_correct}/{len(rows)}")

    # ---- ② 动作分布 + majority baseline
    print("\n② 动作分布（⭐ 不报这一张，「模仿准确率」就是骗人的）")
    print("   " + "-" * 80)
    for pname in makers:
        acts = [p.action for p in all_points if p.policy == pname]
        c = Counter(acts)
        top, n_top = c.most_common(1)[0]
        share = n_top / max(1, len(acts))
        bar = "  ".join(f"{k}:{v}" for k, v in c.most_common())
        print(f"   {pname:<10} n={len(acts):>4}  majority={top}({share:.2f})  ｜ {bar}")
        check(f"{pname} 动作分布未偏斜到「全猜一个」(<{MAJORITY_ALARM:.0%})",
              share < MAJORITY_ALARM,
              f"最常见动作占 {share:.2f} → 模仿准确率必须与 {share:.2f} 比，不是与 0 比")

    # ---- ③ 反事实：这一步重不重要
    #
    # ⚠️ 为什么要**按基线对错拆开算**：
    #    基线本来就答错的题，换什么动作很可能还是错 —— 混在一起算会把"没梯度"高估。
    #    拆开之后两边问的是**两个不同的问题**：
    #      答对的题：哪一步才是关键步？（有梯度 = 关键步）
    #      答错的题：换一步能救回来吗？（有梯度 = **可救**）
    print("\n③ 反事实：在每个决策点强制换成别的动作，看结局变不变")
    print("   " + "-" * 80)
    n_cf = 0
    by_action = defaultdict(lambda: [0, 0])          # 动作 → [有更优, 总]
    stat = {"right": {"better": 0, "worse": 0, "n": 0},
            "wrong": {"better": 0, "worse": 0, "n": 0}}
    per_q = defaultdict(list)                        # (策略,qid) → [(结局, 有更优)]
    for pname, mk in makers.items():
        pts = [p for p in all_points if p.policy == pname]
        for p in pts:
            row = next(x for x in rows if x["qid"] == p.qid)
            # 强制候选 = 环境支持的动作 − 它自己选的那个
            for a in [x for x in available_actions_of(p) if x != p.action]:
                p.cf[a] = counterfactual(r, mk, p.qid, row["question"], p.index, a)
                n_cf += 1
            key = "right" if p.outcome else "wrong"
            stat[key]["n"] += 1
            stat[key]["better"] += int(bool(p.better_exists()))
            stat[key]["worse"] += int(bool(p.worse_exists()))
            by_action[p.action][1] += 1
            by_action[p.action][0] += int(bool(p.better_exists()))
            per_q[(pname, p.qid)].append((p.outcome, bool(p.better_exists())))

    print(f"   跑了 {n_cf} 次反事实重放")
    for k, label in (("right", "基线**答对**的题"), ("wrong", "基线**答错**的题")):
        s = stat[k]
        print(f"   {label}（{s['n']:>3} 个决策点）："
              f"存在**更优**动作 {s['better']:>3} ({s['better']/max(1,s['n']):.3f}) ｜ "
              f"存在**更差**动作 {s['worse']:>3} ({s['worse']/max(1,s['n']):.3f})")

    print("\n   按「当时选的动作」拆（有更优选择的比例）：")
    for a, (g, n) in sorted(by_action.items(), key=lambda x: -x[1][1]):
        print(f"     {a:<9} {g:>3}/{n:<3} ({g/max(1,n):.2f})")

    # ---- ③b ⭐⭐ 最锋利的那一个数：**错题里有多少是"策略能救回来"的**
    #
    #     这是本脚本存在的真正理由 —— 它对应 ToolHorizon 那条
    #     「40% 的题 reward 与策略无关」。这里的同形问题是：
    #     **错了的题，是不是换任何动作都还是错？**（= 错误与策略选择无关）
    #
    #     ⭐ 它同时也是「模仿学习/策略改进的**上界**」：
    #        可救回率 = 0 ⇒ 任何策略在这批题上都不可能比现在更好，
    #        E12 训出来最好也就是打平 —— 那训它干什么。
    print("\n③b ⭐⭐ 错题的「策略可救回率」= E12 的**提升空间上界**")
    print("   " + "-" * 80)
    total_savable, total_wrong = 0, 0
    for pname in makers:
        wrong_q = [(qid, pts) for (pn, qid), pts in per_q.items()
                   if pn == pname and pts and not pts[0][0]]
        savable = [q for q, pts in wrong_q if any(g for _, g in pts)]
        total_savable += len(savable)
        total_wrong += len(wrong_q)
        print(f"   {pname:<10} 答错 {len(wrong_q):>2} 题 ｜ "
              f"其中**换一步能改结局**的 {len(savable):>2} 题 "
              f"({len(savable)/max(1,len(wrong_q)):.2f})")

    check("判据自检：答对的题上**不可能**存在更优动作（定义使然）",
          stat["right"]["better"] == 0,
          f"实测 {stat['right']['better']}；不为 0 说明反事实写错了")
    check("E12 有提升空间：错题里存在可救回的题",
          total_savable > 0,
          f"实测 {total_savable}/{total_wrong}；"
          f"若为 0 ⇒ 错误不由策略选择决定，训出来最好也就是打平")

    # ---- 落盘
    with OUT.open("w", encoding="utf-8") as f:
        for p in all_points:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    print(f"\n   ⇒ 决策日志写入 {OUT.relative_to(_PROJECT)}（{len(all_points)} 条）")

    print("\n" + "=" * 80)
    if FAILS:
        print(f"❌ {len(FAILS)} 条断言没过：")
        for x in FAILS:
            print(f"   · {x}")
    else:
        print("✅ 全部断言通过")
    return 1 if FAILS else 0


def available_actions_of(p: DecisionPoint) -> list:
    """从决策点的特征里反推环境支持哪些动作（决策点存的是特征，不是活状态）。"""
    out = ["search", "answer", "abstain"]
    if p.feats.get("has_locator"):
        out.append("locate")
    if p.feats.get("has_user"):
        out.append("ask")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
