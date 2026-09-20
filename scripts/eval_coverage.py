# -*- coding: utf-8 -*-
"""
块 E8 · 覆盖度策略的标定与对照（零模型 API / 零 GPU / 分钟级）。

    python scripts/eval_coverage.py

━━━ 这个脚本回答两个问题 ━━━

    ① **θ 该取多少？**（标定）
       θ_high / θ_low 不能拍脑袋。在标定集上扫一遍，
       看哪个取值让「答得对 + 拒得准 + 动作少」综合最好。

    ② **覆盖度比规则好多少？**（对照）
       · 规则策略   0/1 判断（E7 的基线）
       · 覆盖度策略 算出来的（E8 的增量）
       同一批题、同一个循环、同一个生成器 —— **只换策略**。

━━━ 指标全部机械可判（不问 LLM 当裁判）━━━

    答案正确率  回答里**引到了 gold 章节** → 1        （检索集 60 条）
    拒答准确率  该拒的题上真的拒答了 → 1              （边界集 45 条）
    误拒率      该答的题上拒答了 → 1（越低越好）      （检索集 60 条）
    硬答率      该拒的题上作答了 → 1（越低越好）      （边界集 45 条）
    平均动作数 / 平均预算消耗

    ⚠️ **诚实边界**：正确率依赖生成层有没有把证据引到 gold 上。
       现在用的是 `GroundedMockLLM`（照证据抄、不会推理），
       所以这个数**主要反映"证据有没有到位 + 有没有被引"**，
       不是真实模型的答题能力。**两档策略共用同一个生成器，所以对照仍然有效。**
       真数字要等有效 API key。
"""

from __future__ import annotations

import sys

# ⚠️ Windows 中文控制台默认 GBK：不设这个，print("⭐") 会抛 UnicodeEncodeError
#    → **判据崩在半路，红绿一个字都读不到**（2026-09-18 实测 eval_retrieval.py）。
#    errors="replace"：宁可显示问号，也不许判据跑到一半死掉。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from agent.loop import AgentLoop, LoopConfig, make_toolbox_factory       # noqa: E402
from agent.generator import make_generator                              # noqa: E402
from agent.policy import CoveragePolicy, RulePolicy                      # noqa: E402
from agent.tools import ScriptedUser                                     # noqa: E402
from retrieval import make_retriever                                      # noqa: E402

LABELS = PROJECT / "data" / "labels.json"
RET_SET = PROJECT / "data" / "eval" / "retrieval_set.json"
BND_SET = PROJECT / "data" / "eval" / "boundary_set.json"
OUT = PROJECT / "reports" / "coverage_eval.json"

BUDGET = 8
USER = ScriptedUser({"conditions": ["hypertension"]})

_fails: List[str] = []

def check(cond, label, detail=""):
    print(f"   {'[OK]' if cond else '[!!]'} {label}" + (f"  —— {detail}" if detail else ""))
    if not cond:
        _fails.append(label)
    return cond

# ---------------------------------------------------------------- 跑一轮

def run_one(retriever, question: str, policy) -> Tuple[Any, Any]:
    loop = AgentLoop(make_toolbox_factory(retriever, llm=make_generator(), top_k=5),
                     policy, LoopConfig(budget=BUDGET))
    lg = loop.run(question, run_id="cov", user=USER)
    return loop.last_stats, lg

def cited_keys(lg) -> set:
    out = set()
    for c in lg.claims:
        out.update(c.cite)
    return out

# ---------------------------------------------------------------- 指标

def evaluate(retriever, policy, ret_rows, bnd_rows, collect: bool = False) -> Dict[str, Any]:
    """在**检索集**上量答得对不对，在**边界集**上量拒得准不准。"""
    n_ans_ok = n_answered = n_abstained_wrong = 0
    n_refuse_ok = n_hard_answer = 0
    actions, costs = [], []
    per_row = []

    for r in ret_rows:
        st, lg = run_one(retriever, r["question"], policy)
        cited = cited_keys(lg)
        ok = st.terminated_by == "answer" and r["gold_cite_key"] in cited
        n_ans_ok += 1 if ok else 0
        if st.terminated_by == "answer":
            n_answered += 1
        else:
            n_abstained_wrong += 1
        actions.append(st.n_actions)
        costs.append(st.cost_spent)
        if collect:
            per_row.append({"qid": r["qid"], "question": r["question"],
                            "gold": r["gold_cite_key"], "cited": sorted(cited),
                            "correct": ok, "terminated_by": st.terminated_by,
                            "n_actions": st.n_actions})

    bnd_by_class: Dict[str, List[int]] = defaultdict(lambda: [0, 0])   # [拒对, 总数]
    for r in bnd_rows:
        st, lg = run_one(retriever, r["question"], policy)
        cls = r.get("boundary_class", "?")
        bnd_by_class[cls][1] += 1
        if st.terminated_by == "abstain":
            n_refuse_ok += 1
            bnd_by_class[cls][0] += 1
        else:
            n_hard_answer += 1
        actions.append(st.n_actions)
        costs.append(st.cost_spent)

    n_ret, n_bnd = len(ret_rows), len(bnd_rows)
    return {
        "accuracy": n_ans_ok / max(1, n_ret),           # 答对率（检索集）
        "wrong_abstain_rate": n_abstained_wrong / max(1, n_ret),   # 误拒率
        "refuse_accuracy": n_refuse_ok / max(1, n_bnd),  # 拒答准确率（边界集）
        "hard_answer_rate": n_hard_answer / max(1, n_bnd),         # 硬答率
        "answered_rate": n_answered / max(1, n_ret),
        "avg_actions": sum(actions) / max(1, len(actions)),
        "avg_cost": sum(costs) / max(1, len(costs)),
        "n_retrieval": n_ret, "n_boundary": n_bnd,
        "boundary_by_class": {c: {"refused": v[0], "n": v[1]}
                              for c, v in sorted(bnd_by_class.items())},
        "per_row": per_row,
    }

# ---------------------------------------------------------------- θ 标定

def calibrate(retriever, ret_rows, bnd_rows,
              highs=(0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 0.9),
              lows=(0.0, 0.2, 0.3, 0.45)) -> List[Dict[str, Any]]:
    """扫 θ。

    ⚠️ 为什么扫网格而不是"试一个看着行就定"：
       单点试验会**把第一次碰到的好看数字当成结论**。
       网格能看出"哪个区间稳"，而不是"哪个点刚好"。

    ⚠️ 用**子集**扫（每档前 N 条），不然 20 组 × 105 条太慢；
       选出 θ 之后再**全量复跑**验证 —— 这一步不能省。
    """
    rows = []
    for hi in highs:
        for lo in lows:
            if lo >= hi:
                continue
            pol = CoveragePolicy(theta_high=hi, theta_low=lo)
            m = evaluate(retriever, pol, ret_rows, bnd_rows)
            rows.append({"theta_high": hi, "theta_low": lo,
                         "accuracy": m["accuracy"],
                         "refuse_accuracy": m["refuse_accuracy"],
                         "wrong_abstain_rate": m["wrong_abstain_rate"],
                         "hard_answer_rate": m["hard_answer_rate"],
                         "avg_actions": m["avg_actions"],
                         # 综合分：答对 + 拒准 − 误拒 − 硬答。
                         # 四项等权 —— **这是价值判断，不是定理**，所以写出来让人能改。
                         "score": (m["accuracy"] + m["refuse_accuracy"]
                                   - m["wrong_abstain_rate"] - m["hard_answer_rate"])})
    return rows

# ---------------------------------------------------------------- 主

def main() -> int:
    print("=" * 78)
    print("块 E8 · 覆盖度策略：θ 标定 + 三档对照")
    print("=" * 78)

    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    ret_rows = json.loads(RET_SET.read_text(encoding="utf-8"))["rows"]
    bnd_rows = json.loads(BND_SET.read_text(encoding="utf-8"))["rows"]
    r = make_retriever(labels, use_locator=True)
    print(f"\n语料 {len(labels)} 份药 ｜ 检索集 {len(ret_rows)} 条 ｜ 边界集 {len(bnd_rows)} 条")
    print(f"预算 {BUDGET} ｜ 用户模拟器已接（可 ask）\n")

    # ---- ① θ 标定（用子集扫，省时间）
    print("① θ 标定（子集扫描：检索集 24 条 + 边界集 20 条）")
    print("-" * 78)
    sub_ret, sub_bnd = ret_rows[:24], bnd_rows[:20]
    grid = calibrate(r, sub_ret, sub_bnd)
    grid.sort(key=lambda x: (-x["score"], x["avg_actions"]))
    print(f"   {'θ_high':>7}{'θ_low':>7}{'答对率':>9}{'拒答率':>9}{'误拒率':>9}"
          f"{'硬答率':>9}{'动作数':>8}{'综合分':>9}")
    for g in grid[:8]:
        print(f"   {g['theta_high']:>7.2f}{g['theta_low']:>7.2f}{g['accuracy']:>9.3f}"
              f"{g['refuse_accuracy']:>9.3f}{g['wrong_abstain_rate']:>9.3f}"
              f"{g['hard_answer_rate']:>9.3f}{g['avg_actions']:>8.2f}{g['score']:>9.3f}")
    best = grid[0]
    print(f"\n   → 选 θ_high={best['theta_high']}, θ_low={best['theta_low']}"
          f"（子集综合分 {best['score']:.3f}）")

    # ---- ② 全量复跑（标定出来的 θ 必须在全量上验一遍）
    print("\n② 全量复跑（子集选出来的 θ，拿到全部 105 条上验）")
    print("-" * 78)
    print(f"   {'策略':<18}{'答对率':>9}{'拒答率':>9}{'误拒率':>9}{'硬答率':>9}"
          f"{'动作数':>8}{'预算':>8}")
    results = {}
    for name, pol in (
        ("① 规则（0/1）", RulePolicy()),
        (f"② 覆盖度(θ={best['theta_high']:.2f}/{best['theta_low']:.2f})",
         CoveragePolicy(theta_high=best["theta_high"], theta_low=best["theta_low"])),
    ):
        m = evaluate(r, pol, ret_rows, bnd_rows, collect=True)
        results[name] = m
        print(f"   {name:<18}{m['accuracy']:>9.3f}{m['refuse_accuracy']:>9.3f}"
              f"{m['wrong_abstain_rate']:>9.3f}{m['hard_answer_rate']:>9.3f}"
              f"{m['avg_actions']:>8.2f}{m['avg_cost']:>8.2f}")

    base = list(results.values())[0]
    new = list(results.values())[1]
    print(f"\n   ⭐ 增量： 答对率 {base['accuracy']:.3f} → {new['accuracy']:.3f}"
          f"（{new['accuracy'] - base['accuracy']:+.3f}）")
    print(f"            拒答率 {base['refuse_accuracy']:.3f} → {new['refuse_accuracy']:.3f}"
          f"（{new['refuse_accuracy'] - base['refuse_accuracy']:+.3f}）")
    print(f"            硬答率 {base['hard_answer_rate']:.3f} → {new['hard_answer_rate']:.3f}"
          f"（{new['hard_answer_rate'] - base['hard_answer_rate']:+.3f}）")

    # ---- 拒答按类别拆开看：**剩下的硬答是哪一类？**
    print("\n   拒答率按边界类别拆开（覆盖度策略）：")
    print(f"      {'类别':<18}{'拒对/总':>10}{'拒答率':>9}")
    for cls, v in new["boundary_by_class"].items():
        rate = v["refused"] / max(1, v["n"])
        flag = "  ← 还漏" if rate < 1.0 else ""
        print(f"      {cls:<18}{v['refused']:>4}/{v['n']:<5}{rate:>9.3f}{flag}")

    # ---- ③ 判据
    print("\n③ 判据")
    check(new["hard_answer_rate"] < base["hard_answer_rate"],
          f"硬答率下降（{base['hard_answer_rate']:.3f} → {new['hard_answer_rate']:.3f}）",
          "覆盖度该治好的主要病")
    check(new["refuse_accuracy"] > base["refuse_accuracy"],
          f"拒答准确率上升（{base['refuse_accuracy']:.3f} → {new['refuse_accuracy']:.3f}）")
    check(new["accuracy"] >= base["accuracy"] - 0.05,
          f"答对率不显著变差（{base['accuracy']:.3f} → {new['accuracy']:.3f}）",
          "⚠️ 靠「多拒答」换来的拒答率不算赢 —— 这条判据就是防这个")
    check(new["avg_actions"] <= base["avg_actions"] + 0.5,
          f"平均动作数不暴涨（{base['avg_actions']:.2f} → {new['avg_actions']:.2f}）")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "budget": BUDGET,
        "grid": grid,
        "chosen": {"theta_high": best["theta_high"], "theta_low": best["theta_low"]},
        "results": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {OUT}")

    print("\n" + "=" * 78)
    if _fails:
        print(f"❌ 判据未通过（{len(_fails)} 项）：")
        for f in _fails:
            print("   -", f)
        return 1
    print("✅ E8 判据全过")
    print("=" * 78)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
