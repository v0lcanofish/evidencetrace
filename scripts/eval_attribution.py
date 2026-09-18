# -*- coding: utf-8 -*-
"""
E13 判据 + 交付物：**跨层归因**（错题指认层）。

━━━ 这个块要解决什么 ━━━

`Ledger.attribute()` 早就写好了（三层：检索 / 利用 / 推理），
**但零调用点** —— 只在账本自检里跑过一次。
这是本项目反复出现的那类问题：**写了，但没接进流程。**

E13 把它接上，产出**错误层级分布**：
    答错的题，到底卡在「找不到证据」「找到了没用」还是「用了还答错」？
**分布指向哪一层，改进就该往哪一层使劲。**

━━━ 判据 ━━━

    ① 三层**各自指认正确**（构造三种故障注入，逐层验）
    ② ⭐ **分布的和 = 被归因的题数**（不许漏题、不许重复归因）
    ③ ⭐ 三层的**建议互不相同**且指向本层（指错层 = 建议指错方向）
    ④ 答对的题报 `none` —— **不虚报病根**
    ⑤ ⭐ **引错章节**必须报检索层，不是推理层（LOINC 粒度的老洞）
    ⑥ ⭐⭐ **瓶颈搬层**（E6 × E13）：检索层升级前后，失败的题**是同一批**，
       只是归因层从 retrieval 变成了 utilization —— **问题被推走了，不是被消灭了**

跑法：
    PYTHONIOENCODING=utf-8 D:/anaconda/envs/myenv/python.exe scripts/eval_attribution.py
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
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from agent.loop import AgentLoop, LoopConfig, make_toolbox_factory   # noqa: E402
from agent.mocks import GroundedMockLLM                               # noqa: E402
from agent.policy import CoveragePolicy                               # noqa: E402
from agent.report import looks_like_abstention                        # noqa: E402
from ledger.ledger import Ledger, Claim, Doc, Step                    # noqa: E402
from retrieval import make_retriever                                 # noqa: E402

LABELS = _PROJECT / "data" / "labels.json"
R_SET = _PROJECT / "data" / "eval" / "retrieval_set.json"
N_RUN = 60          # 跑多少条真实题（全量，几十秒内）


def _mk_loop(r):
    return AgentLoop(make_toolbox_factory(r, llm=GroundedMockLLM(), top_k=5),
                     CoveragePolicy(), LoopConfig(budget=8))


def _run_dist(r, rows):
    """
    跑一批题，返回 (qid → 归因层, Counter, 非 none 里有多少条其实是**拒答**)。

    ⚠️ 最后一个数必须报出来：`attribute()` 的 correct 判据是 `gold ⊆ cited`，
       **拒答没有引用 → cited 为空 → 必然落进 utilization 分支**。
       不拆开的话，"召回了不用"和"策略主动拒答"会被混成同一个数。
    """
    layers, dist, n_abstain = {}, Counter(), 0
    for row in rows:
        gold = {row["gold_cite_key"]}
        lp = _mk_loop(r)
        lg = lp.run(row["question"], run_id=row["qid"])
        res = lg.attribute(gold, correct=gold.issubset(lg.cited_keys()))
        layers[row["qid"]] = res["layer"]
        dist[res["layer"]] += 1
        if res["layer"] != "none" and looks_like_abstention(lp.last_stats.answer_text or ""):
            n_abstain += 1
    return layers, dist, n_abstain


# ================================================================ ① 三种故障注入
def injected_cases():
    """
    手工构造三种故障，**每种只坏一层** —— 这样才能验"归因指得准不准"。

    ⚠️ 三种必须**互斥**：如果一组数据同时坏两层，就分不清归因报的是哪一层。
    """
    d_gold = Doc(doc_id="gold-0001", section="Dosage and Administration",
                 loinc="34068-7", text="...", drug="allopurinol")
    d_other = Doc(doc_id="gold-0001", section="Drug Interactions",
                  loinc="34073-7", text="...", drug="allopurinol")   # 同药、不同节
    gold = {d_gold.cite_key}

    out = []

    # ---- 故障 A：检索层 —— gold 章节根本没召回
    lg = Ledger("inj-retrieval", "q", seed=0)
    lg.record(Step(step=1, action="retrieve", query="q", retrieved=[d_other],
                   used_in_report=True))
    lg.add_claim(Claim(claim_id="c1", text="t", cite=[d_other.cite_key]))
    out.append(("检索层：gold 章节没召回", lg, gold, False, "retrieval"))

    # ---- 故障 B：利用层 —— 召回了全套 gold，但没进回答
    lg = Ledger("inj-utilization", "q", seed=0)
    lg.record(Step(step=1, action="retrieve", query="q", retrieved=[d_gold],
                   used_in_report=False, reason_dropped="context_budget"))
    lg.add_claim(Claim(claim_id="c1", text="t", cite=[]))
    out.append(("利用层：召回了没引", lg, gold, False, "utilization"))

    # ---- 故障 C：推理层 —— 证据全召回也全引了，但结论错
    lg = Ledger("inj-reasoning", "q", seed=0)
    lg.record(Step(step=1, action="retrieve", query="q", retrieved=[d_gold],
                   used_in_report=True))
    lg.add_claim(Claim(claim_id="c1", text="t", cite=[d_gold.cite_key]))
    out.append(("推理层：证据齐了还错", lg, gold, False, "reasoning"))

    # ---- 对照：答对 → none
    lg = Ledger("ok", "q", seed=0)
    lg.record(Step(step=1, action="retrieve", query="q", retrieved=[d_gold],
                   used_in_report=True))
    lg.add_claim(Claim(claim_id="c1", text="t", cite=[d_gold.cite_key]))
    out.append(("对照：答对了", lg, gold, True, "none"))

    # ---- ⑤ 引错章节（同药、不同节）→ 必须报检索层
    lg = Ledger("inj-wrong-section", "q", seed=0)
    lg.record(Step(step=1, action="retrieve", query="q", retrieved=[d_other],
                   used_in_report=True))
    lg.add_claim(Claim(claim_id="c1", text="t", cite=[d_other.cite_key]))
    out.append(("引错章节（同药不同节）", lg, gold, False, "retrieval"))
    return out


def main() -> int:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    r = make_retriever(labels, use_locator=True)

    print("=" * 78)
    print("E13 判据 · 跨层归因（错题指认层）")
    print("=" * 78)

    fails = []

    def check(cond, label, detail=""):
        print(f"   {'[OK]' if cond else '[!!]'} {label}" + (f"  —— {detail}" if detail else ""))
        if not cond:
            fails.append(label)

    # ================================================================ ① 故障注入
    print("\n① 三种故障注入：每种只坏一层，看归因指得准不准\n")
    suggestions = {}
    for name, lg, gold, correct, want in injected_cases():
        res = lg.attribute(gold, correct=correct)
        check(res["layer"] == want, f"{name} → {want}", f"实测 {res['layer']}")
        if want != "none":
            suggestions[want] = res["suggestion"]

    # ================================================================ ③ 建议互不相同
    print("\n③ 三层的建议必须互不相同（指错层 = 建议指错方向）\n")
    check(len(set(suggestions.values())) == len(suggestions),
          "三层的建议两两不同", f"{suggestions}")
    for k, v in suggestions.items():
        print(f"     {k:<12} → 「{v}」")

    # ================================================================ ⑤ 真实题上的分布
    print(f"\n⑤ 跑 {N_RUN} 条真实评测题，看归因能不能真的跑起来\n")
    rows = json.loads(R_SET.read_text(encoding="utf-8"))["rows"][:N_RUN]
    n_answered = len(rows)
    layers, dist, n_abstain = _run_dist(r, rows)

    print(f"   跑完 {n_answered} 条")
    for k in ("none", "retrieval", "utilization", "reasoning"):
        print(f"     {k:<12} {dist.get(k, 0):>3} 条")

    # ② 分布的和 = 题数（**不许漏题、不许重复归因**）
    check(sum(dist.values()) == n_answered,
          "⭐ 归因分布之和 = 被归因的题数（不漏不重）",
          f"{sum(dist.values())} vs {n_answered}")

    # ④ 判据②的反面：如果全部答对，就不该有病根
    if dist.get("none", 0) == n_answered:
        check(True, "全部答对 → 全部报 none（不虚报病根）", f"none = {n_answered}")
    else:
        check(dist.get("none", 0) > 0,
              "有答对的题 → 那些题报 none",
              f"none = {dist.get('none', 0)} / {n_answered}")
        print(f"     ⚠️ 诚实边界：非 none 的 {n_answered - dist.get('none', 0)} 条里，"
              f"**{n_abstain} 条其实是拒答**（correct 判据是 gold⊆cited，"
              f"拒答没有引用 → 必然落进 utilization）。")

    # ================================================================ ⑥ 瓶颈搬层
    # ⭐ 2026-09-18：E6 把检索层修好之后重跑，归因分布**整体搬了一层** ——
    #    这是"改进到底有没有用"最直接的证据，所以锁成常驻判据。
    print("\n⑥ 检索层升级（E6）前后：同一批题、同一个循环，**只换检索器**\n")
    r_old = make_retriever(labels, kind="bm25", use_locator=True)
    layers_old, dist_old, _ = _run_dist(r_old, rows)

    print("   配置                        retrieval  utilization   非 none 合计")
    print("   " + "-" * 62)
    for tag, dd in (("① 旧 BM25（乘性 boost）", dist_old), ("② 新混合（三路+重排）", dist)):
        non = n_answered - dd.get("none", 0)
        print(f"   {tag:<26} {dd.get('retrieval', 0):>7}  {dd.get('utilization', 0):>10}"
              f"   {non:>10}")

    old_fail = {k for k, v in layers_old.items() if v != "none"}
    new_fail = {k for k, v in layers.items() if v != "none"}

    check(dist_old.get("retrieval", 0) > 0,
          "旧检索器上确实存在『gold 没召回』的题",
          f"retrieval = {dist_old.get('retrieval', 0)}")
    check(dist.get("retrieval", 0) == 0,
          "⭐ 升级后『gold 没召回』清零",
          f"retrieval = {dist.get('retrieval', 0)}")
    check(old_fail == new_fail,
          "⭐⭐ 瓶颈**搬层**而不是缩小：新旧失败的题**完全同一批**",
          f"旧 {len(old_fail)} 条 / 新 {len(new_fail)} 条 / 交集 {len(old_fail & new_fail)} 条")
    if old_fail != new_fail:
        print(f"       只在旧的里：{[layers_old[k] for k in sorted(old_fail - new_fail)][:6]}")
        print(f"       只在新的里：{[layers[k] for k in sorted(new_fail - old_fail)][:6]}")

    print("     ⇒ 结论：检索层的病**不是被治好了，是被推到了下一层**。")
    print("       改进往哪使劲，由这次归因指：**改上下文压缩 / 排序**（利用层的建议）。")

    print()
    print("=" * 78)
    if fails:
        print(f"❌ 判据未过（{len(fails)} 项）：")
        for f in fails:
            print("   -", f)
        return 1
    print("✅ E13 判据全过：三层指认准、建议不混、分布不漏题、答对不虚报")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
