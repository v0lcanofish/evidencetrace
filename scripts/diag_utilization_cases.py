# -*- coding: utf-8 -*-
"""
E13 的诊断工具：归因报 `utilization` 时，**逐条看现场**。

    PYTHONIOENCODING=utf-8 python scripts/diag_utilization_cases.py

━━━ 为什么需要它（2026-09-18 建）━━━

`eval_attribution.py` 的 ⑥ 量出「瓶颈从检索层搬到利用层」——
但"utilization 8 条"这个数**本身说明不了病根**，它至少混着两种完全不同的情况：

    ① 召回了却没用     证据在池子里，生成层挑了**另一节**去引
    ② 拒答被算进来     策略认为证据够 → 生成层说"不充分" → 拒答
                      （`attribute()` 的 correct 判据是 `gold ⊆ cited`，
                        **拒答没有引用 → 必然落进 utilization 分支**）

⚠️ 这两种的**正确动作不同**：① 要改生成前的证据排序/剪裁，② 要查两层判据为什么打架。
   不拆开就会拿一个混口径的数去指导改进 —— 那是"阈值不能拍"的反面教材。

对照跑法：`ET_RETRIEVER=bm25 ... diag_utilization_cases.py`（换回旧检索器看同一批题）
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT))

from agent.loop import AgentLoop, LoopConfig, make_toolbox_factory   # noqa: E402
from agent.mocks import GroundedMockLLM                              # noqa: E402
from agent.policy import CoveragePolicy, RulePolicy                  # noqa: E402
from agent.report import looks_like_abstention                       # noqa: E402
from retrieval import make_retriever                                 # noqa: E402

LABELS = _PROJECT / "data" / "labels.json"
R_SET = _PROJECT / "data" / "eval" / "retrieval_set.json"
N = 60

import json


def mk(r, pol):
    return AgentLoop(make_toolbox_factory(r, llm=GroundedMockLLM(), top_k=5),
                     pol, LoopConfig(budget=8))


def main() -> int:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    r = make_retriever(labels, use_locator=True)
    rows = json.loads(R_SET.read_text(encoding="utf-8"))["rows"][:N]

    for pol_name, pol in (("CoveragePolicy", CoveragePolicy()), ("RulePolicy", RulePolicy())):
        print("=" * 78)
        print(f"策略 = {pol_name}")
        print("=" * 78)
        dist = Counter()
        rows_out = []
        for row in rows:
            q, gold = row["question"], {row["gold_cite_key"]}
            lp = mk(r, pol)
            lg = lp.run(q, run_id=row["qid"])
            st = lp.last_stats
            cited = lg.cited_keys()
            correct = gold.issubset(cited)
            res = lg.attribute(gold, correct=correct)
            dist[res["layer"]] += 1
            if res["layer"] != "none":
                actions = [s.action for s in lg.steps]
                ans = st.answer_text or ""
                abstained = looks_like_abstention(ans)
                cov = getattr(pol, "last_coverage", None)
                cov_txt = ""
                if cov is not None:
                    cov_txt = (f"coverage={cov.score:.2f} "
                               f"缺={[f'{s.name}:{s.key}' for s in cov.missing()]}")
                # 生成层**拿到**了哪些证据（按章节名）
                ev_secs = [f"{d.section[:22]}" for d in lp.last_state.evidence]
                rows_out.append({
                    "q": q, "layer": res["layer"], "actions": actions,
                    "abstained": abstained, "empty_answer": not ans.strip(),
                    "term": st.terminated_by,
                    "n_claims": len(lg.claims), "n_cited": len(cited),
                    "n_cited_gold": len(cited & gold),
                    "n_retrieved_gold": len(lg.retrieved_keys() & gold),
                    "answer_head": ans.strip().replace("\n", " ")[:90],
                    "cov": cov_txt, "ev_secs": ev_secs,
                })
        for k in ("none", "retrieval", "utilization", "reasoning"):
            print(f"   {k:<12} {dist.get(k, 0):>3}")
        print(f"\n   非 none 共 {len(rows_out)} 条，逐条看是不是拒答：\n")
        n_abs = 0
        for i, o in enumerate(rows_out, 1):
            if o["abstained"] or o["empty_answer"]:
                n_abs += 1
            print(f"   [{i}] layer={o['layer']}  拒答={o['abstained']}  "
                  f"空答案={o['empty_answer']}  论断={o['n_claims']}  "
                  f"引用={o['n_cited']}  引用命中gold={o['n_cited_gold']}  "
                  f"召回gold={o['n_retrieved_gold']}")
            print(f"       动作={o['actions']}  {o['cov']}")
            print(f"       Q: {o['q'][:70]}")
            print(f"       生成层看到的证据章节: {o['ev_secs']}")
            print(f"       A: {o['answer_head']}")
        print(f"\n   ⇒ 非 none 里 **拒答 {n_abs} / {len(rows_out)}**")
        print(f"   ⇒ 扣掉拒答后真正的 utilization = "
              f"{sum(1 for o in rows_out if o['layer'] == 'utilization' and not o['abstained'] and not o['empty_answer'])}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
