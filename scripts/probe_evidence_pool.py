# -*- coding: utf-8 -*-
"""
E13b 探针：归因报 utilization 的题，**生成层实际看到的证据池长什么样**。

    PYTHONIOENCODING=utf-8 python scripts/probe_evidence_pool.py

━━━ 为什么需要它（2026-09-18 建）━━━

E13 的 ⑥ 量出「瓶颈从检索层搬到利用层」（同一批题，retrieval 8 → utilization 8）。
但 `diag_utilization_cases.py` 只告诉我们**结果**（"gold 在池子里、引了别的节"），
没告诉我们**池子的结构**：

    · 焦点药是谁？        （coverage 的 scope）
    · 池子里有几条是**别的药**的？（跨药污染 —— 生成层没法知道该忽略哪条）
    · 定位到的那一节，池子里有几条？
    · gold 在不在池子里？在池子里排第几？（"只增不减"的直接后果就是它被埋在后面）
    · 生成层真正的判据（词面重合）给每条打了多少分？

⭐ 这一版**不修任何东西**，只测量。先看清楚，再决定剪裁该怎么剪 ——
   「阈值不能拍」的前提是先把分布量出来。
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT))

from agent.coverage import compute_coverage, corpus_drugs_from, question_terms  # noqa: E402
from agent.loop import AgentLoop, LoopConfig, make_toolbox_factory             # noqa: E402
from agent.mocks import GroundedMockLLM                                        # noqa: E402
from agent.policy import CoveragePolicy                                        # noqa: E402
from agent.report import build_evidence_block, looks_like_abstention           # noqa: E402
from retrieval import make_retriever                                           # noqa: E402

LABELS = _PROJECT / "data" / "labels.json"
R_SET = _PROJECT / "data" / "eval" / "retrieval_set.json"
N = 60


def pool_row(question: str, gold: str, docs, located, known_drugs) -> dict:
    """一条题的证据池结构。**纯测量，不做任何剪裁。**"""
    cov = compute_coverage(question, docs, located, known_drugs)
    asked = [s.key for s in cov.slots if s.name == "drug" and s.key != "(未识别)"]
    scope = {d.lower() for d in asked}
    lo = set((located or {}).get("loincs") or [])

    in_scope = [d for d in docs if (d.drug or "").lower() in scope]
    out_scope = [d for d in docs if (d.drug or "").lower() not in scope]
    in_loc = [d for d in docs if d.loinc in lo]

    gold_idx = next((i for i, d in enumerate(docs) if d.cite_key == gold), None)

    # 生成层真正的判据：与问题的词面重合（复刻 GroundedMockLLM._report 的打分）
    qw = GroundedMockLLM._content_words(question)
    overlap = [len(qw & GroundedMockLLM._content_words(d.text)) for d in docs]
    best = max(overlap) if overlap else 0

    return {
        "q": question, "gold": gold,
        "n_pool": len(docs), "n_in_scope": len(in_scope), "n_out_scope": len(out_scope),
        "n_in_located": len(in_loc), "located": sorted(lo),
        "asked": asked,
        "gold_in_scope": bool(gold_idx is not None and (docs[gold_idx].drug or "").lower() in scope),
        "gold_in_located": bool(gold_idx is not None and docs[gold_idx].loinc in lo),
        "gold_pos": gold_idx,                       # 在池子里第几条（0=最前）
        "gold_overlap": overlap[gold_idx] if gold_idx is not None else None,
        "max_overlap": best,
        "gold_is_argmax": bool(gold_idx is not None and overlap and overlap[gold_idx] == best),
        "pool_secs": Counter(d.section[:26] for d in docs).most_common(),
    }


def main() -> int:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    r = make_retriever(labels, use_locator=True)
    known = corpus_drugs_from(r)
    rows = json.loads(R_SET.read_text(encoding="utf-8"))["rows"][:N]

    loop = AgentLoop(make_toolbox_factory(r, llm=GroundedMockLLM(), top_k=5),
                     CoveragePolicy(), LoopConfig(budget=8))

    bad = []
    for row in rows:
        q, gold = row["question"], row["gold_cite_key"]
        lg = loop.run(q, run_id=row["qid"])
        st = loop.last_state
        cited = lg.cited_keys()
        res = lg.attribute({gold}, correct={gold}.issubset(cited))
        if res["layer"] == "none":
            continue
        bad.append((row, st, pool_row(q, gold, st.evidence, st.located, known)))

    print("=" * 88)
    print(f"归因非 none 的题：{len(bad)} 条（策略=CoveragePolicy, 生成器=GroundedMockLLM）")
    print("=" * 88)

    print(f"\n{'#':<3}{'池子':>5}{'同药':>5}{'异药':>5}{'定位节':>6}"
          f"{'gold同药':>9}{'gold定位节':>10}{'gold位置':>9}{'gold分':>7}{'最高分':>7}{'gold是argmax':>13}")
    print("-" * 88)
    for i, (row, st, p) in enumerate(bad, 1):
        print(f"{i:<3}{p['n_pool']:>5}{p['n_in_scope']:>5}{p['n_out_scope']:>5}"
              f"{p['n_in_located']:>6}"
              f"{str(p['gold_in_scope']):>9}{str(p['gold_in_located']):>10}"
              f"{str(p['gold_pos']):>9}{str(p['gold_overlap']):>7}{p['max_overlap']:>7}"
              f"{str(p['gold_is_argmax']):>13}")

    print("\n" + "=" * 88)
    print("逐条现场")
    print("=" * 88)
    for i, (row, st, p) in enumerate(bad, 1):
        abst = looks_like_abstention(st.answer_text or "")
        print(f"\n[{i}] {p['q']}")
        print(f"    焦点药={p['asked']}  定位到={p['located']}")
        print(f"    池子 {p['n_pool']} 条 = 同药 {p['n_in_scope']} + 异药 {p['n_out_scope']}"
              f" ｜ 落在定位章节的 {p['n_in_located']} 条")
        print(f"    gold：在池子里第 {p['gold_pos']} 条 ｜ 同药={p['gold_in_scope']} "
              f"定位节={p['gold_in_located']} ｜ 生成器给它 {p['gold_overlap']} 分，"
              f"全场最高 {p['max_overlap']} 分 → gold 是 argmax={p['gold_is_argmax']}")
        print(f"    池子章节分布: {p['pool_secs']}")
        print(f"    {'【拒答】' if abst else ''}A: {(st.answer_text or '').strip()[:100]}")

    # ---- 汇总（这些是**与生成器无关**的系统事实）
    n_out = sum(1 for _, _, p in bad if p["n_out_scope"] > 0)
    n_gold_lost_pos = sum(1 for _, _, p in bad if p["gold_pos"] is not None and p["gold_pos"] >= 5)
    n_argmax = sum(1 for _, _, p in bad if p["gold_is_argmax"])
    n_abs = sum(1 for _, st, _ in bad if looks_like_abstention(st.answer_text or ""))

    print("\n" + "=" * 88)
    print("汇总（前三条与生成器无关，是系统事实；第四条依赖生成器）")
    print("=" * 88)
    print(f"  · 池子里混进**别的药**的题：        {n_out}/{len(bad)}")
    print(f"  · gold 被埋在池子第 5 条以后的题：  {n_gold_lost_pos}/{len(bad)}"
          f"    ← 「只增不减」的直接后果")
    print(f"  · gold 压根不在池子里的题：         "
          f"{sum(1 for _, _, p in bad if p['gold_pos'] is None)}/{len(bad)}")
    print(f"  · 生成器挑中的就是 gold 的题：      {n_argmax}/{len(bad)}"
          f"    ← 这一条**依赖生成器**，换真模型会变")
    print(f"  · 其中结局是拒答的：                {n_abs}/{len(bad)}"
          f"    ← 需单独报，`attribute()` 的拒答必然落进 utilization")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
