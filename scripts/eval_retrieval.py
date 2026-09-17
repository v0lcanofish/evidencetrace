# -*- coding: utf-8 -*-
"""
块 E6 · 检索层判据（零模型 / 零 GPU / 秒级）。

    python scripts/eval_retrieval.py

用 E5 造好的「检索集」量三层各自的贡献：

    ① BM25 单独              → 基线
    ② BM25 + 章节定位         → ⭐ 本项目的独有能力
    ③ 稠密检索                → 待加（需要 BGE-M3）

**每加一层量一次，差值就是那一层的贡献** —— 这就是面试要讲的"从 X 到 Y"。

━━━ 两个指标 ━━━

    Recall@5   该命中的章节，进 top-5 了吗（0/1，按题平均）
    MRR        第一个命中的排名倒数（越靠前越高，上限 1.0）

    ⭐ 还按**两档**分开报：
        direct      关键词能匹配（药名+实体词就在原文里）
        paraphrase  同义改写（关键词匹配不上，只能靠语义）
       **两档之差 = 稠密检索要补的坑有多大。**
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from retrieval import BM25Retriever, build_chunks          # noqa: E402

LABELS = PROJECT / "data" / "labels.json"
RET_SET = PROJECT / "data" / "eval" / "retrieval_set.json"

_fails: list[str] = []


def check(cond, label, detail=""):
    print(f"   {'[OK]' if cond else '[!!]'} {label}" + (f"  —— {detail}" if detail else ""))
    if not cond:
        _fails.append(label)
    return cond


def hit_of(docs, row: Dict, k: int = 5) -> int:
    """
    命中判定：top-k 里有没有**来自 gold 章节**的 chunk。

    ⭐ 机械可判 —— 比的是 `(doc_id, loinc)`，不涉及任何语义判断。

    ⚠️ 口径修正（2026-09-17 实测踩到）：
       「注意事项」这一类问题，**不同药落在不同的 LOINC 上** ——
         metformin 分成 Precautions(34069-5) + Warnings(34071-1)，
         warfarin  合成一节 Warnings and Precautions(43685-7)。
       所以命中判定必须用 `gold_loinc_any`（**可接受集合**），不能只认一个 ——
       只认一个会把"引对了邻近章节"误判成失败，人为把指标压低。
    """
    any_lo = set(row.get("gold_loinc_any") or [row.get("gold_loinc")])
    gold_doc = row["gold_cite_key"].split("#")[0]
    for d in docs[:k]:
        if d.doc_id == gold_doc and d.loinc in any_lo:
            return d.rank
    return 0


def evaluate(retriever, rows, k: int = 5):
    """返回 (整体指标, 分档指标, 逐题明细)"""
    by_tier = defaultdict(lambda: {"n": 0, "recall": 0, "rr": 0.0})
    detail = []
    for r in rows:
        docs = retriever(r["question"], k)
        rank = hit_of(docs, r, k)
        tier = r["tier"]
        by_tier[tier]["n"] += 1
        by_tier[tier]["recall"] += 1 if rank else 0
        by_tier[tier]["rr"] += (1.0 / rank) if rank else 0.0
        detail.append({"qid": r["qid"], "tier": tier, "rank": rank,
                       "question": r["question"], "gold": r["gold_cite_key"],
                       "top1": docs[0].cite_key if docs else ""})
    n = len(rows)
    rec = sum(d["rank"] > 0 for d in detail) / max(1, n)
    mrr = sum(1.0 / d["rank"] for d in detail if d["rank"]) / max(1, n)
    out = {"n": n, "recall@5": rec, "mrr": mrr}
    tiers = {t: {"n": v["n"], "recall@5": v["recall"] / max(1, v["n"]),
                 "mrr": v["rr"] / max(1, v["n"])} for t, v in by_tier.items()}
    return out, tiers, detail


def main() -> int:
    print("=" * 78)
    print("块 E6 · 检索层判据")
    print("=" * 78)

    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    ret_set = json.loads(RET_SET.read_text(encoding="utf-8"))
    rows = ret_set["rows"]
    print(f"\n语料：{len(labels)} 份药 ｜ 检索集：{len(rows)} 条\n")

    # ---- 分块体检
    chunks = build_chunks(labels)
    lens = sorted(len(c.text) for c in chunks)
    print("① 分块")
    check(len(chunks) >= 100, f"切出 {len(chunks)} 个 chunk（来自 36 节）")
    check(lens[len(lens) // 2] < 700, f"中位长度 {lens[len(lens)//2]} 字符（不会太长）")
    check(all(c.content_hash and c.chunk_id for c in chunks),
          "每个 chunk 都带 chunk_id + content_hash（引用契约）")

    # ---- 三层对比
    print("\n② 三层对比（每加一层量一次）")
    r_base = BM25Retriever(labels, use_locator=False)
    r_locate = BM25Retriever(labels, use_locator=True)

    base, base_t, _ = evaluate(r_base, rows)
    loc, loc_t, detail = evaluate(r_locate, rows)

    print(f"\n   {'配置':<26}{'Recall@5':>10}{'MRR':>10}{'n':>7}")
    print(f"   {'-'*53}")
    print(f"   {'① BM25 单独（基线）':<24}{base['recall@5']:>10.3f}{base['mrr']:>10.3f}{base['n']:>7}")
    print(f"   {'② + 章节定位':<24}{loc['recall@5']:>10.3f}{loc['mrr']:>10.3f}{loc['n']:>7}")
    delta = loc["recall@5"] - base["recall@5"]
    print(f"   {'   ⭐ 章节定位的贡献':<24}{delta:>+10.3f}")

    print(f"\n   {'分档':<26}{'Recall@5':>10}{'MRR':>10}{'n':>7}")
    print(f"   {'-'*53}")
    for t in ("direct", "paraphrase"):
        bt, lt = base_t.get(t, {}), loc_t.get(t, {})
        if not bt:
            continue
        print(f"   {f'① BM25 · {t}':<24}{bt['recall@5']:>10.3f}{bt['mrr']:>10.3f}{bt['n']:>7}")
        print(f"   {f'② +定位 · {t}':<24}{lt['recall@5']:>10.3f}{lt['mrr']:>10.3f}{lt['n']:>7}")

    gap = (loc_t.get("direct", {}).get("recall@5", 0)
           - loc_t.get("paraphrase", {}).get("recall@5", 0))
    print(f"\n   ⭐ direct − paraphrase = {gap:+.3f}")
    print(f"      ↑ 这个差就是**稠密检索要补的坑**有多大")

    # ---- 章节定位本身准不准
    print("\n③ 章节定位的准确率（只看它该判的那部分）")
    locator = r_locate.locator
    n_loc, n_ok = 0, 0
    for r in rows:
        if r["intent"] not in ("interaction", "contraindication"):
            continue
        res = locator(r["question"])
        n_loc += 1
        if set(r.get("gold_loinc_any") or [r["gold_loinc"]]) & set(res["loincs"]):
            n_ok += 1
    check(n_loc > 0, f"覆盖 {n_loc} 条带实体的题")
    print(f"   定位命中率 {n_ok}/{n_loc} = {n_ok/max(1,n_loc):.3f}")

    # ---- 断言（判据）
    print("\n④ 判据")
    check(base["recall@5"] > 0.30,
          f"BM25 基线 Recall@5 > 0.30（实测 {base['recall@5']:.3f}）",
          "太低说明语料/索引有问题")
    check(loc["recall@5"] >= base["recall@5"],
          f"加章节定位**不劣化**（{loc['recall@5']:.3f} ≥ {base['recall@5']:.3f}）")
    check(n_ok / max(1, n_loc) > 0.85,
          f"章节定位命中率 > 0.85（实测 {n_ok/max(1,n_loc):.3f}）")
    check(gap > 0, f"paraphrase 档明显更难（差 {gap:+.3f}）—— 这正是稠密检索要解决的",
          "如果两档差不多，说明 paraphrase 造得不够'改写'")

    # ---- 存盘
    out = PROJECT / "reports" / "retrieval_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n": len(rows),
        "baseline_bm25": base, "with_locator": loc,
        "by_tier": {"baseline": base_t, "with_locator": loc_t},
        "direct_minus_paraphrase": gap,
        "locator_accuracy": n_ok / max(1, n_loc),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {out}")

    # ---- 失败样例（给下一步指方向）
    print("\n⑤ 失败样例（top-5 一条都没命中）—— 这些是下一步要修的")
    fails = [d for d in detail if d["rank"] == 0][:5]
    for d in fails:
        print(f"   [{d['tier']}] {d['question'][:66]}")
        print(f"        gold={d['gold'][-12:]}  top1={d['top1'][-12:] if d['top1'] else '(空)'}")

    print("\n" + "=" * 78)
    if _fails:
        print(f"❌ 检索层判据未通过（{len(_fails)} 项）：")
        for f in _fails:
            print("   -", f)
        return 1
    print("✅ 检索层判据全过")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
