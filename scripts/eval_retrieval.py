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
LOC_SET = PROJECT / "data" / "eval" / "section_locate_set.json"

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
    n_sec = sum(len(L.get("sections", [])) for L in labels)
    print("① 分块")
    # ⚠️ "来自 N 节" 原来是写死的 36（6 份药时代的数）。语料扩到 50 份之后
    #    这句话就成了假的 —— 打印出来的东西必须跟着数据走。
    check(len(chunks) >= 100, f"切出 {len(chunks)} 个 chunk"
                              f"（来自 {len(labels)} 份药的 {n_sec} 节）")
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
    if abs(gap) < 0.05:
        print("      ⚠️ **两档没有差别** —— 这个指标现在量不出「稠密检索的价值」。")
        print("         原因（2026-09-17 查清）：改写档里**仍然带着药名**，")
        print("         而药名是 BM25 最强的信号，所以它根本没变难。")
        print("         ⛔ 之前报的 0.192 / 0.115 **是假象**：那时两档的意图构成不同")
        print("            （direct 只从相互作用/禁忌生成，paraphrase 覆盖 5 种意图），")
        print("            量的是「题不一样」，不是「改写更难」。")
        print("         要真正量出稠密检索的价值，改写档必须**去掉药名**，")
        print("         靠上文/指代把它还原出来 —— 那是 E10（记忆与指代）的活。")
    else:
        print(f"      ↑ 这个差就是**稠密检索要补的坑**有多大")

    # ---- 章节定位本身准不准
    #
    # ⚠️ 2026-09-17 改口径：原来只在**检索集**里挑带实体的那部分（28 条，只有
    #    「相互作用/禁忌」两类意图）——样本又小又偏。
    #    现在改用**章节定位集**（50 条，5 种意图全覆盖）。
    #    ⭐ 这也让 `section_locate_set.json` 第一次有了**指名道姓的消费者** ——
    #      之前它和另外两个集一样，只有元校验脚本读过，没进任何判据（就是"堆测评"）。
    print("\n③ 章节定位的准确率（用章节定位集，5 种意图全覆盖）")
    locator = r_locate.locator
    if LOC_SET.exists():
        loc_rows = json.loads(LOC_SET.read_text(encoding="utf-8"))["rows"]
    else:
        print("   ⚠️ 没有 section_locate_set.json，退回用检索集里的带实体题")
        loc_rows = [r for r in rows if r["intent"] in ("interaction", "contraindication")]
    n_loc, n_ok = 0, 0
    miss_by_intent = defaultdict(int)
    for r in loc_rows:
        res = locator(r["question"])
        n_loc += 1
        if set(r.get("gold_loinc_any") or [r["gold_loinc"]]) & set(res["loincs"]):
            n_ok += 1
        else:
            miss_by_intent[r.get("intent", "?")] += 1
    check(n_loc > 0, f"覆盖 {n_loc} 条定位题")
    print(f"   定位命中率 {n_ok}/{n_loc} = {n_ok/max(1,n_loc):.3f}")
    if miss_by_intent:
        print(f"   漏检按意图：{dict(miss_by_intent)}")

    # ---- 断言（判据）
    print("\n④ 判据")
    check(base["recall@5"] > 0.30,
          f"BM25 基线 Recall@5 > 0.30（实测 {base['recall@5']:.3f}）",
          "太低说明语料/索引有问题")
    check(loc["recall@5"] >= base["recall@5"],
          f"加章节定位**不劣化**（{loc['recall@5']:.3f} ≥ {base['recall@5']:.3f}）")
    check(n_ok / max(1, n_loc) > 0.85,
          f"章节定位命中率 > 0.85（实测 {n_ok/max(1,n_loc):.3f}）")
    # ⛔ 这条判据 2026-09-17 **撤销**（原话是 `check(gap > 0, "paraphrase 档明显更难")`）。
    #
    #    撤销的理由不是"它红了"，是**它量错了东西**：
    #      两档成对化之后 gap = 0.000。原来那个 +0.192 / +0.115 来自
    #      direct 档只覆盖 2 种意图、paraphrase 档覆盖 5 种 —— 比的是不同的题。
    #    ⚠️ **判据红了就去放宽阈值，等于把发现真相的机会扔掉。**
    #       这次红的是一个**本来就不该信的数字**，撤掉它才是对的。
    check(loc_t.get("paraphrase", {}).get("n", 0) >= 10,
          f"paraphrase 档样本量够（n={loc_t.get('paraphrase', {}).get('n', 0)}）",
          "样本太少的话这个差值没有意义")

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
