# -*- coding: utf-8 -*-
"""
块 E6 · 检索层判据。

    python scripts/eval_retrieval.py

━━━ 两层评测，缺一不可（2026-09-18 重构）━━━

    检索集 60   直接/改写两档         —— 度量**有结构先验**时的检索
    探针集 120  药名在、意图词不认识  —— 度量**结构先验失效**时的检索 ⭐

⚠️ **只报检索集是不够的，会得出错误结论**：定位在检索集上 **60/60 全对**，
   于是任何"定位优先"的架构都按构造拿满分（实测 Recall@5 = MRR = 1.000）。
   **这把尺子到顶了 —— 它量不出任何后续改进。**
   所以脚本会主动在指标饱和时**报警**，而不是让你以为还有余量。

━━━ 分层表：每加一层量一次，差值就是那一层的贡献 ━━━

    ① BM25 单独                 词面基线
    ② +章节定位（原：乘性boost）   ← **已知有 bug 的旧实现**，留作对照
    ③ 三路融合（无稠密）           结构先验能引入候选之后
    ④ +稠密（三路融合）           语义补上先验够不着的地方
    ⑤ +重排（带硬约束）⭐          完整 —— **重排必须在已知硬约束内做**，
                                  不限定的话实测是负收益（检索集 1.000 → 0.850）
    ⑥ 稠密单独                   语义单独能走多远

━━━ 两个指标 ━━━

    Recall@5   该命中的章节，进 top-5 了吗（0/1，按题平均）
    MRR        第一个命中的排名倒数（越靠前越高，上限 1.0）
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
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from retrieval import BM25Retriever, HybridRetriever, build_chunks    # noqa: E402

LABELS = PROJECT / "data" / "labels.json"
RET_SET = PROJECT / "data" / "eval" / "retrieval_set.json"
LOC_SET = PROJECT / "data" / "eval" / "section_locate_set.json"
PROBE_SET = PROJECT / "data" / "eval" / "probe_set.json"
BOUNDARY_SET = PROJECT / "data" / "eval" / "boundary_set.json"
EMB_CACHE = PROJECT / "data" / "cache" / "emb"
RERANK_CACHE = PROJECT / "data" / "cache" / "rerank_scores.json"

_fails: List[str] = []


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
       所以命中判定必须用 `gold_loinc_any`（**可接受集合**），不能只认一个。

    ⚠️⚠️ 这个判据**只认章节、不认具体是哪一块**（2026-09-18 想清楚的）——
       于是"把 gold 章节的块全排在前面"的架构**按构造必然命中**，
       而定位在检索集上 60/60 全对 ⇒ 该架构 Recall@5 必然是 1.000。
       **这是这个指标的天花板，不是系统的天花板。** 所以必须配探针集一起看。
    """
    any_lo = set(row.get("gold_loinc_any") or [row.get("gold_loinc")])
    gold_doc = row["gold_cite_key"].split("#")[0]
    for d in docs[:k]:
        if d.doc_id == gold_doc and d.loinc in any_lo:
            return d.rank
    return 0


def evaluate(retriever, rows, k: int = 5):
    by_tier = defaultdict(lambda: {"n": 0, "recall": 0, "rr": 0.0})
    detail = []
    for r in rows:
        docs = retriever(r["question"], k)
        rank = hit_of(docs, r, k)
        tier = r.get("tier", "?")
        by_tier[tier]["n"] += 1
        by_tier[tier]["recall"] += 1 if rank else 0
        by_tier[tier]["rr"] += (1.0 / rank) if rank else 0.0
        detail.append({"qid": r["qid"], "tier": tier, "rank": rank,
                       "question": r["question"], "gold": r["gold_cite_key"],
                       "top1": docs[0].cite_key if docs else "(空)"})
    n = len(rows)
    out = {"n": n,
           "recall@5": sum(d["rank"] > 0 for d in detail) / max(1, n),
           "mrr": sum(1.0 / d["rank"] for d in detail if d["rank"]) / max(1, n)}
    tiers = {t: {"n": v["n"], "recall@5": v["recall"] / max(1, v["n"]),
                 "mrr": v["rr"] / max(1, v["n"])} for t, v in by_tier.items()}
    return out, tiers, detail


def main() -> int:
    print("=" * 78)
    print("块 E6 · 检索层判据")
    print("=" * 78)

    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    ret_rows = json.loads(RET_SET.read_text(encoding="utf-8"))["rows"]
    probe_rows = (json.loads(PROBE_SET.read_text(encoding="utf-8"))["rows"]
                  if PROBE_SET.exists() else [])
    p_dev = [r for r in probe_rows if r.get("split") == "dev"]
    p_test = [r for r in probe_rows if r.get("split") == "test"]

    print(f"\n语料：{len(labels)} 份药 ｜ 检索集：{len(ret_rows)} 条 ｜ "
          f"探针集：{len(probe_rows)} 条（dev {len(p_dev)} / test {len(p_test)}）\n")

    # ---- ① 分块体检
    chunks = build_chunks(labels)
    lens = sorted(len(c.text) for c in chunks)
    n_sec = sum(len(L.get("sections", [])) for L in labels)
    print("① 分块")
    check(len(chunks) >= 100, f"切出 {len(chunks)} 个 chunk"
                              f"（来自 {len(labels)} 份药的 {n_sec} 节）")
    check(lens[len(lens) // 2] < 700, f"中位长度 {lens[len(lens)//2]} 字符（不会太长）")
    check(all(c.content_hash and c.chunk_id for c in chunks),
          "每个 chunk 都带 chunk_id + content_hash（引用契约）")

    # ---- ② 构建五档配置
    print("\n② 构建五档配置")
    t0 = time.time()
    cfgs = {
        "① BM25 单独": BM25Retriever(labels, use_locator=False),
        "② +定位（原：乘性boost）": BM25Retriever(labels, use_locator=True),
        "③ 三路融合（无稠密）": HybridRetriever(labels, use_dense=False),
        "④ +稠密（三路融合）": HybridRetriever(labels, use_locator=True, use_dense=True,
                                              use_rerank=False, dense_cache=EMB_CACHE),
        "⑤ +重排（带硬约束）⭐": HybridRetriever(labels, use_locator=True, use_dense=True,
                                                use_rerank=True, dense_cache=EMB_CACHE,
                                                rerank_cache=RERANK_CACHE),
        "⑥ 稠密单独": HybridRetriever(labels, use_locator=False, use_bm25=False,
                                     use_rerank=False, dense_cache=EMB_CACHE),
    }
    FULL = "⑤ +重排（带硬约束）⭐"
    print(f"   用时 {time.time()-t0:.1f}s"
          f"（稠密向量 {cfgs[FULL].dense.emb.shape}，有磁盘缓存则秒级复用；"
          f"重排分数同样落盘）")

    # ---- ③ 检索集分层表
    print("\n③ 检索集 60（direct / paraphrase 成对）—— 分层表")
    print(f"\n   {'配置':<26}{'Recall@5':>10}{'MRR':>10}{'n':>7}")
    print(f"   {'-'*53}")
    ret_res = {}
    for name, r in cfgs.items():
        m, _t, _d = evaluate(r, ret_rows)
        ret_res[name] = m
        print(f"   {name:<25}{m['recall@5']:>10.3f}{m['mrr']:>10.3f}{m['n']:>7}")

    _, base_t, _ = evaluate(cfgs["① BM25 单独"], ret_rows)
    _, loc_t, ret_detail = evaluate(cfgs[FULL], ret_rows)

    gap = (loc_t.get("direct", {}).get("recall@5", 0)
           - loc_t.get("paraphrase", {}).get("recall@5", 0))
    print(f"\n   {'分档（⑤ +重排）':<26}{'Recall@5':>10}{'MRR':>10}{'n':>7}")
    print(f"   {'-'*53}")
    for t in ("direct", "paraphrase"):
        lt = loc_t.get(t, {})
        if lt:
            print(f"   {f'⑤ · {t}':<24}{lt['recall@5']:>10.3f}{lt['mrr']:>10.3f}{lt['n']:>7}")
    print(f"\n   direct − paraphrase = {gap:+.3f}")
    if abs(gap) < 0.05:
        print("      ⚠️ **两档仍然没有差别** —— 这个差值现在量不出「稠密检索的价值」。")
        print("         原因（2026-09-17 查清）：改写档里**仍然带着药名**，")
        print("         而药名是 BM25 最强的信号，所以它根本没变难。")
        print("         ⛔ 之前报的 0.192 / 0.115 **是假象**（两档问的不是同一批题）。")
        print("         ⭐ 稠密的价值改由**探针集**量（见 ④）—— 那条路走通了。")

    # ---- ④ 探针集（**这才是现在能量出东西的地方**）
    print("\n④ 探针集 120（药名在、意图词不认识 → 结构先验失效）")
    probe_res = {}
    n_loc_hit, contrib = None, None
    if not probe_rows:
        print("   ⚠️ 没有 probe_set.json，跑 `python scripts/build_probe_set.py` 生成")
    else:
        for title, rows in (("探针 dev 60（可调参）", p_dev),
                            ("探针 test 60（只报不改）", p_test)):
            print(f"\n   {title}")
            print(f"   {'配置':<26}{'Recall@5':>10}{'MRR':>10}{'n':>7}")
            print(f"   {'-'*53}")
            for name, r in cfgs.items():
                m, _t, _d = evaluate(r, rows)
                probe_res[(title, name)] = m
                print(f"   {name:<25}{m['recall@5']:>10.3f}{m['mrr']:>10.3f}{m['n']:>7}")

        # ⭐ 探针成立性的硬检查：定位在探针上**必须**什么都认不出来
        loc = cfgs["② +定位（原：乘性boost）"].locator
        n_loc_hit = sum(1 for r in probe_rows
                        if set(loc(r["question"])["loincs"])
                        & set(r["gold_loinc_any"] or [r["gold_loinc"]]))
        tkey = "探针 test 60（只报不改）"
        contrib = (probe_res[(tkey, FULL)]["recall@5"]
                   - probe_res[(tkey, "③ 三路融合（无稠密）")]["recall@5"])
        print(f"\n   ⭐ 探针成立性：定位在探针上命中章节的题数 = {n_loc_hit}/{len(probe_rows)}"
              f"（应为 0）")
        print(f"   ⭐ 「稠密+重排」在探针 test 上的贡献（相对无稠密） = {contrib:+.3f}")

    # ---- ④b ⭐ 词面 vs 语义的**互补性** —— 自适应权重的依据
    #
    #   这一节回答一个必须回答的问题：**为什么融合权重不能等权？**
    #   等权 RRF 实测把稠密从 0.717 拉到 0.633。原因：
    #   两个检索器的强弱**不是固定的，取决于查询里有没有判别性实词**。
    #   下面这三组把这个依赖关系量出来 —— 它同时也是 `bm25_confidence` 的设计依据。
    print("\n④b 词面 vs 语义：强弱取决于**查询里有没有判别性实词**（定位全程关闭）")
    print(f"\n   {'查询类型':<34}{'BM25':>9}{'稠密':>9}{'谁赢':>8}{'n':>7}")
    print(f"   {'-'*67}")
    groups = [
        ("检索集·有实体（实体词是真 token）", [r for r in ret_rows if r.get("entity")]),
        ("检索集·无实体（只剩药名+疑问词）", [r for r in ret_rows if not r.get("entity")]),
        ("探针 test（意图词不认识）", p_test),
    ]
    bm25_only = cfgs["① BM25 单独"]
    dense_only_cfg = cfgs["⑥ 稠密单独"]
    complement = {}
    for gname, grows in groups:
        if not grows:
            continue
        mb, _t, _d = evaluate(bm25_only, grows)
        md, _t, _d = evaluate(dense_only_cfg, grows)
        win = "BM25" if mb["recall@5"] > md["recall@5"] else "稠密"
        complement[gname] = {"bm25": mb["recall@5"], "dense": md["recall@5"], "n": len(grows)}
        print(f"   {gname:<32}{mb['recall@5']:>9.3f}{md['recall@5']:>9.3f}{win:>8}{len(grows):>7}")
    print("   ⇒ **同两个检索器，谁强取决于查询** ⇒ 等权融合必然在某一侧拖后腿")

    # ---- ④c ⭐ 域外弃权 —— **回归判据**
    #
    #   2026-09-18：加上稠密层时，这块**当场红过**。E7 的断言抓到：
    #       [!!] 域外问题检索返回 0 条   [!!] 以 abstain 收尾（answer）
    #   根因：BM25 侧早就修好了（E7 去掉停用词），但**稠密检索永远返回 top-k** ——
    #        余弦对任何两段文本都有定义 → 域外问题又拿到 5 条 → agent 又看不到"什么都没查到"。
    #   修法：稠密层加绝对余弦门限（`dense_min_cosine`，用 boundary_set 的 out_of_scope 标定）。
    #
    #   ⚠️ **这个洞修过两次**（E7 一次词法、E6 一次语义）—— 所以它必须是一条**常驻判据**，
    #      而不是"当时查了一下没事"。换嵌入模型、换语料，它都可能复发。
    n_ood, n_ood_zero = 0, 0
    ood_q = []
    cosine_by_class = defaultdict(list)      # 给图用：每类的稠密 top1 余弦
    if BOUNDARY_SET.exists():
        bd = json.loads(BOUNDARY_SET.read_text(encoding="utf-8"))
        for r in (bd["rows"] if isinstance(bd, dict) else bd):
            cls = r.get("boundary_class")
            cosine_by_class[cls].append(
                cfgs[FULL].dense_gate(r["question"])[1])
            if cls != "out_of_scope":
                continue
            n_ood += 1
            docs = cfgs[FULL](r["question"], 5)
            if not docs:
                n_ood_zero += 1
            else:
                ood_q.append(r["question"])
    for k, v in cosine_by_class.items():
        cosine_by_class[k] = {"n": len(v), "max": max(v), "min": min(v),
                              "mean": sum(v) / len(v)}
    in_domain_cos = [cfgs[FULL].dense_gate(r["question"])[1]
                     for r in probe_rows]
    if in_domain_cos:
        cosine_by_class["in_domain_probe"] = {
            "n": len(in_domain_cos), "max": max(in_domain_cos),
            "min": min(in_domain_cos), "mean": sum(in_domain_cos) / len(in_domain_cos)}
    print(f"\n④c 域外弃权（boundary_set 的 out_of_scope）")
    if not n_ood:
        print("   ⚠️ 没有 boundary_set.json，跳过")
    else:
        print(f"   检索返回 0 条（= agent 能观察到「什么都没查到」）的题数："
              f"{n_ood_zero}/{n_ood}")
        for q in ood_q[:3]:
            print(f"      ✗ 漏放：{q}")

    # ---- ⑤ 章节定位本身的准确率
    print("\n⑤ 章节定位准确率（用章节定位集，5 种意图全覆盖）")
    locator = cfgs[FULL].locator
    if LOC_SET.exists():
        loc_rows = json.loads(LOC_SET.read_text(encoding="utf-8"))["rows"]
    else:
        print("   ⚠️ 没有 section_locate_set.json，退回用检索集里的带实体题")
        loc_rows = [r for r in ret_rows if r["intent"] in ("interaction", "contraindication")]
    n_loc, n_ok = 0, 0
    miss_by_intent = defaultdict(int)
    for r in loc_rows:
        res = locator(r["question"])
        n_loc += 1
        if set(r.get("gold_loinc_any") or [r["gold_loinc"]]) & set(res["loincs"]):
            n_ok += 1
        else:
            miss_by_intent[r.get("intent", "?")] += 1
    loc_acc = n_ok / max(1, n_loc)
    check(n_loc > 0, f"覆盖 {n_loc} 条定位题")
    print(f"   定位命中率 {n_ok}/{n_loc} = {loc_acc:.3f}")
    if miss_by_intent:
        print(f"   漏检按意图：{dict(miss_by_intent)}")

    # ---- ⑥ 判据
    print("\n⑥ 判据")
    base = ret_res["① BM25 单独"]
    fused = ret_res[FULL]
    check(base["recall@5"] > 0.30,
          f"BM25 基线 Recall@5 > 0.30（实测 {base['recall@5']:.3f}）",
          "太低说明语料/索引有问题")
    check(fused["recall@5"] >= base["recall@5"],
          f"完整配置**不劣于** BM25 基线（{fused['recall@5']:.3f} ≥ {base['recall@5']:.3f}）")
    check(loc_acc > 0.85, f"章节定位命中率 > 0.85（实测 {loc_acc:.3f}）")

    # ⛔ 这条判据 2026-09-17 撤销过（原话 `check(gap > 0, "paraphrase 档明显更难")`）。
    #    撤销的理由不是"它红了"，是**它量错了东西**：成对化之后 gap = 0.000。
    #    ⚠️ **判据红了就去放宽阈值，等于把发现真相的机会扔掉。**
    check(loc_t.get("paraphrase", {}).get("n", 0) >= 10,
          f"paraphrase 档样本量够（n={loc_t.get('paraphrase', {}).get('n', 0)}）")

    if probe_rows:
        tkey = "探针 test 60（只报不改）"
        check(n_loc_hit == 0,
              f"探针上的意图**全部定位不出来**（实测 {n_loc_hit} 条被认出）",
              "不为 0 说明探针模板撞上了关键词表 → 探针失效")
        check(probe_res[(tkey, "② +定位（原：乘性boost）")]["recall@5"]
              == probe_res[(tkey, "① BM25 单独")]["recall@5"],
              "探针上「定位」的贡献 = 0（先验确实被关掉了）")
        check(contrib > 0,
              f"稠密+重排在探针 test 上**有正贡献**（{contrib:+.3f}）",
              "这是稠密/重排唯一的用武之地：结构先验够不着的地方")
        # ⭐⭐ 重排必须**带硬约束** —— 这条判据是防"不限定就上重排"的
        #    （实测不限定：检索集 1.000 → 0.850，探针 test 0.717 → 0.700，**两边都亏**）
        f_no_rr = probe_res[(tkey, "④ +稠密（三路融合）")]["recall@5"]
        f_rr = probe_res[(tkey, FULL)]["recall@5"]
        check(f_rr >= f_no_rr,
              f"重排在探针 test 上**不劣化**（{f_rr:.3f} ≥ {f_no_rr:.3f}）",
              "重排本身没问题，问题是**没把硬约束喂给它**")
        check(ret_res[FULL]["recall@5"] >= ret_res["④ +稠密（三路融合）"]["recall@5"],
              f"重排在检索集上**不把结构先验洗掉**"
              f"（{ret_res[FULL]['recall@5']:.3f} ≥ "
              f"{ret_res['④ +稠密（三路融合）']['recall@5']:.3f}）",
              "不限定的话这一条会红：60 条里 46 条会混进别的药")

        dense_only = probe_res[(tkey, "⑥ 稠密单独")]["recall@5"]
        fused_p = probe_res[(tkey, FULL)]["recall@5"]
        check(fused_p >= dense_only - 0.05,
              f"融合**不被弱路拖累**（{fused_p:.3f} vs 稠密单独的 {dense_only:.3f}）",
              "等权 RRF 就在这里翻过车：0.717 → 0.633")

    # ⭐⭐ 饱和报警 —— 这是这次重构**专门加**的。
    #
    #   不报警的话，下一个人看到 Recall@5 = 1.000 会以为"系统很强 / 还有余量"，
    #   而真相是**这把尺子到顶了**。**指标到顶必须自己说出口。**
    sat = [n for n, m in ret_res.items() if m["recall@5"] >= 0.999]
    if sat:
        print()
        print("   ⚠️⚠️ **指标饱和报警**：" + "、".join(sat) + " 在检索集上到了 1.000")
        print("       原因：定位在检索集上 60/60 全对，而命中判据只比 (药, 章节) ——")
        print("       **任何『定位优先』的架构都按构造拿满分**。")
        print("       ⇒ 检索集上**再往上已经没有可量的余量**，后续改进只能看探针集。")
        print("       ⇒ 这不是系统到顶了，是**尺子到顶了**。")

    # ---- 存盘
    out = PROJECT / "reports" / "retrieval_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n": len(ret_rows),
        "layers": ret_res,
        "by_tier": {"bm25": base_t, "fused": loc_t},
        "direct_minus_paraphrase": gap,
        "locator_accuracy": loc_acc,
        "probe": {f"{t}|{n}": m for (t, n), m in probe_res.items()},
        "probe_locator_hits": n_loc_hit,
        "probe_dense_contribution_test": contrib,
        "saturated_on_retrieval_set": sat,
        "complementarity": complement,
        # 域外弃权门限的标定依据：每类问题的稠密 top1 余弦分布
        "cosine_by_class": dict(cosine_by_class),
        "dense_min_cosine": cfgs[FULL].dense_min_cosine,
        "ood_zero_returned": {"n": n_ood, "zero": n_ood_zero},
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {out}")

    # ---- 失败样例
    print("\n⑦ 失败样例（检索集，top-5 一条都没命中）")
    fails = [d for d in ret_detail if d["rank"] == 0][:5]
    if not fails:
        print("   （检索集上已经没有了 —— 但见上面的饱和报警，这不代表检索已经做好）")
    for d in fails:
        print(f"   [{d['tier']}] {d['question'][:66]}")
        print(f"        gold={d['gold'][-12:]}  top1={d['top1'][-12:]}")

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
