# -*- coding: utf-8 -*-
"""
块 E5 · **评测集自己的自检**（零模型 / 零依赖 / 秒级）。

    python scripts/eval_eval_sets.py

为什么评测集也要自检：
    造尺子的人自己不校准，后面所有数字都是假的。
    这套检查盯三件事：
      ① **机械可判** —— 每条 gold 都必须是能程序判定对错的（不是"让 LLM 打分"）
      ② **不重复**   —— 同一句话不能出现两次（凑数）
      ③ **覆盖够**   —— 每份药、每种意图都要有题，不能全压在一两个药上
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
EVAL = PROJECT / "data" / "eval"

_fails: list[str] = []


def check(cond, label, detail=""):
    print(f"   {'[OK]' if cond else '[!!]'} {label}" + (f"  —— {detail}" if detail else ""))
    if not cond:
        _fails.append(label)
    return cond


def load(name):
    p = EVAL / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    print("=" * 78)
    print("块 E5 · 评测集自检")
    print("=" * 78)

    man = load("eval_manifest.json")
    if not man:
        print("❌ 没有 eval_manifest.json —— 先跑 scripts/build_eval_sets.py")
        return 1
    print(f"\n语料：{man['corpus']['drugs']} 份药 ｜ {man['corpus']['sections']} 节")
    print(f"合计：{man['total_rows']} 条\n")

    sets = {k: load(v["file"]) for k, v in man["sets"].items() if v.get("file")}
    drugs = None

    # ============================================================ ① 规模
    # ⚠️ 阈值**从清单里的限量表读**，不写死。
    #    写死的后果实测过：生成器把 352 条精简到 170 条之后，
    #    校验器还在按 v6 时代的"≥80 / ≥40 / ≥15"判，当场报了 8 项失败 ——
    #    **两边各说各话**。阈值只有一个来源，改一处就够。
    print("① 五类评测集都在，条数达到清单里定的量")
    caps = man.get("caps") or {}
    for k, v in man["sets"].items():
        d = sets.get(k)
        n = d["n"] if d else 0
        want = caps.get(k, v.get("n", 0))
        check(n >= want, f"{k:16s} {n:4d} 条（限量 {want}）", v["tests"])

    # ============================================================ ② 机械可判
    print("\n② 每条 gold 都是**机械可判**的（这是「不靠 LLM 当裁判」的底线）")

    loc = sets["section_locate"]["rows"]
    bad = [r for r in loc if not r.get("gold_loinc") or not re.fullmatch(r"\d{5}-\d", r["gold_loinc"])]
    check(not bad, f"章节定位：gold_loinc 全是合法 LOINC 码", f"{len(bad)} 条不合规")

    bnd = sets["boundary"]["rows"]
    allowed_actions = {"refuse_guard", "refuse_escalate", "refuse_no_source", "refuse_no_evidence"}
    bad = [r for r in bnd if r.get("expect_action") not in allowed_actions]
    check(not bad, "边界集：expect_action 全在允许集合里", f"越界取值 {len(bad)} 条")

    ret = sets["retrieval"]["rows"]
    bad = [r for r in ret if "#" not in (r.get("gold_cite_key") or "")]
    check(not bad, "检索集：gold_cite_key 全是 setid#loinc 格式", f"{len(bad)} 条不合规")

    itt = sets["intent"]["rows"]
    bad = [r for r in itt if not r.get("referent")]
    check(not bad, "意图集：每条都有 referent（指代对象）", f"{len(bad)} 条缺")

    e2e = sets["e2e"]["rows"]
    bad = [r for r in e2e if len(r.get("turns", [])) < 2]
    check(not bad, "端到端集：每组至少 2 轮", f"{len(bad)} 组不足")

    # ============================================================ ③ 不重复
    print("\n③ 没有重复题（凑数会稀释指标）")
    for k, d in sets.items():
        rows = d["rows"]
        if k == "e2e":
            keys = [tuple(r["turns"]) for r in rows]
        elif k == "intent":
            keys = [tuple(r["turns"]) for r in rows]
        else:
            keys = [r["question"] for r in rows]
        dup = [x for x, c in Counter(keys).items() if c > 1]
        check(not dup, f"{k:16s} 无重复（{len(keys)} 条）",
              f"重复 {len(dup)} 组" if dup else "")

    # ============================================================ ④ 覆盖
    #
    # ⚠️ 这里**不假装"每份药都有题"**。限量之后做不到，也不该要求：
    #    50 条题铺不满 50 份药。要断言的是"**没有全压在一两个药上**"这个下限。
    #    真实覆盖数照实打出来 —— 数字小就是小，别用"✅"糊过去。
    MIN_DRUG_COVER = 8
    n_corpus = man["corpus"]["drugs"]
    print(f"\n④ 覆盖度：不能全压在一两个药上（语料共 {n_corpus} 份药）")
    for k in ("section_locate", "retrieval", "intent", "e2e"):
        d = sets[k]
        # ⚠️ intent 集里"药名"存在 `referent` 字段（它测的就是指代消解），
        #    别的集存在 `drug`。这里要兼容两种，不然自检自己会崩。
        by_drug = Counter(r.get("drug") or r.get("referent") for r in d["rows"])
        mn, mx = min(by_drug.values()), max(by_drug.values())
        n_d = len(by_drug)
        check(n_d >= min(MIN_DRUG_COVER, n_corpus),
              f"{k:16s} 覆盖 {n_d:2d} / {n_corpus} 份药（下限 {MIN_DRUG_COVER}）",
              f"分布 {mn}–{mx} 条/药")
        if drugs is None:
            drugs = sorted(by_drug)

    by_intent = Counter(r["intent"] for r in loc)
    check(len(by_intent) >= 4, f"章节定位：意图覆盖 {len(by_intent)} 种",
          f"{dict(by_intent)}")
    if len(by_intent) < 5:
        print("      ⚠️ 已知缺口：`interaction` 类题偏薄 —— 很多药的相互作用实体"
              "抽取返回 0（不是限量限没的，是抽取抽不出来）")

    # ============================================================ ⑤ 边界覆盖
    print("\n⑤ 边界集四类都有")
    cls = Counter(r["boundary_class"] for r in bnd)
    for c in ("out_of_scope", "beyond_label", "unknown_drug", "missing_section"):
        check(cls.get(c, 0) >= 5, f"{c:18s} {cls.get(c,0):3d} 条")

    # ============================================================ ⑥ 检索两档
    print("\n⑥ 检索集两档：数量相当 **而且成对**（差值是「语义检索的贡献」）")
    tier = Counter(r["tier"] for r in ret)
    check(tier.get("direct", 0) >= 10, f"direct    {tier.get('direct',0):3d} 条")
    check(tier.get("paraphrase", 0) >= 10, f"paraphrase{tier.get('paraphrase',0):4d} 条")

    # ⭐ 成对性 —— 这条是 2026-09-17 补的，补它是因为踩过一次**静默的错误**：
    #    原来 direct 档只从「相互作用/禁忌」生成、paraphrase 档覆盖全部 5 个意图，
    #    两档**根本不是同一批题**。于是 `direct − paraphrase` 这个头条指标
    #    一直在比较不同的意图构成 —— 两边都算出个像样的数，**没人会怀疑**。
    #    现在要求：同一道题必须同时有 direct 和 paraphrase 两种问法。
    pairs: Dict[Any, set] = {}
    for r in ret:
        pairs.setdefault((r["drug"], r["intent"], r["entity"]), set()).add(r["tier"])
    complete = sum(1 for v in pairs.values() if len(v) == 2)
    check(complete == len(pairs),
          f"成对性：{complete}/{len(pairs)} 道题两档齐全",
          "⚠️ 不齐的话 direct−paraphrase 会被「题不一样」污染")
    n_half = len(pairs) - complete
    if n_half:
        print(f"      ❌ {n_half} 道题只有单档 —— 差值不可信")

    # ============================================================ 抽样
    print("\n⑦ 抽样（肉眼确认不是垃圾）")
    for k in ("section_locate", "boundary", "retrieval", "intent", "e2e"):
        r = sets[k]["rows"][0]
        if k == "intent":
            q = "  →  ".join(r["turns"])
            gold = f"{r['anaphor']} = {r['referent']}"
        elif k == "e2e":
            q = "  →  ".join(r["turns"][:2])
            gold = r["scenario"]
        else:
            q = r["question"]
            gold = r.get("gold_section") or r.get("boundary_class") or r.get("gold_cite_key", "")[:24]
        print(f"   [{k}] {q[:74]}")
        print(f"           gold = {gold}")

    print("\n" + "=" * 78)
    if _fails:
        print(f"❌ 评测集自检未通过（{len(_fails)} 项）：")
        for f in _fails:
            print("   -", f)
        return 1
    total = sum(d["n"] for d in sets.values())
    print(f"✅ 评测集自检全绿 —— {total} 条，全部机械可判、无重复、覆盖达标")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
