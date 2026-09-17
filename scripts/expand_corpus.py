# -*- coding: utf-8 -*-
"""
块 E5 · 扩语料 6 → 50 份药。

    python scripts/expand_corpus.py [--target 50] [--dry-run]

━━━ 为什么必须扩（不是为了"看起来数据多"）━━━

    E8 要做的**证据覆盖度**算法，本质是判断"我收集到的证据够不够回答这个问题"。
    在只有 6 份药的语料里，这个问题**几乎不存在** —— 随手一查就碰到对的药。
    **干扰项不够多，覆盖度就没有意义。**

    所以扩语料是 E8 的前置条件，不是"堆数据"。

━━━ 🔴 一条必须守住的红线：只增不改 ━━━

    已有的 6 份药 **一个字都不动**。

    原因：评测集全部按**现有 setid** 写死 ——
        · `data/eval/retrieval_set.json`   46 条，每条带 gold_cite_key
        · `data/questions_v1.json`         15 道手写题，gold 指向具体章节
        · `data/eval/e2e_set.json`         26 条多轮脚本
    DailyMed 的 setid 是**版本相关的 UUID**：重抓一次就可能换一个。
    重抓 = 整个评测集当场报废，而且**失败是静默的**（gold 指向不存在的章节 → 该题恒为失败）。

    （这个坑 E4 已经踩过一次：手写题 q06 的 gold 指向了 metformin 没有的章节。）
    所以本脚本的默认行为是**只补缺的，绝不覆盖已有的**。

━━━ 另一个会踩的点：**药名 ≠ 一份说明书** ━━━

    同一个药名会命中一堆厂家（「omeprazole」能返回 20 份不同厂家的说明书）。
    挑法沿用 E3：**优先要有 DRUG INTERACTIONS 章的**（那是处方药的特征），
    否则这份说明书抽不出我们关心的 8 节，进语料只会占位置。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from scripts.fetch_labels import (get, find_setid, parse_sections, WANTED)   # noqa: E402

OUT = PROJECT / "data" / "labels.json"

# fetch_labels 里的 BASE 是查询用的基址；取单份 XML 的路径前缀单列出来 ——
# 两个字符串长得太像，混用一次就会静默抓到错的东西
BASE_SPL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls"

# 候选药名：常见处方药（要有 DRUG INTERACTIONS 章）。
# ⚠️ 顺序即抓取顺序 —— 固定顺序才能让"抓到哪 50 份"可复现。
CANDIDATES = [
    # 心血管
    "amlodipine", "losartan", "metoprolol", "simvastatin", "rosuvastatin",
    "carvedilol", "diltiazem", "verapamil", "digoxin", "amiodarone",
    "hydrochlorothiazide", "furosemide", "spironolactone", "clopidogrel",
    # 内分泌 / 代谢
    "glipizide", "sitagliptin", "empagliflozin", "insulin glargine",
    "allopurinol", "colchicine", "methotrexate",
    # 精神 / 神经
    "sertraline", "fluoxetine", "escitalopram", "bupropion", "venlafaxine",
    "duloxetine", "trazodone", "quetiapine", "risperidone", "olanzapine",
    "lithium", "lamotrigine", "levetiracetam", "phenytoin", "valproic acid",
    "gabapentin", "zolpidem", "alprazolam", "lorazepam", "cyclobenzaprine",
    # 消化
    "pantoprazole", "esomeprazole", "famotidine", "ondansetron",
    # 抗感染
    "azithromycin", "amoxicillin", "cephalexin", "doxycycline",
    "ciprofloxacin", "fluconazole", "acyclovir", "valacyclovir",
    "hydroxychloroquine",
    # 呼吸 / 免疫
    "montelukast", "prednisone", "albuterol", "fluticasone",
    # 镇痛
    "tramadol", "oxycodone", "naproxen", "ibuprofen", "celecoxib",
    # 泌尿 / 其他
    "tamsulosin", "finasteride", "sildenafil", "levothyroxine",
]


def load_existing() -> dict:
    if not OUT.exists():
        return {"labels": []}
    return json.loads(OUT.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=50, help="语料目标份数")
    ap.add_argument("--dry-run", action="store_true", help="只列计划，不联网")
    ap.add_argument("--pause", type=float, default=0.35, help="每次抓取之间的间隔（秒）")
    args = ap.parse_args()

    print("=" * 78)
    print("块 E5 · 扩语料（只增不改）")
    print("=" * 78)

    data = load_existing()
    have = {L["drug"]: L for L in data["labels"]}
    n_have_existing = len(have)
    print(f"\n现有语料：{n_have_existing} 份药 —— "
          f"{', '.join(sorted(have))}")
    print(f"🔒 这 {n_have_existing} 份**一个字都不动**（评测集按它们的 setid 写死了）")

    need = max(0, args.target - len(have))
    todo = [d for d in CANDIDATES if d not in have][:need]
    print(f"\n目标 {args.target} 份 → 还要抓 {need} 份，候选 {len(todo)} 个")

    if args.dry_run:
        for d in todo:
            print("   ", d)
        return 0
    if not todo:
        print("\n已经够了，不用抓。")
        return 0

    added, failed, skipped = [], [], []
    for i, drug in enumerate(todo, 1):
        # 达到目标就停（有些药抓不到，所以按"成功数"而不是"尝试数"计）
        if len(have) + len(added) >= args.target:
            break
        print(f"\n[{i}/{len(todo)}] {drug}")
        try:
            sid = find_setid(drug)
        except Exception as e:                                  # noqa: BLE001
            print(f"   ✗ 找 setid 失败：{type(e).__name__}: {e}")
            failed.append((drug, f"{type(e).__name__}"))
            continue
        if not sid:
            print("   ✗ 没有可用结果")
            failed.append((drug, "no result"))
            continue
        try:
            xml = get(f"{BASE_SPL}/{sid}.xml").decode("utf-8", "replace")
        except Exception as e:                                  # noqa: BLE001
            print(f"   ✗ 取全文失败：{e}")
            failed.append((drug, "fetch xml"))
            continue

        secs = parse_sections(xml)
        # ⚠️ 抽到的章节太少（比如只有 2 节）说明这份说明书结构不合我们的语料规格，
        #    放进去只会稀释检索，而且大概率不是处方药标签
        if len(secs) < 3:
            print(f"   ✗ 只抽到 {len(secs)} 节，太薄，跳过")
            skipped.append((drug, f"only {len(secs)} sections"))
            continue

        lo = sorted(s["loinc"] for s in secs)
        print(f"   ✓ setid={sid[:18]}...  {len(secs)} 节  {lo}")
        added.append({"drug": drug, "setid": sid, "sections": secs})
        time.sleep(args.pause)

    data["labels"].extend(added)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    n_sec = sum(len(L["sections"]) for L in data["labels"])
    print("\n" + "=" * 78)
    print(f"✅ 语料：{n_have_existing} → {len(data['labels'])} 份药，共 {n_sec} 个章节")
    print(f"   已存 → {OUT}")
    if failed:
        print(f"   ⚠️ 抓失败 {len(failed)} 个：{', '.join(f'{d}({w})' for d, w in failed)}")
    if skipped:
        print(f"   ⚠️ 太薄跳过 {len(skipped)} 个：{', '.join(f'{d}({w})' for d, w in skipped)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
