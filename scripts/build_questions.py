# -*- coding: utf-8 -*-
"""
E4 第 2 步 · 自建问题集 v1（15 题）+ 每题标 gold 证据集。

和 ToolHorizon 的扩题器不同：那边是**程序化造 800 道**，这边**只能手写**——
因为「回答这个问题该引哪几节说明书」需要读内容才能判，程序判不了。
**这正是「垂类做深」的实质**：通用 benchmark 做不到这个标注粒度。

三档难度：
    single_hop   1 节能答（对照 baseline 用）
    multi_hop    必须综合同一份药的 2 节
    cross_drug   涉及两种药，要各引一节

⚠️ gold 里的每一节，都必须真实存在于 data/labels.json —— 脚本会逐条校验。

跑法（纯 CPU）：
  cd 代码库/projects/EvidenceTrace
  PYTHONIOENCODING=utf-8 python scripts/build_questions.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
LABELS = PROJECT / "data" / "labels.json"
OUT = PROJECT / "data" / "questions_v1.json"

# setid 前缀（脚本会展开成完整 setid）
SID = {
    "warfarin": "ecd271ad", "metformin": "ded33248", "atorvastatin": "61c29f52",
    "lisinopril": "9470b102", "levothyroxine": "6f4bc4be", "omeprazole": "54612d26",
}

# LOINC 短名，写起来清楚点
LOINC = {
    "IND": "34067-9",       # Indications and Usage
    "CI": "34070-3",        # Contraindications
    "DI": "34073-7",        # Drug Interactions
    "DOSE": "34068-7",      # Dosage and Administration
    "WP": "43685-7",        # Warnings and Precautions
    "WARN": "34071-1",      # Warnings（老版标签用这个，如 metformin）
    "PREC": "34069-5",      # Precautions
}

QUESTIONS = [
    # ---------------- single_hop (5)
    dict(qid="q01", diff="single_hop",
         question="Can I take metformin if I have kidney disease?",
         gold=[("metformin", "CI", "禁忌节明确写了肾功能不全者禁用")]),
    dict(qid="q02", diff="single_hop",
         question="What conditions is lisinopril approved to treat?",
         gold=[("lisinopril", "IND", "适应症节列了高血压/心衰/心梗")]),
    dict(qid="q03", diff="single_hop",
         question="Is warfarin safe to take during pregnancy?",
         gold=[("warfarin", "CI", "禁忌节第一条就是妊娠")]),
    dict(qid="q04", diff="single_hop",
         question="What is the usual dose of omeprazole for an active duodenal ulcer?",
         gold=[("omeprazole", "DOSE", "用法用量节")]),
    dict(qid="q05", diff="single_hop",
         question="Who should NOT be given levothyroxine?",
         gold=[("levothyroxine", "CI", "禁忌：未纠正的肾上腺皮质功能不全")]),

    # ---------------- multi_hop (7)：必须综合同一份药的两节
    dict(qid="q06", diff="multi_hop",
         question=("My patient has type 2 diabetes and impaired kidney function. "
                   "Can they still take metformin, and what is the main danger if they do?"),
         gold=[("metformin", "CI", "肾功能不全禁用"),
               ("metformin", "WARN", "乳酸酸中毒是主要风险（该标签用 34071-1 而非 43685-7）")]),
    dict(qid="q07", diff="multi_hop",
         question=("I take atorvastatin. What are the serious muscle-related risks, "
                   "and which co-administered drugs make them more likely?"),
         gold=[("atorvastatin", "WP", "肌病/横纹肌溶解"),
               ("atorvastatin", "DI", "哪些药会加重")]),
    dict(qid="q08", diff="multi_hop",
         question=("I am pregnant. Can I take warfarin, and is there any exception "
                   "for women with a mechanical heart valve?"),
         gold=[("warfarin", "CI", "妊娠禁用 + 机械瓣例外"),
               ("warfarin", "WP", "胎儿危害")]),
    dict(qid="q09", diff="multi_hop",
         question=("I have been prescribed lisinopril. Which other medicines should I "
                   "avoid or use cautiously, and are any of them contraindicated outright?"),
         gold=[("lisinopril", "DI", "NSAID/利尿剂/锂盐等"),
               ("lisinopril", "CI", "与阿利吉仑合用是禁忌")]),
    dict(qid="q10", diff="multi_hop",
         question=("My patient is starting omeprazole. Which other drugs interact with it, "
                   "and what is omeprazole actually approved to treat?"),
         gold=[("omeprazole", "DI", "抗逆转录病毒药等"),
               ("omeprazole", "IND", "适应症：溃疡/H.pylori/反流")]),
    dict(qid="q11", diff="multi_hop",
         question=("A patient has a history of angioedema. Can they take lisinopril, "
                   "and what is the risk of adding an NSAID for their arthritis?"),
         gold=[("lisinopril", "CI", "血管性水肿史禁用"),
               ("lisinopril", "DI", "NSAID → 肾损伤 + 降压效果下降")]),
    dict(qid="q12", diff="multi_hop",
         question=("I take metformin and my doctor wants to add glyburide. "
                   "What does the label say about combining them, and how is that dosed?"),
         gold=[("metformin", "DI", "与格列本脲的相互作用研究"),
               ("metformin", "DOSE", "合并磺脲类的用法")]),

    # ---------------- cross_drug (3)：涉及两种药
    dict(qid="q13", diff="cross_drug",
         question=("I am on both atorvastatin and lisinopril. What interactions does each "
                   "label warn about?"),
         gold=[("atorvastatin", "DI", "他汀侧：地高辛/避孕药等"),
               ("lisinopril", "DI", "普利侧：NSAID/锂盐等")]),
    dict(qid="q14", diff="cross_drug",
         question=("My patient has both diabetes and high cholesterol and is asking whether "
                   "metformin and atorvastatin can be taken together. What should be checked?"),
         gold=[("metformin", "CI", "先看肾功能能不能用二甲双胍"),
               ("atorvastatin", "WP", "他汀侧的肌肉风险")]),
    dict(qid="q15", diff="cross_drug",
         question=("I take warfarin together with omeprazole. What bleeding-related concerns "
                   "and interactions should I watch for?"),
         gold=[("warfarin", "DI", "出血风险 + CYP 相关"),
               ("omeprazole", "DI", "PPI 侧的相互作用")]),
]


def main() -> int:
    print("=" * 76)
    print("E4 · 自建问题集 v1（15 题）")
    print("=" * 76)

    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    setid_of = {L["drug"]: L["setid"] for L in labels}
    # (drug, loinc) 是否存在
    have = {(L["drug"], s["loinc"]) for L in labels for s in L["sections"]}

    out, errs = [], []
    for q in QUESTIONS:
        gold = []
        for drug, lk, why in q["gold"]:
            loinc = LOINC[lk]
            if (drug, loinc) not in have:
                errs.append(f"{q['qid']}: gold 指向不存在的章节 {drug}/{loinc}")
                continue
            gold.append({"drug": drug,
                         "setid": setid_of[drug],
                         "loinc": loinc,
                         "cite_key": f"{setid_of[drug]}#{loinc}",
                         "why": why})
        out.append({"qid": q["qid"], "question": q["question"],
                    "difficulty": q["diff"], "gold": gold})

    # ---------------- 报告
    print(f"\n共 {len(out)} 题\n")
    dc = Counter(q["difficulty"] for q in out)
    gc = Counter(len(q["gold"]) for q in out)
    print(f"难度分布： {dict(dc)}")
    print(f"gold 节数： {dict(sorted(gc.items()))}  ← 几节能答")
    print()
    print("-" * 76)
    for q in out:
        print(f"  {q['qid']}  [{q['difficulty']:<10}] {q['question'][:60]}")
        for g in q["gold"]:
            print(f"        → {g['drug']:<14} {g['loinc']}  {g['why']}")
        print()

    # ---------------- 校验
    if errs:
        print("❌ 校验未通过：")
        for e in errs:
            print("   -", e)
        return 1

    # 每题必须至少 1 条 gold，且 cite_key 唯一
    for q in out:
        if not q["gold"]:
            print(f"❌ {q['qid']} 没有 gold")
            return 1
        keys = [g["cite_key"] for g in q["gold"]]
        if len(keys) != len(set(keys)):
            print(f"❌ {q['qid']} gold 里有重复章节")
            return 1

    OUT.write_text(json.dumps({"n": len(out), "questions": out},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 76)
    print(f"✅ 15 题全部通过校验（gold 都指向真实存在的章节）")
    print(f"   single_hop {dc.get('single_hop',0)} ｜ multi_hop {dc.get('multi_hop',0)}"
          f" ｜ cross_drug {dc.get('cross_drug',0)}")
    print(f"   已存 → {OUT}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
