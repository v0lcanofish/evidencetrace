# -*- coding: utf-8 -*-
"""
块 E6 · 探针集 —— **让结构先验失效的题**。

    python scripts/build_probe_set.py

━━━ 为什么必须造这个集 ━━━

2026-09-18 实测：现有检索集（60 条）里，章节定位**60/60 全对**。
后果 ⇒ **任何"定位优先"的架构都按构造拿满分**（实测 C 方案 Recall@5 = 1.000 / MRR = 1.000）。
也就是说 **Recall@5 这张表已经不量检索质量了，它量的是定位准确率。**

稠密检索加进来也量不出贡献 —— 不是它没用，是**这把尺子量不到**。

━━━ 探针怎么设计 ━━━

结构先验 = 「识别出药 + 识别出意图 → 查哪一节」。
**药名仍然给出来**（否则就不是检索问题，是指代消解问题，那是 E10 的活），
**只把意图词换成关键词表里没有的说法** —— 于是：

    drug 识别得到 → intent 识别不到 → loincs=[] → 先验失效 → 退回纯检索

⭐ 关键：**度量与检索集完全一致**（Recall@5 over (doc, loinc)），
   唯一变的是"先验在不在"。这样两张表的数字可以直接比。

━━━ dev / test 协议（回答"那我把词表补全不就行了？"）━━━

这是个一定会被问到的问题，所以要**用实验回答，不能嘴上回答**：

    dev  档  允许拿它去**补 LOCATE_RULES 关键词**（模拟"人工枚举"）
    test 档  **换一套完全不同的问法**，补完词表后不许再动

    如果补词在 dev 上救回来了、在 test 上没救回来
    → **枚举法的天花板是量出来的，不是猜的**。

⚠️ 诚实边界：模板是手写的，不是真人问法；每类意图只有 4 种说法，
   覆盖面比真实用户窄。**这个集的用途是"量先验失效时的检索能力"，不是"模拟真实分布"。**
"""

import sys

# ⚠️ Windows 中文控制台默认 GBK：不设这个，print("⭐") 会抛 UnicodeEncodeError
#    → **判据崩在半路，红绿一个字都读不到**（2026-09-18 实测 eval_retrieval.py）。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from retrieval import SectionLocator                                  # noqa: E402

LABELS = PROJECT / "data" / "labels.json"
OUT = PROJECT / "data" / "eval" / "probe_set.json"

# ---------------------------------------------------------------- 模板
#
# ⚠️ 每条模板都**故意避开** `retrieval/retriever.py` 里 LOCATE_RULES 的所有关键词。
#    构造完会用 SectionLocator 逐条复验 `detect_intent(q) is None` ——
#    有一条能被认出来，就说明这条模板不合格（脚本会打出来）。
#
#    顺序：0,1 → dev（可以拿来补词）；2,3 → test（补完词也不许碰）
TEMPLATES = {
    "dosage": [
        "{drug} — what's the usual amount for an adult?",
        "How often am I supposed to take {drug}?",
        "What strength of {drug} is normally given?",
        # ⚠️ 这条踩过坑：初版写的是 "how much of it per day"，
        #    恰好撞上 LOCATE_RULES["dosage"] 的 "how much" → 探针当场失效。
        #    **是脚本里那条自检把它拦下来的** —— 手写的词表迟早会漏，所以要机器复验。
        "They handed me {drug}; what's the daily amount?",
    ],
    "indication": [
        "Why did my doctor give me {drug}?",
        "What is {drug} supposed to accomplish?",
        "What's the point of taking {drug}?",
        "What condition is {drug} meant to help with?",
    ],
    "contraindication": [
        "Who should not be given {drug}?",
        "Are there people who must never be given {drug}?",
        "Is {drug} a bad idea for someone with liver problems?",
        "Which patients ought to stay away from {drug}?",
    ],
    "interaction": [
        "Does {drug} play nicely with other medicines?",
        "Anything I shouldn't have alongside {drug}?",
        "Can I drink alcohol while I'm on {drug}?",
        "Would other pills interfere with {drug}?",
    ],
    "precaution": [
        "Anything I need to look out for while on {drug}?",
        "What should I be aware of while taking {drug}?",
        "Is there anything unusual to expect with {drug}?",
        "What else should I know before starting {drug}?",
    ],
}

# 意图 → 该查哪一节（与 retriever.INTENT_TO_LOINC 一致，但 precaution 是多节）
INTENT_LOINC = {
    "dosage": ["34068-7"],
    "indication": ["34067-9"],
    "contraindication": ["34070-3"],
    "interaction": ["34073-7"],
    "precaution": ["34069-5", "34071-1", "43685-7"],
}

N_DRUGS = 6          # 每个模板取几份药


def main() -> int:
    print("=" * 78)
    print("块 E6 · 探针集（结构先验失效的题）")
    print("=" * 78)

    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    locator = SectionLocator(labels)

    rows, bad = [], []
    for intent, tpls in TEMPLATES.items():
        wanted = INTENT_LOINC[intent]
        # 只挑**这份药确实有该节**的题 —— 否则拒答是对的，量不出检索
        pool = []
        for L in labels:
            have = [s["loinc"] for s in L.get("sections", []) if s["loinc"] in wanted]
            if have:
                pool.append((L["drug"], L["setid"], have))
        pool.sort(key=lambda x: x[0])
        if len(pool) < N_DRUGS:
            print(f"   ⚠️ {intent}: 只有 {len(pool)} 份药有该节，不足 {N_DRUGS}")
        for ti, tpl in enumerate(tpls):
            split = "dev" if ti < 2 else "test"
            # ⭐ 换模板就换一批药：避免"某个药恰好特别好/特别差"直接变成模板之间的差
            off = (ti * 3) % max(1, len(pool))
            pick = (pool + pool)[off:off + N_DRUGS]
            for drug, setid, have in pick:
                q = tpl.format(drug=drug)
                # ⭐ 复验：这条题**必须**定位不出意图，否则它就不是"先验失效"的探针
                got = locator.detect_intent(q)
                if got is not None:
                    bad.append((q, got))
                    continue
                rows.append({
                    "qid": f"prb{len(rows):04d}",
                    "split": split, "intent": intent, "template_id": ti,
                    "drug": drug, "tier": "probe",
                    "question": q,
                    "gold_cite_key": f"{setid}#{have[0]}",
                    "gold_loinc": have[0], "gold_loinc_any": have,
                    "entity": "",
                    "note": "结构先验失效：意图词不在 LOCATE_RULES 里",
                })

    print(f"\n① 生成 {len(rows)} 条")
    by = {}
    for r in rows:
        by[(r["split"], r["intent"])] = by.get((r["split"], r["intent"]), 0) + 1
    for split in ("dev", "test"):
        sub = {k[1]: v for k, v in by.items() if k[0] == split}
        print(f"   {split:<5} {sum(sub.values()):>4} 条   {sub}")

    # ---- 判据
    print("\n② 判据")
    fails = []
    def check(cond, label, detail=""):
        print(f"   {'[OK]' if cond else '[!!]'} {label}" + (f"  —— {detail}" if detail else ""))
        if not cond:
            fails.append(label)
        return cond

    check(not bad, f"每条探针的意图都**定位不出来**（否则它不算探针）",
          "" if not bad else f"{len(bad)} 条被认出来了，例：{bad[0]}")
    check(len(rows) >= 100, f"总题量够（{len(rows)} ≥ 100）")
    dev_n = sum(1 for r in rows if r["split"] == "dev")
    test_n = sum(1 for r in rows if r["split"] == "test")
    check(dev_n >= 40 and test_n >= 40, f"dev/test 都够（{dev_n} / {test_n}）")
    # ⭐ 药名必须**在**问句里 —— 探针量的是"意图定位失效"，不是"药名识别失效"
    check(all(r["drug"] in r["question"] for r in rows),
          "药名都出现在问句里（探针不考指代消解，那是 E10）")
    # ⭐ gold 章节必须真的在这份药里，否则这题无解，会污染指标
    have_map = {L["drug"]: {s["loinc"] for s in L.get("sections", [])} for L in labels}
    check(all(set(r["gold_loinc_any"]) & have_map[r["drug"]] for r in rows),
          "gold 章节都真实存在于对应药品的语料里")

    # ---- 存盘
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n": len(rows),
        "purpose": "让结构先验失效的题 —— 量定位失效时检索层的能力",
        "metric": "与 retrieval_set 完全一致：Recall@5 over (doc_id, loinc)",
        "protocol": "templates 0,1 = dev（可用来补词）；2,3 = test（补完词不许再动）",
        "consumer": "scripts/eval_retrieval.py",
        "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {OUT}")

    print("\n③ 样例")
    for r in rows[:4]:
        print(f"   [{r['split']}/{r['intent']}] {r['question']}")

    print("\n" + "=" * 78)
    if fails:
        print(f"❌ 探针集判据未通过（{len(fails)} 项）：")
        for f in fails:
            print("   -", f)
        return 1
    print("✅ 探针集判据全过")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
