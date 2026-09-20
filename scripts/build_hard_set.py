# -*- coding: utf-8 -*-
"""
造一套**有分辨力**的评测题 —— 修 B 线「量不出收益」的根因。

    python scripts/build_hard_set.py                 # → data/eval/hard_set.json
    python scripts/build_hard_set.py --selftest      # 只跑自检

━━━ 为什么要有这个（2026-09-19）━━━

换真模型（deepseek-chat）之后，原有 60 道检索题**答对 60/60**：

    tier   : direct 30 + paraphrase 30   ← 一半是同一问题的改写
    gold   : **60/60 全是单值**          ← 每题只需找到 1 条证据
    ⇒ K=1 和 K=None 拿一样的分 ⇒ 判据「K=1 必须低于 K=None」**结构上永远过不了**

所以 E13b「证据选择的收益」和 E12「决策器」**都量不出东西** —— 不是真模型太强，
是这套题从一开始就没有"做得更好"的空间。**天花板效应。**

━━━ 解法：让每题需要**多条**证据（gold=2）━━━

两条路子，gold 都**从标签里机械抽**（`setid#loinc`），不用 LLM 编：

  题型 A · **同药多节**
      "我在吃 {药}。剂量是多少，另外要避免和什么一起用？"
      gold = {setid#34068-7 剂量, setid#34073-7 相互作用}       —— 两节都必须引

  题型 B · **跨药不对称**
      "{A} 和 {B} 都在我的用药清单上。**两张说明书各自提没提对方？**"
      gold = {setid_A#34073-7, setid_B#34073-7}
      ⇒ 答"B 没提"也**必须引 B 的节**才能证明查过 ⇒ gold=2 是**必需**的，不是硬规定

⚠️ 一条走过的弯路（记下来免得再走）：本来想用"**互相**提及"的药对造跨药题，
   实测**全语料只有 1 对**互相提及（fluoxetine ↔ olanzapine）。
   ⇒ 单靠互提造不出题量。改成"单向提及 + 不对称提问"后，可用药对 **84** 条。

━━━ 红线 ━━━

gold **只做机械抽取**：章节存在、loinc 对得上、正文提到对方药名。
**不用 LLM 生成 gold** —— 那就变成"编造"了。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_PROJECT = Path(__file__).resolve().parents[1]
LABELS = _PROJECT / "data" / "labels.json"
OUT = _PROJECT / "data" / "eval" / "hard_set.json"

LOINC_DOSAGE = "34068-7"
LOINC_INTERACTIONS = "34073-7"
MIN_SECTION_CHARS = 200          # 节太短 = 题目没东西可答

# ---- 题型 A 的模板（确定性，不用 LLM 改写 —— 保证可复现）----
T_A = "I'm taking {drug}. What's the usual dose, and what other medicines should I avoid taking with it?"
# ---- 题型 B：**必须**问"各自提没提对方"，否则只要引一边就够，gold=2 就成了硬规定 ----
T_B = ("Both {a} and {b} are on my medicine list. "
       "Does each label warn about the other one?")


# ---------------------------------------------------------------- 抽取


def load_labels(path: Path = LABELS) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["labels"]


def sections_of(drug_rec, loinc: str) -> List[Dict[str, Any]]:
    """取某 loinc 的节。⚠️ 有的标签同一 loinc 出现两次，全都返回。"""
    return [s for s in drug_rec["sections"] if s.get("loinc") == loinc]


def body_of(drug_rec, loinc: str) -> str:
    return "\n".join(s.get("text", "") for s in sections_of(drug_rec, loinc))


def mentions(text: str, name: str) -> bool:
    return bool(re.search(rf"\b{re.escape(name)}\b", text, re.I))


def cite_key(drug_rec, loinc: str) -> str:
    """账本的引用键 = `setid#loinc`（见 ledger.ledger._first_step_of）。"""
    return f"{drug_rec['setid']}#{loinc}"


# ---------------------------------------------------------------- 组题


def build(labels: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_name = {x["drug"].lower(): x for x in labels}
    names = sorted(by_name)
    rows: List[Dict[str, Any]] = []
    dropped = []

    # ---- 题型 A · 同药多节 ----
    for name in names:
        rec = by_name[name]
        if len(body_of(rec, LOINC_DOSAGE)) < MIN_SECTION_CHARS:
            dropped.append((f"A:{name}", "剂量节太短或缺"));continue
        if len(body_of(rec, LOINC_INTERACTIONS)) < MIN_SECTION_CHARS:
            dropped.append((f"A:{name}", "相互作用节太短或缺"));continue
        rows.append({
            "qid": f"hardA-{name}",
            "kind": "multi_section",
            "question": T_A.format(drug=name),
            "gold_cite_keys": [cite_key(rec, LOINC_DOSAGE),
                               cite_key(rec, LOINC_INTERACTIONS)],
            "drugs": [name],
            "note": "剂量 + 相互作用两节；答全必须引两条，K=1 必然缺一条",
        })

    # ---- 题型 B · 跨药不对称 ----
    seen = set()
    for a in names:
        ra = by_name[a]
        for b in names:
            if a == b or (a, b) in seen:
                continue
            tb = body_of(by_name[b], LOINC_INTERACTIONS)
            if not mentions(tb, a):           # 只取 B 提到 A 的方向
                continue
            seen.add((a, b))
            if len(body_of(ra, LOINC_INTERACTIONS)) < MIN_SECTION_CHARS:
                dropped.append((f"B:{a}->{b}", "被提到的那个药自身没有相互作用节"))
                continue
            rows.append({
                "qid": f"hardB-{a}-{b}",
                "kind": "cross_drug",
                "question": T_B.format(a=a, b=b),
                "gold_cite_keys": [cite_key(ra, LOINC_INTERACTIONS),
                                   cite_key(by_name[b], LOINC_INTERACTIONS)],
                "drugs": [a, b],
                "note": (f"实测：{b} 的相互作用节提到 {a}；"
                         f"{a} 那边{'也提到' if mentions(body_of(ra, LOINC_INTERACTIONS), b) else '**没提到**'}"
                         f" ⇒ 不对称"),
            })

    # ⭐ 交错排列两种题型。下游 `eval_select` 会 `rows[:N]` 截断（N=60），
    #    不交错的话前 60 条全是"同药多节"，跨药那类一道都进不来。
    a_rows = [r for r in rows if r["kind"] == "multi_section"]
    b_rows = [r for r in rows if r["kind"] == "cross_drug"]
    mixed: List[Dict[str, Any]] = []
    for i in range(max(len(a_rows), len(b_rows))):
        if i < len(a_rows):
            mixed.append(a_rows[i])
        if i < len(b_rows):
            mixed.append(b_rows[i])
    rows = mixed

    return {
        "n": len(rows),
        "note": ("有分辨力的补充评测集。每题 gold=2，从标签机械抽取（setid#loinc），"
                 "不用 LLM 生成。目的：让 K 扫描重新有分辨力 —— "
                 "见 scripts/build_hard_set.py 顶部。"),
        "loinc": {"dosage": LOINC_DOSAGE, "interactions": LOINC_INTERACTIONS},
        "rows": rows,
        "_dropped": dropped,
    }


# ---------------------------------------------------------------- 自检


def _selftest(data: Dict[str, Any], labels: List[Dict[str, Any]]) -> int:
    fails = []
    rows = data["rows"]
    by_name = {x["drug"].lower(): x for x in labels}
    valid_keys = {(x["setid"], s["loinc"]) for x in labels for s in x["sections"]}

    def chk(cond, label):
        print(f"   {'[OK]' if cond else '[!!]'} {label}")
        if not cond:
            fails.append(label)

    print("=" * 84)
    print(f"hard_set 自检 ｜ {len(rows)} 条题")
    print("=" * 84)

    print("\n① 规模与配比")
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    chk(len(rows) >= 60, f"题量 ≥60（实测 {len(rows)}）—— 太少撑不起 K 扫描")
    chk(len(kinds) == 2, f"两种题型都在（{kinds}）")

    print("\n② ⭐ 每道题 gold 必须 = 2（这是整套题存在的理由）")
    bad = [r["qid"] for r in rows if len(r["gold_cite_keys"]) != 2]
    chk(not bad, f"全部 gold=2（违规 {len(bad)} 条：{bad[:3]}）")

    print("\n③ gold 必须是**真实存在**的引用键（setid#loinc）")
    unknown = [k for r in rows for k in r["gold_cite_keys"]
               if tuple(k.split("#")) not in valid_keys]
    chk(not unknown, f"全部引用键都能在语料里找到（找不到 {len(unknown)} 条：{unknown[:2]}）")

    print("\n④ ⭐ 跨药题：被提到的那一边**正文里确实提到**对方药名")
    miss = []
    for r in rows:
        if r["kind"] != "cross_drug":
            continue
        a, b = r["drugs"]
        if not mentions(body_of(by_name[b], LOINC_INTERACTIONS), a):
            miss.append(r["qid"])
    chk(not miss, f"机械抽取与正文一致（不一致 {len(miss)} 条）")

    print("\n⑤ 同药多节题：两条 gold 必须**来自不同 loinc**（否则 K 没有意义）")
    same = []
    for r in rows:
        if r["kind"] != "multi_section":
            continue
        loincs = [k.split("#")[1] for k in r["gold_cite_keys"]]
        if len(set(loincs)) != 2:
            same.append(r["qid"])
    chk(not same, f"两节 loinc 不同（相同 {len(same)} 条）")

    print("\n⑥ 题面不能泄露答案（不能出现 loinc / setid）")
    leak = [r["qid"] for r in rows
            if any(t in r["question"] for t in ("34073", "34068", "setid", "#"))]
    chk(not leak, f"题面干净（泄露 {len(leak)} 条）")

    print("\n⑦ 被丢掉的（分母要报出来）")
    print(f"   丢弃 {len(data['_dropped'])} 条")
    for q, why in data["_dropped"][:5]:
        print(f"     · {q}: {why}")

    print("\n" + "=" * 84)
    if fails:
        print(f"❌ {len(fails)} 项没过")
        return 1
    print("✅ 全部通过")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    labels = load_labels()
    data = build(labels)
    rc = _selftest(data, labels)
    if a.selftest:
        return rc
    if rc:
        print("\n⛔ 自检没过，**不写盘**。")
        return rc

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {p}（{data['n']} 条）")
    print("\n下一步：")
    print("  ET_REAL_LLM=1 python scripts/eval_select.py --set hard   # 跑 K 扫描看有没有分辨力")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
