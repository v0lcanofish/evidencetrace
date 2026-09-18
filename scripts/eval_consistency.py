# -*- coding: utf-8 -*-
"""
E11 判据：**引用偏不偏**（跨药一致性核验）。

━━━ 三条判据 ━━━

    ① 不对称**可检测**      20 对单向强警告全中、方向正确（其中 7 对人工核对过原句）
    ② **不误报**            4 对双向一致的**不许**判成不对称
    ③ ⭐ **不许选择性引用**  一方说了、另一方也说了，答案只引一边 → 拦
    ④ ⭐ **沉默要如实标注** 对方确实没说 → 答案必须说明，不许让用户以为它表了态

⚠️ ③ 和 ④ 的正确动作**完全不同**（前者补检索、后者加说明），
   所以必须分开报 —— 混成一个的话 agent 会白白重试。

跑法：
    PYTHONIOENCODING=utf-8 python scripts/eval_consistency.py
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
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from agent.consistency import (ABSENT, MENTION, WARN, DrugIndex,  # noqa: E402
                               audit_citation_balance)
from retrieval import make_retriever                                 # noqa: E402

LABELS = _PROJECT / "data" / "labels.json"

# ⚠️⚠️ **数字口径锁定**：全量扫描下「甲强警告、乙沉默」是 **20 对**（剔掉表格伪句后）。
#     这个数**必须和代码用同一个 WARN_RE** —— 2026-09-18 我用一个手写的窄词表
#     （漏了 `risk of`）扫出"7 对"并写进了报告，**实际是 20 对**，差的 13 对全是真警告。
#     所以下面除了逐条验，还加了一条**锁定总数**的断言：口径一漂就红。
N_ASYMMETRIC_TOTAL = 20

# 人工核对过原句的 7 对（是上面 20 对的**子集**，挑的是最教科书的）
KNOWN_ASYM = [
    ("amiodarone", "digoxin"),
    ("amiodarone", "warfarin"),
    ("furosemide", "lithium"),
    ("bupropion", "phenytoin"),
    ("amlodipine", "simvastatin"),
    ("sitagliptin", "metformin"),
    ("empagliflozin", "metformin"),
]


def main() -> int:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    r = make_retriever(labels)
    idx = DrugIndex(r.chunks, [c.drug for c in r.chunks])

    print("=" * 78)
    print("E11 判据 · 引用偏不偏（跨药一致性核验）")
    print("=" * 78)
    print(f"  语料：{len(idx.drugs)} 份说明书")

    fails = []

    def check(cond, label, detail=""):
        print(f"   {'[OK]' if cond else '[!!]'} {label}" + (f"  —— {detail}" if detail else ""))
        if not cond:
            fails.append(label)

    # ================================================================ ① 不对称可检测
    print("\n① 不对称可检测（甲警告、乙沉默）\n")
    for a, b in KNOWN_ASYM:
        c = idx.check_pair(a, b)
        ok = c.is_asymmetric and c.speaker == a and c.silent == b
        check(ok, f"{a} → {b}", f"{c.describe()}")
        if a == "amiodarone" and b == "digoxin":
            check("by half" in c.a_side.sentence or "discontinue" in c.a_side.sentence.lower(),
                  "　└ 抓到的是**强警告原句**", f"「{c.a_side.sentence[:64]}」")

    # ================================================================ ② 不误报
    print("\n② 不误报：双方一致的不许判成不对称\n")
    # 从实测里挑：NEG×NEG 的一致对 + plain×plain 的一致对
    consistent = [("clopidogrel", "omeprazole"), ("clopidogrel", "esomeprazole"),
                  ("allopurinol", "colchicine")]
    for a, b in consistent:
        c = idx.check_pair(a, b)
        check(not c.is_asymmetric, f"{a} ↔ {b} 判为一致", c.describe())

    # 反向也要查：真正不对称的对，不许被判成一致
    c = idx.check_pair("digoxin", "amiodarone")
    check(c.is_asymmetric and c.speaker == "amiodarone",
          "反向查（digoxin, amiodarone）方向仍然正确",
          f"speaker={c.speaker} silent={c.silent}")

    # ================================================================ ③ 选择性引用
    print("\n③ ⭐ 不许选择性引用（一方说了、另一方也说了，只引一边）\n")
    c = idx.check_pair("clopidogrel", "omeprazole")     # 双方都 warn
    check(c.kind == "both_warn", "（前置）这一对是双方都警告", c.kind)

    only_one = audit_citation_balance(c, cited_drugs={"clopidogrel"})
    check(not only_one.ok, "只引一边 → 判为**选择性引用**",
          only_one.problems[0]["kind"] if only_one.problems else "（没报）")

    both = audit_citation_balance(c, cited_drugs={"clopidogrel", "omeprazole"})
    check(both.ok, "两边都引 → 通过", both.summary())

    # ================================================================ ④ 沉默要如实标注
    print("\n④ ⭐ 对方沉默时：不是选择性引用，但必须**如实标注**\n")
    c = idx.check_pair("amiodarone", "digoxin")
    only_a = audit_citation_balance(c, cited_drugs={"amiodarone"})
    kinds = {p["kind"] for p in only_a.problems}
    check(not only_a.ok and "silent_unstated" in kinds,
          "只引警告方 → 报的是 `silent_unstated`（而不是 selective_citation）",
          f"kinds = {sorted(kinds)}")
    check("selective_citation" not in kinds,
          "⭐ 且**没有**误报成选择性引用 —— 两者的正确动作不同",
          "前者要加说明，后者要去补检索；混成一个 agent 会白白重试")

    # ================================================================ 边界
    print("\n⑤ 边界\n")
    c2 = idx.check_pair("amiodarone", "aspirin")        # aspirin 不在语料里
    check(c2.kind == "both_absent", "域外药（aspirin 不在库里）→ 双方都 absent",
          c2.kind)
    c3 = idx.check_pair("aspirin", "digoxin")
    check(c3.kind == "both_absent" or c3.kind == "asymmetric_b",
          "域外药当主语时不崩", c3.kind)

    # ================================================================ ⑤ 总数锁定
    # ⚠️ 这条是**防口径漂移**的：2026-09-18 我用一个手写窄词表扫出"7 对"并写进了报告，
    #    而代码的宽词表给出的是 20 对 —— 差的 13 对全是真警告。
    #    所以把总数写死：**以后口径一变就红**，逼人先查清楚再改。
    total = sum(1 for a in idx.drugs for b in idx.drugs
                if a != b and idx.check_pair(a, b).kind == "asymmetric_a")
    check(total == N_ASYMMETRIC_TOTAL,
          f"⭐ 全量扫描的单向强警告数锁定为 {N_ASYMMETRIC_TOTAL}",
          f"实测 {total} —— 对不上说明 WARN_RE 或语料变了，**别直接改这个常量**，先查清楚")

    n_asym = sum(1 for a, b in KNOWN_ASYM if idx.check_pair(a, b).is_asymmetric)
    print(f"\n   人工核对过原句的 7 对：{n_asym}/{len(KNOWN_ASYM)} 全中"
          f"（它们是上面 {total} 对里最教科书的那个子集）")

    print()
    print("=" * 78)
    if fails:
        print(f"❌ 判据未过（{len(fails)} 项）：")
        for f in fails:
            print("   -", f)
        return 1
    print("✅ E11 判据全过：不对称检得出、一致的不误报、选择性引用拦得住、沉默要标注")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
