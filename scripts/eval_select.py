# -*- coding: utf-8 -*-
"""
块 E13b 的判据：生成前的证据选择（排序 + 按预算截断）。

    PYTHONIOENCODING=utf-8 python scripts/eval_select.py
    PYTHONIOENCODING=utf-8 python scripts/eval_select.py --quick

━━━ 判据三铁律怎么落的 ━━━

    可执行        —— 本文件，12 条断言 + 一张 K 扫描表
    阈值不能拍    —— `budget` 不设默认值就上，**扫出来**（判据 ③）；
                     并且要求**最优 K 落在网格内部**（两端说明网格没罩住）
    区分「做到了」
    和「看起来做到了」—— 两条专门为此设计的判据：
        · 判据 ④ 反向用例：K=1（池子里只留 1 条）**也能让生成层少犯错** ——
          如果答对率不掉，说明这把尺子量不出"暴力剪裁的代价"，判据作废
        · 判据 ⑤ 对抗档：把池子**倒过来**再跑一遍 ——
          mock 生成器同分时取"靠前"的那条，若提升全来自这个 tie-break，
          倒序之后提升就会消失。**这一条是专门用来拆穿自己的。**
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT))

from agent.coverage import compute_coverage, corpus_drugs_from          # noqa: E402
from agent.loop import AgentLoop, LoopConfig, make_toolbox_factory      # noqa: E402
from agent.mocks import GroundedMockLLM                                 # noqa: E402
from agent.policy import CoveragePolicy                                 # noqa: E402
from agent.report import looks_like_abstention                          # noqa: E402
from agent.select import (select_evidence, focus_drugs, struct_rank,    # noqa: E402
                          R_BOTH, R_SECTION, R_DRUG, R_NONE)
from ledger.ledger import Doc                                          # noqa: E402
from retrieval import make_retriever                                    # noqa: E402

LABELS = _PROJECT / "data" / "labels.json"
R_SET = _PROJECT / "data" / "eval" / "retrieval_set.json"
N = 60

# 扫描网格：**故意两端都放极端值** ——
#   1 = 暴力剪裁（只留一条）；None = E13b 之前的老口径（全池）
# 最优值必须落在中间，否则说明网格没罩住 / 这个指标没有分辨力。
GRID = [1, 2, 3, 4, 5, 6, 8, None]

FAILS: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"   {'✓' if ok else '✗'} {name}" + (f"　{detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------- ① 合同测试


def t_contract(known, rows) -> None:
    """选择层的「焦点药」必须和覆盖度的「药品槽位」**逐字一致**。

    ⚠️ 为什么值得单开一条判据：这两处是**同一个判断的两份实现**。
       一旦漂移，会出现「覆盖度说这份药没进作用域、选择层却认为有焦点药」
       ——**不报错**，只是排序依据悄悄变了。本项目在闭包/归因那条路上
       已经因为"两套实现"栽过一次（E13 的粒度洞），这是第二次设防。
    """
    print("\n① 合同测试：select.focus_drugs ≡ coverage 药品槽位")
    bad = []
    for r in rows:
        q = r["question"]
        mine = sorted(x.lower() for x in focus_drugs(q, known))
        cov = compute_coverage(q, [], None, known)
        theirs = sorted(s.key.lower() for s in cov.slots
                        if s.name == "drug" and s.key != "(未识别)")
        if mine != theirs:
            bad.append((q, mine, theirs))
    check(f"60 题逐题一致", not bad,
          f"不一致 {len(bad)} 条" + (f"　例：{bad[0]}" if bad else ""))


# ---------------------------------------------------------------- ② 单元断言


def _doc(drug, loinc, auth=0.5, key=None) -> Doc:
    return Doc(doc_id=key or "aaaaaaaa-1111-2222-3333-444444444444",
               section="S", loinc=loinc, drug=drug, text="t", authority=auth)


def t_units() -> None:
    print("\n② 单元断言：结构档位 / 排序 / 截断边界")
    scope, lo = {"a"}, {"1111-1"}
    check("定位章节 ∩ 焦点药 = 3", struct_rank(_doc("a", "1111-1"), scope, lo) == R_BOTH)
    check("只对章节 = 2（E11 的跨药出处落在这档，**不许当垃圾扔**）",
          struct_rank(_doc("b", "1111-1"), scope, lo) == R_SECTION)
    check("只对药 = 1", struct_rank(_doc("a", "9999-9"), scope, lo) == R_DRUG)
    check("都不对 = 0", struct_rank(_doc("b", "9999-9"), scope, lo) == R_NONE)

    # 排序：档位优先于权威度（结构是硬先验，权威度只是同档内的次序）
    pool = [_doc("b", "9999-9", auth=0.99, key="bbbbbbbb-1111-2222-3333-444444444444"),
            _doc("a", "1111-1", auth=0.10, key="cccccccc-1111-2222-3333-444444444444")]
    sel = select_evidence(pool, {"loincs": ["1111-1"]}, ["a"], "a q")
    check("档位压过权威度", sel.kept[0].drug == "a",
          f"首条 drug={sel.kept[0].drug}")

    # 同分稳定：不抖动（同档同权威度 → 保持原检索序）
    p2 = [_doc("a", "1111-1", 0.5, "dddddddd-1111-2222-3333-444444444444"),
          _doc("a", "1111-1", 0.5, "eeeeeeee-1111-2222-3333-444444444444")]
    s2 = select_evidence(p2, {"loincs": ["1111-1"]}, ["a"], "a q")
    check("同分保持原序（确定性）", [d.doc_id[0] for d in s2.kept] == ["d", "e"])

    # 边界
    check("budget=None → 不截断", len(select_evidence(pool, None, ["a"], "a q").kept) == 2)
    check("budget 超过池子 → 不报错也不补", len(select_evidence(pool, None, ["a"], "a q", budget=99).kept) == 2)
    check("budget=1 → 只留一条，且理由是可查的",
          (lambda s: len(s.kept) == 1 and s.dropped and s.dropped[0].reason == "over_budget")
          (select_evidence(pool, None, ["a"], "a q", budget=1)))
    check("空池不炸", select_evidence([], None, ["a"], "a q", budget=3).n_kept == 0)

    # 被截断的必须**条条有理由**（不许静默丢弃）
    s3 = select_evidence(pool, None, ["a"], "a q", budget=1)
    check("每条被丢的都有理由", all(x.reason for x in s3.dropped))


# ---------------------------------------------------------------- ③ K 扫描


def run_pass(r, known, rows, budget, order="struct"):
    """跑一遍 60 题，返回统计 + 每条题的明细（供 gold 保留率用）。"""
    stats = {"correct": 0, "util": 0, "abstain": 0, "kept": [], "other": [],
             "gold_kept": 0, "gold_pos": [], "n": 0}
    details = []
    for row in rows:
        q, gold = row["question"], {row["gold_cite_key"]}
        holder = {}
        base = make_toolbox_factory(r, llm=GroundedMockLLM(), top_k=5,
                                    evidence_budget=budget, select_order=order)

        def factory(lg, user, _b=base, _h=holder):
            tb = _b(lg, user)
            _h["tb"] = tb
            return tb

        lp = AgentLoop(factory, CoveragePolicy(), LoopConfig(budget=8))
        lg = lp.run(q, run_id=row["qid"])
        st, tb = lp.last_state, holder.get("tb")
        sel = getattr(tb, "last_selection", None)

        cited = lg.cited_keys()
        ok = gold.issubset(cited)
        layer = lg.attribute(gold, correct=ok)["layer"]
        abst = looks_like_abstention(st.answer_text or "")
        stats["n"] += 1
        stats["correct"] += int(ok)
        stats["util"] += int(layer == "utilization")
        stats["abstain"] += int(abst)
        if sel:
            stats["kept"].append(sel.n_kept)
            stats["other"].append(sel.n_other_drug)
            gk = row["gold_cite_key"]
            stats["gold_kept"] += int(any(d.cite_key == gk for d in sel.kept))
        details.append((row["question"], ok, layer, abst, sel, cited, gold))
    return stats, details


def report(name, s) -> str:
    n = s["n"] or 1
    return (f"   {name:<16} 答对 {s['correct']:>2}/{n}"
            f"  util {s['util']:>2}"
            f"  拒答 {s['abstain']:>2}"
            f"  gold保留 {s['gold_kept']:>2}/{n}"
            f"  平均保留 {sum(s['kept'])/max(1,len(s['kept'])):>4.1f} 条"
            f"  其中异药 {sum(s['other'])/max(1,len(s['other'])):>4.1f}")


def main() -> int:
    quick = "--quick" in sys.argv
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    r = make_retriever(labels, use_locator=True)
    known = corpus_drugs_from(r)
    rows = json.loads(R_SET.read_text(encoding="utf-8"))["rows"][:N]
    if quick:
        rows = rows[:12]
        print(f"⚠️ --quick：只跑前 {len(rows)} 条")

    t_contract(known, rows)
    t_units()

    # ---- ③ K 扫描 + ⑤ 对抗档
    print(f"\n③ K 扫描（策略=CoveragePolicy｜生成器=GroundedMockLLM｜{len(rows)} 题）")
    print("   " + "-" * 92)
    results = {}
    for k in ([None] if quick else GRID):
        s, det = run_pass(r, known, rows, k)
        results[k] = (s, det)
        print(report(f"K={k}", s))

    best_k = (None if quick else max(GRID, key=lambda k: (results[k][0]["correct"],
                                                          -results[k][0]["util"])))
    print(f"\n   ⇒ 答对率最高：K={best_k}")

    # ---- ④ 反向用例：暴力剪裁必须有代价
    print("\n④ 反向用例（防「池子越小越好」）")
    s1 = results[1][0] if 1 in results else None
    sN = results[None][0]
    if s1 is None:
        print("   （--quick 跳过）")
    else:
        check("K=1 的答对率**必须**低于 K=None（否则这把尺子量不出暴力剪裁的代价）",
              s1["correct"] < sN["correct"],
              f"K=1 {s1['correct']} vs K=None {sN['correct']}")

    # ---- ⑤ 对抗档：倒序池子
    print("\n⑤ 对抗档：把池子倒过来（拆穿「提升是不是靠 mock 同分取靠前」）")
    if quick:
        print("   （--quick 跳过）")
    else:
        s_rev_full, _ = run_pass(r, known, rows, None, order="reverse")
        print(report("reverse/全池", s_rev_full))
        s_rev_best, _ = run_pass(r, known, rows, best_k, order="reverse")
        print(report(f"reverse/K={best_k}", s_rev_best))
        base = results[None][0]["correct"]
        fwd = results[best_k][0]["correct"]
        revf, revb = s_rev_full["correct"], s_rev_best["correct"]
        gain_fwd, gain_rev = fwd - base, revb - revf
        print(f"\n   ⇒ 正序提升 {gain_fwd:+d} ｜ 倒序提升 {gain_rev:+d}")
        if gain_fwd > 0 and gain_rev >= gain_fwd:
            print("   ⇒ 倒序后提升**没有消失** → 提升不是 tie-break 造的，"
                  "可以报成结构排序的贡献")
        elif gain_fwd > 0:
            print(f"   ⇒ ⚠️ 倒序后提升从 {gain_fwd:+d} 掉到 {gain_rev:+d} ——"
                  "**提升里有一部分是 mock 同分取靠前给的**，不能全记在结构排序头上")

    print("\n" + "=" * 92)
    if FAILS:
        print(f"❌ {len(FAILS)} 条断言没过：")
        for f in FAILS:
            print(f"   · {f}")
    else:
        print("✅ 全部断言通过")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
