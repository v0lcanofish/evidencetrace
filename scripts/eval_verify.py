# -*- coding: utf-8 -*-
"""
E9 判据：**坏引用 100% 抓住**。

━━━ 判据怎么写的，以及为什么这么写 ━━━

判据不是"跑一遍看看"，是**构造出每一种坏引用，逐条断言被抓住**。

坏引用的种类是**从代码的核验分支反推**出来的（verify.py 里三个 K_*），不是拍脑袋列的 ——
这样"判据覆盖了所有分支"这件事才有依据。每加一个分支，这里就要加一条用例。

⚠️ 光测"坏引用被抓"不够，还要测**好引用不被误伤**：
    一个把所有引用都判坏的核验器，也能通过"坏引用 100% 抓住"。
    所以第 1、5 条是**反向用例** —— 它们防的是"核验器过于激进"。

跑法：
    PYTHONIOENCODING=utf-8 python scripts/eval_verify.py
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

from agent.verify import (K_NO_LOINC, K_NOT_IN_CORPUS, K_NOT_RETRIEVED,  # noqa: E402
                          strict_cite_ok, verify_answer)
from retrieval import make_retriever                                 # noqa: E402

LABELS = _PROJECT / "data" / "labels.json"

# 一条**不存在**的 setid（UUID 形状合法，但语料里没有）——
# 专门用来测 not_in_corpus：形状对、内容是编的，是最难抓的一种
FAKE_SETID = "deadbeef-0000-4000-8000-000000000000"


def main() -> int:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    r = make_retriever(labels)

    corpus_keys = {f"{c.doc_id}#{c.loinc}" for c in r.chunks if c.loinc}
    # 取一份真实证据（问阿司匹林不太合适，用一份确定在库里的药）
    drug = r.chunks[0].drug
    docs = list(r(f"{drug} warnings", 5, restrict={"drug": drug}))
    ev_keys = {d.cite_key for d in docs}

    print("=" * 78)
    print("E9 判据 · 坏引用 100% 抓住")
    print("=" * 78)
    print(f"  语料：{len(r.chunks)} 个 chunk ｜ {len(corpus_keys)} 个 (setid#loinc) 引用键")
    print(f"  本轮证据：{len(ev_keys)} 条（药 = {drug}）")

    good = sorted(ev_keys)[0] if ev_keys else None
    # 语料里有、但本轮**没检索到**的一条
    not_retrieved = sorted(corpus_keys - ev_keys)[0] if corpus_keys - ev_keys else None
    assert good and not_retrieved, "语料/证据构造失败，判据无效"

    # ---------------------------------------------------------------- 用例
    # 每条 = (名字, 答案文本, 期望的 ok, 期望抓到的 kind 或 None, 传不传 corpus_keys)
    cases = [
        ("① 好引用（反向用例：不许误伤）",
         f"The drug can cause dizziness [{good}].", True, None, True),

        ("② 有效拒答（反向用例：拒答≠引用坏）",
         "The evidence is insufficient to answer this question.", True, None, True),

        ("③ 没有章节码 [setid]",
         f"The drug can cause dizziness [{good.split('#')[0]}].", False, K_NO_LOINC, True),

        ("④ 语料里有、但本轮没检索到",
         f"The drug can cause dizziness [{not_retrieved}].", False, K_NOT_RETRIEVED, True),

        ("⑤ 编造的 setid（语料里没有）",
         f"The drug can cause dizziness [{FAKE_SETID}#34073-7].", False, K_NOT_IN_CORPUS, True),

        ("⑥ 真 setid + 不存在的章节码",
         f"The drug can cause dizziness [{good.split('#')[0]}#99999-9].", False, K_NOT_IN_CORPUS, True),

        ("⑦ 有正文、一条引用都没有 → malformed",
         "The drug can cause dizziness and nausea.", False, None, True),

        ("⑧ 空字符串 → 不算拒答，算没答",
         "", False, None, True),

        ("⑨ 三条引用里混一条坏的 → 只抓坏的那条",
         f"Fact one [{good}]. Fact two [{not_retrieved}]. Fact three [{good}].",
         False, K_NOT_RETRIEVED, True),

        ("⑩ 不传 corpus_keys 时，编的 setid 也要抓住（降级为 not_retrieved）",
         f"The drug can cause dizziness [{FAKE_SETID}#34073-7].", False, K_NOT_RETRIEVED, False),
    ]

    fails = []

    def check(cond, label, detail=""):
        print(f"   {'[OK]' if cond else '[!!]'} {label}" + (f"  —— {detail}" if detail else ""))
        if not cond:
            fails.append(label)

    print("\n① 逐条用例\n")
    for name, text, want_ok, want_kind, use_corpus in cases:
        v = verify_answer(text, ev_keys, corpus_keys if use_corpus else None)
        ok_right = (v.ok == want_ok)
        if want_kind is None:
            kind_right = True
            got = "—"
        else:
            kind_right = want_kind in v.bad_kinds
            got = ",".join(sorted(v.bad_kinds)) or "（没抓到）"
        check(ok_right and kind_right, name,
              f"ok={v.ok}(期望{want_ok}) 抓到={got}")

    # ---------------------------------------------------------------- 覆盖率
    print("\n② 分支覆盖率（从代码的核验分支反推，而不是凭感觉列用例）\n")
    covered = set()
    for name, text, _w, want_kind, use_corpus in cases:
        if want_kind:
            covered.add(want_kind)
    check(covered == {K_NO_LOINC, K_NOT_RETRIEVED, K_NOT_IN_CORPUS},
          "三个核验分支全部有用例覆盖",
          f"覆盖 {sorted(covered)}")

    # 反向用例必须存在 —— 否则"全判坏"的核验器也能过判据
    n_positive = sum(1 for _n, _t, w, _k, _c in cases if w)
    check(n_positive >= 2,
          "⭐ 有反向用例（好引用 / 有效拒答）—— 防'核验器过于激进'",
          f"{n_positive} 条")

    # ---------------------------------------------------------------- 严格格式
    print("\n③ 严格格式判据\n")
    check(strict_cite_ok(f"text [{good}]"), "带章节码 = 严格格式")
    check(not strict_cite_ok(f"text [{good.split('#')[0]}]"),
          "不带章节码 = 不严格")

    # ---------------------------------------------------------------- 纯函数
    print("\n④ 核验器是纯函数（不改输入、可重放）\n")
    ev_before = set(ev_keys)
    _ = verify_answer(f"x [{good}].", ev_keys, corpus_keys)
    check(ev_keys == ev_before, "核验不修改传进去的 evidence_keys")

    print()
    print("=" * 78)
    if fails:
        print(f"❌ 判据未过（{len(fails)} 项）：")
        for f in fails:
            print("   -", f)
        return 1
    print(f"✅ 判据全过：{len(cases)} 条用例，坏引用 100% 抓住，好引用未误伤")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
