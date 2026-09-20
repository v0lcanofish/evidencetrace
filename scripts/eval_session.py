# -*- coding: utf-8 -*-
"""
E10 判据：**多轮一致**。

━━━ "多轮一致"拆成四条可机械判的 ━━━

    ① 指代消解正确      第 2 轮不带药名 → 解析出的药 == 第 1 轮那个
    ② 跨轮复用生效      同药同节不重查 → **动作数下降**
    ③ 核验标准不放松    多轮下坏引用照样 100% 抓住
    ④ ⭐ **不串味**      换药的轮次**不许**沿用上一轮

⭐ 第 ④ 条是防"记忆变幻觉"的关键：
   **过度消解也是错** —— 用户明确问了另一个药，用记忆覆盖它就是新的错误，
   不是"记性好"。记忆的价值在于"该记的时候记住，**该忘的时候忘掉**"。

跑法：
    PYTHONIOENCODING=utf-8 python scripts/eval_session.py
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

from agent.loop import AgentLoop, LoopConfig, make_toolbox_factory   # noqa: E402
from agent.mocks import MockLLM# noqa: E402
from agent.generator import make_generator                              # noqa: E402
from agent.policy import CoveragePolicy, RulePolicy                   # noqa: E402
from agent.session import Session                                     # noqa: E402
from retrieval import make_retriever                                 # noqa: E402

LABELS = _PROJECT / "data" / "labels.json"

def _mk_loop(r, llm=None, policy=None, cfg=None):
    return AgentLoop(make_toolbox_factory(r, llm=llm or make_generator(), top_k=5),
                     policy or CoveragePolicy(), cfg or LoopConfig(budget=8))

def main() -> int:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    r = make_retriever(labels, use_locator=True)
    known = sorted({c.drug for c in r.chunks if c.drug})

    # 挑三份**确定在库里**的药当对话主角
    A, B, C = known[0], known[1], known[2]

    print("=" * 78)
    print("E10 判据 · 多轮一致")
    print("=" * 78)
    print(f"  语料：{len(known)} 份药 ｜ 本轮对话主角：{A} → {B} → {C}")

    fails = []

    def check(cond, label, detail=""):
        print(f"   {'[OK]' if cond else '[!!]'} {label}" + (f"  —— {detail}" if detail else ""))
        if not cond:
            fails.append(label)

    # ================================================================ ① 指代消解
    print("\n① 指代消解（纯规则，零成本，先把消解本身打准）\n")
    s = Session(known_drugs=known)
    cases = [
        (f"What does {A} interact with?", ["explicit", [A]],
         "点名了药、无线索词 → 全新问题"),
        (f"What about taking it with {B}?", ["from_context", [A, B]],
         "⭐ 点名新药+线索词 → 焦点药也要带上（不然用户真正在问的药会丢）"),
        ("Is it safe during pregnancy?", ["from_context", [B]],
         "纯指代 → 用焦点药"),
        (f"Tell me about {C}.", ["explicit", [C]],
         "⭐ 换药：点名了另一个药 → 只取新的，**绝不掺上一轮的**（防串味）"),
        ("What is the price of tea in China?", ["none", []],
         "域外问题、无线索词 → 不猜"),
    ]
    for q, (want_mode, want_drugs), why in cases:
        res = s.resolve(q)
        ok = res.mode == want_mode and res.drugs == want_drugs
        check(ok, f"{q[:46]}", f"{why} ｜ 实测 {res.mode} {res.drugs}")
    check(s.focus == [C], "⭐ 焦点只被『点名的药』更新（域外问题不改焦点）",
          f"焦点 = {s.focus}")

    # 「不串味」单独再打一发：**没有线索词**的换药
    s2 = Session(known_drugs=known)
    s2.resolve(f"side effects of {A}")
    r2 = s2.resolve(f"side effects of {B}")
    check(r2.drugs == [B] and A not in r2.drugs,
          f"⭐ 不串味：从 {A} 换到 {B} 时不带旧药", f"实测 {r2.drugs}")

    # ================================================================ ② 跨轮复用
    print("\n② 跨轮复用（同药同节不重查 → 动作数下降）\n")
    # ⚠️ 测试用例必须问**语料里真采了的章节**。
    #    第一版用的是 "What are the side effects of X?" —— 而"副作用"那节
    #    语料里根本没有（E8 的 hard_fail 就是治这个的），于是策略直接硬拒答：
    #    证据 0 条、没有答案、没有引用可抓。**两条判据红在那里，是测试问错了问题。**
    q1 = f"What does {A} interact with?"

    s_solo = Session(known_drugs=known)
    lp = _mk_loop(r)
    s_solo.ask(q1, lp)
    solo_actions = s_solo.turns[0].n_actions
    solo_ev = len(s_solo.evidence)

    # 同一个问题再问一遍 —— 但这次是**带着上一轮证据**的多轮会话
    lp2 = _mk_loop(r)
    s_ctx = Session(known_drugs=known)
    s_ctx.ask(q1, lp2)
    before = len(s_ctx.evidence)
    rec2 = s_ctx.ask(q1, lp2)             # 完全重复的一轮
    ctx_actions = rec2.n_actions
    new_ev = rec2.n_new_evidence

    check(ctx_actions < solo_actions,
          "⭐ 带着上一轮证据再问同一题 → **动作数下降**",
          f"单轮 {solo_actions} → 多轮第 2 轮 {ctx_actions}")
    check(new_ev < solo_ev or new_ev == 0,
          "⭐ 第 2 轮**几乎没有新证据**（复用生效，不是重查）",
          f"第 1 轮拿到 {solo_ev} 条，第 2 轮新增 {new_ev} 条")

    # ================================================================ ③ 核验不放松
    print("\n③ 核验标准不放松（多轮下坏引用照样抓）\n")
    # ⚠️ **必须先立前置断言**：这一条验的是"核验在不在"，
    #    而如果策略因为覆盖度不够直接拒答了，压根没有引用可抓 ——
    #    那时 `failed_cites` 为空，断言会**因为错误的原因通过/失败**。
    #    第一版就是这么栽的：换了问题后覆盖度不够，三条里两条红得莫名其妙。
    qbad = "Can I take warfarin with ibuprofen?"
    lp3 = _mk_loop(r, llm=MockLLM(bad_citation=True))
    s3 = Session(known_drugs=known)
    rec3 = s3.ask(qbad, lp3)
    # ⚠️ 要读的是 **TurnRecord**，不是 lp3.last_state ——
    #    Session 为了不让上一轮的失败污染下一轮，每轮开头会清掉 state 的
    #    收尾字段，所以 last_state 里已经读不到了。**状态该清，记录不能清。**
    reached = any(t.startswith("answer") for t in lp3.last_trace)
    check(reached, "（前置）策略走到了 answer —— 这一步才验得到核验",
          "没走到的话下面的断言没有意义")
    check(rec3.n_bad_cites > 0, "多轮第 1 轮里坏引用照样被拦",
          f"抓到 {[c['kind'] for c in rec3.bad_cites][:2]}")
    check(not rec3.answered, "被打回的答案没有被采纳")
    # 再跑一轮，确认第 2 轮也没有放水
    rec4 = s3.ask(f"What about taking it with {B}?", lp3)
    ev4 = lp3.last_state.evidence_keys()
    bad4 = [k for c in s3.last_ledger.claims for k in c.cite if k not in ev4]
    check(not rec4.answered or not bad4, "第 2 轮同样没有交出坏答案",
          f"坏引用 {len(bad4)} 条 ｜ 第2轮抓到的类型 "
          f"{[c['kind'] for c in rec4.bad_cites][:2]}")

    # ================================================================ ④ 端到端多轮
    print("\n④ 端到端：三轮真实追问，全程可追溯\n")
    lp5 = _mk_loop(r)
    s5 = Session(known_drugs=known)
    convo = [q1, f"What about taking it with {B}?", "Is it contraindicated?"]
    for q in convo:
        rec = s5.ask(q, lp5)
        print(f"   轮{rec.turn}  {rec.resolution.mode:<12} 药={rec.resolution.drugs}"
              f"  动作={rec.n_actions}  证据={rec.n_evidence}")
    check(s5.turns[1].resolution.drugs == [A, B],
          "第 2 轮解析出**两个**药", f"{s5.turns[1].resolution.drugs}")
    check(s5.turns[2].resolution.drugs == [B],
          "第 3 轮解析出第 2 轮点名的那个药", f"{s5.turns[2].resolution.drugs}")
    check(all(not t.malformed for t in s5.turns), "三轮都没有 malformed")
    check(s5.turns[-1].n_evidence >= s5.turns[0].n_evidence,
          "证据**跨轮累积**（不是每轮清零）",
          f"{[t.n_evidence for t in s5.turns]}")

    print()
    print("=" * 78)
    if fails:
        print(f"❌ 判据未过（{len(fails)} 项）：")
        for f in fails:
            print("   -", f)
        return 1
    print("✅ E10 判据全过：指代消解准、跨轮复用生效、核验不放松、换药不串味")
    print("=" * 78)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
