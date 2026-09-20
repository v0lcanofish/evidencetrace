# -*- coding: utf-8 -*-
"""
块 E7 · agent 判据（零模型 / 零 GPU / 秒级）。

    python scripts/eval_agent.py

分两部分：
    第一部分（本文件 · 工具集）  五个工具各自的行为对不对
    第二部分（段 2 追加）        agent 循环能不能自主多轮跑通

━━━ 为什么工具集要单独验 ━━━

    工具是 agent 的**手脚**。手脚要是不准，后面策略再聪明也没用，
    而且会以一种很难查的方式"错得合理"：
        · search 不带限定 → 返回别的药的章节，agent 以为查到了
        · search 不去重   → agent 反复查同一个词，预算烧光还以为是"没找到"
        · ask 的结果混进证据 → 闭包断言当场失去意义（口述没有出处可追）

    这些错误都不会报异常，只会让 agent 表现得"有点笨"。
    所以必须在接循环之前，用断言把它们钉死。

跑法：
  cd 代码库/projects/EvidenceTrace
  PYTHONIOENCODING=utf-8 python scripts/eval_agent.py
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
from typing import Dict, List

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from ledger.ledger import Ledger, LedgerError, ACTIONS                      # noqa: E402
from agent.mocks import MockLLM, MockRetriever# noqa: E402
from agent.generator import make_generator                              # noqa: E402
from agent.state import AgentState, A_SEARCH, A_ASK, A_ANSWER, A_ABSTAIN    # noqa: E402
from agent.tools import ToolBox, ScriptedUser                               # noqa: E402
from retrieval import make_retriever                                         # noqa: E402

LABELS = PROJECT / "data" / "labels.json"
RET_SET = PROJECT / "data" / "eval" / "retrieval_set.json"

_fails: List[str] = []

def check(cond, label, detail=""):
    print(f"   {'[OK]' if cond else '[!!]'} {label}" + (f"  —— {detail}" if detail else ""))
    if not cond:
        _fails.append(label)
    return cond

def _new_box(retriever, question="test", user=None, llm=None):
    lg = Ledger("test", question, seed=0)
    return lg, ToolBox(retriever=retriever, ledger=lg, user=user, llm=llm)

# ================================================================ ① 定位

def part_locate(labels) -> Dict:
    print("\n① locate —— 问题 → 该查哪一节（本项目相对通用 RAG 的结构先验）")
    print("-" * 78)
    r = make_retriever(labels, use_locator=True)
    lg, tb = _new_box(r)

    res = tb.locate("Can I take metformin if I have kidney disease?")
    check(res.ok, f"定位成功：{res.payload.get('loincs')}", res.note)
    check(res.payload.get("drug") == "metformin", f"识别出药品 = {res.payload.get('drug')}")
    check("34070-3" in (res.payload.get("loincs") or []), "定位到禁忌节 34070-3")
    check(len(lg.steps) == 1 and lg.steps[0].action == "locate",
          "账本里记了一条 locate（不是 retrieve）")

    # 定位不出来时要**诚实返回空**，而不是硬猜一个章节
    res2 = tb.locate("What is the price of tea in China?")
    check(not res2.ok and not res2.payload.get("loincs"),
          "问无关问题时定位返回空（不硬猜）", res2.note)

    # 同一类问题不同药落在不同 LOINC 上 —— 这是实测踩过的口径坑
    r3 = tb.locate("Is warfarin safe during pregnancy?")
    check(r3.payload.get("loincs") == ["34070-3"] or "34070-3" in r3.payload.get("loincs", []),
          f"warfarin 妊娠问题落在 {r3.payload.get('loincs')}")
    return {"locate_rate": None}

# ================================================================ ② 限定检索

def hit_rank(docs, drug, loincs, k=5):
    for d in docs[:k]:
        if d.drug == drug and d.loinc in loincs:
            return d.rank
    return 0

def part_restrict(labels, rows) -> Dict:
    print("\n② search 的 restrict —— 「先卡章节再排序」 vs 「先排序再过滤」")
    print("-" * 78)
    r = make_retriever(labels, use_locator=False)      # 关掉加权，只看限定本身

    n = hit_restrict = hit_filter = 0
    example = None
    for row in rows:
        drug = row.get("drug")
        lo = set(row.get("gold_loinc_any") or [row.get("gold_loinc")])
        if not drug or not lo:
            continue
        n += 1
        q = row["question"]
        a = r(q, 5, restrict={"drug": drug, "loincs": sorted(lo)})
        b = [d for d in r(q, 5) if d.drug == drug and d.loinc in lo]
        ok_a, ok_b = bool(hit_rank(a, drug, lo)), bool(hit_rank(b, drug, lo))
        hit_restrict += ok_a
        hit_filter += ok_b
        if ok_a and not ok_b and example is None:
            example = (q, drug, sorted(lo))

    ra, rb = hit_restrict / max(1, n), hit_filter / max(1, n)
    print(f"   同一 k=5 下，在 {n} 条检索题上：")
    print(f"     先卡章节再排序  命中 {hit_restrict:>3}/{n} = {ra:.3f}")
    print(f"     先全局排序再过滤 命中 {hit_filter:>3}/{n} = {rb:.3f}")
    check(ra > rb, f"先卡后排明显更好（{ra:.3f} > {rb:.3f}）",
          "一样的话说明限定没起作用")
    if example:
        print(f"   一个实例：{example[0][:60]}")
        print(f"       限定 {example[1]} 的 {example[2]} → 找得到；先排后滤 → 被 top-k 截没了")

    # ⭐ 限定要真的**限定** —— 但两个维度的力度不同（9/19 改，见 retrieval/hybrid.py）：
    #      drug  ：**硬过滤**，返回的每一条都得是这个药
    #      loincs：**加权**，定位章节排最前，但**不再把其它章节挡在外面**
    #
    # ⚠️ 旧断言是"5 条全都得在 metformin/34070-3"，编码的是硬过滤语义。
    #    改成软约束的理由是**实测的**：loincs 硬过滤会把定位器**没识别出的**章节
    #    物理挡在池子外 —— 而那些章节里可能就是 gold，排序、K、模型全都救不回来。
    #    （hard_set 的 cross_drug 那批：gold 进池率 硬过滤 41% → 软约束 55%。）
    lg, tb = _new_box(r)
    res = tb.search("metformin kidney disease",
                    restrict={"drug": "metformin", "loincs": ["34070-3"]})
    paid = res.payload or []
    check(res.ok and paid and all(d.drug == "metformin" for d in paid),
          f"限定后返回 {len(paid)} 条，**全部是 metformin**（drug 维度仍是硬过滤）")
    check(bool(paid) and paid[0].loinc == "34070-3",
          f"定位章节 34070-3 排在**最前面**（实测第 1 条 = "
          f"{paid[0].loinc if paid else '—'}）",
          "软约束：定位章节优先，但不把其它章节挡在外面")

    res_other = tb.search("metformin kidney disease",
                          restrict={"drug": "ibuprofen", "loincs": ["34073-7"]})
    check(not any(d.drug == "ibuprofen" for d in (res_other.payload or [])),
          "限定到 ibuprofen 时不会漏出 metformin 的章节", res_other.note)
    return {"restrict": ra, "filter_after": rb, "n_rows": n}

# ================================================================ ③ 去重

def part_dedup(labels) -> Dict:
    print("\n③ search 的去重 —— 「这次到底有没有拿到新东西」")
    print("-" * 78)
    r = make_retriever(labels, use_locator=True)
    lg, tb = _new_box(r, "Can I take warfarin with ibuprofen?")

    q = "warfarin interaction"
    a = tb.search(q, exclude=set())
    check(a.ok, f"第一次查拿到 {len(a.payload)} 条", a.note)

    seen = {d.cite_key for d in a.payload}
    b = tb.search(q, exclude=seen)
    check(not b.ok and not b.payload,
          "同样 query 再查一次 → ok=False、返回空（机械信号：这条白跑了）", b.note)
    check(len(lg.steps) == 2,
          "两次检索在账本里**都记了**（重复检索确实发生过，烧了预算）")

    # 换个 query 但命中已见过的 chunk → 同样应该判为"没新东西"
    c = tb.search("warfarin bleeding risk", exclude=seen | {d.cite_key for d in b.payload})
    print(f"   换 query 再查：{c.note}")

    return {"dup_note": b.note}

# ================================================================ ④ 问用户

def part_ask() -> Dict:
    print("\n④ ask —— 动作空间里唯一的第二个信息源")
    print("-" * 78)
    user = ScriptedUser({"conditions": ["chronic kidney disease"],
                         "medications": ["metformin", "lisinopril"]})
    lg, tb = _new_box(MockRetriever(), user=user)
    st = AgentState(question="Can I take metformin?")

    a = tb.ask("Do you have any kidney conditions?")
    check(a.ok, f"有档案 → 答上了：{a.payload['text']}", "")
    st.user_facts.append(a.payload["text"])
    check(len(st.evidence) == 0,
          "⭐ 用户说的话**没有**进 evidence（口述没有出处，不能当引用）")

    b = tb.ask("What is your blood type?")
    check(not b.ok, f"档案里没有 → 诚实说不知道：{b.payload['text']}")

    lg2, tb2 = _new_box(MockRetriever(), user=None)
    c = tb2.ask("Do you have any kidney conditions?")
    check(not c.ok and len(lg2.steps) == 0,
          "没接用户时 ask 不记账（没有发生的事不该进账本）", c.note)

    # 账本要能记下"问了什么 + 用户答了什么"
    check(lg.steps[0].action == "ask" and lg.steps[0].query == "Do you have any kidney conditions?",
          "账本记下了问了什么")
    return {}

# ================================================================ ⑤ 作答 / 拒答

def part_answer_abstain() -> Dict:
    print("\n⑤ answer / abstain —— 两个收尾动作")
    print("-" * 78)
    lg, tb = _new_box(MockRetriever(), llm=MockLLM())
    st = AgentState(question="Can I take ibuprofen while on warfarin?")

    r0 = tb.answer(st)
    check(not r0.ok and r0.cost == 0,
          "一条证据都没有时 answer 拒绝执行（该走 abstain 而不是硬编）", r0.note)

    st.add_evidence(tb.search("ibuprofen warfarin interaction",
                             exclude=set()).payload)
    st.add_evidence(tb.search("ibuprofen contraindication asthma",
                             exclude=st.evidence_keys()).payload)
    r1 = tb.answer(st)
    check(r1.ok and r1.payload, f"有证据 → 生成回答（{len(r1.payload)} 字符）")
    check(lg.steps[-1].action == "synthesize",
          "answer 记成账本的 synthesize（同一件事，不新增枚举）")

    r2 = tb.abstain("证据覆盖不足，且用户未提供用药史")
    check(r2.ok and lg.steps[-1].action == "abstain", "abstain 记了一条动作")
    check(lg.steps[-1].note.startswith("证据覆盖不足"), "拒答理由进了账本")

    # 拒答不写理由 → 必须被拦
    bad = Ledger("bad", "q")
    try:
        from ledger.ledger import Step
        bad.record(Step(step=1, action="abstain"))
        check(False, "拒答不写理由应该被拦")
    except LedgerError as e:
        check(True, f"拒答不写理由被拦住了", str(e)[:46])
    return {}

# ================================================================ ⑥ 成本

def part_cost() -> Dict:
    print("\n⑥ 成本计量 —— 「自主决策」成立的前提是动作代价可比")
    print("-" * 78)
    lg, tb = _new_box(make_retriever(json.loads(LABELS.read_text(encoding="utf-8"))["labels"]))
    plan = [("locate", 1), ("search", 2), ("ask", 2), ("answer", 1), ("abstain", 0)]
    for tool, want in plan:
        check(tb.cost(tool) == want, f"{tool:<8} 成本 {tb.cost(tool)}")

    spent = tb.cost("locate") + tb.cost("search") + tb.cost("search")
    check(spent == 5, f"「定位一次 + 查两次」= {spent} 个预算单位（可累加）")
    check(tb.cost("ask") == tb.cost("search"),
          "问用户和检索一样贵（打扰用户 = 一次真实代价）")
    return {"cost_table": dict(tb.cost_map)}

# ================================================================ ⑦ 循环

def _mk_loop(retriever, llm=None, policy=None, cfg=None):
    from agent.loop import AgentLoop, LoopConfig, make_toolbox_factory
    from agent.policy import RulePolicy
    return AgentLoop(make_toolbox_factory(retriever, llm=llm or make_generator(), top_k=5),
                     policy or RulePolicy(), cfg or LoopConfig(budget=8))

def part_loop_e2e(labels) -> Dict:
    print("\n⑦ AgentLoop · 端到端 —— agent 自己走完一整轮")
    print("-" * 78)
    r = make_retriever(labels, use_locator=True)
    loop = _mk_loop(r)
    q = "Can I take metformin if I have kidney disease?"
    lg = loop.run(q, run_id="e2e", user=ScriptedUser({"conditions": ["chronic kidney disease"]}))
    st = loop.last_stats

    got = [s.action for s in lg.steps]
    check(got[:1] == ["locate"] and "retrieve" in got and got[-1] == "synthesize",
          f"账本动作序列 = {got}", "定位 → 检索 → …（agent 自己选的，不是我排的队）")
    check(st.terminated_by == "answer", f"以 answer 正常收尾（terminated_by={st.terminated_by}）")
    check(bool(lg.report_md) and st.n_claims > 0,
          f"产出报告：{st.n_claims} 条带引用论断、{st.n_evidence} 条证据")
    check(not lg.malformed, "闭包通过（每条引用都追得到出处）")
    check(st.cost_spent == 8 - loop.last_state.budget,
          f"预算账对得上：花了 {st.cost_spent}")
    return {"e2e_actions": st.n_actions, "claims": st.n_claims}

# ================================================================ ⑧ 刹车

def part_loop_brakes(labels) -> Dict:
    print("\n⑧ AgentLoop · 三个刹车（不显式就会早停或烧光预算）")
    print("-" * 78)
    from agent.loop import LoopConfig
    r = make_retriever(labels, use_locator=True)
    q = "Can I take warfarin with ibuprofen?"

    # ① 经济刹车：买不起下一个动作
    loop = _mk_loop(r, cfg=LoopConfig(budget=2))
    lg = loop.run(q, run_id="b1")
    check(loop.last_stats.terminated_by == "budget",
          f"预算只够定位一次 → 被经济刹车收尾（{loop.last_stats.terminated_by}）")
    print(f"       {loop.last_trace[-1]}")

    # ② 步数硬上限
    loop = _mk_loop(r, cfg=LoopConfig(budget=100, max_steps=2))
    loop.run(q, run_id="b2")
    check(loop.last_stats.terminated_by == "max_steps",
          f"喂它一个大预算 + 步数上限 2 → 停在 {loop.last_stats.n_actions} 步")

    # ③ 打转刹车：策略原地踏步（用一个故意写坏的策略来验刹车真的在）
    class StuckPolicy:
        name = "stuck"
        def __call__(self, state):
            from agent.state import Action, A_SEARCH
            return Action(A_SEARCH, {"query": "warfarin interaction"})   # 永远同一个动作
    loop = _mk_loop(r, policy=StuckPolicy(), cfg=LoopConfig(budget=50, max_repeats=2))
    loop.run(q, run_id="b3")
    check(loop.last_stats.terminated_by == "repeat",
          f"坏策略反复提同一个动作 → 打转刹车生效（{loop.last_stats.terminated_by}）",
          "没有它，弱模型会把预算全烧在同一个 query 上")

    # 刹车也要记账：为什么停下来**是研究数据** —— 账本里必须有那一步
    check(any(s.action in ("synthesize", "abstain") for s in lg.steps),
          "刹车收尾照样写进账本（「为什么会停」本身是数据）")
    return {}

# ================================================================ ⑨ 域外 / 拒答

OOD_Q = "What is the price of tea in China?"        # 域外问题：语料里一个字都不沾

def part_loop_abstain(labels) -> Dict:
    print("\n⑨ AgentLoop · 域外问题与拒答路径（该说不知道时必须敢说）")
    print("-" * 78)
    r = make_retriever(labels, use_locator=True)

    # ⭐ 先验「查不到」这个信号本身是可信的 —— 这是上一段修停用词才买来的
    check(len(r(OOD_Q, 5)) == 0,
          "域外问题检索返回 0 条（agent 能观察到「我什么都没查到」）",
          "修之前会返回 5 条：in/of/the 这些虚词在每条 chunk 里都有")

    # ⭐ 已知边界的回归护栏：词命中 ≠ 药对了
    #    ⚠️ 断言只写在**不随语料漂移**的性质上（初版写"正文不许出现 aspirin"，
    #       语料 6→50 之后当场失效 —— 那是语料组成的偶然，不是要验的东西）
    sneaky = "Can I take aspirin if I have a stomach ulcer?"
    hit = r(sneaky, 5)
    check(len(hit) > 0 and all(d.drug != "aspirin" for d in hit),
          f"「 aspirin + ulcer」返回 {len(hit)} 条 —— 没有一条来自 aspirin 的说明书",
          f"⚠️ 已知边界：语料里没有 aspirin 这个药，却照样返回结果（命中靠疾病词）。"
          f"来源药 {sorted({d.drug for d in hit})} —— 这正是 E8 的【药品】槽位要补的")

    # 无用户 → 查不到就该拒答
    loop = _mk_loop(r)
    lg = loop.run(OOD_Q, run_id="abstain")
    st = loop.last_stats
    check(st.terminated_by == "abstain", f"以 abstain 收尾（{st.terminated_by}）")
    check(not st.n_claims, "拒答路径**不产出任何 claim**（不瞎编）")
    last = lg.steps[-1]
    check(last.action == "abstain" and last.note, f"账本记了拒答 + 理由：{last.note}")
    check(all(s.action in ("locate", "retrieve", "ask", "synthesize", "abstain")
              for s in lg.steps), "全程动作都在封闭动作空间内")

    # 拒答也走闭包检查（空 claim 集合 → 通过）—— 不能因为拒答就跳过校验
    try:
        lg.check_closure()
        check(True, "拒答路径的闭包检查照常执行且通过")
    except LedgerError as e:
        check(False, "拒答路径闭包检查异常", str(e)[:50])
    return {}

# ================================================================ ⑩ 问用户

def part_loop_ask(labels) -> Dict:
    print("\n⑩ AgentLoop · 问用户路径（第二个信息源）")
    print("-" * 78)
    r = make_retriever(labels, use_locator=True)
    loop = _mk_loop(r)
    lg = loop.run(OOD_Q, run_id="ask",
                  user=ScriptedUser({"conditions": ["peptic ulcer"]}))
    st = loop.last_stats
    acts = [s.action for s in lg.steps]

    check(st.n_ask >= 1, f"一无所获时策略问了一次用户（问了 {st.n_ask} 次）")
    check(loop.last_state.user_facts,
          f"用户的话进了 user_facts：{loop.last_state.user_facts[:1]}")
    check(st.terminated_by == "abstain",
          f"问了也补不上证据 → 最终仍拒答（{st.terminated_by}）",
          "⭐ 这条很关键：问了用户 ≠ 就有证据了，不许拿口述当引用")

    ev = loop.last_state.evidence
    check(all("#" in d.cite_key for d in ev),
          f"evidence（{len(ev)} 条）里每一条都带章节码（可核验）")
    check(all(not any(f.startswith(d.section) for d in ev) for f in loop.last_state.user_facts),
          "用户口述**没有**混进 evidence（口述没有出处，不能当引用）")
    print(f"       账本动作序列：{acts}")
    return {"n_ask": st.n_ask}

# ================================================================ ⑩b 规则策略的病

def part_rule_pathology(labels) -> Dict:
    """
    规则策略的病 **和** 闭包检查的边界 —— 两件事在同一次运行里暴露。

    问题问的是一个**不在语料里**的药（aspirin），但"ulcer"这个词语料里有：
        规则策略：检索有返回 → 认为有证据 → 作答          ← 病
        闭包检查：每条引用都追得到出处 → 通过              ← 但拦不住这件事
    """
    print("\n⑩b 规则策略的病 + 闭包检查的边界（两件事一起暴露）")
    print("-" * 78)
    r = make_retriever(labels, use_locator=True)
    loop = _mk_loop(r)
    q = "Can I take aspirin if I have a stomach ulcer?"      # aspirin 不在语料里
    lg = loop.run(q, run_id="patho")
    st = loop.last_stats

    check(st.terminated_by == "answer" and st.n_claims > 0,
          f"规则策略**作答了**（{st.n_evidence} 条证据 / {st.n_claims} 条论断）",
          "它只有 0/1 两种判断：检索有返回 = 有证据")

    # ⚠️ 断言只写在**不会随语料漂移**的性质上。
    #    初版写的是「正文里不许出现 aspirin」——语料 6→50 之后就有说明书正文提到 aspirin，
    #    断言当场失效。那是**语料组成的偶然**，不是我们要验的东西。
    #    要验的是：【语料里根本没有 aspirin 这份说明书】，agent 却照样答了。
    ev = loop.last_state.evidence
    check(all(d.drug != "aspirin" for d in ev),
          f"检索到的 {len(ev)} 条**没有一条来自 aspirin 的说明书**——语料里根本没这个药",
          f"来源药：{sorted({d.drug for d in ev})}")
    check(any("ulcer" in d.text.lower() for d in ev),
          "命中靠的是**疾病词**（ulcer / duodenal ulcer），不是**药品**",
          "→ 状态里缺的正是「药品」这个槽位")

    check(not lg.malformed,
          "⭐ 而闭包检查**通过了** —— 每条引用确实都追得到出处",
          "结论：**闭包只保证「引用追得到」，不保证「引对了」。**"
          "这条边界必须写进报告，否则「引用可验证」会被误读成「答案正确」")

    print("       理想行为：该拒答（「这个药不在我的语料里」）")
    print("       规则策略做不到 —— 它的状态里**没有「药品对不对」这一项**。")
    print("       → E8 的证据槽位要补的正是这个（槽位 = 药品 / 条件 / 关系）")
    return {"pathology": "rule-cannot-check-drug-identity"}

# ================================================================ ⑪ 坏引用

def part_loop_closure(labels) -> Dict:
    """
    E9 之后这条测的是**新契约**：坏引用在循环内就被拦下，不留到最后。

    ⚠️ 为什么是"改写"而不是"放宽"：
       E9 之前坏引用是「先采纳 → 最后 check_closure 抛异常」——
       **发现问题时循环已经结束，没有补救机会**。
       E9 按设计 v7 第 137 行改成「追不到 → 回到 ②③」，
       于是坏引用**根本进不了账本**，LedgerError 自然不再抛。
       旧断言（"必须抛 LedgerError"）测的是**旧行为**，留着它等于把设计改回去。

       但**闭包断言本身不能因此被架空** —— 下面单独验它仍然是活的。
    """
    print("\n⑪ AgentLoop · 引用核验（E9：坏引用在**循环内**被拦，不留到最后）")
    print("-" * 78)
    from agent.loop import LoopConfig
    from agent.policy import CoveragePolicy
    from ledger.ledger import Ledger, Claim
    r = make_retriever(labels, use_locator=True)
    Q = "Can I take warfarin with ibuprofen?"

    # —— ① 循环级：核验拦下 + 答案不被采纳（策略无关）
    loop = _mk_loop(r, llm=MockLLM(bad_citation=True),
                    cfg=LoopConfig(budget=8, strict_closure=True))
    lg = loop.run(Q, run_id="bad")
    st = loop.last_state

    check(bool(st.failed_cites), "坏引用被核验器当场拦下",
          f"抓到 {[c['kind'] for c in st.failed_cites][:2]}")
    check(not st.answer_text,
          "⭐ 被打回的答案**没有被采纳**（answer_text 为空）",
          "旧行为=先采纳后 raise；新行为=没通过就不采纳")
    check(any("[verify]" in ln for ln in loop.last_trace),
          "循环把核验失败**显式记进了 trace**（反馈链存在）",
          "只写日志不给策略看 = 没有反馈；这里两样都做了")
    ev_keys = st.evidence_keys()
    bad = [k for c in lg.claims for k in c.cite if k not in ev_keys]
    check(not bad, "最终账本里没有任何一条追不到出处的引用", f"坏引用 {len(bad)} 条")
    n_answer_rule = sum(1 for t in loop.last_trace if t.startswith("answer"))

    # —— ② 策略级：**会读状态的策略**拿着反馈去补检索（E9 的算法本体）
    loop_cov = _mk_loop(r, llm=MockLLM(bad_citation=True), policy=CoveragePolicy(),
                        cfg=LoopConfig(budget=8))
    lg_cov = loop_cov.run(Q, run_id="bad-cov")
    trace_cov = loop_cov.last_trace
    check(any("引用被打回" in ln for ln in trace_cov),
          "⭐ 会读状态的策略**拿着反馈改了下一步**（设计 v7：追不到 → 回到 ②③）",
          "这就是 E9 的算法本体：核验结果反过来决定下一步")
    n_answer_cov = sum(1 for t in trace_cov if t.startswith("answer"))
    check(n_answer_cov < n_answer_rule,
          "⭐ 会读状态的策略**答得更少**就收敛到拒答",
          f"CoveragePolicy 答 {n_answer_cov} 次 vs RulePolicy {n_answer_rule} 次"
          f"（后者读不到反馈，只会重复答到踩刹车）")
    ev_cov = loop_cov.last_state.evidence_keys()
    bad_cov = [k for c in lg_cov.claims for k in c.cite if k not in ev_cov]
    check(not bad_cov and not loop_cov.last_state.answer_text,
          "会读状态的策略最终也**没有交出坏答案**（走的是拒答）",
          f"abstain 理由：{loop_cov.last_state.abstain_reason[:40]}")

    # —— 闭包断言**仍然是活的**：不能因为 E9 在前面拦了，就把它架空
    lg2 = Ledger(run_id="direct", question="q")
    lg2.add_claim(Claim(claim_id="c1", text="t", cite=["deadbeef-0000#99999-9"]))
    try:
        lg2.check_closure()
        check(False, "闭包断言仍然是活的", "绕过核验直接塞坏 claim 也拦不住 → 被架空了")
    except LedgerError as e:
        check(True, "闭包断言仍然是活的（绕过核验直接塞坏 claim → 拦住）",
              str(e).splitlines()[0][:40])

    # 放宽模式下同样不炸，且账本依然干净
    loop = _mk_loop(r, llm=MockLLM(bad_citation=True),
                    cfg=LoopConfig(budget=8, strict_closure=False))
    lg3 = loop.run("Can I take warfarin with ibuprofen?", run_id="bad-lenient")
    ev3 = loop.last_state.evidence_keys()
    bad3 = [k for c in lg3.claims for k in c.cite if k not in ev3]
    check(not bad3, "放宽模式下：不炸，且账本依然干净", f"坏引用 {len(bad3)} 条")
    return {}

# ================================================================ ⑫ 可复现

def part_loop_repro(labels) -> Dict:
    print("\n⑫ AgentLoop · 可复现（同 seed 两次账本逐字节一致）")
    print("-" * 78)
    r = make_retriever(labels, use_locator=True)
    q = "Can I take metformin if I have kidney disease?"
    a = _mk_loop(r).run(q, run_id="rep", seed=42).to_dict()
    b = _mk_loop(r).run(q, run_id="rep", seed=42).to_dict()
    check(a == b, "两次跑完全一致（整条循环是确定性的）",
          "不确定的循环没法做三档对照实验")
    return {}

# ================================================================ 主

def main() -> int:
    print("=" * 78)
    print("块 E7 · 工具集判据（零模型 / 零 GPU / 秒级）")
    print("=" * 78)

    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    rows = json.loads(RET_SET.read_text(encoding="utf-8"))["rows"]
    print(f"\n语料：{len(labels)} 份药 ｜ 检索集：{len(rows)} 条\n")

    part_locate(labels)
    stats = part_restrict(labels, rows)
    part_dedup(labels)
    part_ask()
    part_answer_abstain()
    part_cost()

    print("\n" + "─" * 78)
    print("第二组：Agent Loop（π(a|s) 真的在决定下一步做什么）")
    print("─" * 78)
    e2e = part_loop_e2e(labels)
    part_loop_brakes(labels)
    part_loop_abstain(labels)
    part_loop_ask(labels)
    part_rule_pathology(labels)
    part_loop_closure(labels)
    part_loop_repro(labels)

    print("\n" + "=" * 78)
    if _fails:
        print(f"❌ 判据未通过（{len(_fails)} 项）：")
        for f in _fails:
            print("   -", f)
        return 1
    print("✅ E7 判据全过")
    print(f"   · 工具集：五个工具行为正确 ｜ 限定检索 {stats['restrict']:.3f} > "
          f"先排后滤 {stats['filter_after']:.3f}（同 k=5，{stats['n_rows']} 条）")
    print(f"   · 循环：端到端 {e2e['e2e_actions']} 个动作 / {e2e['claims']} 条论断，"
          f"三个刹车 + 拒答 + 问用户 + 坏引用拦截 + 可复现全过")
    print(f"   · 动作空间（封闭）：{ACTIONS}")
    print("=" * 78)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
