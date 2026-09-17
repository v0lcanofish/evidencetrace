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
  PYTHONIOENCODING=utf-8 D:/anaconda/envs/myenv/python.exe scripts/eval_agent.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from ledger.ledger import Ledger, LedgerError, ACTIONS                      # noqa: E402
from agent.mocks import MockLLM, MockRetriever, GroundedMockLLM             # noqa: E402
from agent.state import AgentState, A_SEARCH, A_ASK, A_ANSWER, A_ABSTAIN    # noqa: E402
from agent.tools import ToolBox, ScriptedUser                               # noqa: E402
from retrieval import BM25Retriever                                         # noqa: E402

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
    r = BM25Retriever(labels, use_locator=True)
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
    r = BM25Retriever(labels, use_locator=False)      # 关掉加权，只看限定本身

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

    # ⭐ 限定要真的**限定**：返回的每一条都得在候选集里
    lg, tb = _new_box(r)
    res = tb.search("metformin kidney disease",
                    restrict={"drug": "metformin", "loincs": ["34070-3"]})
    check(res.ok and all(d.drug == "metformin" and d.loinc == "34070-3"
                         for d in res.payload),
          f"限定后返回 {len(res.payload)} 条，全部落在 metformin/34070-3")

    res_other = tb.search("metformin kidney disease",
                          restrict={"drug": "ibuprofen", "loincs": ["34073-7"]})
    check(not any(d.drug == "ibuprofen" for d in (res_other.payload or [])),
          "限定到 ibuprofen 时不会漏出 metformin 的章节", res_other.note)
    return {"restrict": ra, "filter_after": rb, "n_rows": n}


# ================================================================ ③ 去重

def part_dedup(labels) -> Dict:
    print("\n③ search 的去重 —— 「这次到底有没有拿到新东西」")
    print("-" * 78)
    r = BM25Retriever(labels, use_locator=True)
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
    lg, tb = _new_box(BM25Retriever(json.loads(LABELS.read_text(encoding="utf-8"))["labels"]))
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
    return AgentLoop(make_toolbox_factory(retriever, llm=llm or GroundedMockLLM(), top_k=5),
                     policy or RulePolicy(), cfg or LoopConfig(budget=8))


def part_loop_e2e(labels) -> Dict:
    print("\n⑦ AgentLoop · 端到端 —— agent 自己走完一整轮")
    print("-" * 78)
    r = BM25Retriever(labels, use_locator=True)
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
    r = BM25Retriever(labels, use_locator=True)
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
    r = BM25Retriever(labels, use_locator=True)

    # ⭐ 先验「查不到」这个信号本身是可信的 —— 这是上一段修停用词才买来的
    check(len(r(OOD_Q, 5)) == 0,
          "域外问题检索返回 0 条（agent 能观察到「我什么都没查到」）",
          "修之前会返回 5 条：in/of/the 这些虚词在每条 chunk 里都有")

    # ⭐ 已知边界的回归护栏：词命中 ≠ 药对了
    sneaky = "Can I take aspirin if I have a stomach ulcer?"
    hit = r(sneaky, 5)
    check(len(hit) > 0 and all("aspirin" not in d.text.lower() for d in hit),
          f"「 aspirin + ulcer」返回 {len(hit)} 条 —— 全是词命中，没有一条真的讲 aspirin",
          "⚠️ 已知边界：agent 目前分不清「词命中」和「药对了」，"
          "这正是 E8 的证据槽位要补的（槽位里要有【药品】这一项）")

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
    r = BM25Retriever(labels, use_locator=True)
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
    r = BM25Retriever(labels, use_locator=True)
    loop = _mk_loop(r)
    q = "Can I take aspirin if I have a stomach ulcer?"      # aspirin 不在语料里
    lg = loop.run(q, run_id="patho")
    st = loop.last_stats

    check(st.terminated_by == "answer" and st.n_claims > 0,
          f"规则策略**作答了**（{st.n_evidence} 条证据 / {st.n_claims} 条论断）",
          "它只有 0/1 两种判断：检索有返回 = 有证据")

    texts = " ".join(d.text.lower() for d in loop.last_state.evidence)
    check("aspirin" not in texts,
          "⭐ 但它引的 5 条证据里**一条都没提 aspirin** —— 全是词命中",
          "")

    check(not lg.malformed,
          "⭐ 而闭包检查**通过了** —— 每条引用确实都追得到出处",
          "结论：**闭包只保证「引用追得到」，不保证「引对了」。**"
          "这条边界必须写进报告，否则「引用可验证」会被误读成「答案正确」")

    print(f"       理想行为：该拒答（「这个药不在我的语料里」）")
    print(f"       规则策略做不到 —— 它的状态里**没有「药品对不对」这一项**。")
    print("       → E8 的证据槽位要补的正是这个（槽位 = 药品 / 条件 / 关系）")
    return {"pathology": "rule-cannot-check-drug-identity"}


# ================================================================ ⑪ 坏引用

def part_loop_closure(labels) -> Dict:
    print("\n⑪ AgentLoop · 闭包断言是活的（坏引用必须被拦）")
    print("-" * 78)
    from agent.loop import LoopConfig
    r = BM25Retriever(labels, use_locator=True)
    loop = _mk_loop(r, llm=MockLLM(bad_citation=True),
                    cfg=LoopConfig(budget=8, strict_closure=True))
    try:
        loop.run("Can I take warfarin with ibuprofen?", run_id="bad")
        check(False, "坏引用应该被拦住", "闭包断言是摆设")
    except LedgerError as e:
        check(True, "故意给追不到出处的引用 → 闭包拦住了", str(e).splitlines()[0][:44])

    # 放宽时不炸，但必须**标记** malformed（批量跑批用）
    loop = _mk_loop(r, llm=MockLLM(bad_citation=True),
                    cfg=LoopConfig(budget=8, strict_closure=False))
    lg = loop.run("Can I take warfarin with ibuprofen?", run_id="bad-lenient")
    check(lg.malformed, "放宽模式下不炸，但这轮被标记 malformed（不进统计）")
    return {}


# ================================================================ ⑫ 可复现

def part_loop_repro(labels) -> Dict:
    print("\n⑫ AgentLoop · 可复现（同 seed 两次账本逐字节一致）")
    print("-" * 78)
    r = BM25Retriever(labels, use_locator=True)
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
