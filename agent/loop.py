# -*- coding: utf-8 -*-
"""
块 E7 · Agent Loop —— 把「状态 / 工具 / 策略」串成一个自主循环。

    while 没停:
        s = (q, E, T, b)          ← 看状态
        a = π(s)                  ← 策略选动作（唯一由策略决定的地方）
        执行 a，写账本，扣预算       ← 工具干活
    直到 a ∈ {answer, abstain}，或触发刹车

━━━ ⭐ 这个文件和 E2 的 research_agent.py 差在哪 ━━━

    E2  控制流是**写死的**：拆问题 → 逐个检索 → 取前 3 → 写报告
        agent 没有任何决定权，它只是在跑我排好的队

    E7  控制流是**策略给的**：下一步做什么由 π(a|s) 决定
        我不知道它会走哪条路 —— 这才是"自主"

    ⭐ E2 那份**不删**，留作对照组：
       "同一批题，流水线 vs agent 循环"本身就是一组结果。

━━━ 终止条件（必须全部显式，否则 agent 不是早停就是烧光预算）━━━

    ① 策略选了 answer / abstain          —— 正常收尾（策略自己决定停）
    ② 预算不够再做一个动作                —— 经济刹车，**强制收尾**
    ③ 步数达到硬上限                     —— 兜住策略失控
    ④ 策略连续提出已试过的动作            —— 循环刹车（防止原地打转）

    ⚠️ ②③④ 都是**强制收尾**，不是异常。它们要写进账本（带理由），
       因为"agent 为什么会停"本身就是研究数据 ——
       E8 要看的正是"规则策略有多少比例是被预算逼停的"。

━━━ 引用契约仍然是最硬的那条线 ━━━

    agent 自由度越高，"某条引用到底哪来的"就越难追。
    所以循环的最后一步永远是同一件事：
        **把回答里的每条引用拿回账本里对一遍（闭包检查）**。
    对不上 → 这轮标记 malformed，不进统计。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ledger.ledger import Ledger, LedgerError, Claim
from agent.report import split_claims, looks_like_abstention
from agent.select import budget_from_env, order_from_env
from agent.verify import verify_answer
from agent.state import (
    AgentState, Action, TERMINAL_ACTIONS,
    A_LOCATE, A_SEARCH, A_ASK, A_ANSWER, A_ABSTAIN,
)
from agent.tools import ToolBox, ToolResult, ScriptedUser
from agent.policy import Policy


# ---------------------------------------------------------------- 配置


@dataclass
class LoopConfig:
    """循环的刹车参数。**三个都是"防失控"，不是"算法"。**"""

    budget: int = 8          # b：动作预算（成本单位见 tools.DEFAULT_COST）
    max_steps: int = 12      # 硬上限：策略提出多少动作都要停
    max_repeats: int = 2     # 连续提出已试过的动作几次 → 刹车
    strict_closure: bool = True   # 闭包对不上时抛不抛（批量跑批设 False）


@dataclass
class RunStats:
    """一轮的可测量结果 —— E8 的三档对照就靠这些数。"""

    question: str = ""
    terminated_by: str = ""          # answer | abstain | budget | max_steps | repeat
    n_actions: int = 0
    n_search: int = 0
    n_ask: int = 0
    n_locate: int = 0
    cost_spent: int = 0
    n_evidence: int = 0
    n_claims: int = 0
    malformed: bool = False
    answer_text: str = ""
    abstain_reason: str = ""
    # ⭐ 策略选了 answer，但生成层说"证据不足" ——
    #    这是一个**独立的行为信号**，不是错误：说明"策略以为够了、实际不够"。
    #    E8 的三档对照里，这个数越小说明停止准则越准。
    answered_but_insufficient: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


# ---------------------------------------------------------------- 工厂


def make_toolbox_factory(retriever: Any, llm: Any = None, top_k: int = 5,
                         cost: Optional[Dict[str, int]] = None,
                         evidence_budget: Optional[int] = None,
                         select_order: Optional[str] = None
                         ) -> Callable[[Ledger, Optional[ScriptedUser]], ToolBox]:
    """把「检索器 + LLM」封成一个**按轮造工具箱**的工厂。

    ⚠️ 为什么要工厂而不是直接传 ToolBox：
       工具箱绑着账本，账本是**一轮一个**的。每个用户的档案也不同。
       传一个现成的工具箱 = 所有轮共用一个账本，账本立刻失去意义。
    """
    # ⚠️ E13b 的两个旋钮在这里**统一落到一处**（和 retrieval/factory.py 收口
    #    「14 处写死 BM25Retriever」是同一个道理）：参数缺省时读环境变量，
    #    免得每个判据脚本各写一份 —— 那种分散正是「写了但没接进流程」的温床。
    ev_budget = evidence_budget if evidence_budget is not None else budget_from_env()
    order = select_order or order_from_env()

    def factory(ledger: Ledger, user: Optional[ScriptedUser]) -> ToolBox:
        return ToolBox(retriever=retriever, ledger=ledger, user=user,
                       llm=llm, top_k=top_k, cost=cost,
                       evidence_budget=ev_budget, select_order=order)
    return factory


# ---------------------------------------------------------------- 循环


class AgentLoop:
    """一轮完整的自主研究。**它自己不决定任何动作 —— 那是策略的事。**"""

    def __init__(self,
                 toolbox_factory: Callable[..., ToolBox],
                 policy: Policy,
                 config: Optional[LoopConfig] = None):
        self.toolbox_factory = toolbox_factory
        self.policy = policy
        self.cfg = config or LoopConfig()
        # 最近一轮的产物（批量跑批时逐轮读）
        self.last_state: Optional[AgentState] = None
        self.last_stats: Optional[RunStats] = None
        self.last_trace: List[str] = []

    # ------------------------------------------------------------ 主循环

    def run(self, question: str, run_id: str = "run-0", seed: int = 42,
            user: Optional[ScriptedUser] = None, verbose: bool = False,
            carry: Optional[AgentState] = None) -> Ledger:
        """
        Args:
            carry: ⭐ E10 的跨轮继承口子 —— 上一轮的 AgentState（多轮会话用）。
                   **只继承证据和用户口述**，不继承动作账本/预算/定位：
                     · 证据继承   → 第 2 轮不必重查第 1 轮查过的药（复用的机制在这）
                     · 账本不继承 → 它是**每轮内部**打转刹车的依据；
                                    跨轮继承会让策略"因为上一轮查过所以不许再查"，那是错的
                     · 预算不继承 → 每轮是独立的一次咨询
                     · 定位不继承 → 它是"这问题该查哪一节"，换问题就作废
        """
        lg = Ledger(run_id=run_id, question=question, seed=seed)
        tb = self.toolbox_factory(lg, user)
        st = AgentState(question=question, budget=self.cfg.budget,
                        has_user=tb.user is not None,
                        has_locator=tb.locator is not None,
                        known_drugs=_corpus_drugs(tb.retriever),
                        drug_by_setid=_drug_by_setid(tb.retriever),
                        evidence=list(carry.evidence) if carry else [],
                        user_facts=list(carry.user_facts) if carry else [])
        # ⭐ E10：把继承来的证据**登记进本轮的账本**。
        #    ⚠️ 不登记的话，第 2 轮的引用会全部闭包失败 ——
        #       因为本轮没重查（证据是继承的），账本里自然没有那一步。
        #    登记用 `CARRIED_STEP` 标记来源，**报告里看得见"这条来自上一轮"**。
        if carry and carry.evidence:
            lg.mark_carried(carry.evidence)
        # E9：语料级引用键，引用核验判 not_in_corpus 要用。一轮算一次，不算贵。
        self._corpus_keys = _corpus_keys(tb.retriever)
        trace: List[str] = []
        repeats = 0
        terminated_by = ""

        while True:
            # ---- 刹车 ③：步数硬上限
            if len(lg.steps) >= self.cfg.max_steps:
                self._force_finish(tb, st, trace, "达到步数硬上限")
                terminated_by = "max_steps"
                break

            act = self.policy(st)
            cost = tb.cost(act.action)
            terminal = act.action in TERMINAL_ACTIONS

            # ---- 刹车 ②：买不起下一个动作 → 强制收尾
            if not terminal and cost > st.budget:
                self._force_finish(tb, st, trace, "预算不够再做一个动作")
                terminated_by = "budget"
                break

            # ---- 刹车 ④：原地打转
            #   ⚠️ E9 之后**不能再加 `not terminal`**：answer 可能被引用核验打回，
            #      于是它会被反复提出。旧写法把 terminal 排除在外，导致一个被打回的答案
            #      能一路重试到 max_steps（2026-09-18 实测重试了 10 次才被步数上限拦住）。
            if st.has_tried(act.action, act.arg):
                repeats += 1
                if repeats >= self.cfg.max_repeats:
                    self._force_finish(tb, st, trace, "策略连续提出已试过的动作")
                    terminated_by = "repeat"
                    break
            else:
                repeats = 0

            # ---- 执行
            res = self._execute(act, tb, st)
            st.budget -= res.cost
            st.step = len(lg.steps)
            st.mark_tried(act.action, act.arg, res.ok, res.note)
            line = self._fmt(act, res)
            trace.append(line)
            if verbose:
                print("  " + line)

            if terminal:
                # ⭐ E9：answer **核验过了才算真的答了**。
                #    核验没过（answer_text 被作废、failed_cites 有记录）→ **不终止**，
                #    退回循环把失败信息交给策略 —— 设计 v7："追不到 → 回到 ②③"。
                #    ⚠️ 不这么改的话，"核验"就只是个事后判死，反馈链是断的。
                if act.action == A_ANSWER and not st.answer_text:
                    trace.append("[verify] 引用核验未通过 → 退回循环，带着失败信息重检索")
                    if verbose:
                        print("  [verify] 引用核验未通过 → 退回循环")
                    continue
                terminated_by = act.action
                break

        # ---- 收尾：解析引用 + 闭包检查（无论怎么停，这一步都要走）
        self._finalize(lg, st)

        stats = self._stats(lg, st, terminated_by)
        self.last_state, self.last_stats, self.last_trace = st, stats, trace
        return lg

    # ------------------------------------------------------------ 执行

    def _execute(self, act: Action, tb: ToolBox, st: AgentState):
        if act.action == A_LOCATE:
            r = tb.locate(str(act.arg))
            if r.payload:
                st.located = r.payload
            return r

        if act.action == A_SEARCH:
            arg = act.arg if isinstance(act.arg, dict) else {"query": act.arg}
            r = tb.search(str(arg.get("query", "")),
                          restrict=arg.get("restrict"),
                          exclude=st.evidence_keys(),      # ⭐ 去重交给工具，不靠策略回忆
                          k=arg.get("k"))
            if r.payload:
                st.add_evidence(r.payload)
            return r

        if act.action == A_ASK:
            r = tb.ask(str(act.arg))
            # ⚠️ 用户说的话进 user_facts，**不进 evidence** ——
            #    口述没有出处（chunk_id/content_hash），混进证据 = 闭包断言作废
            if r.ok and r.payload:
                st.user_facts.append(r.payload["text"])
            return r

        if act.action == A_ANSWER:
            r = tb.answer(st)
            if not r.ok:
                return r
            # ⭐ E9 的落点：**答案生成之后，先过机械核验，再决定采不采纳**。
            #    这是设计 v7 第 137 行那句"核验结果反过来决定下一步"的实现位置。
            #    核验器是纯函数（agent/verify.py），判据在 scripts/eval_verify.py。
            v = verify_answer(r.payload or "", st.evidence_keys(),
                              getattr(self, "_corpus_keys", None))
            if v.ok:
                st.answer_text = r.payload
                st.failed_cites = []
                return r
            # 核验不过 → **不采纳**：答案作废、坏引用写进 state（策略读得到）
            st.answer_text = ""
            st.failed_cites = [x.to_dict() for x in v.bad]
            return ToolResult(
                tool=r.tool, ok=False, payload=None, cost=r.cost,
                note=f"引用核验未通过（{len(v.bad)}/{v.n_cites} 条）：{v.summary()}")

        if act.action == A_ABSTAIN:
            st.abstain_reason = str(act.arg or "（策略未给理由）")
            return tb.abstain(st.abstain_reason)

        raise ValueError(f"循环不认识的动作：{act.action!r}")     # 动作空间是封闭的

    def _force_finish(self, tb: ToolBox, st: AgentState, trace: List[str], reason: str):
        """刹车触发了：有证据就答，没证据就拒答。**照样记账，因为"为什么会停"是数据。**"""
        # ⭐ E9：如果之前有引用核验失败、而且**一直没答出合法答案** —— 不许硬答。
        #    给一条没有出处的用药建议是本项目的红线；宁可拒答。
        #    ⚠️ 注意条件是 `and not st.answer_text`：如果能答出合法答案，正常走下面的分支。
        if st.failed_cites and not st.answer_text:
            reason = f"{reason}；且引用核验一直没过（{len(st.failed_cites)} 条坏引用）"
            st.abstain_reason = reason
            r = tb.abstain(reason)
            act = Action(A_ABSTAIN, reason)
            st.mark_tried(act.action, act.arg, r.ok, f"[刹车] {reason}")
            st.step = len(trace)
            trace.append(f"[刹车] {reason} → {act.action}（{r.note}）")
            return
        if st.evidence:
            r = tb.answer(st)
            st.answer_text = r.payload or ""
            act = Action(A_ANSWER)
        else:
            st.abstain_reason = reason
            r = tb.abstain(reason)
            act = Action(A_ABSTAIN, reason)
        st.mark_tried(act.action, act.arg, r.ok, f"[刹车] {reason}")
        st.step = len(trace)
        trace.append(f"[刹车] {reason} → {act.action}（{r.note}）")

    # ------------------------------------------------------------ 收尾

    def _finalize(self, lg: Ledger, st: AgentState):
        if not st.answer_text:
            # 拒答路径：没有报告、没有 claim —— 闭包检查空转通过（这是对的）
            return
        lg.report_md = st.answer_text
        for i, (text, cites) in enumerate(split_claims(st.answer_text), start=1):
            lg.add_claim(Claim(claim_id=f"c{i}", text=text, cite=cites))
        if not lg.claims:
            # ⚠️ 「答了但没有一条引用」有**两种**，不能一律判成格式错误：
            #      ① 生成层直接说"证据不足，答不了" → 这是**有效拒答**，格式没坏
            #      ② 真编了一段没出处的话 → 这才是 malformed
            #    混为一谈的后果：一个本该算"拒答"的 run 被记成"格式崩了"，
            #    然后 E8 的拒答准确率就被污染了。（这个坑是拿 mock 跑第一遍时
            #    看到 malformed=true 才发现的 —— 断言没红，是**数字不对劲**。）
            if not looks_like_abstention(st.answer_text):
                lg.malformed = True
        try:
            lg.check_closure()
        except LedgerError:
            lg.malformed = True
            if self.cfg.strict_closure:
                raise

    def _stats(self, lg: Ledger, st: AgentState, terminated_by: str) -> RunStats:
        return RunStats(
            question=lg.question,
            terminated_by=terminated_by,
            n_actions=len(st.tried),
            n_search=st.count(A_SEARCH),
            n_ask=st.count(A_ASK),
            n_locate=st.count(A_LOCATE),
            cost_spent=self.cfg.budget - st.budget,
            n_evidence=len(st.evidence),
            n_claims=len(lg.claims),
            malformed=lg.malformed,
            answer_text=st.answer_text,
            abstain_reason=st.abstain_reason,
            answered_but_insufficient=bool(st.answer_text)
            and looks_like_abstention(st.answer_text),
        )

    @staticmethod
    def _fmt(act: Action, res) -> str:
        arg = act.arg if not isinstance(act.arg, dict) else act.arg.get("query", "")
        head = f"{act.action}({str(arg)[:40]})"
        why = f"  ｜策略依据：{act.note}" if act.note else ""
        return f"{head:<44} ok={res.ok!s:<5} cost={res.cost}  {res.note}{why}"


def _corpus_drugs(retriever) -> List[str]:
    """语料里有哪些药 —— 从检索器的 chunk 上取（E8 的覆盖度要用）。"""
    chunks = getattr(retriever, "chunks", None)
    if not chunks:
        return []
    return sorted({c.drug for c in chunks if getattr(c, "drug", "")})


def _corpus_keys(retriever) -> set:
    """
    语料里**全部** (setid#loinc) 引用键 —— E9 的引用核验要拿它判 not_in_corpus。

    ⚠️ 为什么必须传它：不传的话，"引用一个语料里根本没有的章节"会被降级成
       `not_retrieved`（因为"不在证据里"包含了"语料里也没有"）。
       两者**补救动作完全不同**：
         not_retrieved  → 再检索一次就行
         not_in_corpus  → 模型在编，检索一百次也变不出来
       把这两个混成一个，策略会白白重试。
    """
    chunks = getattr(retriever, "chunks", None)
    if not chunks:
        return set()
    return {f"{c.doc_id}#{c.loinc}" for c in chunks if getattr(c, "loinc", None)}


def _drug_by_setid(retriever) -> Dict[str, str]:
    """setid → 药名。E9 的定向补检索要用（见 AgentState.drug_by_setid 的注释）。"""
    chunks = getattr(retriever, "chunks", None)
    if not chunks:
        return {}
    return {c.doc_id: c.drug for c in chunks if getattr(c, "drug", "")}


# ---------------------------------------------------------------- 打印


def render_trace(lg: Ledger, trace: Optional[List[str]] = None,
                 stats: Optional[RunStats] = None) -> str:
    """把一轮走的路打出来 —— **这是"agent 真的在自主"唯一的可视化证据**。"""
    out = [f"问题：{lg.question}", ""]
    if trace:
        for i, line in enumerate(trace, 1):
            out.append(f"  {i:>2}. {line}")
    else:
        for s in lg.steps:
            got = f"  拿到 {len(s.retrieved)} 条" if s.retrieved else ""
            out.append(f"  {s.step:>2}. {s.action:<10}{s.note or ''}{got}")
    out.append("")
    if lg.report_md:
        out.append("回答：")
        out.append("  " + lg.report_md.replace("\n", "\n  "))
    else:
        out.append(f"拒答：{stats.abstain_reason if stats else '（无）'}")
    if stats:
        out.append("")
        out.append(f"  [停止原因={stats.terminated_by}｜动作 {stats.n_actions} 次"
                   f"（查 {stats.n_search}/问 {stats.n_ask}/定位 {stats.n_locate}）"
                   f"｜花了 {stats.cost_spent} 个预算单位"
                   f"｜证据 {stats.n_evidence} 条｜论断 {stats.n_claims} 条]")
    return "\n".join(out)


__all__ = ["AgentLoop", "LoopConfig", "RunStats", "make_toolbox_factory", "render_trace"]
