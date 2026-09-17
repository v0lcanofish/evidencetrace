# -*- coding: utf-8 -*-
"""
块 E7 · 工具集 —— agent 的手脚。

五个工具，每个 = **一次动作**（和 state.py 的动作空间一一对应）：

    工具                        作用                            成本
    ─────────────────────────────────────────────────────────────
    locate(query)               问题 → 该查哪一节（LOINC）         1
    search(query, restrict)     拿 query 换证据                    2
    ask(question)               问用户                             2
    answer(state)               用已有证据作答（收尾）              1
    abstain(reason)             拒答（收尾）                       0

━━━ ⭐ 为什么「成本」必须写进代码 ━━━

    "agent 自己决定下一步做什么"这句话要成立，**动作的代价必须可比**。
    如果 locate 和 search 都免费，策略就永远选"再查一轮"——
    所谓的"自主决策"退化成"把预算烧光为止"。

    成本单位不是钱，是**三种真实代价的折算**：
        ① token   —— 检索结果要占上下文（search 最贵）
        ② 打扰    —— 问用户是有社交成本的（ask 也贵）
        ③ 决策    —— locate 纯本地规则，几乎免费

    ⚠️ **诚实边界**：下面这组数值是**先验设定，不是实测标定**。
       它们只保证"量级关系对"（search/ask 比 locate 贵），
       绝对值没有实验依据。E8 会用评测集把阈值和成本一起标定出来。

━━━ ⭐ 为什么每个工具都要写账本 ━━━

    因为 E1 定的那条不变量：**报告里每一条引用，都要能在账本里追到出处**。
    agent 自由度越高，"某条引用到底哪来的"就越难追 —— 账本是唯一的刹车。
    所以工具的记账不是"埋点"，是**正确性要求**。

━━━ ⚠️ 关于「重复动作」的处理 ━━━

    search 会带 `exclude`（已见过的引用键）：重复检索**照样记账**
    （它确实发生了一次，烧了预算），但返回值里没有新东西 → `ok=False`。
    这样策略拿到的是"这次白跑了"的**机械信号**，而不是靠它自己回忆。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from ledger.ledger import (
    Ledger, Step, Doc,
    ACTION_LOCATE, ACTION_RETRIEVE, ACTION_ASK, ACTION_SYNTHESIZE, ACTION_ABSTAIN,
)
from agent.report import SYNTH_PROMPT, USER_NOTE_TEMPLATE, build_evidence_block
from agent.state import AgentState


# ---------------------------------------------------------------- 成本表

DEFAULT_COST: Dict[str, int] = {
    "locate": 1,
    "search": 2,
    "ask": 2,
    "answer": 1,
    "abstain": 0,
}


@dataclass
class ToolResult:
    """一次工具调用的结果。

    ⚠️ `ok` 的含义是「**这次动作拿到有用东西了吗**」，不是「有没有报错」。
       这个区别是刻意的：策略要判断的是"要不要换一条路"，
       而"没报错但什么也没拿到"恰恰最需要它知道。
    """

    tool: str
    ok: bool
    payload: Any = None
    cost: int = 0
    note: str = ""


# ---------------------------------------------------------------- 用户模拟器

# 槽位表：问哪类问题 → 去档案里取哪个槽位。
# ⚠️ 顺序有意义：先匹配具体的。
USER_SLOTS = (
    ("conditions", ("condition", "disease", "diagnos", "kidney", "renal", "diabet",
                    "asthma", "pregnan", "liver", "heart", "ulcer", "bleed")),
    ("medications", ("medication", "medicine", "taking", "drug", "prescription",
                     "blood thinner", "anticoagul", "supplement")),
    ("allergies", ("allerg",)),
    ("age", ("age", "how old")),
    ("alcohol", ("alcohol", "drink")),
)


class ScriptedUser:
    """
    最小用户模拟器 —— 让 `ask` 这条路径**可测**。

    每个场景带一份档案（profile），agent 问的时候按槽位取；取不到就诚实说不知道。

    ⚠️ **诚实边界（要写进报告）**：这是**模拟用户**，不是真人。
       它能验证"agent 会不会在信息不足时去问、问了会不会用上"，
       **不能**替代真实用户研究。真实用户的回答模糊/矛盾/答非所问，
       这些都超出了这里的建模范围。

    为什么要让它存在：`ask` 是动作空间里唯一的**第二个信息源**。
       没有它，π(a|s) 就退化成"检索几轮"，agent 与 RAG 的差别就只剩多轮——
       而那点差别撑不起"它自己决定每一步做什么"这句话。
    """

    def __init__(self, profile: Optional[Dict[str, Any]] = None, name: str = "patient"):
        self.profile = profile or {}
        self.name = name

    def answer(self, question: str) -> Dict[str, Any]:
        """返回 {"answered": bool, "slot": str|None, "text": str}。确定性、不联网。"""
        q = (question or "").lower()
        for slot, kws in USER_SLOTS:
            if not any(k in q for k in kws):
                continue
            val = self.profile.get(slot)
            if val:
                text = val if isinstance(val, str) else ", ".join(str(v) for v in val)
                return {"answered": True, "slot": slot,
                        "text": f"Yes — my {slot}: {text}."}
            # 问到了点子上，但档案里没有 → 诚实说不知道
            # ⭐ 这条路径很重要：agent 必须能接受"问了也白问"，
            #    而不是把用户没说过的当成用户说过的（那就变成幻觉了）
            return {"answered": False, "slot": slot,
                    "text": f"I'm not sure about my {slot}."}
        return {"answered": False, "slot": None,
                "text": "I don't have that information."}


# ---------------------------------------------------------------- 工具集


class ToolBox:
    """
    agent 能用的全部工具。**它自己不决定调哪个** —— 那是策略的事。

    职责边界（务必守住）：
        ToolBox  执行动作 + 记账 + 报告"这次拿到了什么"
        Policy   决定下一个动作是什么
        Loop     串起来 + 管预算 + 判终止
    """

    def __init__(self,
                 retriever: Callable[..., List[Doc]],
                 ledger: Ledger,
                 locator: Any = None,
                 user: Optional[ScriptedUser] = None,
                 llm: Optional[Callable[[str], str]] = None,
                 top_k: int = 5,
                 cost: Optional[Dict[str, int]] = None):
        self.retriever = retriever
        self.ledger = ledger
        # locator 缺省时从检索器身上取（BM25Retriever 自带 SectionLocator）
        self.locator = locator if locator is not None else getattr(retriever, "locator", None)
        self.user = user
        self.llm = llm
        self.top_k = top_k
        self.cost_map = {**DEFAULT_COST, **(cost or {})}
        # 只算一次：检索器支不支持"限定章节再排序"
        self._retriever_takes_restrict = _accepts_kw(self.retriever, "restrict")

    def cost(self, tool: str) -> int:
        return self.cost_map.get(tool, 1)

    def _next_step(self) -> int:
        return len(self.ledger.steps) + 1

    # ------------------------------------------------------------ locate

    def locate(self, query: str) -> ToolResult:
        """问题 → 该查哪一节。

        ⭐ 这是本项目相对"通用 RAG"的**结构先验**：
           DailyMed 的章节带 LOINC 医学编码，所以"该查哪一节"不是语义猜，
           而是查表 —— 而且可解释："我查相互作用那节，因为你在问能不能一起吃。"
        """
        if self.locator is None:
            return ToolResult("locate", False, note="没有章节定位器，这步跳过")

        r = self.locator(query)
        loincs = r.get("loincs") or []
        note = r.get("reason") or "定位不出来，交给检索兜底"
        self.ledger.record(Step(step=self._next_step(), action=ACTION_LOCATE,
                                query=query, note=note))
        return ToolResult("locate", ok=bool(loincs), payload=r,
                          cost=self.cost("locate"), note=note)

    # ------------------------------------------------------------ search

    def search(self, query: str, restrict: Optional[Dict[str, Any]] = None,
               exclude: Optional[Set[str]] = None, k: Optional[int] = None) -> ToolResult:
        """检索证据。

        Args:
            restrict: {"drug": str|None, "loincs": [...]}
                      ⭐ **先卡章节再排序**，不是"排完序再过滤"——
                         后者会在该章节其实有料时，因为全局排名低而误判成"这节没东西"。
            exclude:  已见过的引用键，用于"这次有没有拿到新东西"
        """
        k = k or self.top_k
        exclude = exclude or set()

        if self._retriever_takes_restrict:
            docs = list(self.retriever(query, k, restrict=restrict))
        else:
            docs = list(self.retriever(query, k))
            if restrict:
                docs = _apply_restrict(docs, restrict)

        fresh = [d for d in docs if d.cite_key not in exclude]
        n_dup = len(docs) - len(fresh)

        # ⚠️ 账本里记**过滤后**的 fresh —— 重复条目不该再做一次归因输入
        self.ledger.record(Step(step=self._next_step(), action=ACTION_RETRIEVE,
                                query=query, retrieved=fresh))

        if not docs:
            return ToolResult("search", False, payload=[], cost=self.cost("search"),
                              note="这条 query 一无所获")
        if not fresh:
            return ToolResult("search", False, payload=[], cost=self.cost("search"),
                              note=f"拿到 {n_dup} 条但**全是已见过的**（重复检索）")
        note = f"新拿到 {len(fresh)} 条" + (f"，另有 {n_dup} 条重复" if n_dup else "")
        return ToolResult("search", True, payload=fresh,
                          cost=self.cost("search"), note=note)

    # ------------------------------------------------------------ ask

    def ask(self, question: str) -> ToolResult:
        """问用户 —— 动作空间里唯一的第二个信息源。

        ⚠️ 用户说的话 **不进 evidence**，进 state.user_facts。
           原因：证据要能引用、能核验（chunk_id + content_hash）；
           用户口述没有出处，把它混进证据就是把"可核验"和"不可核验"搅在一起，
           闭包断言立刻失去意义。
        """
        if self.user is None:
            return ToolResult("ask", False, cost=0, note="没有用户可问（离线跑批未接）")

        r = self.user.answer(question)
        self.ledger.record(Step(step=self._next_step(), action=ACTION_ASK,
                                query=question, note=r["text"]))
        return ToolResult("ask", ok=bool(r["answered"]), payload=r,
                          cost=self.cost("ask"), note=r["text"])

    # ------------------------------------------------------------ answer

    def answer(self, state: AgentState) -> ToolResult:
        """用已有证据作答。**答案不是策略编的，是工具按证据生成的。**"""
        if not state.evidence:
            return ToolResult("answer", False, cost=0,
                              note="一条证据都没有 —— 这条路径该走 abstain")

        # ⚠️ 用户口述要插在**问题之后、证据之前**，不能拼在末尾 ——
        #    拼在末尾会被并进最后一条证据的正文里，等于伪造了那段说明书的内容。
        #    （这个坑写 mock LLM 的时候就会暴露：它按证据块切分找关键词，
        #      用户的一句话会被当成"这段证据里写了 X"。）
        q = state.question
        if state.user_facts:
            q = q + "\n" + USER_NOTE_TEMPLATE.format(
                facts="\n".join(f"- {f}" for f in state.user_facts))
        prompt = SYNTH_PROMPT.format(q=q, evidence=build_evidence_block(state.evidence))
        text = self.llm(prompt) if self.llm else ""

        self.ledger.record(Step(step=self._next_step(), action=ACTION_SYNTHESIZE,
                                note=f"用了 {len(state.evidence)} 篇证据"))
        return ToolResult("answer", bool(text), payload=text,
                          cost=self.cost("answer"),
                          note=f"用 {len(state.evidence)} 篇证据生成回答")

    # ------------------------------------------------------------ abstain

    def abstain(self, reason: str) -> ToolResult:
        """拒答。**拒答是一个决策，不是"什么都没发生"。**"""
        self.ledger.record(Step(step=self._next_step(), action=ACTION_ABSTAIN,
                                note=reason))
        return ToolResult("abstain", True, payload=reason,
                          cost=self.cost("abstain"), note=reason)


# ---------------------------------------------------------------- 小工具


def _accepts_kw(fn: Any, name: str) -> bool:
    """这个可调用对象收不收 `name` 这个关键字参数。"""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _apply_restrict(docs: List[Doc], restrict: Dict[str, Any]) -> List[Doc]:
    """给不支持 restrict 的检索器兜底：**排序后过滤**。

    ⚠️ 这条路径和"先卡后排"不等价：全局排名靠后的章节会被 top-k 截掉，
       于是"这节其实有料"被误判成"这节没东西"。
       所以它是兜底，不是正路 —— 支持 restrict 的检索器优先走正路。
    """
    drug = restrict.get("drug")
    loincs = set(restrict.get("loincs") or [])
    out = []
    for d in docs:
        if drug and not _doc_belongs_to(d, drug):
            continue
        if loincs and d.loinc not in loincs:
            continue
        out.append(d)
    return out


def _doc_belongs_to(d: Doc, drug: str) -> bool:
    """这条证据是不是这份药的。

    Doc 现在有 drug 字段（E7 加的），但**老账本 JSON 里没有** ——
    读回旧账本时 drug 是空串，所以留一条按 title 兜底的路
    （title 格式见 chunker.to_doc：`chunk_id|content_hash|药名 / 章节名`）。
    """
    if d.drug:
        return d.drug.lower() == drug.lower()
    return drug.lower() in (d.title or "").lower()


__all__ = ["ToolBox", "ToolResult", "ScriptedUser", "DEFAULT_COST", "USER_SLOTS"]
