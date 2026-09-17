# -*- coding: utf-8 -*-
"""
块 E7 · agent 的状态与动作空间 —— 策略 π(a|s) 的「右边」和「左边」。

    s = (q, E, T, b)
        问题 q ｜ 已收集的证据 E ｜ 已试过的动作 T ｜ 剩余预算 b

    a ∈ { locate, search, ask, answer, abstain }

━━━ 为什么单开一个文件，而不是塞进 loop.py ━━━

    设计 v7 里，策略有三档实现（规则 / LLM / 训练），要能**对照**。
    对照的前提是**循环和策略互不认识**：

        loop.py   只认 State，不认策略是想出来的还是算出来的
        policy.py 只认 State，不认循环怎么调工具

    State 是两者的**公共契约**，谁都不该独占它 —— 所以它必须独立成一个模块。
    如果塞进 loop.py，policy.py 就得 import loop.py，
    策略一改就可能碰坏循环，三档对照就无从谈起。

━━━ 动作 → 账本的映射 ━━━

    agent 词汇         账本动作           为什么
    ─────────────────────────────────────────────────────────
    search      →     retrieve      同一件事，不新增枚举
    answer      →     synthesize    同一件事，不新增枚举
    locate / ask / abstain          E7 真正新增的三个（见 ledger.py）

    ⚠️ 这个映射表是整个 E7 唯一"两套词汇"的地方，集中在这里，
       不让它散落到各个模块 —— 否则以后改动作空间要满仓库找。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ledger.ledger import (
    Doc,
    ACTION_LOCATE, ACTION_RETRIEVE, ACTION_ASK, ACTION_SYNTHESIZE, ACTION_ABSTAIN,
)

# ---------------------------------------------------------------- 动作空间

A_LOCATE = "locate"          # 定位：问题 → 该查哪一节（LOINC）
A_SEARCH = "search"          # 检索：拿 query 换证据
A_ASK = "ask"                # 问用户：拿问题换用户的实际情况
A_ANSWER = "answer"          # 作答：收尾动作
A_ABSTAIN = "abstain"        # 拒答：收尾动作

ACTIONS = (A_LOCATE, A_SEARCH, A_ASK, A_ANSWER, A_ABSTAIN)

# agent 词汇 → 账本词汇
LEDGER_ACTION: Dict[str, str] = {
    A_LOCATE: ACTION_LOCATE,
    A_SEARCH: ACTION_RETRIEVE,
    A_ASK: ACTION_ASK,
    A_ANSWER: ACTION_SYNTHESIZE,
    A_ABSTAIN: ACTION_ABSTAIN,
}

# 收尾动作：选到它，这一轮就结束（循环的终止条件之一）
TERMINAL_ACTIONS = (A_ANSWER, A_ABSTAIN)


@dataclass
class Action:
    """策略的输出：选一个动作，并给它一个参数。

    arg 的含义随动作变：
        locate  → query 字符串
        search  → {"query": str, "restrict": {"drug":…, "loincs":[…]} | None}
        ask     → 要问用户的话
        answer  → None（答案由工具用已有证据生成，不是策略编的）
        abstain → 拒答理由
    """

    action: str
    arg: Any = None

    def __post_init__(self):
        if self.action not in ACTIONS:
            raise ValueError(f"未知动作 {self.action!r}；动作空间是 {ACTIONS}")


@dataclass
class TriedAction:
    """试过的动作（T）。

    ⭐ 它存在的唯一理由：**让"别重复走死路"变成机械可判**。
       没有 T，策略只能靠上下文猜"这个 query 我是不是查过了"，
       弱模型必然反复查同一个词，把预算烧光。
    """

    action: str
    arg: Any
    ok: bool            # 这次动作**拿到有用东西了吗**（不是"有没有报错"）
    note: str = ""      # 一句话结果，人读 + 写进 trace

    def key(self) -> str:
        """去重键：同一个动作 + 同一个参数 = 同一次尝试。"""
        return f"{self.action}|{_norm_arg(self.arg)}"


def _norm_arg(arg: Any) -> str:
    """把参数规范化成可比较的字符串（大小写/空白不敏感）。"""
    if isinstance(arg, dict):
        q = _norm_arg(arg.get("query", ""))
        r = arg.get("restrict") or {}
        lo = ",".join(sorted(r.get("loincs") or []))
        return f"{q}|{r.get('drug') or ''}|{lo}"
    return re.sub(r"\s+", " ", str(arg or "").strip().lower())


# ---------------------------------------------------------------- 状态


@dataclass
class AgentState:
    """agent 的全部记忆。**除了它，循环和策略之间没有别的共享状态。**"""

    question: str

    # ---- E：已收集的证据（每条都带引用契约 chunk_id + content_hash）
    evidence: List[Doc] = field(default_factory=list)

    # ---- T：已试过的动作
    tried: List[TriedAction] = field(default_factory=list)

    # ---- b：剩余预算（单位见 tools.py 的成本表）
    budget: int = 8

    # ---- 定位轨迹：locate 的结果，供 search 复用（"查哪一节"的答案）
    located: Optional[Dict[str, Any]] = None

    # ---- 用户说过的话：证据之外的第二个信息源，**不能和证据混为一谈**
    #      （证据能引用、能核验；用户口述不能 —— 这是本项目的一条红线）
    user_facts: List[str] = field(default_factory=list)

    # ---- 收尾时填
    answer_text: str = ""
    abstain_reason: str = ""

    # ---- 环境能力：动作空间**实际**有哪些动作可选
    #      ⭐ 这两项属于状态，不属于策略 —— "有没有用户可问"是 agent 的处境，
    #         不是策略的偏好。策略必须知道它不能选一个环境不支持的动作。
    has_user: bool = False
    has_locator: bool = False

    # 步数（= 账本里的 step 数，由循环维护）
    step: int = 0

    # ------------------------------------------------------------ 证据

    def evidence_keys(self) -> Set[str]:
        """已收集证据的引用键（setid#LOINC）。"""
        return {d.cite_key for d in self.evidence}

    def add_evidence(self, docs: List[Doc]) -> List[Doc]:
        """收新证据，返回**真正新增**的那部分（重复的按 cite_key 丢掉）。

        ⚠️ 去重按 cite_key（setid#LOINC）而不是 doc_id ——
           一份说明书有多节共用同一个 setid，按 doc_id 去重会把第二三节全滤掉。
           （这个坑 E2 接编排时踩过一次，见 ledger.py 的 _first_step_of 注释。）
        """
        have = self.evidence_keys()
        fresh = [d for d in docs if d.cite_key not in have]
        for d in fresh:
            have.add(d.cite_key)
        self.evidence.extend(fresh)
        return fresh

    def n_evidence_by_section(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for d in self.evidence:
            if d.loinc:
                out[d.loinc] = out.get(d.loinc, 0) + 1
        return out

    # ------------------------------------------------------------ 动作历史

    def mark_tried(self, action: str, arg: Any, ok: bool, note: str = "") -> TriedAction:
        t = TriedAction(action=action, arg=arg, ok=ok, note=note)
        self.tried.append(t)
        return t

    def tried_keys(self) -> Set[str]:
        return {t.key() for t in self.tried}

    def has_tried(self, action: str, arg: Any) -> bool:
        return TriedAction(action, arg, True).key() in self.tried_keys()

    def count(self, action: str) -> int:
        return sum(1 for t in self.tried if t.action == action)

    def last_tried(self, action: str) -> Optional[TriedAction]:
        for t in reversed(self.tried):
            if t.action == action:
                return t
        return None

    # ------------------------------------------------------------ 给策略看

    def summary(self) -> str:
        """人读（也喂给 LLM 策略）的状态摘要。

        ⭐ 三档策略对照时，这个摘要就是策略①②③的**同一份输入** ——
           输入一样，输出不同，差异才归因得到"策略"头上。
        """
        lines = [f"question: {self.question}",
                 f"budget_left: {self.budget}",
                 f"evidence: {len(self.evidence)} 条"]
        by_sec = self.n_evidence_by_section()
        if by_sec:
            lines.append("  by_section: " + ", ".join(f"{k}×{v}" for k, v in sorted(by_sec.items())))
        for d in self.evidence:
            lines.append(f"  - [{d.cite_key}] {d.section}: {d.text[:80]}...")
        if self.located:
            lines.append(f"located: drug={self.located.get('drug')} "
                         f"intent={self.located.get('intent')} "
                         f"loincs={self.located.get('loincs')}")
        if self.user_facts:
            lines.append("user said:")
            for f in self.user_facts:
                lines.append(f"  - {f}")
        if self.tried:
            lines.append("tried:")
            for t in self.tried:
                lines.append(f"  - {t.action}({_norm_arg(t.arg)}) -> ok={t.ok} {t.note[:60]}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "budget": self.budget,
            "step": self.step,
            "evidence": [d.cite_key for d in self.evidence],
            "located": self.located,
            "user_facts": self.user_facts,
            "tried": [{"action": t.action, "arg": _norm_arg(t.arg), "ok": t.ok, "note": t.note}
                      for t in self.tried],
            "answer_text": self.answer_text,
            "abstain_reason": self.abstain_reason,
        }


__all__ = [
    "AgentState", "Action", "TriedAction",
    "A_LOCATE", "A_SEARCH", "A_ASK", "A_ANSWER", "A_ABSTAIN",
    "ACTIONS", "TERMINAL_ACTIONS", "LEDGER_ACTION",
]
