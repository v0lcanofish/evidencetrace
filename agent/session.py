# -*- coding: utf-8 -*-
"""
多轮会话（E10）—— 记忆与状态。

━━━ 为什么需要这一层 ━━━

`AgentLoop.run(question)` 是**单轮**的：一个问题、一个账本、一个答案。
但真实的用药咨询是**多轮追问**，而项目立项动机的**第一条**就是它：

    用户：华法林有什么副作用？
    agent：……[答案]
    用户：那和阿司匹林一起吃呢？      ← "那…呢" 指代上一轮的"副作用"
    agent：……
    用户：孕妇能吃吗？                ← 主语没了，指代前面聊的药

**三件事要解决**：
    ① **指代缺失** —— 第 2、3 轮的问题里**没有药名**，检索直接打偏
    ② **重复劳动** —— 第 1 轮查过的药，第 2 轮又查一遍
    ③ **标准不能松** —— 多轮下的引用照样要过 E9 的核验

━━━ 记忆记什么（这是设计决策，不是实现细节）━━━

**不记"全部对话"，只记三样：**

    · 已确认的实体（药名，按提及顺序）  ← 指代消解靠它
    · 累积的证据（cite_key → Doc）      ← 跨轮复用靠它
    · 用户口述过的事实                  ← 和证据严格分开（老规矩）

⭐ **为什么只记实体不记整段对话**：记整段对话 = 把上一轮的**结论**也带进下一轮，
   而结论可能是错的 —— 错的东西会在多轮里**滚雪球**。
   实体是**已被机械确认过的**（药名是从库里的 known_drugs 里认出来的），风险低得多。

━━━ 一条反面判据：不串味 ━━━

    **过度消解也是错。**

    用户第 3 轮明确问了另一个药，agent 却沿用上一轮的焦点药 —— 这是**新的错误**，
    不是"记性好"。所以判据里专门有一条：换药时必须解析到**新药**。

    记忆的价值在"该记的时候记住"，**同样在于"该忘的时候忘掉"**。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# ---- 指代线索词 ------------------------------------------------------------
# ⚠️ 这份词表**必须保守**。宽一点会把"问另一个药"也误判成指代，
#    而误判的后果比漏判严重（漏判 = 少一次消解；误判 = 答错药）。
PRONOUN_CUES = (
    "it", "that", "this", "these", "those", "them", "they",
    "the same", "same one", "also", "too", "as well", "what about",
    "how about", "instead",
)

# 把上面那串编成一个词边界正则 —— 用 `in` 直接判会把 "item" 里的 "it" 也算上
_CUE_RE = re.compile(r"\b(" + "|".join(re.escape(c) for c in PRONOUN_CUES) + r")\b",
                     re.IGNORECASE)


@dataclass
class Resolution:
    """一次指代消解的结果。**判据直接打这个对象**，不打问题的文本。"""

    raw: str                      # 用户原话
    question: str                 # 送给 agent 的问题（消解后）
    drugs: List[str] = field(default_factory=list)   # 解析出的药
    mode: str = "none"            # explicit ｜ from_context ｜ none
    cue: str = ""                 # 命中的线索词（方便排查"为什么消解了"）

    def to_dict(self):
        return {"raw": self.raw, "question": self.question, "drugs": self.drugs,
                "mode": self.mode, "cue": self.cue}


@dataclass
class TurnRecord:
    """一轮的记账 —— 判据和可视化都读它。"""

    turn: int
    resolution: Resolution
    n_actions: int = 0
    n_search: int = 0
    n_evidence: int = 0          # 本轮结束时**累计**证据数
    n_new_evidence: int = 0      # 本轮的增量（跨轮复用的直接读数）
    answered: bool = False
    abstained: bool = False
    malformed: bool = False
    # ⭐ 本轮被核验打回的引用。**必须在清状态之前抄一份留下来** ——
    #    `AgentState.failed_cites` 每一轮开头会被清（否则上一轮的失败会污染
    #    下一轮的策略决策），但"这一轮发生过什么"是记录，**记录不能跟着清**。
    bad_cites: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def n_bad_cites(self) -> int:
        return len(self.bad_cites)

    def to_dict(self):
        return {**self.__dict__, "resolution": self.resolution.to_dict()}


# ---------------------------------------------------------------- 会话


class Session:
    """
    一次多轮会话。**它自己不检索、不生成** —— 只维护跨轮状态 + 做指代消解。

    用法：
        s = Session(known_drugs=[...])
        s.ask("What are the side effects of warfarin?")     # 第 1 轮
        s.ask("What about with ibuprofen?")                 # 第 2 轮，自动带上 warfarin
    """

    def __init__(self, known_drugs: Sequence[str], budget_per_turn: int = 8):
        self.known_drugs = [d for d in known_drugs if d]
        self.budget_per_turn = budget_per_turn
        # 焦点药：最近一次被**明确点名**的药（按提及顺序，最近的在最后）
        self.focus: List[str] = []
        # 跨轮累积
        self.evidence: list = []
        self.user_facts: List[str] = []
        self.turns: List[TurnRecord] = []
        self._carry = None            # 上一轮的 AgentState（给下一轮当底子）
        self.last_ledger = None       # 最近一轮的账本（判据要查 claims）

    # ------------------------------------------------------------ 指代消解

    def resolve(self, question: str) -> Resolution:
        """
        **机械消解，不问 LLM。**

        ⚠️ 判据是**线索词**，不是"有没有点名药"。三种情况要分开：

            ① 点名了药 + **没有**线索词   → 全新问题，**只取点名的**
                 "What are the side effects of metformin?"
            ② 点名了药 + **有**线索词     → 是"和它一起呢"，**焦点药也要带上**
                 "What about taking it with ibuprofen?"   ← it = 上一轮的药
                 这一条 2026-09-18 才补上：原来的写法只取点名的药，
                 **会把用户真正在问的那个药丢掉**（问的是"华法林+布洛芬"，
                 解析出来只剩布洛芬）。这是设计里的洞，不是实现 bug。
            ③ 没点名药 + 有线索词 + 有焦点 → 纯指代，用焦点药
                 "Is it safe during pregnancy?"           ← it = 上一轮的药

        ⭐ **本轮的焦点更新只由"点名的药"驱动**（`focus = named`），
           不由线索词驱动 —— 否则一句 "what about that" 就能把焦点改掉。
        """
        low = (question or "").lower()
        named = [d for d in self.known_drugs if d.lower() in low]
        m = _CUE_RE.search(low)
        cue = m.group(1).lower() if m else ""

        # ② / ③：有线索词且会话里有焦点
        if cue and self.focus:
            drugs = list(self.focus)
            for d in named:                     # 新点名的药追加在焦点后面
                if d not in drugs:
                    drugs.append(d)
            if named:
                self.focus = named              # ⭐ 焦点只被"点名的药"更新
            ctx_only = [d for d in drugs if d not in named]
            # 把上下文补出来的药**写进问题文本**，而不是只塞进 restrict：
            # 生成层也要知道在说哪个药，否则它写出来的答案照样是飘的。
            resolved = (f"{question} ({', '.join(ctx_only)})"
                        if ctx_only else question)
            return Resolution(question, resolved, drugs, "from_context", cue)

        # ①：点名了药、无线索词 —— 全新问题，**绝不掺上一轮的药**（防串味）
        if named:
            self.focus = named
            return Resolution(question, question, list(named), "explicit")

        # 其余一律不猜：可能是域外的药（aspirin 不在库里），也可能真没提药
        return Resolution(question, question, [], "none", cue)

    # ------------------------------------------------------------ 跑一轮

    def ask(self, question: str, loop, **run_kwargs) -> TurnRecord:
        """
        跑一轮。`loop` 是一个 AgentLoop 实例（注入进来，方便换策略做对照）。

        每轮**继承**上一轮的证据和用户口述，但**不继承**动作账本和预算 ——
        ⭐ 这个划分不是随便定的：
            · 证据要继承 → 否则第 2 轮会重查第 1 轮查过的药（复用失效）
            · 动作账本不继承 → 它是**每轮内部**打转刹车的依据，
              跨轮继承会让策略"因为上一轮查过所以不许再查"，那是错的
            · 预算不继承 → 每轮是独立的一次咨询
        """
        res = self.resolve(question)
        lg = loop.run(res.question, run_id=f"turn{len(self.turns)+1}",
                      carry=self._carry, **run_kwargs)
        st = loop.last_state
        stats = loop.last_stats
        self.last_ledger = lg

        before = len(self._carry.evidence) if self._carry is not None else 0
        rec = TurnRecord(
            turn=len(self.turns) + 1,
            resolution=res,
            n_actions=stats.n_actions,
            n_search=getattr(stats, "n_search", 0),
            n_evidence=len(st.evidence),
            n_new_evidence=max(0, len(st.evidence) - before),
            answered=bool(st.answer_text),
            abstained=bool(st.abstain_reason) and not st.answer_text,
            malformed=bool(getattr(lg, "malformed", False)),
            bad_cites=list(st.failed_cites),     # ⚠️ 必须在下面清状态**之前**抄
        )
        self.turns.append(rec)
        self.evidence = list(st.evidence)
        self.user_facts = list(st.user_facts)
        # ⚠️ carry 里要把**上一轮的收尾字段清掉**：它们是上一轮的产物，
        #    带进下一轮会让 `_finalize` 误以为下一轮已经答过了；
        #    `failed_cites` 不清的话，上一轮被打回的引用会**污染下一轮的策略决策**
        #    （策略会以为本轮也答被拒了，直接去补检索）。
        #    ⭐ 但**记录已经抄进 TurnRecord 了**（上面那行），所以清的是状态、不是历史。
        st.answer_text = ""
        st.abstain_reason = ""
        st.failed_cites = []
        self._carry = st
        return rec

    # ------------------------------------------------------------ 读数

    def summary(self) -> Dict[str, Any]:
        return {
            "n_turns": len(self.turns),
            "final_focus": list(self.focus),
            "context_resolved": sum(1 for t in self.turns
                                   if t.resolution.mode == "from_context"),
            "turns": [t.to_dict() for t in self.turns],
        }


__all__ = ["Session", "Resolution", "TurnRecord", "PRONOUN_CUES"]
