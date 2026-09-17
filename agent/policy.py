# -*- coding: utf-8 -*-
"""
块 E7 · 策略 π(a|s) 的接口 + 第一档实现（规则）。

    π(a | s)  ←  ⭐ 这就是本项目的算法核心

━━━ 三档实现的接口（设计 v7）━━━

    档  实现             性质                    块
    ────────────────────────────────────────────────
    ①   RulePolicy       启发式，零成本            E7（本文件 · 占位基线）
    ②   LLMPolicy        把状态给 LLM 让它选       E8（待写）
    ③   TrainedPolicy    用 ①② 的决策日志 SFT      E12（需要轻量 GPU）

    ⚠️ 三档**共用同一个接口**，循环一行不用改 ——
       这是"把行动决策从规则 → LLM → 训练"能做成**对照实验**的前提。
       如果三档各自改循环，量出来的差异就分不清是"策略不同"还是"实现不同"。

━━━ 规则策略为什么故意做得这么笨 ━━━

    它**只能判断"有证据 / 没证据"**（0/1），不会算"证据够不够"。
    所以它必然表现成两种病之一：
        · 拿到一条证据就急着答 → 答不全
        · 一条都没拿到就反复折腾 → 或者草率拒答

    **这不是偷懒，这是基线。** E8 的覆盖度算法要证明的就是
    "把 0/1 换成算出来的覆盖度，好多少" —— 没有这个笨基线，那个数字无从谈起。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agent.coverage import Coverage, compute_coverage
from agent.state import (
    AgentState, Action,
    A_LOCATE, A_SEARCH, A_ASK, A_ANSWER, A_ABSTAIN,
)


class Policy(ABC):
    """策略接口。**输入状态，输出动作** —— 除此之外不许碰别的。"""

    name: str = "policy"

    @abstractmethod
    def __call__(self, state: AgentState) -> Action:
        ...  # pragma: no cover

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


class RulePolicy(Policy):
    """
    ① 档：规则。

    规则表（**从上往下，第一条"还没试过"的胜出**）：

        R1  没定过位                    → locate(原问题)
        R2  定位到了、那一节还没查过      → search(原问题, 限定=定位结果)
        R3  没裸查过                    → search(原问题)         ← 兜底：不限定
        R4  已经有证据                   → answer
        R5  查了但一无所获、有用户、没问过 → ask(一句通用追问)
        R6  其他                        → answer / abstain（看有没有证据）

    ⚠️ 每条规则都要检查 `has_tried` —— 规则策略没有记忆，**状态就是它的记忆**。
       不检查的话它会永远提同一个动作，把预算烧光（这正是弱模型的通病）。
    """

    name = "rule"

    def __init__(self, locate_first: bool = True, ask_when_empty: bool = True):
        self.locate_first = locate_first
        self.ask_when_empty = ask_when_empty

    def __call__(self, state: AgentState) -> Action:
        q = state.question

        # R0 预算见底 → 直接收尾，别提出一个买不起的动作
        if state.budget <= 0:
            return self._finish(state, "预算耗尽")

        # R1 定位（环境没装定位器就跳过）
        if self.locate_first and state.has_locator and not state.has_tried(A_LOCATE, q):
            return Action(A_LOCATE, q)

        # R2 限定检索：只在定位到的那份药的、那几个章节里查
        loc = state.located or {}
        if loc.get("loincs"):
            arg = {"query": q, "restrict": {"drug": loc.get("drug"),
                                            "loincs": list(loc["loincs"])}}
            if not state.has_tried(A_SEARCH, arg):
                return Action(A_SEARCH, arg)

        # R3 裸检索兜底：定位失败 / 限定查空时，退回全局查一次
        if not state.has_tried(A_SEARCH, q):
            return Action(A_SEARCH, q)

        # R4 有证据就答（⚠️ 这就是它的病：不问"够不够"）
        if state.evidence:
            return Action(A_ANSWER)

        # R5 一无所获 → 问用户一次（规则策略唯一的补救手段）
        if (self.ask_when_empty and state.has_user
                and state.count(A_ASK) == 0 and not state.has_tried(A_ASK, _ASK_TEXT)):
            return Action(A_ASK, _ASK_TEXT)

        # R6 兜底
        return self._finish(state, "检索不到任何证据")

    @staticmethod
    def _finish(state: AgentState, reason: str) -> Action:
        """收尾：有证据就答，没证据就拒答。**拒答必须带理由。**"""
        return Action(A_ANSWER) if state.evidence else Action(A_ABSTAIN, reason)


# 规则策略的追问是**固定一句** —— 它没有能力识别"缺了哪一半"。
# E8 的覆盖度算法会算出"还缺哪个槽位"，从而生成**针对性的**追问。
_ASK_TEXT = "Could you tell me more about your situation or any conditions you have?"


class CoveragePolicy(Policy):
    """
    ①' 档：**覆盖度策略**（块 E8 的核心增量）。

    和 `RulePolicy` 的唯一区别是**「够了没」怎么判**：

        RulePolicy      bool(state.evidence)          ← 0/1，有返回就算够
        CoveragePolicy  coverage(s) >= θ_high         ← 算出来的

    ⭐ 但这个区别会**一路传导到行动选择上**：

        不够 → 不是"再查一轮"，而是**用「缺哪个槽位」生成下一句 query**。
               （RulePolicy 不知道缺什么，只能把原问题重复查，把预算烧光。）

        该停 → 不是"有证据就答"，而是**覆盖度真的够了才答**。
               （这直接治掉 `aspirin + 胃溃疡` 那道病：coverage=1/4 → 拒答。）

    ━━━ θ 从哪来 ━━━

        ⚠️ **不能拍**。`theta_high` / `theta_low` 必须在标定集上扫出来
        （`scripts/eval_coverage.py`）。构造函数的默认值只是**扫描起点**，
        不是结论。谁要是直接把默认值写进报告，那个数字就是编的。
    """

    name = "coverage"

    def __init__(self, theta_high: float = 0.75, theta_low: float = 0.45,
                 max_terms: int = 4, ask_when_stuck: bool = True):
        self.theta_high = theta_high
        self.theta_low = theta_low
        self.max_terms = max_terms
        self.ask_when_stuck = ask_when_stuck
        # 最近一次的覆盖度读数 —— 标定脚本和归因都要读它
        self.last_coverage: Optional[Coverage] = None
        # 每次决策的读数都留档（标定时要按阈值重新切，不能只留最后一次）
        self.coverage_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------ 主逻辑

    def __call__(self, state: AgentState) -> Action:
        q = state.question
        cov = self._coverage(state)
        note = cov.describe()

        # R0 预算见底
        if state.budget <= 0:
            return self._finish(state, cov, "预算耗尽")

        # R1 定位（这一步没有争议，和规则策略一致）
        if state.has_locator and not state.has_tried(A_LOCATE, q):
            return Action(A_LOCATE, q, note=note + "  ｜先定位该查哪一节")

        # R1.5 ⭐ **硬拒答**：问的是我们没采的那一节 —— 确定的"没有"，
        #      不跟"还不够"用同一个量纲，直接拒，不看分数。
        #      （不做这一步的话，`side`/`effects` 这类到处都有的词会把分数抬起来。）
        if cov.hard_fail:
            return Action(A_ABSTAIN, cov.hard_fail, note=note)

        # R2 ⭐ 停：覆盖度够高才答
        if state.evidence and cov.score >= self.theta_high:
            return Action(A_ANSWER, None,
                          note=note + f"  ｜≥θ_high({self.theta_high}) → 停")

        loc = state.located or {}

        # R3 还没有证据 → 用定位到的章节做**精确检索**（精度最高的一步）
        if not state.evidence and loc.get("loincs"):
            arg = {"query": q, "restrict": {"drug": loc.get("drug"),
                                            "loincs": list(loc["loincs"])}}
            if not state.has_tried(A_SEARCH, arg):
                return Action(A_SEARCH, arg,
                              note=note + "  ｜按定位的章节精确查")

        # R4 ⭐ 缺口驱动：**缺哪个槽位就查哪个词**（规则策略做不到这一步）
        mq = cov.missing_query(q)
        if mq and not state.has_tried(A_SEARCH, {"query": mq}):
            return Action(A_SEARCH, {"query": mq},
                          note=note + f"  ｜缺口→补查「{mq}」")

        # R5 定位有结果但还没按它查过
        if loc.get("loincs"):
            arg = {"query": q, "restrict": {"drug": loc.get("drug"),
                                            "loincs": list(loc["loincs"])}}
            if not state.has_tried(A_SEARCH, arg):
                return Action(A_SEARCH, arg, note=note + "  ｜补一次限定检索")

        # R6 全局兜底查
        if not state.has_tried(A_SEARCH, {"query": q}):
            return Action(A_SEARCH, {"query": q}, note=note + "  ｜全局兜底查")

        # R7 中间地带 → 问用户（低覆盖时问了也没用，先别打扰）
        if (self.ask_when_stuck and state.has_user and state.count(A_ASK) == 0
                and cov.score >= self.theta_low):
            return Action(A_ASK, _ASK_TEXT, note=note + "  ｜中间地带→问用户")

        # R8 招都使完了，覆盖度还是不够 → **拒答**（这是和规则策略最大的行为差异）
        return self._finish(state, cov, f"覆盖度 {cov.score:.2f} 仍不足")

    # ------------------------------------------------------------ 内部

    def _coverage(self, state: AgentState) -> Coverage:
        cov = compute_coverage(state.question, state.evidence, state.located,
                               state.known_drugs, self.max_terms)
        self.last_coverage = cov
        self.coverage_log.append({
            "step": state.step, "n_evidence": len(state.evidence),
            **cov.to_dict(),
        })
        return cov

    def _finish(self, state: AgentState, cov: Coverage, reason: str) -> Action:
        """收尾：⭐ **只有覆盖度 ≥ θ_low 才允许答**，否则拒答。

        ⚠️ 这一行就是整个 E8 的落点。
           RulePolicy 在这里写的是 `if state.evidence:` —— 有证据就答，
           而那正是"aspirin 那题引了 5 条不相干的证据还作答"的原因。
        """
        note = cov.describe()
        if state.evidence and cov.score >= self.theta_low:
            return Action(A_ANSWER, None, note=note + f"  ｜{reason} → 答")
        return Action(A_ABSTAIN, reason, note=note + f"  ｜{reason} → 拒答")


__all__ = ["Policy", "RulePolicy", "CoveragePolicy"]
