# -*- coding: utf-8 -*-
"""
块 E12 前置 · 决策点的采集与反事实重放 —— **训练决策器之前必须先量有没有东西可学**。

━━━ 为什么先做这个，而不是直接开训 ━━━

设计 v7 里策略③是「用 ①② 的决策日志 **SFT 小模型**」。
但在花 GPU 之前必须先答一个问题：

    **这些决策点里，有多少是「选什么动作都一样」的？**

ToolHorizon 那边已经栽过一次同形的坑：**40% 的题的 reward 与策略无关**
（gold 一条写动作都没有 → 什么都不做也满分）→ 放进 RL 会人为制造零梯度。
同一个病在这里的表现是：

    如果某个决策点上「换成别的动作，最后结局不变」，
    那么这个决策点**没有梯度** —— 模仿它、优化它，都学不到东西。
    它的标签只是"当时那套规则碰巧走到这儿"，不是"这是对的"。

⭐ 所以本模块提供两件东西，**都不训练、不联网、纯 CPU**：

    LoggingPolicy   把每个决策点的 (状态特征, 动作) 记下来  → 决策日志
    ForcedPolicy    在第 i 个决策点**强制换个动作**，其余交给原策略 → 反事实

    量出来的那个数（有梯度的决策点占比）就是 **E12 的开工判据**：
    占比太低 ⇒ 先别训，训了也是拟合噪声。

━━━ 两条边界（写进报告）━━━

    ① 反事实换的是**动作类型**，参数用规范默认值 ——
       "换动作"和"换参数"是两件事，本模块只量前者。
    ② 单步反事实不是因果推断：改了第 i 步，第 i+1 步的状态就变了，
       策略后面的选择也跟着变。量的是**策略 + 环境的联合反应**，
       这正是我们要的（"这一步的选择会不会改变结局"），但它不是"控制变量"。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from agent.coverage import compute_coverage
from agent.policy import Policy
from agent.select import focus_drugs
from agent.state import (
    AgentState, Action,
    A_LOCATE, A_SEARCH, A_ASK, A_ANSWER, A_ABSTAIN, ACTIONS,
)

# 强制动作时用的规范参数（见边界①）
FORCE_ARGS: Dict[str, Any] = {
    A_LOCATE: None,      # None → 用当轮问题
    A_SEARCH: None,      # None → 用当轮问题（裸查）
    A_ASK: "Could you tell me more about your situation or any conditions you have?",
    A_ANSWER: None,
    A_ABSTAIN: "（反事实强制拒答）",
}


# ---------------------------------------------------------------- 状态特征

def state_features(state: AgentState) -> Dict[str, Any]:
    """把状态抽成**机械可判**的特征向量。

    ⚠️ 全部是"算出来的"，没有一处是"让模型感觉"：
       覆盖度是算的（`coverage.py`），定位命中是比字段，缺口是数出来的。
       这和策略①②的输入是**同一套东西** —— 三档对照才有意义。
    """
    cov = compute_coverage(state.question, state.evidence, state.located,
                           state.known_drugs)
    loc = state.located or {}
    scope = {d.lower() for d in focus_drugs(state.question, state.known_drugs)}
    lo = set(loc.get("loincs") or [])

    n_hit = sum(1 for d in state.evidence
                if (d.drug or "").lower() in scope and d.loinc in lo)
    n_other = sum(1 for d in state.evidence
                  if scope and (d.drug or "").lower() not in scope)

    by_name = {s.name: s for s in cov.slots}
    return {
        # 预算与进度
        "budget": state.budget,
        "step": state.step,
        # 证据
        "n_evidence": len(state.evidence),
        "n_hit_evidence": n_hit,            # 定位章节 ∩ 焦点药 —— 最该要的那种
        "n_other_drug_evidence": n_other,
        # 定位
        "has_located": int(bool(lo)),
        "n_located_loincs": len(lo),
        # 覆盖度（算出来的）
        "coverage": round(cov.score, 4),
        "hard_fail": int(bool(cov.hard_fail)),
        "slot_drug_filled": int(bool(by_name.get("drug") and by_name["drug"].filled)),
        "slot_section_filled": int(bool(by_name.get("section") and by_name["section"].filled)),
        "n_slot_missing": len(cov.missing()),
        # 动作历史
        "n_tried_locate": state.count(A_LOCATE),
        "n_tried_search": state.count(A_SEARCH),
        "n_tried_ask": state.count(A_ASK),
        # E9 的核验反馈
        "has_bad_cites": int(bool(state.has_bad_cites)),
        "n_failed_cites": len(state.failed_cites),
        # 环境能力
        "has_user": int(state.has_user),
        "has_locator": int(state.has_locator),
    }


# ---------------------------------------------------------------- 决策点


@dataclass
class DecisionPoint:
    """一个决策点：**策略在状态 s 下选了动作 a**。"""

    qid: str
    policy: str
    index: int                              # 本题内的第几个决策（0 起）
    action: str
    arg_repr: str
    feats: Dict[str, Any] = field(default_factory=dict)
    # ---- 结果（跑完才填）
    outcome: Optional[bool] = None          # 这条轨迹最后答对了没
    abstained: bool = False
    # ---- 反事实：{被强制的动作: 结局}
    cf: Dict[str, bool] = field(default_factory=dict)

    def better_exists(self) -> Optional[bool]:
        """**存在比当时更好的动作吗**（当时做错了，换个动作能对）。

        ⭐ 这个才是"可学信号" —— 策略改进的空间全在这里。
        """
        if self.outcome is None or not self.cf:
            return None
        return (not self.outcome) and any(self.cf.values())

    def worse_exists(self) -> Optional[bool]:
        """**存在比当时更差的动作吗**（当时做对了，换个动作会错）。"""
        if self.outcome is None or not self.cf:
            return None
        return bool(self.outcome) and any(not v for v in self.cf.values())

    def has_gradient(self) -> Optional[bool]:
        """结局对这一步敏不敏感（双向）。

        ⚠️⚠️ **不要拿它当"可学信号"，它会被 abstain 污染。**
            2026-09-18 实测：第一版判据只用了这一个量，结果是
            「答对的题 412/412 = 1.000、答错的题 0/66 = 0.000」——
            干净得可疑。原因是反事实候选里含 `abstain`：
            任何一条答对的轨迹，随便哪一步强制拒答都会让结局变错，
            于是"有梯度"对每条正确轨迹的**每一个**决策点恒真。
            **它量的是"拒答会不会搞砸"，不是"这一步重不重要"。**
            留这个方法是为了记录这个坑；判据请用 better_exists。
        """
        if self.outcome is None or not self.cf:
            return None
        return any(v != self.outcome for v in self.cf.values())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["better_exists"] = self.better_exists()
        d["worse_exists"] = self.worse_exists()
        d["has_gradient"] = self.has_gradient()
        return d


# ---------------------------------------------------------------- 包装策略


class LoggingPolicy(Policy):
    """包住任意策略，把每个决策点记下来。**不改变被包策略的任何行为。**"""

    def __init__(self, base: Policy, sink: List[DecisionPoint],
                 qid: str = "", name: Optional[str] = None):
        self.base = base
        self.sink = sink
        self.qid = qid
        self.name = name or getattr(base, "name", "policy")
        self.n_calls = 0

    def __call__(self, state: AgentState) -> Action:
        act = self.base(state)
        self.sink.append(DecisionPoint(
            qid=self.qid, policy=self.name, index=self.n_calls,
            action=act.action, arg_repr=str(act.arg)[:80],
            feats=state_features(state)))
        self.n_calls += 1
        return act


class ForcedPolicy(Policy):
    """在第 `force_at` 个决策点**强制换成 `force_action`**，其余交给原策略。

    ⭐ 为什么这样能当反事实：循环是**确定性**的（同 seed、无随机、无时间），
       所以第 i 步之前的状态与基线轨迹**逐字节一致** ——
       在第 i 步强制换动作，等价于"在同一条轨迹的同一个岔口拐了另一条路"。

    ⚠️ 第 i 步之后轨迹会分叉，原策略会基于新状态重新决策 —— **这是对的**，
       我们要量的就是"这一步的选择会不会改变最终结局"，不是"强行走完原计划"。
    """

    def __init__(self, base: Policy, force_at: int, force_action: str,
                 question: str = ""):
        self.base = base
        self.force_at = force_at
        self.force_action = force_action
        self.question = question
        self.n_calls = 0
        self.fired = False

    def __call__(self, state: AgentState) -> Action:
        i = self.n_calls
        self.n_calls += 1
        if i != self.force_at:
            return self.base(state)
        self.fired = True
        arg = FORCE_ARGS.get(self.force_action, None)
        if self.force_action == A_LOCATE and arg is None:
            arg = state.question
        if self.force_action == A_SEARCH and arg is None:
            arg = {"query": state.question}
        return Action(self.force_action, arg, note="[反事实] 强制")


def available_actions(state: AgentState) -> List[str]:
    """环境**实际支持**的动作 —— 没有定位器就别强制 locate，没有用户就别强制 ask。

    ⚠️ 不筛的话，反事实会拿"这个环境根本做不到的动作"去比，
       量出来的"结局变了"全是假的。
    """
    out = [A_SEARCH, A_ANSWER, A_ABSTAIN]
    if state.has_locator:
        out.append(A_LOCATE)
    if state.has_user:
        out.append(A_ASK)
    return out


__all__ = ["DecisionPoint", "LoggingPolicy", "ForcedPolicy",
           "state_features", "available_actions", "FORCE_ARGS", "ACTIONS"]
