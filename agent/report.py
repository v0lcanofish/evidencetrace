# -*- coding: utf-8 -*-
"""
报告生成与解析 —— 线性流水线（E2）和 agent 循环（E7）**共用的一份**。

    from agent.report import SYNTH_PROMPT, build_evidence_block, split_claims, CITE_RE

━━━ 为什么要抽出来 ━━━

    E7 之后项目里同时存在两条链路：

        E2  research_agent.py   线性流水线  ← 保留，当**对照组**
        E7  loop.py             agent 循环  ← 主体

    两条链路必须写出**同一种引用格式**——因为闭包断言（每一条引用都要能在
    账本里追到出处）靠正则把"报告里的引用"和"账本里的检索"对上。
    格式一旦有两份实现，迟早会漂移，漂移之后闭包检查就会开始误报/漏报。

    ⭐ 更重要的是：报告格式是**数据契约**，不是某条链路的实现细节。

━━━ 引用格式硬约束 ━━━

    [<setid>#<loinc>]        例：[ecd271ad-…#34070-3]

    · 必须带 LOINC —— 引入章节码的**全部意义**就是能验"引的是不是那一节"
    · 格式不对 → 该 run 标记 malformed，不进统计（不猜、不修）
"""

from __future__ import annotations

import re
from typing import List, Tuple

# 引用格式：[setid] 或 [setid#LOINC]
CITE_RE = re.compile(r"\[([0-9a-fA-F-]{8,})(?:#(\d+-\d))?\]")

# ---------------------------------------------------------------- 提示词

PLAN_PROMPT = """You are a research planner. Break the question into 2-3 sub-questions
that must be answered from drug labels.

Question: {q}

Return one sub-question per line, no numbering, no extra text."""

SYNTH_PROMPT = """You are a medical information assistant. Answer the question using ONLY
the evidence below. Every sentence that states a fact MUST cite its source.

Cite in this exact format: [<setid>#<loinc>]
Example: Ibuprofen is contraindicated in patients on anticoagulants [a1b2c3d4-...#34073-7].

If the evidence is insufficient, say so explicitly instead of guessing.

Question: {q}

Evidence:
{evidence}

Answer:"""

# 用户说过的话是**另一类信息**：能用来收窄问题，但**不能当引用**。
# 所以它进 prompt 时单独成块、单独标注，不和证据混在一起。
USER_NOTE_TEMPLATE = """
The user has told you the following about themselves (context only —
NEVER cite this as evidence, it is not a document):
{facts}
"""


# ---------------------------------------------------------------- 组装


def build_evidence_block(docs) -> str:
    """把证据列表拼成 prompt 里的证据块。

    每条的抬头就是**引用键**——模型抄过去就是合规引用，不用自己拼格式。
    """
    return "\n\n".join(
        f"[{d.doc_id}#{d.loinc}] {d.section}\n{d.text}" for d in docs
    )


def split_claims(report: str) -> List[Tuple[str, List[str]]]:
    """把报告切成「一句 + 它的引用」。

    只保留**带引用**的句子 —— 没有引用的句子不是 claim，是废话/过渡句。
    """
    out: List[Tuple[str, List[str]]] = []
    for sent in re.split(r"(?<=[.!?])\s+", (report or "").strip()):
        if not sent.strip():
            continue
        cites = []
        for m in CITE_RE.finditer(sent):
            sid, loinc = m.group(1), m.group(2)
            cites.append(f"{sid}#{loinc}" if loinc else sid)
        if cites:
            out.append((sent.strip(), cites))
    return out


def looks_like_abstention(text: str) -> bool:
    """报告是不是在说"答不了"。

    ⚠️ 这是**粗判**，只用于生成侧自检（"证据不足时别硬答"）。
       真正的拒答判定走 `abstain` 动作 + 账本，不走文本匹配。
    """
    t = (text or "").lower()
    return any(k in t for k in ("insufficient", "cannot answer", "unable to answer",
                                "don't have enough", "do not have enough"))


__all__ = ["CITE_RE", "PLAN_PROMPT", "SYNTH_PROMPT", "USER_NOTE_TEMPLATE",
           "build_evidence_block", "split_claims", "looks_like_abstention"]
