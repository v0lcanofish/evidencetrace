# -*- coding: utf-8 -*-
"""
编排引擎 —— 「提问 → 查资料 → 写答案」的循环。

设计原则：

  1. **LLM 和检索器都是普通函数（callable）**
     真模型（DeepSeek）和假模型（Mock）签名一样 → **能直接互换，不改一行编排代码**。
     这也是为什么能先在本地把逻辑验对、再花钱接 API。

  2. **每一步都往账本写一条**（E1 定下的规矩）
     不写就追不到来源，闭包断言就无从谈起。

  3. **引用格式硬约束 `[setid#LOINC]`**
     闭包断言靠它把「报告里的引用」和「账本里的检索」对上。
     格式不对 → 该 run 标记 malformed，不进统计。

  4. **确定性**
     固定 seed、固定检索顺序（不用 set/dict 遍历序）→ 同 seed 两次结果完全一致。

跑法：被 scripts/ 下的脚本 import，不单独跑。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ledger.ledger import (
    Ledger, Step, Claim, Doc,
    ACTION_PLAN, ACTION_RETRIEVE, ACTION_SYNTHESIZE,
    DROP_CONTEXT_BUDGET, DROP_IRRELEVANT, LedgerError,
)

# 引用格式：[setid] 或 [setid#LOINC]
CITE_RE = re.compile(r"\[([0-9a-fA-F-]{8,})(?:#(\d+-\d))?\]")


# ---------------------------------------------------------------- 接口

# 大模型：给 prompt，返回文本。真模型/假模型同一个签名。
LLM = Callable[[str], str]
# 检索器：给 query 和 k，返回 Doc 列表。
Retriever = Callable[[str, int], List[Doc]]


@dataclass
class AgentConfig:
    max_steps: int = 8            # 最多检索几次（防弱模型无限查）
    top_k: int = 5                # 每次取几篇
    context_budget: int = 3       # 最终报告最多用几篇（其余记 reason_dropped）


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


# ---------------------------------------------------------------- 编排


class ResearchAgent:
    """一次研究的完整循环。所有写入都走账本，不绕过。"""

    def __init__(self, llm: LLM, retriever: Retriever, config: Optional[AgentConfig] = None):
        self.llm = llm
        self.retriever = retriever
        self.cfg = config or AgentConfig()

    # ------------------------------------------------------------ 主循环

    def run(self, question: str, run_id: str = "run-0", seed: int = 42) -> Ledger:
        lg = Ledger(run_id=run_id, question=question, seed=seed)

        # ---- ① 拆子问题
        subqs = self._plan(question, lg)

        # ---- ② 逐个子问题检索
        seen_ids = set()
        for i, sq in enumerate(subqs[: self.cfg.max_steps], start=1):
            docs = self.retriever(sq, self.cfg.top_k)
            # ⚠️ 按 cite_key（setid#LOINC）去重，不是 doc_id ——
            #    一份说明书有多节、共用同一个 setid，按 doc_id 去重会把第二三节全滤掉
            docs = [d for d in docs if d.cite_key not in seen_ids]
            for d in docs:
                seen_ids.add(d.cite_key)
            if not docs:
                continue
            lg.record(Step(step=len(lg.steps) + 1, action=ACTION_RETRIEVE,
                           query=sq, retrieved=docs,
                           used_in_report=None))                      # 后面回填

        # ---- ③ 选证据 + 写报告
        all_docs = self._all_docs(lg)
        used = all_docs[: self.cfg.context_budget]                    # 确定性取前 k 个
        report = self._synthesize(question, used)

        # ---- ④ 回填 used_in_report / reason_dropped
        used_ids = {d.cite_key for d in used}
        for s in lg.steps:
            if s.action == ACTION_RETRIEVE:
                s.used_in_report = any(d.cite_key in used_ids for d in s.retrieved)
                # ⚠️ 检索到了但没进报告 → 必须给原因，否则「利用层」归因无从判定
                if s.used_in_report is False:
                    s.reason_dropped = DROP_CONTEXT_BUDGET
                elif s.used_in_report and any(d.cite_key not in used_ids for d in s.retrieved):
                    s.reason_dropped = DROP_CONTEXT_BUDGET

        lg.record(Step(step=len(lg.steps) + 1, action=ACTION_SYNTHESIZE,
                       note=f"用了 {len(used)} 篇证据"))

        # ---- ⑤ 解析引用 → 逐条 Claim
        lg.report_md = report
        for cid, (text, cites) in enumerate(self._split_claims(report), start=1):
            lg.add_claim(Claim(claim_id=f"c{cid}", text=text, cite=cites))

        # ---- ⑥ 格式校验 + 闭包
        if not lg.claims or any(not c.cite for c in lg.claims):
            lg.malformed = True
        try:
            lg.check_closure()
        except LedgerError:
            lg.malformed = True
            raise
        return lg

    # ------------------------------------------------------------ 子步骤

    def _plan(self, question: str, lg: Ledger) -> List[str]:
        raw = self.llm(PLAN_PROMPT.format(q=question))
        subs = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        subs = subs or [question]                                     # 兜底：拆不出来就用原问题
        lg.record(Step(step=len(lg.steps) + 1, action=ACTION_PLAN,
                       note=f"拆出 {len(subs)} 个子问题"))
        return subs

    def _all_docs(self, lg: Ledger) -> List[Doc]:
        """按检索顺序把所有 Doc 拉平（保持顺序 → 确定性）。"""
        out, seen = [], set()
        for s in lg.steps:
            for d in s.retrieved:
                if d.cite_key not in seen:          # 按节去重，见上面 run() 的注释
                    seen.add(d.cite_key)
                    out.append(d)
        return out

    def _synthesize(self, question: str, docs: List[Doc]) -> str:
        if not docs:
            return "The available evidence is insufficient to answer this question."
        ev = "\n\n".join(
            f"[{d.doc_id}#{d.loinc}] {d.section}\n{d.text}" for d in docs
        )
        return self.llm(SYNTH_PROMPT.format(q=question, evidence=ev))

    def _split_claims(self, report: str):
        """把报告切成「一句 + 它的引用」。"""
        out = []
        for sent in re.split(r"(?<=[.!?])\s+", report.strip()):
            if not sent.strip():
                continue
            cites = []
            for m in CITE_RE.finditer(sent):
                sid, loinc = m.group(1), m.group(2)
                cites.append(f"{sid}#{loinc}" if loinc else sid)
            if cites:
                out.append((sent.strip(), cites))
        return out


__all__ = ["ResearchAgent", "AgentConfig", "LLM", "Retriever", "CITE_RE"]
