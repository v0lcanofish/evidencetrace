# -*- coding: utf-8 -*-
"""
假 LLM + 假检索器 —— 让整条链路能【不联网、不花钱】跑起来。

为什么要它们（和 ToolHorizon 的 mock 自测一个道理）：
    **先把逻辑验对，再花钱。** 真模型（DeepSeek）+ 真数据源（DailyMed）
    换进来只是把这两个函数换掉，编排代码一行不用改。

关键性质：
    · **确定性** —— 不看时间、不用随机数；同样的输入永远给同样的输出
    · **不碰网络** —— 断网也能跑（这是自测的判据之一）
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from ledger.ledger import Doc


# ---------------------------------------------------------------- 固定语料

# 玩具语料：3 份药品说明书 × 各 2 个章节。
# doc_id 用真 setid 的格式（UUID），LOINC 用真的相互作用/禁忌章节码。
FIXTURE: List[Dict] = [
    {
        "doc_id": "5a709591-2fab-98e7-e063-6394a90ac50a",
        "title": "IBUPROFEN TABLET",
        "section": "Drug Interactions",
        "loinc": "34073-7",
        "text": ("Warfarin: ibuprofen may increase the anticoagulant effect of warfarin. "
                 "Patients on anticoagulant therapy should be monitored closely and "
                 "concurrent use should be avoided if possible."),
        "keywords": ["ibuprofen", "warfarin", "interaction", "anticoagulant",
                     "bleeding", "blood thinner"],
    },
    {
        "doc_id": "5a709591-2fab-98e7-e063-6394a90ac50a",
        "title": "IBUPROFEN TABLET",
        "section": "Contraindications",
        "loinc": "34070-3",
        "text": ("Ibuprofen is contraindicated in patients with a history of asthma, "
                 "urticaria, or allergic-type reactions after taking aspirin or other "
                 "NSAIDs."),
        "keywords": ["ibuprofen", "contraindication", "asthma", "allergy", "nsaid"],
    },
    {
        "doc_id": "704e99b1-c14a-40bf-9d5f-0d0f2c2b8a11",
        "title": "AMOXICILLIN CAPSULE",
        "section": "Drug Interactions",
        "loinc": "34073-7",
        "text": ("Probenecid: concurrent use with amoxicillin decreases the renal tubular "
                 "secretion of amoxicillin, resulting in increased and prolonged blood "
                 "levels of amoxicillin."),
        "keywords": ["amoxicillin", "probenecid", "interaction", "antibiotic"],
    },
    {
        "doc_id": "704e99b1-c14a-40bf-9d5f-0d0f2c2b8a11",
        "title": "AMOXICILLIN CAPSULE",
        "section": "Dosage and Administration",
        "loinc": "34068-7",
        "text": ("The usual adult dose of amoxicillin is 500 mg every 12 hours or 250 mg "
                 "every 8 hours, depending on the severity of the infection."),
        "keywords": ["amoxicillin", "dose", "dosage", "administration"],
    },
    {
        "doc_id": "ded33248-44ef-45f6-9f4a-1a2b3c4d5e6f",
        "title": "METFORMIN HYDROCHLORIDE TABLET",
        "section": "Contraindications",
        "loinc": "34070-3",
        "text": ("Metformin is contraindicated in patients with severe renal impairment "
                 "(eGFR below 30 mL/min/1.73 m2). Lactic acidosis is a rare but serious "
                 "metabolic complication."),
        "keywords": ["metformin", "contraindication", "renal", "kidney", "lactic"],
    },
    {
        "doc_id": "ded33248-44ef-45f6-9f4a-1a2b3c4d5e6f",
        "title": "METFORMIN HYDROCHLORIDE TABLET",
        "section": "Drug Interactions",
        "loinc": "34073-7",
        "text": ("Carbonic anhydrase inhibitors and other drugs that cause acidosis may "
                 "increase the risk of lactic acidosis with metformin."),
        "keywords": ["metformin", "interaction", "acidosis", "drug"],
    },
]


# ---------------------------------------------------------------- 假检索器


class MockRetriever:
    """按关键词打分返回文档 —— 确定性的，不联网。

    打分方式故意做得简单：query 里的词命中 keywords 就加分。
    真实的 BM25 接进来时换掉这个类就行（接口一样）。
    """

    def __init__(self, fixture: List[Dict] = None):
        self.docs = fixture if fixture is not None else FIXTURE

    def _score(self, query: str, entry: Dict) -> int:
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        return len(q & set(entry["keywords"]))

    def __call__(self, query: str, k: int = 5) -> List[Doc]:
        scored = []
        for i, e in enumerate(self.docs):
            sc = self._score(query, e)
            if sc > 0:
                scored.append((sc, -i, e))          # -i 保证同分时按原始顺序 → 确定性
        scored.sort(reverse=True)
        out = []
        for rank, (sc, _, e) in enumerate(scored[:k], start=1):
            out.append(Doc(
                doc_id=e["doc_id"], section=e["section"], loinc=e["loinc"],
                text=e["text"], title=e["title"],
                rank=rank, score=float(sc),
                authority=0.9 if e["section"] == "Contraindications" else 0.8,
                recency=0.9,
            ))
        return out


# ---------------------------------------------------------------- 假 LLM


class MockLLM:
    """按 prompt 类型返回预置文本 —— 确定性，不联网。

    它做两件事：
      · 认出「拆问题」的 prompt → 返回固定的子问题
      · 认出「写报告」的 prompt → 返回带**合规引用**的报告
        （引用格式必须和 CITE_RE 对得上，否则闭包断言会失败 —— 这正是要验的）
    """

    def __init__(self, bad_citation: bool = False):
        # bad_citation=True 时故意返回追不到出处的引用，用来验闭包断言真的会拦
        self.bad_citation = bad_citation

    def __call__(self, prompt: str) -> str:
        if "research planner" in prompt:
            return self._plan(prompt)
        return self._report(prompt)

    # ---- 拆问题
    @staticmethod
    def _plan(prompt: str) -> str:
        return ("ibuprofen warfarin interaction\n"
                "ibuprofen contraindication asthma")

    # ---- 写报告
    def _report(self, prompt: str) -> str:
        ev = prompt.split("Evidence:", 1)[-1]
        if self.bad_citation:
            return "This claim cites something that was never retrieved [deadbeef-0000#99999-9]."

        # 把证据切成 (引用键, 正文) 段，按关键词挑【对的】那篇
        # ⚠️ 不按顺序瞎挑 —— 一开始就是按位置挑的，结果把哮喘禁忌
        #    引到了 metformin 头上（闭包还照样通过，因为那篇确实被检索了）。
        #    **这正说明「存在」≠「相关」：闭包只保证追得到出处，不保证引对了。**
        blocks = re.findall(r"\[([0-9a-fA-F-]{8,}#\d+-\d)\]\s*(.*?)(?=\n\n\[|\Z)", ev, re.S)
        if not blocks:
            return "The available evidence is insufficient to answer this question."

        def pick(*words):
            for key, text in blocks:
                low = text.lower()
                if all(w in low for w in words):
                    return key
            return None

        c1 = pick("warfarin")               # 相互作用那条
        c2 = pick("asthma")                 # 禁忌那条
        if not (c1 and c2):
            return "The available evidence is insufficient to answer this question."
        return (f"Ibuprofen interacts with warfarin and may increase bleeding risk "
                f"[{c1}]. "
                f"Ibuprofen is contraindicated in patients with a history of asthma "
                f"[{c2}].")


__all__ = ["MockRetriever", "MockLLM", "FIXTURE"]
