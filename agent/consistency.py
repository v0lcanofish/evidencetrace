# -*- coding: utf-8 -*-
"""
跨药一致性核验（E11）—— 引用**偏不偏**。

━━━ 为什么需要它，以及它和 E9 的分工 ━━━

    E9 的核验管「引用**真不真**」  —— 这条引用追得到出处吗？
    E11 的核验管「引用**偏不偏**」 —— 只引了支持一方、漏了另一方？

**"选择性引用"是真实世界里最害人的那种。**

    问：我能同时吃 A 和 B 吗？
    A 的说明书：「Avoid concomitant use.」
    B 的说明书：**完全没提 A**

    答案只引 A → 用户以为**双方都警告了**（其实 B 根本没表态）
    答案只引 B → 用户以为**没事**（其实 A 明确警告了）

    两种都是误导，而且**两种都能通过 E9 的核验**（引用确实追得到出处）。

━━━ ⚠️ 一条被实测推翻的设计假设 ━━━

设计 v7 的语料一节写着：

    「6 份药 → 50 份药 …… **冲突消解需要多份说明书**」

**这个假设不成立。** 2026-09-18 全量扫了 50 份说明书：

    95 对药互相提及，「同一药对两边说法相反」的**只有 1 对，而且是假阳性**
    （SYNJARDY 是 empagliflozin + metformin 的**复方制剂**，不是冲突）

    **原因**：FDA 说明书在同一套监管要求下写，**它们本来就高度一致**。
    **"多几份说明书"造不出冲突。**

真正的「不一致」是**信息不对称**（实测分布，剔掉表格伪句后）：

    mention × absent   67 对     甲普通提及、乙没提
    ⭐ warn × absent   20 对     **甲明确警告、乙完全没提**
    mention × mention   4 对     双方都提（一致，不该报）
    warn    × warn      4 对     双方都警告（一致，不该报）

那 20 对是真东西，例如：

    amiodarone → digoxin    「Reduce digoxin by half or discontinue.」
    digoxin                 **没提 amiodarone**
    losartan   → lithium    「Lithium: Risk of lithium toxicity.」
    sertraline → warfarin   「Increased Risk of Bleeding: ...」

⚠️⚠️ **数字踩过一次坑，记在这**：我一开始用手写的**窄**词表（不含 `risk of`）扫，
    得出「7 对」并写进了报告。**实际是 20 对** —— 漏掉的 13 对全是真警告
    （"Risk of lithium toxicity" ×4、"Increased Risk of Bleeding" ×4 ……）。
    **口径不一致的代价就是数字差三倍。** 现在这里和判据用同一个 `WARN_RE`。

━━━ 判据 ━━━

    ① 不对称**可检测**：20 对全中，方向正确（另有 7 对人工核对过原句）
    ② **不误报**：4 对双向一致的**不许**判成不对称
    ③ ⭐ **不许选择性引用**：一方说了、另一方也说了，答案只引一边 → 拦
    ④ ⭐ **沉默要如实标注**：对方确实没说 → 要求答案说明，而不是当成"没事"
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

# 强警告词。
# ⚠️ **必须包含 `risk of`** —— 说明书里最常见的警告写法就是 "Increased risk of X toxicity"。
#    2026-09-18 我临时用手写的窄词表扫过一遍，得出"7 对"，比这里少 15 对 ——
#    差的那些恰恰是真警告（"Risk of lithium toxicity"、"Increased Risk of Bleeding"）。
#    **口径不一致的代价就是数字差三倍。**
WARN_RE = re.compile(
    r"\b(avoid|contraindicat|do not|should not|not recommended|discontinue|"
    r"serious|fatal|reduce\b[^.]{0,30}\bby half|risk of)\b", re.I)

# ⚠️ 但宽词表会捞进**表格标题**：说明书把相互作用写成表格，
#    切句后标题会变成 "(7.1) Table 1: Amiodarone Drug Interactions
#    Concomitant Drug Class/Name Examples Clinical Comment ..." ——
#    它含 `risk of` 之类的词纯属偶然，**不是一句话，是一张表的表头**。
#    实测 22 对里有 3 对是这么来的。不过滤的话，报出来的"警告"是假的。
_TABLE_HEADER_RE = re.compile(
    r"(table\s*\d+|drug class/name|clinical comment|examples\s+clinical|"
    r"concomitant drug class)", re.I)


def looks_like_table_header(sent: str) -> bool:
    """切句切出来的"表头伪句"。**判警告之前先把它排掉。**"""
    return bool(_TABLE_HEADER_RE.search(sent or ""))

# 三种表态
WARN, MENTION, ABSENT = "warn", "mention", "absent"


@dataclass
class Side:
    """一方（某药的说明书）对另一方的表态。"""

    speaker: str                 # 谁在说
    about: str                   # 说的是谁
    label: str = ABSENT          # warn / mention / absent
    sentence: str = ""           # 最重的那一句（有强警告就取强警告那句）
    section: str = ""
    n_hits: int = 0              # 提到对方几次

    def to_dict(self):
        return self.__dict__.copy()


@dataclass
class CrossCheck:
    """一次跨药核验的结果。**判据直接打这个对象。**"""

    a: str
    b: str
    a_side: Side
    b_side: Side

    @property
    def kind(self) -> str:
        la, lb = self.a_side.label, self.b_side.label
        if la == WARN and lb == WARN:
            return "both_warn"           # 双方都警告 —— 一致
        if la == WARN and lb == ABSENT:
            return "asymmetric_a"        # ⭐ 甲警告、乙沉默
        if lb == WARN and la == ABSENT:
            return "asymmetric_b"        # ⭐ 乙警告、甲沉默
        if la == WARN and lb == MENTION:
            return "asymmetric_a"        # 甲强警告、乙只轻描淡写地提了一句
        if lb == WARN and la == MENTION:
            return "asymmetric_b"
        if la == ABSENT and lb == ABSENT:
            return "both_absent"
        return "both_mention"            # 都只是提及 —— 一致

    @property
    def is_asymmetric(self) -> bool:
        return self.kind.startswith("asymmetric")

    @property
    def speaker(self) -> Optional[str]:
        """谁警告了（不对称时）。"""
        if self.kind == "asymmetric_a":
            return self.a
        if self.kind == "asymmetric_b":
            return self.b
        return None

    @property
    def silent(self) -> Optional[str]:
        """谁沉默（不对称时）。"""
        if self.kind == "asymmetric_a":
            return self.b
        if self.kind == "asymmetric_b":
            return self.a
        return None

    def describe(self) -> str:
        m = {"both_warn": "双方都警告（一致）", "both_absent": "双方都没提",
             "both_mention": "双方都只是提及（一致）"}
        if self.kind in m:
            return m[self.kind]
        return (f"⚠️ 不对称：**{self.speaker}** 警告了，"
                f"而 **{self.silent}** 的说明书没提")

    def to_dict(self):
        return {"a": self.a, "b": self.b, "kind": self.kind,
                "asymmetric": self.is_asymmetric, "speaker": self.speaker,
                "silent": self.silent, "describe": self.describe(),
                "a_side": self.a_side.to_dict(), "b_side": self.b_side.to_dict()}


# ---------------------------------------------------------------- 索引


class DrugIndex:
    """drug → 说明书全文（按 chunk 拼）。用来回答"这份药的说明书里有没有提那个药"。"""

    def __init__(self, chunks: Sequence, drug_names: Sequence[str]):
        self.drugs = sorted({d for d in drug_names if d})
        self._low = {d.lower(): d for d in self.drugs}
        txt: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for c in chunks:
            d = getattr(c, "drug", "")
            if d:
                txt[d.lower()].append((getattr(c, "section", "") or "", c.text or ""))
        # 每条 (section, 句子)
        self._sents: Dict[str, List[Tuple[str, str]]] = {}
        for d, parts in txt.items():
            out = []
            for sec, t in parts:
                for s in re.split(r"(?<=[.;])\s+", t):
                    if s.strip():
                        out.append((sec, s.strip()))
            self._sents[d] = out

    def known(self, name: str) -> Optional[str]:
        return self._low.get((name or "").lower())

    def side(self, speaker: str, about: str) -> Side:
        """看 `speaker` 的说明书怎么讲 `about`。"""
        sp = self.known(speaker)
        ab = self.known(about)
        s = Side(speaker=speaker, about=about)
        if not sp or not ab:
            return s
        pat = re.compile(r"\b" + re.escape(ab.lower()) + r"\b", re.I)
        hits = [(sec, sent) for sec, sent in self._sents.get(sp.lower(), [])
                if pat.search(sent.lower())]
        s.n_hits = len(hits)
        if not hits:
            return s                      # label 保持 ABSENT
        s.label = MENTION
        for sec, sent in hits:
            # ⚠️ 表格标题先排掉 —— 它含警告词是偶然的，不是一句话
            if WARN_RE.search(sent) and not looks_like_table_header(sent):
                s.label = WARN
                s.sentence, s.section = sent, sec
                break
        if s.label == MENTION:            # 没有强警告 → 取第一句**正文**当引文
            body = [(sec, x) for sec, x in hits if not looks_like_table_header(x)]
            s.section, s.sentence = (body or hits)[0]
        return s

    def check_pair(self, a: str, b: str) -> CrossCheck:
        return CrossCheck(a=a, b=b, a_side=self.side(a, b), b_side=self.side(b, a))


# ---------------------------------------------------------------- 答案层审计


@dataclass
class CitationAudit:
    """一份答案的"偏不偏"审计结果。"""

    ok: bool
    problems: List[Dict[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        if self.ok:
            return "引用不偏：涉及多药的问题，双方说明书都表态了或都如实标注了"
        return "；".join(f"{p['kind']}：{p['detail']}" for p in self.problems[:2])

    def to_dict(self):
        return {"ok": self.ok, "problems": self.problems}


def audit_citation_balance(pair: CrossCheck,
                           cited_drugs: Set[str]) -> CitationAudit:
    """
    ⭐ **选择性引用检测** —— E11 的判据③。

    ⚠️ 关键在**区分两种情况**，它们的正确动作完全不同：

        · 对方**也说了**（warn / mention 都算），答案却只引了一边
          → **选择性引用**，必须补上另一边的引用（`selective_citation`）

        · 对方**确实没说**（absent）
          → 不是错，但答案必须**如实标注"未提及"**，
            否则用户会以为对方也表了态（`silent_unstated`）

    混成一个的话，前者的正确动作是"去补检索"，后者是"加一句说明" ——
    用错动作会让 agent 白白重试（这个坑 E9 的 not_retrieved / not_in_corpus 踩过一模一样的）。

    Args:
        pair:        跨药核验结果
        cited_drugs: 答案里**实际引用了**的药名集合（小写）
    """
    cited = {d.lower() for d in cited_drugs}
    problems: List[Dict[str, str]] = []

    for side in (pair.a_side, pair.b_side):
        # ⚠️⚠️ 查的是 **speaker 自己的说明书**有没有被引，**不是 about**。
        #    那句话印在 speaker 的说明书里（`amiodarone` 的 Drug Interactions 写着
        #    "Reduce digoxin by half"），所以引用它 = 引 speaker 的说明书。
        #    2026-09-18 这里一开始写成了 `side.about not in cited` —— **写反了**，
        #    于是「已经引了 amiodarone」会被报成"选择性引用（漏了 digoxin）"。
        #    判据④那条红就是这么抓出来的。
        if side.label in (WARN, MENTION) and side.speaker.lower() not in cited:
            problems.append({
                "kind": "selective_citation",
                "detail": (f"{side.speaker} 的说明书对 {side.about} 有表述"
                           f"（{side.label}），但答案**没有引用它**"),
                "missing_cite": side.speaker,
                "quote": side.sentence[:120],
            })

    # 对方沉默时：不算选择性引用，但**要求答案里说明它沉默**
    for side in (pair.a_side, pair.b_side):
        if side.label == ABSENT:
            problems.append({
                "kind": "silent_unstated",
                "detail": (f"{side.speaker} 的说明书**未提及** {side.about} —— "
                           f"答案需要如实说明，不然用户会以为它表过态"),
                "missing_cite": "",
                "quote": "",
            })

    # ⚠️ both_warn / both_mention 且两边都引了 → 干净
    if not problems:
        return CitationAudit(ok=True)
    return CitationAudit(ok=False, problems=problems)


__all__ = ["DrugIndex", "Side", "CrossCheck", "CitationAudit",
           "audit_citation_balance", "looks_like_table_header",
           "WARN", "MENTION", "ABSENT", "WARN_RE"]
