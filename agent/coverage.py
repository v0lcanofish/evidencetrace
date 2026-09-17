# -*- coding: utf-8 -*-
"""
块 E8 · 证据覆盖度 —— 把「证据够不够」从 0/1 变成**算出来的数**。

    规则策略问的是：  "有证据吗？"            → 0 或 1
    覆盖度问的是：    "证据够不够支撑回答？"   → 一个数

━━━ ⭐ 槽位（slot）= 回答这个问题**必须有**的东西 ━━━

    从**问题的结构**推出来，不靠模型感觉：

        "Can I take metformin if I have kidney disease?"
             药品  = metformin      ← 有没有一份证据来自 metformin 的说明书？
             章节  = 34070-3        ← 有没有来自「禁忌」这一节的证据？
             实词  = "kidney"       ← 这个词在证据正文里出现了吗？
             实词  = "disease"      ← 同上

        coverage = 已填槽位 / 需要槽位

━━━ ⭐ 每个槽位「填没填上」都是机械可判的 ━━━

    药品槽位：证据里有没有一份，其 `drug` 出现在问题里      → 比字符串
    章节槽位：证据里有没有来自 locate 定位到的那个 LOINC    → 比字段
    实词槽位：这个词在不在证据正文里                        → 查词

    **没有一处需要问 LLM「你觉得够了吗」。**
    这是和 Self-RAG 那类「让模型自省」的关键区别 —— **自省不可核验，槽位可核验**。

━━━ ⭐ 为什么把「实词」也做成槽位（而不是只看有没有证据）━━━

    实测的病：问 `Can I take aspirin if I have a stomach ulcer?`
    （**aspirin 不在语料里**），检索照样返回 5 条 —— 因为命中了 `ulcer` 这个疾病词
    （famotidine / omeprazole 的适应症）。规则策略看到"有返回"就作答了。

    把实词做成槽位之后，这道题**自己就掉下来了**：
        证据里有 `ulcer` ✓，但没有 `aspirin` ✗、没有 `stomach` ✗，药品槽位也 ✗
        → coverage = 1/4 → 远低于阈值 → 该拒答

    **不需要专门的「这药在不在库里」规则，也不需要任何魔法阈值。**

━━━ ⚠️ 诚实边界 ━━━

    · 这是**词面覆盖**，不是语义覆盖。用户说"肾不好"而说明书写 "renal impairment"
      时，实词槽位会判成没填上（假阴性）。
    · 反过来，实词出现在**无关的上下文**里也会被判成填上（假阳性）。
    · 真正的语义覆盖要靠稠密检索 —— 那是 E6 补完那 0.115 的坑之后的事。
      **当前版本的定位是：把「有返回」和「有内容」区分开，不是做到完美。**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from retrieval.retriever import tokenize as _tokenize


# ---------------------------------------------------------------- 槽位

SLOT_DRUG = "drug"
SLOT_SECTION = "section"
SLOT_TERM = "term"


@dataclass
class Slot:
    """一个证据槽位。`filled` 是**算出来的**，`detail` 说明怎么算出来的。"""

    name: str                       # drug | section | term
    key: str                        # 槽位的具体内容（药名 / LOINC / 那个词）
    filled: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "key": self.key, "filled": self.filled,
                "detail": self.detail}


@dataclass
class Coverage:
    """一次覆盖度计算的结果。**它是策略的输入，也是归因的依据。**"""

    slots: List[Slot] = field(default_factory=list)

    # ⭐ **硬拒答理由**。设了就一定拒答，不看分数。
    #
    #    为什么要有它，而不是"把覆盖度压低一点"：
    #      「问题问的是我们**没采**的那一节」是个**确定的事实**，
    #      不是"证据不够"这种程度问题。把它摊进分数里，会被别的槽位抬起来 ——
    #      实测：`What are the side effects of warfarin?` 里
    #      `side`/`effects` 两个词在说明书正文里**到处都是**，实词槽位全填上，
    #      覆盖度 0.75 → 照样作答。**12 条 missing_section 漏了 5 条。**
    #
    #      硬信号就该当硬信号用：**确定的"没有"不该和"还不够"用同一个量纲。**
    hard_fail: Optional[str] = None

    @property
    def n_total(self) -> int:
        return len(self.slots)

    @property
    def n_filled(self) -> int:
        return sum(1 for s in self.slots if s.filled)

    @property
    def score(self) -> float:
        """coverage ∈ [0, 1]。一个槽位都没有时定义为 0（没东西可支撑）。"""
        return self.n_filled / self.n_total if self.n_total else 0.0

    def missing(self) -> List[Slot]:
        return [s for s in self.slots if not s.filled]

    def missing_query(self, question: str) -> Optional[str]:
        """⭐ **缺口 → 下一句 search query**。

        这是本块最重要的一条连线：覆盖度不只是"够不够"的判据，
        它**直接告诉 agent 该去查什么**。
        规则策略做不到这个 —— 它只知道"再查一轮"，不知道该查什么，
        于是把同一句话重复查，把预算烧光。

        ⚠️ 拼出来的 query 必须**去掉原问题里的虚词和已有药名**，
           否则它会退化成"把原问题再查一遍"。
        """
        missing = self.missing()
        if not missing:
            return None
        # 缺的实词优先；药品/章节缺口用原问题兜底（那是范围问题，不是词的问题）
        terms = [s.key for s in missing if s.name == SLOT_TERM]
        if terms:
            return " ".join(terms)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "n_filled": self.n_filled,
            "n_total": self.n_total,
            "hard_fail": self.hard_fail,
            "slots": [s.to_dict() for s in self.slots],
        }

    def describe(self) -> str:
        """人读的一行摘要（写进 trace / 账本 note）。"""
        if self.hard_fail:
            return f"硬拒答：{self.hard_fail}"
        marks = []
        for s in self.slots:
            tag = {SLOT_DRUG: "药", SLOT_SECTION: "节", SLOT_TERM: "词"}[s.name]
            marks.append(f"{'✓' if s.filled else '✗'}{tag}:{s.key}")
        return f"coverage={self.score:.2f} ({self.n_filled}/{self.n_total})  " + " ".join(marks)


# ---------------------------------------------------------------- 计算

# 问题里不携带检索意图的词。
# ⚠️ 分两层：
#   ① 检索层的 STOPWORDS（查资料时该去的虚词）——复用，别再列一份
#   ② **问句特有的填充词**：`take / much / many / supposed` 这些
#      在**文档里**是正常词（BM25 该保留），但在**问句里**是模板词，
#      当槽位只会稀释覆盖率。
#      实测：`How much allopurinol should I take?` 抽出 `['much','take']` ——
#      两个纯填充词占了全部实词槽位，而它们和"该查哪一节"毫无关系。
_QUESTION_FILLERS = frozenset("""
take takes taking taken took
much many more most less least
supposed typical amount amounts
should could would shall may might must
want need get got give gives giving
know tell say says ask asks
something anything nothing everything
please thanks okay ok
""".split())

_TERM_MIN_LEN = 3


def question_terms(question: str, known_drugs: Sequence[str]) -> List[str]:
    """从问题里抽出**实词**（去停用词、去药名、去重，保序）。

    ⚠️ 去掉药名是**故意的**：药名由「药品槽位」单独管，
       混进实词里会让药名一个词占两个槽位，把覆盖率撑高。
    """
    out: List[str] = []
    seen: Set[str] = set()
    low_q = (question or "").lower()
    for w in _tokenize(question):
        if len(w) < _TERM_MIN_LEN or w in seen or w in _QUESTION_FILLERS:
            continue
        if any(w == d.lower() or w in d.lower().split() for d in known_drugs):
            continue
        seen.add(w)
        out.append(w)
    return out


def compute_coverage(question: str,
                     evidence: Sequence[Any],
                     located: Optional[Dict[str, Any]],
                     known_drugs: Sequence[str],
                     max_terms: int = 4) -> Coverage:
    """算一次覆盖度。

    Args:
        question:    原始问题
        evidence:    已收集的证据（`ledger.Doc`，带 drug / loinc / text）
        located:     locate 的结果 {"drug":…, "loincs":[…]}，没有就传 None
        known_drugs: 语料里有哪些药（用来从问题里识别药名）
        max_terms:   最多取几个实词当槽位。**限量的理由**：问题越长槽位越多，
                     长问题会被"词多"惩罚，和"证据够不够"没关系。
                     ⚠️ 4 是先验值，E8 的标定可以一并扫。
    """
    slots: List[Slot] = []
    ev_drugs = {(d.drug or "").lower() for d in evidence if d.drug}
    ev_loincs = {d.loinc for d in evidence if d.loinc}

    # ---- ① 药品槽位：问题里点名的药，有没有它的说明书进来
    low_q = (question or "").lower()
    asked = [d for d in known_drugs if d.lower() in low_q]
    for d in asked:
        ok = d.lower() in ev_drugs
        slots.append(Slot(SLOT_DRUG, d, ok,
                          "有这份药的章节" if ok else
                          f"**没有**这份药的任何章节（证据来自 {sorted(ev_drugs) or '（空）'}）"))
    if not asked:
        # 问题里没点名任何**库里的**药 —— 这是个信号，不是"没槽位"
        slots.append(Slot(SLOT_DRUG, "(未识别)", False,
                          "问题里没有语料中收录的药名"))

    # ---- ⭐ 作用域：实词和章节**只在"该查的那份药"的范围内**判定
    #
    #     ！！这条是防**自证**的，是本块最容易写错的地方 ！！
    #
    #     漏洞长这样（实测踩到过）：
    #         agent 缺 `stomach` 这个词 → 拿 "take stomach" 去查
    #         → 查回来的证据里**当然有** stomach（不然怎么叫命中）
    #         → 槽位填上 → coverage 涨 → 作答
    #     **"缺什么查什么"会自己满足自己**，停止准则形同虚设 ——
    #     agent 可以无限把自己喂饱，而且每一轮看起来都在"努力补证据"。
    #
    #     修法：把判定范围从"所有收集到的证据"收窄到
    #     **问题里点名的那份药的说明书**。
    #     aspirin 那题问的药根本不在库里 → 作用域是空的 → 一个实词都填不上
    #     → coverage 掉到 0 → 该拒答。**漏洞自动关上。**
    scope = {d.lower() for d in asked}
    in_scope = [d for d in evidence if (d.drug or "").lower() in scope] if scope else []
    texts = " ".join((d.text or "").lower() for d in in_scope)
    scope_loincs = {d.loinc for d in in_scope if d.loinc}

    # ---- ② 章节槽位：定位到的那一节，有没有**该药**的证据进来
    lo = (located or {}).get("loincs") or []
    intent = (located or {}).get("intent")
    hard_fail: Optional[str] = None

    if lo:
        for x in lo:
            slots.append(Slot(SLOT_SECTION, x, x in scope_loincs,
                              f"该药的证据章节 {sorted(scope_loincs) or '（空）'}"))
    else:
        # ⭐ 意图识别出来了、但**没有任何一节可用** ——
        #    说明问的是这份药**没有（或我们没采）**的那一节。
        #    这是确定的事实，直接判硬拒答，不再靠分数。
        detail = "没能定位到该查哪一节"
        if intent:
            hard_fail = (f"问题问的是「{intent}」那一节 —— "
                         f"语料里没有这份药的这一节（不是检索失败，是没采）")
            detail = hard_fail
        slots.append(Slot(SLOT_SECTION, "(未定位)", False, detail))

    # ---- ③ 实词槽位：问题的实词，有没有在**该药**的证据正文里出现
    for w in question_terms(question, known_drugs)[:max_terms]:
        # 词干宽松匹配：说明书里是 "hypersensitivity"，问题里可能是 "hypersensitive"
        hit = w in texts or (len(w) >= 6 and w[: len(w) - 2] in texts)
        slots.append(Slot(SLOT_TERM, w, hit,
                          "出现在证据正文里" if hit else "证据正文里没这个词"))

    return Coverage(slots=slots, hard_fail=hard_fail)


def corpus_drugs_from(retriever: Any) -> List[str]:
    """从检索器里取语料包含哪些药（chunk 上带 drug 字段）。"""
    chunks = getattr(retriever, "chunks", None)
    if not chunks:
        return []
    return sorted({c.drug for c in chunks if getattr(c, "drug", "")})


__all__ = ["Slot", "Coverage", "compute_coverage", "question_terms",
           "corpus_drugs_from", "SLOT_DRUG", "SLOT_SECTION", "SLOT_TERM"]
