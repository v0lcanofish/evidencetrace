# -*- coding: utf-8 -*-
"""
引用核验（E9）—— **机械判，不问 LLM**。

━━━ 为什么需要这个文件 ━━━

E7 收尾时已经能跑 `ledger.check_closure()`，但它有个致命的位置问题：

    它跑在 `loop._finalize()` 里 —— **循环已经结束了**。
    失败了只能写一个 `malformed=True` 收场，**发现问题时已经没机会补救**。

而设计 v7 第 137 行要求的是：

    **反馈：机械核验结果反过来决定下一步** —— 追不到 → 回到 ②③（重新检索）

⭐ **这就是 E9 的全部内容：把核验从「事后判死」改成「循环内的检查点 + 反馈」。**

━━━ 四类坏引用（判据：坏引用 100% 抓住）━━━

    ① no_loinc        引了 [setid] 但没带章节码
                      —— 设计写死"必须带 LOINC"，没有章节码就验不了"引的是不是那一节"
    ② not_retrieved   引用的 (setid#loinc) 本轮**根本没检索到**
                      —— 这是"凭空引用"，闭包检查抓的就是它
    ③ not_in_corpus   引用的 (setid#loinc) **语料里压根没有**
                      —— 比 ② 更狠：连语料都没有，模型是在编
                      （要传 corpus_keys 才检；不传就跳过这一条）
    ④ 有效拒答被当成 0 引用 —— ⚠️ 这个**不算坏引用**，见下

━━━ 一条红线：拒答 ≠ 引用坏了 ━━━

    "证据不足，我答不了" 也是**没有引用**的。
    把它判成 malformed 会污染拒答准确率 —— 这个坑 E7 踩过一次
    （当时是"断言没红，是数字不对劲"才发现的）。
    所以核验层必须能区分：**空引用 = 拒答（合法）** vs **有内容但引用全坏（malformed）**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set

from agent.report import CITE_RE, looks_like_abstention, split_claims

# 引用必须带章节码 —— 用严格版正则单独判，**不动 CITE_RE**
# （CITE_RE 的 loinc 组是可选的，改它会连带影响 split_claims 的解析行为）
_STRICT_CITE_RE = __import__("re").compile(r"\[[0-9a-fA-F-]{8,}#\d+-\d\]")

# 坏引用的四种形态
K_NO_LOINC = "no_loinc"
K_NOT_RETRIEVED = "not_retrieved"
K_NOT_IN_CORPUS = "not_in_corpus"

_REASON = {
    K_NO_LOINC: "引用没带章节码（写成 [setid] 了）—— 没有章节码就验不了'引的是不是那一节'",
    K_NOT_RETRIEVED: "引用的这一节本轮【没检索到】—— 凭空引用",
    K_NOT_IN_CORPUS: "引用的这一节【语料里根本没有】—— 模型在编",
}


@dataclass
class CiteVerdict:
    """一条引用的核验结论。"""

    cite: str
    ok: bool
    kind: str = ""                 # 空 = 通过；否则是上面三个 K_*
    reason: str = ""

    def to_dict(self):
        """写进 `AgentState.failed_cites` 的形态 —— 策略要按这个读。"""
        return {"cite": self.cite, "kind": self.kind, "reason": self.reason}


@dataclass
class AnswerVerdict:
    """整份答案的核验结论。"""

    ok: bool
    is_abstention: bool = False
    n_claims: int = 0
    n_cites: int = 0
    verdicts: List[CiteVerdict] = field(default_factory=list)

    @property
    def bad(self) -> List[CiteVerdict]:
        return [v for v in self.verdicts if not v.ok]

    @property
    def bad_kinds(self) -> Set[str]:
        return {v.kind for v in self.bad}

    def summary(self) -> str:
        if self.is_abstention:
            return "有效拒答（无引用是合法的，不算坏引用）"
        if self.ok:
            return f"核验通过：{self.n_claims} 条论断 / {self.n_cites} 条引用全部追得到出处"
        return (f"核验未通过：{len(self.bad)}/{self.n_cites} 条引用有问题 ｜ "
                + "；".join(f"{v.cite}（{v.reason}）" for v in self.bad[:3]))

    def to_dict(self):
        return {
            "ok": self.ok,
            "is_abstention": self.is_abstention,
            "n_claims": self.n_claims,
            "n_cites": self.n_cites,
            "n_bad": len(self.bad),
            "bad_kinds": sorted(self.bad_kinds),
            "bad": [{"cite": v.cite, "kind": v.kind, "reason": v.reason}
                    for v in self.bad],
        }


def verify_cite(cite: str,
                evidence_keys: Set[str],
                corpus_keys: Optional[Set[str]] = None) -> CiteVerdict:
    """核验单条引用。**先判格式，再判闭包，最后判语料存在性** —— 顺序不能反：
    一条没带章节码的引用根本没法拿去做闭包匹配，先判闭包会给出误导性的理由。"""
    if "#" not in cite:
        return CiteVerdict(cite, False, K_NO_LOINC, _REASON[K_NO_LOINC])
    if cite not in evidence_keys:
        # ⚠️ 语料里有、但本轮没检索到 → 是 not_retrieved 而不是 not_in_corpus。
        #    两者的**补救动作完全不同**：前者再检索一次就行，后者说明模型在编。
        if corpus_keys is not None and cite not in corpus_keys:
            return CiteVerdict(cite, False, K_NOT_IN_CORPUS, _REASON[K_NOT_IN_CORPUS])
        return CiteVerdict(cite, False, K_NOT_RETRIEVED, _REASON[K_NOT_RETRIEVED])
    return CiteVerdict(cite, True)


def verify_answer(answer_text: str,
                  evidence_keys: Set[str],
                  corpus_keys: Optional[Set[str]] = None) -> AnswerVerdict:
    """
    核验一份答案。**纯函数** —— 不碰账本、不改状态，所以能单独喂坏数据打判据。

    Args:
        answer_text:   生成层写出来的报告
        evidence_keys: 本轮已收集证据的 cite_key 集合（`AgentState.evidence_keys()`）
        corpus_keys:   语料里**全部** (setid#loinc) 的集合。传了才检 not_in_corpus。
                       ⚠️ 不传 ≠ 通过，只是"这一层没查" —— 别把不查当成没问题。

    Returns:
        AnswerVerdict。`ok=True` 的三种情形：
          ① 每条引用都追得到
          ② 有效拒答（没有引用，但文本在说"答不了"）
          ③ 没有任何引用、也不是拒答 —— ⚠️ **这种 ok=False**（见下）
    """
    text = (answer_text or "").strip()
    claims = split_claims(text)

    # —— 空引用分两种，必须分开（E7 踩过的坑）
    if not claims:
        if looks_like_abstention(text):
            return AnswerVerdict(ok=True, is_abstention=True)
        # 有正文、但一句带引用的都没有 → 这就是 malformed。
        # 注意 `not text` 也算：答了空字符串等于没答，不能让空串混过去。
        return AnswerVerdict(ok=False, n_claims=0, n_cites=0)

    verdicts: List[CiteVerdict] = []
    for _sent, cites in claims:
        for c in cites:
            verdicts.append(verify_cite(c, evidence_keys, corpus_keys))

    n_bad = sum(1 for v in verdicts if not v.ok)
    return AnswerVerdict(ok=(n_bad == 0), n_claims=len(claims),
                         n_cites=len(verdicts), verdicts=verdicts)


def strict_cite_ok(sent: str) -> bool:
    """句子里的引用是不是**全部**带章节码（严格格式）。

    单独给测试用 —— `split_claims` 会用 `CITE_RE` 把 `[setid]` 也当成一条引用，
    所以要判"格式对不对"得绕过它、用严格版正则单独数一遍。
    """
    loose = len(CITE_RE.findall(sent))
    strict = len(_STRICT_CITE_RE.findall(sent))
    return loose == strict


__all__ = ["CiteVerdict", "AnswerVerdict", "verify_cite", "verify_answer",
           "strict_cite_ok", "K_NO_LOINC", "K_NOT_RETRIEVED", "K_NOT_IN_CORPUS"]
