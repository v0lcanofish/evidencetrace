# -*- coding: utf-8 -*-
"""
块 E13b · 生成前的证据选择（排序 + 按预算截断）。

━━━ 这个块要修的是什么（**实测量出来的，不是猜的**）━━━

E13 的 ⑥ 量出「瓶颈从检索层搬到利用层」：换完检索器后，同一批 60 题里
`retrieval 8 → utilization 8`（失败的是**完全同一批题**）。
`probe_evidence_pool.py` 把这 8 条的**池子结构**量了出来：

    · gold 永远在池子**第 0 条**        ← 推翻了我原来"只增不减把 gold 埋后面"的假设
    · gold 与问题**字面零重合**（8/8）  ← 问的是剂量，而剂量正文里根本不出现药名
      例：`How much allopurinol should I take?` 的 gold 正文是
          "Gout: Prior to initiating treatment assess serum uric acid level…"
          ——**整段没有 "allopurinol" 这个词**
    · 池子里混进**别的药**的证据：4/8 条（异药 4~8 条）

⇒ **病根**：生成层手里是一个**扁平、全量、按检索先后排列**的池子，
  它只能靠"哪条 chunk 和问题字面像"来猜该引哪条 —— 而这条判据在**剂量类问题**上
  必然挑错（药名不出现在正文里，别节的 "alloxxx used" 反而重合更多）。

⇒ 而 agent **明明已经算出来了**：`coverage.py` 的 `scope`（问题点名的那份药）
  和 `located.loincs`（该查哪一节）—— **却一个字都没传给生成层**。
  `ToolBox.answer()` 交给生成的是 `state.evidence` 的全量原序。

━━━ 这一层的定位（务必守住，否则会做成"又一层 RAG"）━━━

    检索层  决定「把哪些 chunk 捞进池子」        —— E6 已做
    选择层  决定「生成时看池子里的哪几条、按什么顺序」  —— 本块
    生成层  照着看到的证据写引用                   —— E9

    ⭐ 本层**不重新算一遍语义相似度**，只做一件事：
      **把 agent 自己在检索阶段已经算出来的结构事实，转达给生成层。**
      排序依据全部来自 state（焦点药 / 定位章节 / 章节权威度），
      **没有一处是"再猜一次哪条和问题像"** —— 否则就是把生成层的病抄一遍。

━━━ ⚠️ 为什么**不做硬剪**（不按药名直接扔）━━━

    E11 实测过：95 对药互相提及，其中 `mention×absent 67` 对 ——
    即「A 的说明书提了 B，B 的说明书**没提** A」。
    所以**别的药的说明书有时恰恰是唯一出处**，按药名硬剪会把 E11 整块场景剪没。
    ⇒ 本层只做「**排序 + 按预算截断**」：异药的证据排在后面，预算够就留着。
       （"丢掉"的动作永远只有一个理由：**排到了预算之外**。）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ledger.ledger import Doc


def budget_from_env() -> Optional[int]:
    """生成预算从环境变量读 —— **所有脚本共用一个旋钮**，免得每个脚本各写一份。

        ET_EVIDENCE_BUDGET=6      → 生成层最多看 6 条
        ET_EVIDENCE_BUDGET=off    → 不截断（E13b 之前的老口径，用于对照）

    ⚠️ 老口径必须是**显式**的 `off`，不能靠"没设就是 off" ——
       否则判据脚本里漏设一次，就会静默跑回旧行为、把结论说反。
    """
    v = (os.environ.get("ET_EVIDENCE_BUDGET") or "").strip().lower()
    if v in ("", "off", "none", "unlimited"):
        return None
    n = int(v)
    if n < 1:
        raise ValueError(f"ET_EVIDENCE_BUDGET 要么是 off，要么 ≥1；收到 {v!r}")
    return n


def order_from_env() -> str:
    """排序档：`struct`（产品路径）/ `reverse`（对抗档，判据 ⑥ 用）。

    ⚠️ 默认值是 `struct` —— 判据脚本里想跑对抗档必须**显式**写 reverse。
    """
    return (os.environ.get("ET_SELECT_ORDER") or "struct").strip().lower()


# ---------------------------------------------------------------- 结构分

# 结构档位：分越高越靠前。
#   3 = 定位章节 ∩ 焦点药   —— 结构先验说"就是这条"（E6 的核心资产）
#   2 = 定位章节            —— 章节对（E11 的跨药出处落在这里，**不能扔**）
#   1 = 焦点药              —— 药对、节不对
#   0 = 都不对              —— 既不是这份药、也不在该查的那一节
R_BOTH, R_SECTION, R_DRUG, R_NONE = 3, 2, 1, 0


def focus_drugs(question: str, known_drugs: Sequence[str]) -> List[str]:
    """问题里点名的、且**语料里真收录了**的药。

    ⚠️ 这份判定必须和 `coverage.compute_coverage` 的**药品槽位**逐字一致 ——
       两处一旦漂移，就会出现「覆盖度说这药的作用域是空的、选择层却认为有焦点药」
       这种**不报错的**自相矛盾。`scripts/eval_select.py` ② 有合同测试钉住它。
    """
    low_q = (question or "").lower()
    return [d for d in known_drugs if d and d.lower() in low_q]


def struct_rank(d: Doc, scope: set, loincs: set) -> int:
    """一条证据的结构档位。**只看 state 里的字段，不读正文。**"""
    in_sec = bool(d.loinc and d.loinc in loincs)
    in_drug = bool(d.drug and d.drug.lower() in scope)
    if in_sec and in_drug:
        return R_BOTH
    if in_sec:
        return R_SECTION
    if in_drug:
        return R_DRUG
    return R_NONE


# ---------------------------------------------------------------- 结果

@dataclass
class Dropped:
    """被截断抛弃的一条证据。**理由必须可查**（不许静默丢弃）。"""

    cite_key: str
    struct: int
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"cite_key": self.cite_key, "struct": self.struct, "reason": self.reason}


@dataclass
class Selection:
    """一次生成前选择的结果。**它同时是给生成层的输入和给归因的证据。**"""

    kept: List[Doc] = field(default_factory=list)
    dropped: List[Dropped] = field(default_factory=list)
    n_pool: int = 0
    budget: Optional[int] = None
    focus: List[str] = field(default_factory=list)
    loincs: List[str] = field(default_factory=list)
    # 池子里有几条是**别的药**的（与生成器无关的系统事实，报告要用）
    n_other_drug: int = 0

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    def struct_hist(self) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for d in self.kept:
            r = struct_rank(d, {x.lower() for x in self.focus}, set(self.loincs))
            out[r] = out.get(r, 0) + 1
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_pool": self.n_pool, "n_kept": self.n_kept, "budget": self.budget,
            "focus": list(self.focus), "loincs": list(self.loincs),
            "n_other_drug": self.n_other_drug,
            "kept": [d.cite_key for d in self.kept],
            "dropped": [x.to_dict() for x in self.dropped],
        }


# ---------------------------------------------------------------- 主函数


def select_evidence(evidence: Sequence[Doc],
                    located: Optional[Dict[str, Any]],
                    known_drugs: Sequence[str],
                    question: str,
                    budget: Optional[int] = None,
                    order: str = "struct") -> Selection:
    """按结构排序，并截断到 `budget` 条。

    Args:
        evidence: 池子（`state.evidence`，全量、原序）
        located:  `{"drug":…, "loincs":[…]}`，没定位到就 None
        budget:   生成层最多看几条。`None` = 不截断（只排序）

    ⚠️ **只排序不截断时，本层对"生成层挑错"是无效的** ——
       实测：gold 与问题字面零重合，而干扰项重合 1~2 分，
       打分排序改不了"谁的分数高"，只有**把干扰项截掉**才改得了候选集。
       所以 budget 是本层唯一真正起作用的旋钮，**它必须标定，不能拍**（判据 ③ 扫）。
    """
    pool = list(evidence)
    scope = {d.lower() for d in focus_drugs(question, known_drugs)}
    loincs = set((located or {}).get("loincs") or [])

    # 稳定排序：先结构档位，再章节权威度，最后**保持原检索序**（同分不抖动）
    idx = {id(d): i for i, d in enumerate(pool)}
    if order == "struct":
        pool.sort(key=lambda d: (-struct_rank(d, scope, loincs),
                                 -(d.authority or 0.0),
                                 idx[id(d)]))
    elif order == "reverse":
        # ⭐ **对抗档**：把池子倒过来。用来验"提升是不是靠 mock 的『同分取靠前』捡来的"
        #    —— 见 scripts/eval_select.py ⑥。不是产品路径，是判据用的。
        pool.reverse()
    else:
        raise ValueError(f"未知 order={order!r}（只有 struct / reverse）")

    n_keep = len(pool) if budget is None else max(0, min(int(budget), len(pool)))
    kept, rest = pool[:n_keep], pool[n_keep:]

    dropped = [Dropped(cite_key=d.cite_key,
                       struct=struct_rank(d, scope, loincs),
                       reason="over_budget")
               for d in rest]

    return Selection(kept=kept, dropped=dropped, n_pool=len(pool), budget=budget,
                     focus=sorted(scope), loincs=sorted(loincs),
                     n_other_drug=sum(1 for d in pool
                                      if scope and (d.drug or "").lower() not in scope))


__all__ = ["select_evidence", "Selection", "Dropped",
           "struct_rank", "focus_drugs", "budget_from_env", "order_from_env",
           "R_BOTH", "R_SECTION", "R_DRUG", "R_NONE"]
