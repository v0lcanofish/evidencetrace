# -*- coding: utf-8 -*-
"""
证据账本 Evidence Ledger —— EvidenceTrace 的核心数据结构。

它要解决的问题：
    最终报告是【黑盒结果】—— 只能告诉你「答得不好」，不能告诉你「为什么不好」。
    账本把每一步的证据状态记下来，让「为什么」变成可机械判定的问题。

三个实体，靠 doc_id 串起来：
    Step（一步动作）──retrieved──▶ Doc（说明书章节）
                                    │ doc_id 是主键
                                    ▼
    Claim（回答里的一条建议）──cite──▶ Doc

两个枚举字段（必须封闭，否则归因无法机械判定）：
    action          : plan | retrieve | read | synthesize
    reason_dropped  : context_budget | judged_irrelevant | duplicate | low_authority | None

核心不变量（closure）：
    ⭐ 回答里每一条 cite，都必须能追到某一步的 retrieved —— 否则账本和报告脱节，整个机制作废。

跑自检：
    python ledger/ledger.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------- 枚举（封闭）

ACTION_PLAN = "plan"
ACTION_RETRIEVE = "retrieve"
ACTION_READ = "read"
ACTION_SYNTHESIZE = "synthesize"
ACTIONS = (ACTION_PLAN, ACTION_RETRIEVE, ACTION_READ, ACTION_SYNTHESIZE)

DROP_CONTEXT_BUDGET = "context_budget"
DROP_IRRELEVANT = "judged_irrelevant"
DROP_DUPLICATE = "duplicate"
DROP_LOW_AUTHORITY = "low_authority"
DROP_REASONS = (DROP_CONTEXT_BUDGET, DROP_IRRELEVANT, DROP_DUPLICATE, DROP_LOW_AUTHORITY)

SOURCE_TYPES = ("dailymed_spl", "pubmed", "openfda", "other")


class LedgerError(ValueError):
    """账本的完整性约束被违反。"""


# ---------------------------------------------------------------- 实体


@dataclass
class Doc:
    """一份被检索到的证据（这里是 DailyMed 说明书的一个章节）。"""

    doc_id: str                     # ⭐ 主键：DailyMed 的 setid（UUID）
    section: str                    # 章节名（人读）
    loinc: Optional[str] = None     # ⭐ LOINC 章节码（机械可判）
    text: str = ""
    source_type: str = "dailymed_spl"
    title: str = ""
    rank: int = 0
    score: float = 0.0
    authority: float = 0.0          # 权威性 0-1
    recency: float = 0.0            # 时效性 0-1
    verifiable: bool = True         # 是否可核验

    def __post_init__(self):
        if not self.doc_id:
            raise LedgerError("Doc.doc_id 不能为空（它是账本的主键）")
        if self.source_type not in SOURCE_TYPES:
            raise LedgerError(f"未知 source_type: {self.source_type}")

    @property
    def cite_key(self) -> str:
        """引用键：回答里就该写这个。带章节码，保证「引的是哪一节」也可核验。"""
        return f"{self.doc_id}#{self.loinc}" if self.loinc else self.doc_id


@dataclass
class Step:
    """一步动作。四类动作的字段要求不同，见 validate()。"""

    step: int
    action: str
    query: Optional[str] = None
    retrieved: List[Doc] = field(default_factory=list)
    used_in_report: Optional[bool] = None      # 只对 retrieve 有意义
    reason_dropped: Optional[str] = None       # ⭐ 利用层归因的唯一依据
    tokens_after: Optional[int] = None
    note: str = ""

    def validate(self) -> None:
        if self.action not in ACTIONS:
            raise LedgerError(f"step {self.step}: 未知 action={self.action!r}")
        if self.reason_dropped is not None and self.reason_dropped not in DROP_REASONS:
            raise LedgerError(f"step {self.step}: 未知 reason_dropped={self.reason_dropped!r}")
        if self.action == ACTION_RETRIEVE and not self.query:
            raise LedgerError(f"step {self.step}: retrieve 动作必须有 query")
        if self.used_in_report is False and self.reason_dropped is None:
            raise LedgerError(
                f"step {self.step}: 检索到了但没用进回答，必须给 reason_dropped "
                f"（否则「利用层」归因无从判定）"
            )


@dataclass
class Claim:
    """回答里的一条论断。"""

    claim_id: str
    text: str
    cite: List[str] = field(default_factory=list)   # 引用键列表
    first_retrieved_at_step: Optional[int] = None   # ⭐ 归因的直接判据
    verifiable: Optional[bool] = None


# ---------------------------------------------------------------- 账本


class Ledger:
    """一次完整研究的账本。所有写入都走 record_* / add_claim，不直接改内部列表。"""

    def __init__(self, run_id: str, question: str, seed: int = 42):
        self.run_id = run_id
        self.question = question
        self.seed = seed
        self.steps: List[Step] = []
        self.claims: List[Claim] = []
        self.report_md: str = ""
        self.truncated: bool = False
        self.malformed: bool = False
        self._doc_index: Dict[str, Doc] = {}        # doc_id -> 第一次出现的 Doc

    # ------------------------------------------------------------ 写

    def record(self, step: Step) -> Step:
        """唯一的写账本入口。不允许绕过（埋点要求）。"""
        step.validate()
        self.steps.append(step)
        for d in step.retrieved:
            # 只记第一次见到的那份（其余是重复检索）
            self._doc_index.setdefault(d.doc_id, d)
        return step

    def add_claim(self, claim: Claim) -> Claim:
        """写一条回答论断。自动回填 first_retrieved_at_step —— 这是归因的判据，不能靠人手填。"""
        for key in claim.cite:
            if claim.first_retrieved_at_step is None:
                s = self._first_step_of(key)
                if s is not None:
                    claim.first_retrieved_at_step = s
        self.claims.append(claim)
        return claim

    # ------------------------------------------------------------ 查

    def _first_step_of(self, cite_key: str) -> Optional[int]:
        """
        找出这条引用第一次出现是在第几步。

        ⚠️ 必须按【完整引用键】匹配，不能只看 doc_id ——
           一份药品说明书有多节，它们共用同一个 setid：

               5a709591-...#34073-7  ← Drug Interactions
               5a709591-...#34070-3  ← Contraindications

           只比 doc_id 的话，「引了 A 药但章节写错」会被判成通过，
           而引入 LOINC 的**全部意义**就是验「引的是不是那一节」。
           （这个漏洞是块 E2 接编排时发现的：去重按 doc_id 做，
             导致同一份药的第二、三节全被过滤掉；顺着查才发现闭包也没验章节。）
        """
        doc_id, _, loinc = cite_key.partition("#")
        for s in self.steps:
            for d in s.retrieved:
                if d.doc_id != doc_id:
                    continue
                if loinc and d.loinc and d.loinc != loinc:
                    continue          # 药对了但章节不对 → 不算找到
                return s.step
        return None

    def retrieved_doc_ids(self) -> set:
        out = set()
        for s in self.steps:
            out.update(d.doc_id for d in s.retrieved)
        return out

    def cited_doc_ids(self) -> set:
        out = set()
        for c in self.claims:
            out.update(k.split("#", 1)[0] for k in c.cite)
        return out

    def dropped(self) -> List[Step]:
        """检索到了但没进回答的步骤（利用层归因的输入）。"""
        return [s for s in self.steps
                if s.action == ACTION_RETRIEVE and s.used_in_report is False]

    # ------------------------------------------------------------ 核心不变量

    def check_closure(self) -> None:
        """
        ⭐ 账本-报告的闭包检查。

        每一条引用都必须能在账本里找到出处，否则「引用可验证率」和「三层归因」都失去意义
        —— 因为追不到来源，就无法判断是检索层、利用层还是推理层的问题。
        """
        problems = []
        for c in self.claims:
            if not c.cite:
                problems.append(f"claim {c.claim_id}: 没有任何引用")
                continue
            for key in c.cite:
                s = self._first_step_of(key)
                if s is None:
                    problems.append(
                        f"claim {c.claim_id}: 引用 {key} 在账本里【追不到】出处"
                        f"（注意：药对了但章节不对也算追不到）")
                elif c.first_retrieved_at_step != s:
                    problems.append(
                        f"claim {c.claim_id}: first_retrieved_at_step={c.first_retrieved_at_step} "
                        f"与实际首次检索步 {s} 不一致")
        if problems:
            raise LedgerError("闭包检查失败：\n  - " + "\n  - ".join(problems))

    # ------------------------------------------------------------ 归因（三层，链式）

    def attribute(self, gold_doc_ids, correct: bool) -> Dict[str, Any]:
        """
        对一条回答做三层归因。**链式判定，第一个不达标的层就是病根。**

        Args:
            gold_doc_ids: 这道题的 gold 证据集（doc_id 集合）
            correct:      这条回答是否正确
        Returns:
            {"layer": "retrieval"|"utilization"|"reasoning"|"none", "evidence": ..., ...}
        """
        gold = set(gold_doc_ids)
        got = self.retrieved_doc_ids() & gold
        cited = self.cited_doc_ids() & gold

        recall = len(got) / len(gold) if gold else 1.0
        utilization = len(cited) / len(got) if got else 0.0

        if correct:
            return {"layer": "none", "recall": recall, "utilization": utilization,
                    "evidence": "回答正确"}

        # ① 检索层：gold 证据没被找到
        if not gold.issubset(self.retrieved_doc_ids()):
            missing = sorted(gold - self.retrieved_doc_ids())
            return {"layer": "retrieval", "recall": recall, "utilization": utilization,
                    "evidence": f"gold {len(gold)} 条，只召回了 {len(got)} 条；"
                                f"缺 {missing}",
                    "suggestion": "改 query 生成 / 扩召回"}

        # ② 利用层：找到了但没进回答
        if not gold.issubset(self.cited_doc_ids()):
            missing = sorted(gold - self.cited_doc_ids())
            reasons = {s.reason_dropped for s in self.dropped()}
            return {"layer": "utilization", "recall": recall, "utilization": utilization,
                    "evidence": f"召回了全套 gold，但只有 {len(cited)} 条进了回答；"
                                f"缺 {missing}；reason_dropped={sorted(r for r in reasons if r)}",
                    "suggestion": "改上下文压缩 / 排序"}

        # ③ 推理层：证据齐了还是答错
        return {"layer": "reasoning", "recall": recall, "utilization": utilization,
                "evidence": "gold 证据全部召回并进了回答，但结论错",
                "suggestion": "换模型 / 加 CoT"}

    # ------------------------------------------------------------ 序列化

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "question": self.question,
            "seed": self.seed,
            "steps": [asdict(s) for s in self.steps],
            "claims": [asdict(c) for c in self.claims],
            "report_md": self.report_md,
            "truncated": self.truncated,
            "malformed": self.malformed,
        }

    def save(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                     encoding="utf-8")
        return p

    @classmethod
    def load(cls, path) -> "Ledger":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        lg = cls(d["run_id"], d["question"], d.get("seed", 42))
        for sd in d["steps"]:
            sd["retrieved"] = [Doc(**x) for x in sd.get("retrieved", [])]
            lg.record(Step(**sd))
        lg.claims = [Claim(**c) for c in d.get("claims", [])]
        lg.report_md = d.get("report_md", "")
        lg.truncated = d.get("truncated", False)
        lg.malformed = d.get("malformed", False)
        return lg


# ---------------------------------------------------------------- 自检


def _selftest() -> int:
    lg = Ledger("run-test", "Can I take ibuprofen with warfarin?", seed=0)

    d1 = Doc(doc_id="aaaa-1111", section="Drug Interactions", loinc="34073-7",
             text="...", rank=1, score=9.1, authority=0.9)
    d2 = Doc(doc_id="bbbb-2222", section="Contraindications", loinc="34070-3",
             text="...", rank=2, score=7.0, authority=0.9)

    lg.record(Step(step=1, action="plan", note="拆成 2 个子问题"))
    lg.record(Step(step=2, action="retrieve", query="ibuprofen warfarin interaction",
                   retrieved=[d1, d2], used_in_report=False,
                   reason_dropped=DROP_CONTEXT_BUDGET))
    lg.add_claim(Claim(claim_id="c1", text="两者同服会增加出血风险。",
                       cite=[d1.cite_key]))
    lg.report_md = "两者同服会增加出血风险 [aaaa-1111#34073-7]。"

    # ---- 1. cite_key 带 LOINC
    assert d1.cite_key == "aaaa-1111#34073-7", d1.cite_key

    # ---- 2. first_retrieved_at_step 自动回填
    assert lg.claims[0].first_retrieved_at_step == 2, lg.claims[0].first_retrieved_at_step

    # ---- 3. 闭包通过
    lg.check_closure()

    # ---- 4. 三层归因：gold 全召回但只用了 1 条 → 利用层
    r = lg.attribute({"aaaa-1111", "bbbb-2222"}, correct=False)
    assert r["layer"] == "utilization", r
    assert "context_budget" in r["evidence"], r["evidence"]

    # ---- 5. 检索层：gold 里有一条根本没召回
    lg2 = Ledger("run-2", "q", seed=0)
    lg2.record(Step(step=1, action="retrieve", query="x", retrieved=[d1],
                    used_in_report=True))
    lg2.add_claim(Claim(claim_id="c1", text="t", cite=[d1.cite_key]))
    r2 = lg2.attribute({"aaaa-1111", "bbbb-2222"}, correct=False)
    assert r2["layer"] == "retrieval", r2

    # ---- 6. 推理层：证据全齐还是错
    lg3 = Ledger("run-3", "q", seed=0)
    lg3.record(Step(step=1, action="retrieve", query="x", retrieved=[d1, d2],
                    used_in_report=True))
    lg3.add_claim(Claim(claim_id="c1", text="t", cite=[d1.cite_key, d2.cite_key]))
    r3 = lg3.attribute({"aaaa-1111", "bbbb-2222"}, correct=False)
    assert r3["layer"] == "reasoning", r3

    # ---- 7. 闭包必须能抓出「引用追不到出处」
    lg4 = Ledger("run-4", "q", seed=0)
    lg4.add_claim(Claim(claim_id="c1", text="t", cite=["ghost-9999#34073-7"]))
    try:
        lg4.check_closure()
        raise AssertionError("闭包检查漏掉了幽灵引用")
    except LedgerError:
        pass

    # ---- 8. 枚举字段必须封闭
    for bad in [
        dict(step=1, action="fly"),
        dict(step=1, action="retrieve", query="q", used_in_report=False),   # 缺 reason
        dict(step=1, action="retrieve", query="q", reason_dropped="whatever"),
    ]:
        try:
            Step(**bad).validate()
            raise AssertionError(f"非法 step 未被拦截: {bad}")
        except LedgerError:
            pass

    # ---- 9. 存盘 / 读回 往返一致
    p = Path(__file__).resolve().parent / "_selftest_ledger.json"
    lg.save(p)
    back = Ledger.load(p)
    assert back.to_dict() == lg.to_dict(), "存盘-读回不一致"
    p.unlink()

    print("ledger self-test: ALL PASS")
    print(f"  闭包 ✓ ｜ 三层归因(检索/利用/推理) ✓ ｜ 枚举校验 ✓ ｜ 序列化往返 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
