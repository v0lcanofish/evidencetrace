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
    action          : plan | retrieve | read | synthesize | locate | ask | abstain
    reason_dropped  : context_budget | judged_irrelevant | duplicate | low_authority | None

核心不变量（closure）：
    ⭐ 回答里每一条 cite，都必须能追到某一步的 retrieved —— 否则账本和报告脱节，整个机制作废。

跑自检：
    python ledger/ledger.py
"""

from __future__ import annotations

import sys

# ⚠️ Windows 中文控制台默认 GBK：不设这个，print("⭐") 会抛 UnicodeEncodeError
#    → **判据崩在半路，红绿一个字都读不到**（2026-09-18 实测 eval_retrieval.py）。
#    errors="replace"：宁可显示问号，也不许判据跑到一半死掉。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------- 枚举（封闭）

ACTION_PLAN = "plan"
ACTION_RETRIEVE = "retrieve"
ACTION_READ = "read"
ACTION_SYNTHESIZE = "synthesize"

# ---- 块 E7 新增：agent 自己选的动作 ----
# ⚠️ agent 说的 `search` / `answer` **不新增枚举** —— 它们就是 retrieve / synthesize，
#    只是换了个更贴合 agent 语境的叫法。理由：三层归因（dropped / attribute）
#    整条链都挂在 retrieve 上，为改名复制一份归因逻辑，两份迟早不一致。
#    这里只补【真正新增】的三个：定位、问用户、拒答。
ACTION_LOCATE = "locate"        # 问题 → 该查哪一节（买路钱，还没拿到证据）
ACTION_ASK = "ask"              # 问用户（证据之外的第二个信息源）
ACTION_ABSTAIN = "abstain"      # 拒答（拒答本身是一个决策，不是"没答"）

ACTIONS = (ACTION_PLAN, ACTION_RETRIEVE, ACTION_READ, ACTION_SYNTHESIZE,
           ACTION_LOCATE, ACTION_ASK, ACTION_ABSTAIN)

DROP_CONTEXT_BUDGET = "context_budget"
DROP_IRRELEVANT = "judged_irrelevant"
DROP_DUPLICATE = "duplicate"
DROP_LOW_AUTHORITY = "low_authority"
DROP_REASONS = (DROP_CONTEXT_BUDGET, DROP_IRRELEVANT, DROP_DUPLICATE, DROP_LOW_AUTHORITY)

SOURCE_TYPES = ("dailymed_spl", "pubmed", "openfda", "other")

# ⭐ E10：`first_retrieved_at_step` 取这个值 = **这条引用来自上一轮**，不是本轮检索的。
#    ⚠️ 用一个**显式的负数哨兵**而不是 None，是因为两者含义完全相反：
#       None = 压根追不到出处（该报错）｜ CARRIED_STEP = 追得到，只是上一轮（合法）。
#       合并的话，闭包就分不清"多轮继承"和"凭空编造"了。
CARRIED_STEP = -1


class LedgerError(ValueError):
    """账本的完整性约束被违反。"""


# ---------------------------------------------------------------- 实体


@dataclass
class Doc:
    """一份被检索到的证据（这里是 DailyMed 说明书的一个章节）。"""

    doc_id: str                     # ⭐ 主键：DailyMed 的 setid（UUID）
    section: str                    # 章节名（人读）
    loinc: Optional[str] = None     # ⭐ LOINC 章节码（机械可判）
    drug: str = ""                  # 药名。E7 的 search 要按「哪份药的哪一节」限定候选集，
                                    # 只有 loinc 不够 —— 所有药都有 34073-7 那一节
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
        # 带 query 的动作：retrieve 查资料 / locate 定位章节 / ask 问用户，
        # 都是「拿一个字符串去换东西」，没有 query 就无从复盘"当时问了什么"
        if self.action in (ACTION_RETRIEVE, ACTION_LOCATE, ACTION_ASK) and not self.query:
            raise LedgerError(f"step {self.step}: {self.action} 动作必须有 query")
        # 拒答必须写理由：拒答是一个【决策】，不是"什么都没发生"。
        # 不写理由，事后无法区分「证据不够该拒」和「模型偷懒拒了」。
        if self.action == ACTION_ABSTAIN and not self.note:
            raise LedgerError(f"step {self.step}: abstain 必须写清理由（note）")
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
        # ⭐ E10：跨轮继承来的证据（多轮会话里上一轮查到的）。
        #
        #    **为什么闭包必须认它们**：E10 让第 2 轮继承第 1 轮的证据，
        #    于是第 2 轮**不会重查**，账本里自然没有那一步 —— 闭包一查就失败。
        #    而那会把 E9 刚立的"核验不放松"变成"核验把正常情况也拦了"。
        #
        #    ⚠️ **但也不能默默放行**：所以用 `CARRIED_STEP` 这个**显式哨兵值**
        #       标记"来自上一轮"，让它在报告里看得见 ——
        #       而不是让它长得像本轮检索的，那才是静默失真。
        self.carried: List[Doc] = []

    def mark_carried(self, docs) -> int:
        """登记跨轮继承的证据，返回**真正新增**的条数。"""
        have = {d.cite_key for d in self.carried}
        fresh = [d for d in docs if d.cite_key not in have]
        self.carried.extend(fresh)
        return len(fresh)

    def carried_keys(self) -> set:
        return {d.cite_key for d in self.carried}

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

        def matches(d) -> bool:
            if d.doc_id != doc_id:
                return False
            # 药对了但章节不对 → 不算找到（引入 LOINC 的全部意义就在这）
            return not (loinc and d.loinc and d.loinc != loinc)

        for s in self.steps:
            for d in s.retrieved:
                if matches(d):
                    return s.step
        # ⭐ E10：本轮没有 → 查**上一轮继承来的**。返回哨兵值，不是 None
        #    （None 的含义是"压根追不到"，两者绝不能混）。
        for d in self.carried:
            if matches(d):
                return CARRIED_STEP
        return None

    def retrieved_keys(self) -> set:
        """
        本轮检索到的**完整引用键**（`setid#LOINC`）。

        ⚠️⚠️ 归因必须用它，**不能用只比 doc_id 的版本**。
            一份说明书有多节共用同一个 setid：

                5a709591-...#34073-7  ← Drug Interactions
                5a709591-...#34070-3  ← Contraindications

            只比 doc_id 的话，「引了 A 药但**章节写错**」会被判成"召回到过"，
            于是归因会把病根指到**推理层**（"证据齐了还错"），
            **而真实原因是引错了章节 —— 归因指错层，建议就指错方向。**

            这个洞 E2 在 `_first_step_of` 里修过一次（引入 LOINC 的意义就在这），
            **但 `attribute` 这条路漏了** —— 2026-09-18 做 E13 接归因时才发现，
            是**同型复发**。
        """
        out = set()
        for s in self.steps:
            out.update(d.cite_key for d in s.retrieved if d.loinc)
        return out

    def cited_keys(self) -> set:
        """回答里引用的**完整引用键**（带 LOINC）。**不许把 loinc 丢掉。**"""
        out = set()
        for c in self.claims:
            out.update(k for k in c.cite if "#" in k)
        return out

    def carried_keys_all(self) -> set:
        """跨轮继承来的引用键（E10）。归因时也要算"召回到过"。"""
        return self.carried_keys()

    def retrieved_doc_ids(self) -> set:
        """⚠️ 只到 doc_id 粒度 —— **归因不要用它**，用 `retrieved_keys()`。"""
        out = set()
        for s in self.steps:
            out.update(d.doc_id for d in s.retrieved)
        return out

    def cited_doc_ids(self) -> set:
        """⚠️ 只到 doc_id 粒度（**会丢掉 LOINC**）—— 归因不要用它。"""
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

    def attribute(self, gold_cite_keys, correct: bool) -> Dict[str, Any]:
        """
        对一条回答做三层归因。**链式判定，第一个不达标的层就是病根。**

        Args:
            gold_cite_keys: 这道题的 gold 证据集，**完整引用键**（`setid#LOINC`）集合。
                            ⚠️ 不接受 doc_id 粒度 —— 见 `retrieved_keys()` 的注释。
            correct:        这条回答是否正确
        Returns:
            {"layer": "retrieval"|"utilization"|"reasoning"|"none", ...}

        ⭐ **链式顺序不能乱**：检索层最早，其次是利用层，最后才是推理层。
           "检索缺一条 **且** 引用也缺一条"的错题，病根是**检索** ——
           后面那层的缺失是它的**后果**，不是独立的问题。
        """
        gold = set(gold_cite_keys)
        retrieved = self.retrieved_keys() | self.carried_keys_all()
        got = retrieved & gold
        cited = self.cited_keys() & gold

        recall = len(got) / len(gold) if gold else 1.0
        utilization = len(cited) / len(got) if got else 0.0

        if correct:
            return {"layer": "none", "recall": recall, "utilization": utilization,
                    "evidence": "回答正确"}

        # ① 检索层：gold 证据没被找到
        if not gold.issubset(retrieved):
            missing = sorted(gold - retrieved)
            return {"layer": "retrieval", "recall": recall, "utilization": utilization,
                    "evidence": f"gold {len(gold)} 条，只召回了 {len(got)} 条；"
                                f"缺 {missing}",
                    "suggestion": "改 query 生成 / 扩召回"}

        # ② 利用层：找到了但没进回答
        if not gold.issubset(cited):
            missing = sorted(gold - cited)
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
    GOLD = {d1.cite_key, d2.cite_key}      # ⚠️ 用 cite_key，不用 doc_id
    r = lg.attribute(GOLD, correct=False)
    assert r["layer"] == "utilization", r
    assert "context_budget" in r["evidence"], r["evidence"]

    # ---- 5. 检索层：gold 里有一条根本没召回
    lg2 = Ledger("run-2", "q", seed=0)
    lg2.record(Step(step=1, action="retrieve", query="x", retrieved=[d1],
                    used_in_report=True))
    lg2.add_claim(Claim(claim_id="c1", text="t", cite=[d1.cite_key]))
    r2 = lg2.attribute(GOLD, correct=False)
    assert r2["layer"] == "retrieval", r2

    # ---- 6. 推理层：证据全齐还是错
    lg3 = Ledger("run-3", "q", seed=0)
    lg3.record(Step(step=1, action="retrieve", query="x", retrieved=[d1, d2],
                    used_in_report=True))
    lg3.add_claim(Claim(claim_id="c1", text="t", cite=[d1.cite_key, d2.cite_key]))
    r3 = lg3.attribute(GOLD, correct=False)
    assert r3["layer"] == "reasoning", r3

    # ---- 6b. ⭐⭐ **引了同一个药的错误章节** → 必须报 retrieval，不是 reasoning
    #      （这个洞 E2 在闭包那边修过一次，`attribute` 这条路上漏了 —— 2026-09-18 同型复发）
    #      gold 是 34070-3 那一节，agent 引的是 34073-7 那一节 —— 同一个药、错的章节
    d_wrong = Doc(doc_id="aaaa-1111", section="Drug Interactions", loinc="34073-7", text="...")
    lg6 = Ledger("run-6", "q", seed=0)
    lg6.record(Step(step=1, action="retrieve", query="x", retrieved=[d_wrong],
                    used_in_report=True))
    lg6.add_claim(Claim(claim_id="c1", text="t", cite=[d_wrong.cite_key]))
    r6 = lg6.attribute({d2.cite_key}, correct=False)      # gold = 34070-3
    assert r6["layer"] == "retrieval", \
        f"引错章节该报 retrieval（gold 章节没被召回），实际 {r6['layer']} —— " \
        f"只比 doc_id 的话这里会误报成 reasoning，**归因指错层**"
    # 反过来：引对了章节，就不该报 retrieval
    lg7 = Ledger("run-7", "q", seed=0)
    lg7.record(Step(step=1, action="retrieve", query="x", retrieved=[d2],
                    used_in_report=True))
    lg7.add_claim(Claim(claim_id="c1", text="t", cite=[d2.cite_key]))
    assert lg7.attribute({d2.cite_key}, correct=True)["layer"] == "none"

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
