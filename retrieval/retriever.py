# -*- coding: utf-8 -*-
"""
块 E6 · 检索器 —— 给 agent 装上"眼睛"。

    from retrieval.retriever import BM25Retriever, SectionLocator

接口对齐现有的 `MockRetriever`：`Retriever = Callable[[str, int], List[Doc]]`
→ **可以直接替换，编排代码一行不用改**。

━━━ 三层，每层都能单独开关（为了量出各自的贡献）━━━

    ① BM25        关键词匹配      —— 基线
    ② 章节定位    问题 → LOINC     —— 本项目的独有能力
    ③ 稠密检索    语义匹配        —— 后面加（需要 BGE-M3）

    ⭐ 每加一层就量一次 Recall@5，差值就是那一层的贡献。
       这就是"从 X 到 Y"的数字，是每一层贡献的 Trade-off 依据。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from retrieval.chunker import Chunk, build_chunks, to_doc

# ---------------------------------------------------------------- BM25
# 复用 `代码库/docs/八股资料/search_retriever.py` 的实现（本仓设计决策：
# 不装 rank-bm25，自己写 —— 零依赖、可复现、英文 tokenizer 正好适配 DailyMed）。
# 这里改成**按 chunk 索引**并返回下标而不是字符串。

_TOK = re.compile(r"[a-z0-9]+")

# ⚠️ 缩写要**在切词之前**剥掉 —— 这是 E7 那个洞的**同一族**，2026-09-18 第二次咬人。
#
#   E7 修的是「整词停用词」：`in/of/the/is` 在每个 chunk 里都有 → 假命中。
#   但 `[a-z0-9]+` 遇到撇号会**从撇号处切断**：
#
#       "What's the stock price of Apple?"  →  ['what', 's', 'the', 'stock', ...]
#                                              ↑ what 被停用词表丢了，**'s' 的残渣漏了进来**
#
#   而 `s` 是所有格残渣（`patient's` / `drug's`），**几乎每个 chunk 里都有**。
#   实测后果：域外问题 `What's the stock price of Apple?` 返回 **46 条证据**，
#   而且 `bm25_confidence` = 0.244 **不为 0**（被 stock/price/apple 抬起来了）
#   → 弃权门限失效 → agent 又答域外问题。
#
#   ⇒ 修法不是在停用词表里加一个 `'s'`（那治不了 `patient's → patient`——
#     我们**希望** `patient` 留下），而是**先把缩写后缀从词干上摘掉**。
#
#   ⚠️ 不顺手把「单字符 token」全删掉：`5 / 1 / 2` 这些是 SPL 的**章节编号**，
#     是正文里的真内容（`( 5.14)`），删了会丢信号。
_CLITIC = re.compile(r"['’](s|re|ve|ll|d|m)\b")     # what's → what ｜ I'd → I
_NOT = re.compile(r"n['’]t\b")                       # don't → do（do 是停用词）

# ⚠️ 停用词必须去掉（块 E7 接 agent 时才暴露的真缺陷）。
#
# 原来不过滤，于是查询 "What is the price of tea in China?" 会被切成
#   {what, is, the, price, of, tea, in, china}
# 其中 in/of/the/is 在**每一个** chunk 里都出现 → BM25 分数 > 0 → 返回一批结果。
# 后果不是"稍微不准"，而是 **agent 永远观察不到「我什么都没查到」这个信号**：
# 它看到"有 5 条证据"就以为查到了，于是域外问题也硬答。
#
# ⭐ 注意这不是"调阈值"，是修一个**词法层面的错误**：
#    虚词不携带检索意图，让它们参与匹配只会制造假命中。
#    （真·相关性阈值是另一件事，属于 E8 的活，要标定，不能拍。）
STOPWORDS = frozenset("""
a an the and or but if then than that this these those there here
is are was were be been being am do does did done doing
of to in on at by for with from into over under about as
i me my we our you your he she it they them his her its their
what which who whom whose when where why how
can could should would shall will may might must
have has had having not no nor so such only own same too very
""".split())


def tokenize(s: str, drop_stop: bool = True) -> List[str]:
    s = (s or "").lower()
    s = _NOT.sub("", s)        # 先处理 n't（don't → do，do 是停用词会被丢掉）
    s = _CLITIC.sub("", s)     # 再摘所有格/缩写后缀（what's → what，patient's → patient）
    out = _TOK.findall(s)
    return [w for w in out if w not in STOPWORDS] if drop_stop else out


class ChunkBM25:
    """BM25 over chunks。k1/b 用标准值（1.5 / 0.75）。"""

    def __init__(self, texts: List[str], k1: float = 1.5, b: float = 0.75):
        self.N = len(texts)
        self.tok = [tokenize(t) for t in texts]
        self.dl = [len(t) for t in self.tok]
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        self.tf = [Counter(t) for t in self.tok]
        df: Counter = Counter()
        for t in self.tok:
            df.update(set(t))
        self.idf = {w: math.log((self.N - df[w] + 0.5) / (df[w] + 0.5) + 1.0)
                    for w in df}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """返回 [(chunk 下标, 分数), ...]，按分数降序。"""
        q = tokenize(query)
        scored: List[Tuple[int, float]] = []
        for i in range(self.N):
            tf, dl = self.tf[i], self.dl[i]
            s = 0.0
            for w in q:
                f = tf.get(w)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                    s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / denom
            if s > 0:
                scored.append((i, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def score_all(self, query: str) -> Dict[int, float]:
        return dict(self.search(query, k=self.N))


# ---------------------------------------------------------------- 章节定位

# 问题里的说法 → 它是想问哪一类（意图）
# ⚠️ 顺序有意义：先匹配具体的，再匹配泛的。
LOCATE_RULES: List[Tuple[str, List[str]]] = [
    # ⚠️ 顺序有意义：先匹配具体的，再匹配泛的。
    #    2026-09-17 补了两类真实问法（块 E7 接 agent 时暴露的漏检）：
    #      · 「does X mess with Y」—— 口语化的相互作用问法，原来只有 mix/combine
    #      · 「safe to take / during pregnancy」—— 原来是 is it safe，漏了这两种搭配
    #    补词前查过语料：妊娠类问题在这些说明书里确实落在禁忌节，所以加 "pregnan"
    #    不会把答案指错节。（加词前先查语料，别凭语感加。）
    # ⚠️ 9/19 补 **一个**词形：表里的 "take it with" 匹配不到 "taking with it"，
    #    于是 "…what other medicines should I avoid taking with it?" 掉到
    #    contraindication（"should i avoid"），**相互作用节整个丢了**（它是 gold 之一）。
    #
    #    ⛔ 我一度还加了 "other medicines / other drugs / other medications" —— **撤掉了**。
    #       理由不是没用，是**它会打坏探针的负控**：
    #       探针集里有 6 条故意用绕开关键词表的措辞（"Does X play nicely with other
    #       medicines?"），补这三个词之后它们全被认出，`eval_retrieval` 的判据
    #       「探针上的意图全部定位不出来」当场变红 —— 而那句断言的 detail 写着
    #       「不为 0 说明探针模板撞上了关键词表 → **探针失效**」。
    #       ⭐ 补词会**同时**提高覆盖率、削弱负控。补之前先跑探针，别顺手加。
    ("interaction", ["interact", "together with", "combine", "combining", "mix",
                     "mixing", "take it with", "taking with", "coadmin", "on top of",
                     "mess with", "mess up", "safe to take with"]),
    ("contraindication", ["contraindicated", "off-limits", "safe for someone",
                          "should i avoid", "if i have", "can i take", "can i still take",
                          "is it safe", "safe to take", "safe during",
                          "pregnan", "breastfeed", "nursing"]),
    ("dosage", ["how much", "how many", "dose", "dosage", "milligram", "mg ",
                "how should i take", "starting dose", "recommended dose"]),
    ("precaution", ["watch out", "careful", "warning", "precaution", "keep an eye",
                    "red flag", "monitor"]),
    # ⚠️ 下面三类是**我们知道它存在、但语料里故意没采**的章节。
    #    识别它们不是为了检索（检索不到），是为了让 agent 能说
    #    "你问的是副作用那一节 —— 我这份资料里没有" ，而不是拿别的章节硬答。
    #
    #    实测背景（E8）：`What are the side effects of warfarin?` 原来识别不出意图，
    #    但 `side`/`effects` 两个词在说明书正文里到处都是 → 实词槽位全填上
    #    → 覆盖度 0.75 → 作答。**12 条 missing_section 漏了 5 条。**
    ("adverse_reactions", ["side effect", "side effects", "adverse reaction",
                           "adverse reactions", "side-effect"]),
    ("clinical_studies", ["clinical trial", "clinical study", "clinical studies",
                          "trial data"]),
    ("mechanism", ["how does it work", "how it works", "mechanism of action",
                   "how does this work"]),
    ("indication", ["used for", "prescribe", "treat", "what is it for",
                    "what does it do", "why was i put on"]),
]

# 意图 → 该查哪一节（可能多节，因为不同药的说明书结构不一样）
INTENT_TO_LOINC: Dict[str, List[str]] = {
    "interaction": ["34073-7"],
    "contraindication": ["34070-3"],
    "dosage": ["34068-7"],
    "indication": ["34067-9"],
    "precaution": ["34069-5", "34071-1", "43685-7"],
    # 这三节的 LOINC 是**真实存在但本语料没采集**的 —— 定位到它们 = 没资料可引。
    # ⭐ 这正是「拒答要有理由」的机械依据：不是"检索失败"，是"这一节我们没采"。
    "adverse_reactions": ["34084-4"],
    "clinical_studies": ["34090-1"],
    "mechanism": ["34083-6"],
}


class SectionLocator:
    """
    ⭐ **本项目的独有能力**：问题 → 该查哪一节。

    DailyMed 的章节带 LOINC 医学编码，所以"该查哪一节"不是靠猜，
    而是**按结构定位**。而且可解释：
        "我引用了相互作用那一节，因为你在问能不能一起吃。"

    ⚠️ 要点：**同一类问题，不同药可能落在不同的 LOINC 上**
       （warfarin 把警告和注意事项合成一节 43685-7，别的药分成两节）。
       所以映射时必须**看这份药实际有哪些节**，不能硬套词表。
    """

    def __init__(self, labels: List[Dict[str, Any]]):
        # {drug: {loinc, ...}} —— 这份药实际有哪些节
        self.have: Dict[str, set] = {
            L["drug"]: {s["loinc"] for s in L.get("sections", [])} for L in labels
        }
        self.drugs = sorted(self.have)

    def detect_intents(self, query: str) -> List[str]:
        """
        ⭐ 问题里命中的**全部**意图（按 `LOCATE_RULES` 顺序，去重）。

        ⚠️ 9/19 之前这里只返回**第一个**命中就收工（`detect_intent`），
           于是「一句话问两件事」的复合问句会被压成一个意图：

               "I'm taking allopurinol. What's the usual dose, and what other
                medicines should I avoid taking with it?"
                 → 只认了 "should i avoid" ⇒ intent=contraindication
                 → loincs=['34070-3']（禁忌节），**剂量节 34068-7 整个丢了**

           后果不是"排错序"，是**gold 压根进不了候选池** ——
           因为策略是拿 `loc['loincs']` 去构造 `restrict` 的
           （`agent/policy.py` 那三处）。池子里没有，后面排序怎么排都救不回来。

           ⚠️ 这个洞是造 `hard_set`（每题 gold=2、一句话问两件事）才踩出来的：
              原来那套 60 题**都是一句话一件事**，所以从没暴露。
        """
        q = (query or "").lower()
        return [intent for intent, kws in LOCATE_RULES if any(k in q for k in kws)]

    def detect_intent(self, query: str) -> Optional[str]:
        """兼容旧接口：**优先级最高**的那个意图（= `detect_intents` 的第一个）。"""
        hits = self.detect_intents(query)
        return hits[0] if hits else None

    def find_drugs(self, query: str) -> List[str]:
        """
        ⭐ 问题里点名的**全部**药品（按在问句里出现的先后）。

        ⚠️ 9/19 之前只有 `find_drug`，**只返回第一个**。于是
           "Both alprazolam and fluoxetine are on my medicine list…" 这种**跨药问题**
           只会限定到一个药，另一个药的证据被 `restrict` 硬过滤挡在池子外 ——
           而它正是 gold 之一。

           实测（硬题集 cross_drug 那 86 道）：gold 进池率只有 41%；
           同批 multi_section（单药两节）是 91%。差距就出在这。
        """
        q = (query or "").lower()
        hits = [(q.index(d), d) for d in self.drugs if d in q]
        return [d for _, d in sorted(hits)]

    def find_drug(self, query: str) -> Optional[str]:
        """兼容旧接口：**最先出现**的那个药（= `find_drugs` 的第一个）。"""
        hits = self.find_drugs(query)
        return hits[0] if hits else None

    def __call__(self, query: str) -> Dict[str, Any]:
        """
        返回定位结果：
            {"drug": ..., "intent": ..., "loincs": [...], "reason": "..."}
        定位不出来时 drug/intent 为 None（交给 BM25 兜底）。
        """
        intents = self.detect_intents(query)
        drugs = self.find_drugs(query)

        # ⭐ 兜底规则：**一个意图线索都没有、但点名了 ≥2 个药** → 按相互作用处理。
        #
        # 为什么需要：跨药题的自然问法（"Both A and B are on my medicine list.
        #   Does each label warn about the other one?"）**一个关键词都不命中**，
        #   于是 loincs 为空、没有章节导向，两个药的块在 BM25 里争 top-k
        #   —— 实测这类题 gold 进池率只有 55%。
        #
        # ⚠️ 为什么加 `not intents` 这个前提：**不加会误伤**。
        #   实测有禁忌类问题句子里带第二个药名（"Can I take esomeprazole if I have
        #   hypersensitivity to …"），裸规则会给它硬塞一个 interaction。
        #   加上"本来就没认出任何意图"之后，影响面收敛到**只有跨药题**：
        #     检索集/探针集/边界集/定位集/意图集/e2e  **各 0 条受影响**
        #     只有 hard_set 的 86 条 cross_drug 触发
        #   ⭐ 这就是"补规则之前先跑全题集量影响面"—— 上一条关键词我正是漏了这一步，
        #     把探针的负控打红了（见 LOCATE_RULES 里的记录）。
        #
        # ⚠️ 另外，现在 loincs 是**加权不是硬过滤**（见 hybrid.py），
        #   所以这条规则就算偶尔判错，代价也只是"排得靠前一点"，不会把别的章节挡在外面。
        if not intents and len(drugs) >= 2:
            intents = ["interaction"]

        drug = drugs[0] if drugs else None
        # 章节取**所有点名药品**的并集 —— 跨药问题两边都要查
        pool_drugs = drugs or ([drug] if drug else [])
        loincs: List[str] = []
        # ⭐ 把**每个**命中的意图对应的章节并起来（一句话问两件事 = 两节都要查）
        for intent in intents:
            cand = INTENT_TO_LOINC.get(intent, [])
            if pool_drugs:
                # ⭐ 只保留**这些药实际有的**那些节
                cand = [x for x in cand
                        if any(d in self.have and x in self.have[d] for d in pool_drugs)]
            for x in cand:
                if x not in loincs:
                    loincs.append(x)
        reason = ""
        if intents and drug and loincs:
            reason = f"「{'/'.join(intents)}」类问题 + 药品 {drug} → 命中章节 {loincs}"
        elif intents:
            reason = (f"「{'/'.join(intents)}」类问题，但"
                      f"{('药品 ' + drug + ' 没有对应章节') if drug else '没识别出药品'}")
        return {"drug": drug,                                # 兼容旧字段（最先出现的那个）
                "drugs": drugs,                              # ⭐ 新增：全部点名药品
                "intent": intents[0] if intents else None,   # 兼容旧字段（最高优先级那个）
                "intents": intents,                          # ⭐ 新增：全部命中
                "loincs": loincs, "reason": reason}


# ---------------------------------------------------------------- 检索器


class BM25Retriever:
    """
    BM25 检索器（可选叠加章节定位）。

    Args:
        use_locator: ⭐ **消融开关**。
            True  → 章节定位命中的 chunk 加权（×locate_boost）
            False → 纯 BM25 基线
        两者的 Recall@5 差值，就是「章节定位」这一层的贡献。
    """

    def __init__(self, labels: List[Dict[str, Any]],
                 use_locator: bool = True,
                 locate_boost: float = 3.0,
                 target_chars: int = 420):
        self.chunks: List[Chunk] = build_chunks(labels, target_chars=target_chars)
        self.index: Dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}
        self.bm25 = ChunkBM25([c.text for c in self.chunks])
        self.locator = SectionLocator(labels)
        self.use_locator = use_locator
        self.locate_boost = locate_boost
        self.last_trace: Dict[str, Any] = {}

    # ---- 引用核验用：按 chunk_id 取回原块
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self.index.get(chunk_id)

    def search_detail(self, query: str, k: int = 5,
                      restrict: Optional[Dict[str, Any]] = None
                      ) -> Tuple[List[Chunk], Dict[str, Any]]:
        """返回 (top-k chunk, 定位轨迹)。轨迹要写进账本 —— 可解释性靠它。

        Args:
            restrict: {"drug": str|None, "loincs": [...]} —— **先卡候选集再排序**。

        ⭐ 两种用法必须分清（这是块 E7 接 agent 时才暴露的）：

            加权（locate_boost）  定位命中的章节**加分**，但全局仍可竞争
                                  → 适合"定位可能错，留条后路"
            限定（restrict）      候选集**只剩**这些章节
                                  → 适合 agent 已经决定"就查这一节"

            ⚠️ 差别不是风格问题：如果 agent 说"查禁忌那节"，我们却先全局排序再过滤，
               禁忌节里那些全局排名靠后的 chunk 会被 top-k 截掉 ——
               agent 会收到"这节没东西"的**错误信号**，然后要么重复查、要么误判拒答。
        """
        loc = self.locator(query) if self.use_locator else {"drug": None, "intent": None,
                                                           "loincs": [], "reason": "定位关闭"}
        scores = self.bm25.score_all(query)

        if restrict:
            allow = self._allowed_indices(restrict)
            scores = {i: s for i, s in scores.items() if i in allow}
        elif self.use_locator and loc["loincs"]:
            for i, c in enumerate(self.chunks):
                if c.loinc in loc["loincs"] and c.doc_id in self._doc_ids_for(loc, query):
                    scores[i] = scores.get(i, 0.0) * self.locate_boost

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        self.last_trace = {"query": query, "locate": loc, "restrict": restrict,
                           "n_candidates": len(scores), "k": k}
        return [self.chunks[i] for i, _ in ranked], self.last_trace

    def _allowed_indices(self, restrict: Dict[str, Any]) -> set:
        """限定模式下，哪些 chunk 在候选集里。"""
        drug = restrict.get("drug")
        loincs = set(restrict.get("loincs") or [])
        return {i for i, c in enumerate(self.chunks)
                if (not drug or c.drug == drug) and (not loincs or c.loinc in loincs)}

    def _doc_ids_for(self, loc: Dict[str, Any], query: str) -> set:
        """定位到的那份药的 setid 集合（药名没识别出来时 = 全部，不加权）。"""
        d = loc.get("drug")
        if not d:
            return set()
        return {c.doc_id for c in self.chunks if c.drug == d}

    def __call__(self, query: str, k: int = 5, restrict: Optional[Dict[str, Any]] = None):
        """Retriever 接口：query, k → List[Doc]

        `restrict` 是 E7 加的第三个参数，**默认 None → 老调用点行为不变**。
        """
        chunks, _ = self.search_detail(query, k, restrict=restrict)
        return [to_doc(c, rank=i + 1, score=1.0 / (i + 1)) for i, c in enumerate(chunks)]


__all__ = ["ChunkBM25", "SectionLocator", "BM25Retriever", "tokenize",
           "INTENT_TO_LOINC", "LOCATE_RULES"]
